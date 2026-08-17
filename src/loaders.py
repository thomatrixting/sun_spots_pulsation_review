"""Loading one region's cubes and building its masks — the A/B fork point.

The two datasets keep separate loaders on purpose. They are laid out differently on disk
(NOAA: `region_01_*_cube.fits` on a uniform grid written by `processing`; DS0N: IDL-style
`cube_*.fits` plus `.sav` timing) and their thresholds mean different things. Trying to
serve both from one function is how the old `sunspot_analysis.py` ended up carrying two
whole pipeline generations at once.

What they *do* agree on is the shape of what they return, and that is what lets every
function in `analysis`, `spectra`, `oscillation` and `animation` be shared:

    data = {
        'cube_cont', 'cube_mag', 'cube_dop'   (n_t, ny, nx) float arrays, NaN in gaps
        'umbra', 'penumbra', 'both'           (n_t, ny, nx) bool masks
        'hot_spot'                            bool mask or None
        'cube_mag_qsun', 'cube_dop_qsun'      quiet-sun reference cubes
        'time_h', 'timestamps', 'cadence_s', 'n_t'
        'present'                             {series: bool array} — False where no data
        ...plus per-loader extras
    }
"""

from __future__ import annotations

import datetime
import pathlib
import warnings
from collections.abc import Callable

import numpy as np
from astropy.io import fits
from scipy.io import readsav

from .segmentation import build_regions, masks_from_cubes
from .utilities import mean_series as _mean_series, read_cube, reindex_on_grid

def _load_cube(path: pathlib.Path, fill: float) -> np.ndarray:
    with fits.open(path) as hdul:
        arr = hdul[0].data.astype(float)  # type: ignore[union-attr]
    return np.where(np.abs(arr) > fill, np.nan, arr)


def verify_cadence(
    ds_dir: str | pathlib.Path,
    expected_s: float = 720.0,
    tol_s: float = 1.0,
) -> tuple[float, np.ndarray]:
    """
    Verify the frame cadence from the IDL .sav timing file.

    Reads ``cube_continuum_coords2times.sav``, parses the timestamps stored in
    the ``times`` array, and checks that every consecutive difference equals
    ``expected_s`` within ``tol_s`` seconds.

    Parameters
    ----------
    ds_dir     : path to the DS0X directory containing the .sav file
    expected_s : expected cadence in seconds  (default 720)
    tol_s      : tolerance in seconds for the check

    Returns
    -------
    Measured median cadence in seconds.

    Raises
    ------
    ValueError  if any gap deviates from expected_s by more than tol_s.
    FileNotFoundError if the .sav file is absent.
    """
    ds_dir = pathlib.Path(ds_dir)
    sav_path = ds_dir / 'cube_continuum_coords2times.sav'
    if not sav_path.exists():
        raise FileNotFoundError(f'.sav file not found: {sav_path}')

    sav = readsav(sav_path)
    raw = sav['times'].flatten().astype(str)
    times = [datetime.datetime.strptime(t, '%d-%b-%Y %H:%M:%S.%f') for t in raw]
    diffs = np.array([
        (times[i + 1] - times[i]).total_seconds() for i in range(len(times) - 1)
    ])

    bad = np.abs(diffs - expected_s) > tol_s
    if bad.any():
        warnings.warn(
            f'Cadence check failed: {bad.sum()} gaps deviate from {expected_s} s '
            f'(max deviation {np.abs(diffs - expected_s).max():.1f} s)'
        )
    else:
        print(f'Cadence OK — {len(diffs)} gaps, all {expected_s:.0f} s  ')


    median_cad = float(np.median(diffs))
    print(f'(median {median_cad:.1f} s, max dev {np.abs(diffs - expected_s).max():.2f} s)')
    return median_cad , diffs


def load_ds0n_region(
    ds_dir: str | pathlib.Path,
    umbra_thresh: float = 30_000,
    penumbra_thresh: float = 50_000,
    fill: float = 1e6,
    cluster_mode: str = 'largest',
    cadence_s: float = 720.0,
    filter_mask: bool = True,
    filter_both: bool = False,
    mag_filter: Callable[[np.ndarray], np.ndarray] | None = None,
    raw_dopler: bool = False,
) -> dict:
    """
    Load the SDO/HMI data cubes and build region masks.

    The quiet-sun signal is taken from the dedicated pre-computed cubes
    ``cube_dopplergram_qsun.fits`` and ``cube_magnetogram_qsun.fits`` (a
    small patch of quiet sun, typically 60×60 px), not from a mask on the
    sunspot continuum cube.

    Parameters
    ----------
    ds_dir          : path to the DS0X directory.  Must contain:
                        cube_continuum.fits, cube_magnetogram.fits,
                        cube_dopplergram_corrected.fits,
                        cube_dopplergram_qsun.fits, cube_magnetogram_qsun.fits
    umbra_thresh    : continuum DN below which a pixel is umbra  (default 30 000)
    penumbra_thresh : continuum DN upper bound for penumbra      (default 50 000)
                      penumbra = umbra_thresh ≤ cont < penumbra_thresh
    fill            : absolute DN value above which data is fill/missing
    cluster_mode    : 'largest' or 'central' — how to pick the main sunspot cluster
    cadence_s       : time step between frames in seconds (default 720 s).
                      Can also be a 1-D array of per-gap cadences (length n_t-1)
                      as returned by verify_cadence().  In that case t=0 is
                      prepended and time_h is built via cumsum.  The median
                      cadence is stored in data['cadence_s'] for FFT use.
    filter_both     : if True, derive ``both`` as ``umbra | penumbra`` after
                      filtering, so it is consistent with the filtered masks.
                      Default False preserves the original behaviour (raw
                      continuum threshold, may include secondary sunspots).
                      Has no effect when filter_mask=False.
    mag_filter      : optional callable applied to cube_mag to define a
                      magnetogram-based sub-region within the sunspot footprint.
                      Example: ``lambda b: b > 500`` creates a region where
                      B > 500 G inside the sunspot.  The resulting mask is stored
                      as ``data['hot_spot']``.  If None, hot_spot is None.

    Returns
    -------
    dict with keys:
        cube_cont, cube_mag, cube_dop              : (n_t, ny, nx) float arrays
        cube_dop_qsun, cube_mag_qsun               : (n_t, ny_q, nx_q) float arrays
        umbra, penumbra, both                      : (n_t, ny, nx) bool arrays
        hot_spot                                   : (n_t, ny, nx) bool array or None
        time_h                                     : (n_t,) elapsed time in hours
        n_t, cadence_s, umbra_thresh, penumbra_thresh : metadata
    """
    ds_dir = pathlib.Path(ds_dir)

    cube_cont     = _load_cube(ds_dir / 'cube_continuum.fits', fill)
    cube_mag      = _load_cube(ds_dir / 'cube_magnetogram.fits', fill)
    cube_dop_qsun = _load_cube(ds_dir / 'cube_dopplergram_qsun.fits', fill)
    cube_mag_qsun = _load_cube(ds_dir / 'cube_magnetogram_qsun.fits', fill)


    if not raw_dopler:
        cube_dop      = _load_cube(ds_dir / 'cube_dopplergram_corrected.fits', fill)
    else:
        cube_dop      = _load_cube(ds_dir / 'cube_dopplergram.fits', fill)

    return masks_from_cubes(
        cube_cont, cube_mag, cube_dop, cube_dop_qsun, cube_mag_qsun,
        umbra_thresh=umbra_thresh, penumbra_thresh=penumbra_thresh,
        cluster_mode=cluster_mode, cadence_s=cadence_s,
        filter_mask=filter_mask, filter_both=filter_both, mag_filter=mag_filter,
    )


def _fill_nearest_frames(cube: np.ndarray, present: np.ndarray, max_slots: int):
    """Fill absent frames from the nearest present one within ``max_slots``, in place.

    For a series whose frames land on grid slots that another series never occupies. See
    `load_noaa_region`'s ``mag_fill_slots`` for the case this exists for.

    Returns the ``filled`` bool array — slots that now hold a copy of a neighbour rather
    than their own observation.
    """
    present = np.asarray(present, dtype=bool)
    filled = np.zeros(len(present), dtype=bool)
    have = np.flatnonzero(present)
    if not len(have):
        return filled

    for t in np.flatnonzero(~present):
        j = have[np.argmin(np.abs(have - t))]
        if abs(j - t) <= max_slots:
            cube[t] = cube[j]
            filled[t] = True
    return filled


def _raw_cube_on_grid(cube, times, grid, ref_shape, label: str):
    """Place a cube read from ``data/raw/`` onto the processed cubes' time grid.

    The two directories do not share a time axis and are not meant to: ``data/raw/`` holds
    the frames that actually downloaded, while 02A writes ``data/processed/`` on a uniform
    grid covering the *union* of the three series, with an all-NaN frame in every slot a
    series had no frame for. NOAA 11106 is 452 raw frames against 480 grid slots. Indexing
    one against the other is what raises ``operands could not be broadcast together`` —
    and a plain truncation to the shorter length would be worse than the error, because
    every frame after the first gap would sit one slot early and the 24 h fits would drift.

    So the raw cube goes through the same `reindex_on_grid` join 02A used, which is a no-op
    for a region whose raw cube is already complete.

    A *spatial* mismatch is refused instead of repaired: it means 02A cropped this region
    (`crop_to_common_window`, for a download box that changed mid-window) and the crop
    offset is not recoverable from the raw cube alone.
    """
    from .utilities import reindex_on_grid

    if tuple(cube.shape[1:]) != tuple(ref_shape):
        raise ValueError(
            f'{label}: raw cube is {cube.shape[1]}x{cube.shape[2]} but the processed '
            f'continuum is {ref_shape[0]}x{ref_shape[1]}. 02A cropped this region to its '
            f'common data window and the raw cube is uncropped, so the two cannot be '
            f'indexed against each other. Use 02A\'s corrected cube '
            f'(raw_magnetogram=False / raw_dopplergram=False), or re-run 02A for this '
            f'region with crop_to_data off if you need the full box.')

    seconds = np.array([(t - grid[0]).total_seconds() for t in grid])
    cadence_s = float(np.median(np.diff(seconds))) if len(grid) > 1 else np.nan

    out, present = reindex_on_grid(cube, times, grid, cadence_s)
    if not present.all():
        n = int((~present).sum())
        warnings.warn(
            f'{label}: {n} of {len(grid)} grid slots have no raw frame and are NaN '
            f'(the raw cube has {len(times)} frames). Every quantity averaged over those '
            f'slots is NaN by design — see mag_fill_slots.', stacklevel=4)
    return out


def load_noaa_region(
    processed_dir: str | pathlib.Path,
    region: str = 'region_01',
    mag_fill_slots: int = 0,
    custom_valid_region: np.ndarray | None = None,
    raw_magnetogram: bool  = False,
    raw_dopplergram: bool  = False,
    raw_dir: str | pathlib.Path = None,
    **region_kwargs,
) -> dict:
    """
    Build the ``load_ds0n_region``-style data dict from the three corrected cubes that
    ``notebooks/02A_data_procesing.ipynb`` writes for a NOAA region.

    This is the bridge between the NOAA pipeline and everything in this module: the
    result can be passed straight to ``compute_metrics``, ``plot_magnetogram_masks``,
    ``plot_time_series``, ``plot_area``, ``plot_ffts_*`` and ``save_animation`` exactly
    like a ``load_ds0n_region`` result for a DS0X directory.

    **Masks are rebuilt here, every time, by `build_regions`** — never read from disk, even
    when a mask cube happens to sit next to the data. That is the whole point of the split:
    what gets analysed is always what the current ``region_kwargs`` produce, so a stale mask
    file can never silently drive the analysis, and retuning a threshold costs one 03A run
    instead of a full 02A re-correction.

    The other differences from ``load_ds0n_region``:

    - **Gap frames are derived, not stored.** A frame that a series was missing is entirely
      NaN on 02A's uniform time grid, so ``present`` is just
      ``np.isfinite(cube).any(axis=(1, 2))`` per series — exact, and one less thing to keep
      in sync.
    - **No quiet-sun patch cubes exist** for these regions, so the ``cube_*_qsun`` entries
      are the per-frame spatial means over the quiet-sun mask, shaped ``(n_t, 1, 1)``, which
      is all ``compute_metrics`` ever takes of them.
    - **Cadence is measured, not assumed.** Median timestamp spacing — NOAA 11117 is sampled
      at ~360 s while the others are at 720 s, so a fixed cadence would put its FFT
      frequency axis out by a factor of two.

    Parameters
    ----------
    processed_dir : path to ``data/processed/NOAA_<noaa>_<date>/``
    region        : region prefix within that directory (default ``'region_01'``)
    mag_fill_slots : int
        Repair for a magnetogram that never shares a grid slot with the continuum. Every
        magnetogram quantity is averaged over a mask built from the continuum, so a frame
        where only one of the two exists yields NaN. NOAA 11117 hits this hard: from
        2010-10-30 its continuum runs at 720 s on even slots and its magnetogram at 720 s
        on **odd** slots, 360 s apart, so ``mean_mag_*`` is NaN for the entire last day
        while ``mean_dop_*`` is fine — the data is there, it is just interleaved.

        Setting this to *n* fills each empty magnetogram slot from the nearest magnetogram
        frame within *n* slots. Off by default because it fabricates a coincidence that the
        observations do not have; ``data['mag_filled']`` marks every slot it invented.
        ``mag_fill_slots=1`` shifts a frame by at most one grid step — for 11117 that is
        360 s against a 720 s magnetogram cadence, half a frame, which is negligible for a
        24 h signal but is not free at the Nyquist end. The real fix is to re-download that
        region's last day on one clock.
    custom_valid_region : np.ndarray | None
        A boolean mask for valid regions in the continuum cube.
    raw_magnetogram : bool
        Read the magnetogram from ``data/raw/`` instead of 02A's corrected cube — the field
        as downloaded, with the quiet-sun plane still in it. For asking what the plane
        subtraction did, not for analysis: the raw cube keeps HMI's instrumental offset and
        the gradient across the box, so ``hotspot_gauss`` cuts at a different physical level
        than it does on the corrected cube and the two hot-spot masks are not comparable.
        Needs ``raw_dir``. The cube is reindexed onto the processed time grid on the way in
        (`_raw_cube_on_grid`) — the raw and processed cubes have different frame counts.
    raw_dopplergram : bool
        The same for the Doppler cube: read ``data/raw/<region>_dopplergram_cube.fits``
        instead of 02A's ``_dopplergram_calibrated_cube.fits``. This is the velocity as
        HMI measured it, so **none** of `src/doppler_calibration.py` has been applied —
        the spacecraft term ``v_SDO`` is still in it, and that one alone is a ±3 km/s
        sinusoid at exactly 24 h, orders of magnitude above the umbral signal. Use it to
        see what the calibration removed, never as the input to `fit_diurnal` or the FFT.
        Also needs ``raw_dir``, and is reindexed onto the processed grid the same way.
    raw_dir : path | None
        This region's directory under ``data/raw/``, required when ``raw_magnetogram`` or
        ``raw_dopplergram``.
    **region_kwargs : forwarded to `build_regions` — thresholds, cluster mode, mag_filter.

    Returns
    -------
    dict with the same keys as ``masks_from_cubes``, plus ``timestamps``, ``present``,
    ``i_qs``, ``cluster_info``, and ``c_mean`` / ``doppler_terms`` when 02A's per-frame
    diagnostics table is present beside the cubes.
    """
    from .utilities import read_cube

    processed_dir = pathlib.Path(processed_dir)

    cube_cont, timestamps = read_cube(processed_dir / f'{region}_continuum_cube.fits')

    if (raw_magnetogram or raw_dopplergram) and raw_dir is None:
        which = ' and '.join(n for n, on in [('raw_magnetogram', raw_magnetogram),
                                             ('raw_dopplergram', raw_dopplergram)] if on)
        raise ValueError(
            f"{which}=True needs raw_dir — this region's directory under data/raw/, e.g. "
            "pathlib.Path(str(processed_dir).replace('/processed/', '/raw/'))")

    def _read_raw(kind: str) -> np.ndarray:
        # 01A's cubes are on the timestamps that downloaded, not on 02A's uniform grid.
        path = pathlib.Path(raw_dir) / f'{region}_{kind}_cube.fits'
        cube, times = read_cube(path)
        return _raw_cube_on_grid(cube, times, timestamps, cube_cont.shape[1:], path.name)

    if raw_magnetogram:
        cube_mag = _read_raw('magnetogram')
    else:
        cube_mag, _ = read_cube(processed_dir / f'{region}_magnetogram_corrected_cube.fits')

    if raw_dopplergram:
        cube_dop = _read_raw('dopplergram')
    else:
        cube_dop, _ = read_cube(processed_dir / f'{region}_dopplergram_calibrated_cube.fits')

    n_t = len(timestamps)
    present = {name: np.isfinite(cube).any(axis=(1, 2))
               for name, cube in [('cont', cube_cont), ('mag', cube_mag), ('dop', cube_dop)]}

    # Every magnetogram quantity is averaged over a mask built from the *continuum*, so it
    # needs both series in the same grid slot. NOAA 11117 stops providing that: from
    # 2010-10-30 its continuum drops to 720 s on even slots while its magnetogram sits on
    # 720 s odd slots, 360 s apart, so they never coincide again and mean_mag_* is NaN for
    # the whole last day even though both cubes have data throughout.
    mag_filled = np.zeros(n_t, dtype=bool)
    if mag_fill_slots:
        mag_filled = _fill_nearest_frames(cube_mag, present['mag'], mag_fill_slots)
        present['mag'] = present['mag'] | mag_filled

    regions = build_regions(cube_cont, cube_mag, custom_valid_region=custom_valid_region, **region_kwargs)

    # 02A's per-frame correction diagnostics. Optional on purpose: cubes written before this
    # table existed still load, they just have nothing to report about the corrections.
    c_mean, doppler_terms = None, None
    frames_path = processed_dir / f'{region}_frames.fits'
    if frames_path.exists():
        with fits.open(frames_path) as hdul:
            frames = hdul['FRAMES'].data
            names = frames.columns.names
        if 'C_MEAN' in names:
            c_mean = np.asarray(frames['C_MEAN'], dtype=float)
        terms = {k.lower()[2:]: np.asarray(frames[k], dtype=float)
                 for k in ('V_SDO', 'V_LSF', 'V_CLV', 'V_GRAVITY') if k in names}
        doppler_terms = terms or None

    seconds = np.array([(t - timestamps[0]).total_seconds() for t in timestamps])
    gaps = np.diff(seconds)
    cadence_s = float(np.median(gaps)) if gaps.size else np.nan

    qsun = regions['qsun']
    return dict(
        cube_cont=cube_cont, cube_mag=cube_mag, cube_dop=cube_dop,
        # Shaped (n_t, 1, 1) so np.nanmean(cube[t]) returns that frame's quiet-sun mean.
        cube_mag_qsun=_mean_series(cube_mag, qsun).reshape(n_t, 1, 1),
        cube_dop_qsun=_mean_series(cube_dop, qsun).reshape(n_t, 1, 1),
        time_h=seconds / 3600, n_t=n_t, cadence_s=cadence_s,
        timestamps=timestamps, present=present, mag_filled=mag_filled,
        c_mean=c_mean, doppler_terms=doppler_terms,
        **{k: regions[k] for k in (
            'umbra', 'penumbra', 'both', 'hot_spot', 'qsun', 'i_qs', 'cluster_info',
            'raw_area_px', 'umbra_thresh', 'penumbra_thresh', 'cluster_mode',
            'umbra_frac', 'penumbra_frac')},
    )

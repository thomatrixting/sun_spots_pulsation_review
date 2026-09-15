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
    """Read a DS0N cube, turning its fill value into NaN.

    Kept in float32, which is what is actually on disk (BITPIX -32) and what HMI's
    precision justifies. The previous `astype(float)` upcast to float64 and so doubled
    the memory for no extra information: DS00 is 901x299x794, i.e. 1.7 GB per cube in
    float64 against 0.86 GB in float32, and this loader reads four of them. Every mean
    over these cubes is still accumulated in float64 — see `utilities.mean_series` — so
    the numbers are unchanged.
    """
    with fits.open(path) as hdul:
        arr = hdul[0].data.astype(np.float32)  # type: ignore[union-attr]
    return np.where(np.abs(arr) > fill, np.float32(np.nan), arr)


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
    (median_s, gaps_s)
        ``median_s`` is the measured median cadence in seconds, and ``gaps_s`` the full
        array of ``n_t - 1`` measured gaps between consecutive frames. **Pass ``gaps_s``
        straight into ``load_ds0n_region(cadence_s=...)``** when a dataset has jumps: that
        is what makes ``time_h`` the real elapsed time (built by cumsum) instead of an
        assumed uniform grid.

    Raises
    ------
    FileNotFoundError if the .sav file is absent.

    A gap deviating from ``expected_s`` by more than ``tol_s`` only warns — the timing is
    reported, never rejected, because an irregular cadence is a thing to feed forward, not
    a thing to refuse.
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

    # A gap that is not positive means the timing file is out of order, which is a different
    # complaint from "the cadence varies": cumsum still gives the right *total* elapsed time,
    # but one frame sits in the past and any interpolation over that axis is meaningless
    # until it is dealt with. DS00 and DS05 each carry one such frame.
    backwards = np.flatnonzero(diffs <= 0)
    if backwards.size:
        warnings.warn(
            f'{ds_dir.name}: {backwards.size} timestamp(s) out of order — frame(s) '
            f'{(backwards + 1).tolist()[:5]} are dated before the frame preceding them '
            f'(most negative gap {diffs.min() / 3600:.1f} h). time_h will not be monotonic.'
        )

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


def _frames_out_of_order(gaps_s: np.ndarray) -> np.ndarray:
    """Indices of frames whose timestamp is inconsistent with both of its neighbours'.

    Built from the per-gap cadences `verify_cadence` returns. An inversion between two
    adjacent frames does not by itself say which of the two is wrong, so the test is which
    one has to go for the order to come back: frame k is the culprit when it sits outside
    its neighbours *and* those neighbours agree with each other once it is gone. That
    catches a frame dated in the past (a dip — DS00's frame 837 and DS05's frame 362) and
    one dated in the future (a spike) alike, and blames only the frame responsible.
    """
    times = np.concatenate([[0.0], np.cumsum(np.asarray(gaps_s, dtype=float))])
    before = np.r_[-np.inf, times[:-1]]
    after = np.r_[times[1:], np.inf]
    return np.flatnonzero(((times <= before) | (times >= after)) & (before < after))


def _drop_frames(cubes: dict, gaps_s: np.ndarray, drop: np.ndarray):
    """Remove `drop` frames from every cube and re-measure the gaps across the holes."""
    times = np.concatenate([[0.0], np.cumsum(np.asarray(gaps_s, dtype=float))])
    keep = np.setdiff1d(np.arange(times.size), drop)
    return ({name: cube[keep] for name, cube in cubes.items()},
            np.diff(times[keep]))


def _elapsed_s(gaps_s, n_t: int) -> np.ndarray:
    """Elapsed seconds per frame, from either a scalar cadence or the measured gaps."""
    gaps = np.asarray(gaps_s, dtype=float)
    if gaps.ndim == 0:
        return np.arange(n_t) * float(gaps)
    return np.concatenate([[0.0], np.cumsum(gaps)])


def _crop_to_time_limit(cubes: dict, gaps_s, time_limit):
    """Keep only the frames inside `time_limit` hours, and re-measure the gaps.

    Returns `(cubes, gaps_s, t0_h, kept)`. `t0_h` is the elapsed time of the first frame
    kept, which the caller adds back to `time_h` so the axis keeps counting from the first
    frame of the *whole* cube: cropping to (100, 350) gives a series labelled 100-350 h,
    not 0-250 h, so a period or a window means the same thing cropped or not.
    """
    tmin, tmax = time_limit
    n_t = len(next(iter(cubes.values())))
    times_s = _elapsed_s(gaps_s, n_t)
    times_h = times_s / 3600

    keep = np.ones(n_t, dtype=bool)
    if tmin is not None:
        keep &= times_h >= tmin
    if tmax is not None:
        keep &= times_h <= tmax

    kept = np.flatnonzero(keep)
    if kept.size < 2:
        raise ValueError(
            f'time_limit={time_limit} h keeps {kept.size} frame(s); this cube covers '
            f'{times_h[0]:.2f}-{times_h[-1]:.2f} h.')

    return ({name: cube[kept] for name, cube in cubes.items()},
            np.diff(times_s[kept]), float(times_h[kept[0]]), kept)


def load_ds0n_region(
    ds_dir: str | pathlib.Path,
    umbra_thresh: float = 30_000,
    penumbra_thresh: float = 50_000,
    fill: float = 1e6,
    cluster_mode: str = 'largest',
    cadence_s: float | np.ndarray = 720.0,
    filter_mask: bool = True,
    filter_both: bool = False,
    mag_filter: Callable[[np.ndarray], np.ndarray] | None = None,
    raw_dopler: bool = False,
    cluster_xlim: tuple[int, int] | None = None,
    cluster_ylim: tuple[int, int] | None = None,
    normalize_mag_by_mu: bool = True,
    drop_out_of_order: bool = False,
    time_limit: tuple[float | None, float | None] | None = None,
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
    cluster_xlim, cluster_ylim : (int, int) or None
                      Restrict where the cluster search (`filter_mask=True`) looks for a
                      spot, as pixel index ranges on x (columns) / y (rows). Use this when
                      the box contains more than one sunspot and the wrong one gets picked.
                      See `segmentation.masks_from_cubes` for exactly what it restricts.
    drop_out_of_order : bool
                      Drop frames whose timestamp is out of order — dated before the frame
                      preceding them, or after the one following them. DS00 (frame 837) and
                      DS05 (frame 362) each carry exactly one, ~190 h and ~73 h in the past,
                      and `time_h` is not monotonic while they are in. Off by default: the
                      timing file is reported as it is, and `verify_cadence` warns. Needs
                      `cadence_s` to be the measured gap array — a scalar cadence carries no
                      timestamps to judge. The frames removed are listed in
                      `data['dropped_frames']`.
    normalize_mag_by_mu : bool
                      Divide `cube_mag` by this region's `cube_mu.fits` right after loading,
                      before masks (and therefore `hot_spot`) are built — so `mag_filter`'s
                      threshold applies to the mu-normalized field, not the raw one. Default
                      True. `cube_mu.fits` ships alongside every DS0N dataset already, same
                      grid as `cube_magnetogram.fits`.
    time_limit      : (tmin, tmax) in HOURS from the first frame, or None for the whole
                      cube. Every cube is cropped to that stretch before the masks are
                      built, so everything downstream — metrics, spectra, the animation —
                      sees only those frames. `(0, 350)` keeps the start of the series up to
                      hour 350; either bound may be None to leave that end alone, e.g.
                      `(350, None)`. Both ends are inclusive.

                      `time_h` still counts from the first frame of the *whole* cube, so a
                      crop of `(100, 350)` is labelled 100-350 h. The cut is by time, not by
                      frame index, which is what makes it mean the same thing on a dataset
                      whose cadence jumps. Applied after `drop_out_of_order`, so a frame with
                      a bad timestamp cannot drag the window with it.

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

    if normalize_mag_by_mu:
        cube_mu = _load_cube(ds_dir / 'cube_mu.fits', fill)
        cube_mag = cube_mag / cube_mu

    if not raw_dopler:
        cube_dop      = _load_cube(ds_dir / 'cube_dopplergram_corrected.fits', fill)
    else:
        cube_dop      = _load_cube(ds_dir / 'cube_dopplergram.fits', fill)

    dropped = np.array([], dtype=int)
    if drop_out_of_order:
        if np.asarray(cadence_s).ndim == 0:
            warnings.warn('drop_out_of_order needs the measured gap array from '
                          'verify_cadence as cadence_s; a scalar cadence has no timestamps '
                          'to check. Nothing dropped.')
        else:
            dropped = _frames_out_of_order(cadence_s)
            if dropped.size:
                cubes = dict(cube_cont=cube_cont, cube_mag=cube_mag, cube_dop=cube_dop,
                             cube_dop_qsun=cube_dop_qsun, cube_mag_qsun=cube_mag_qsun)
                if normalize_mag_by_mu:
                    # It goes out in `result` and gets averaged per mask in compute_metrics,
                    # so it has to lose the same frames as everything else.
                    cubes['cube_mu'] = cube_mu
                cubes, cadence_s = _drop_frames(cubes, cadence_s, dropped)
                cube_cont, cube_mag, cube_dop = (cubes['cube_cont'], cubes['cube_mag'],
                                                 cubes['cube_dop'])
                cube_dop_qsun, cube_mag_qsun = cubes['cube_dop_qsun'], cubes['cube_mag_qsun']
                if normalize_mag_by_mu:
                    cube_mu = cubes['cube_mu']
                print(f'Dropped {dropped.size} out-of-order frame(s): {dropped.tolist()}')
                # One pass fixes an isolated bad frame, which is what these files carry. Two
                # adjacent ones would need another, so say so rather than hand back an axis
                # that still runs backwards somewhere.
                if np.any(cadence_s <= 0):
                    warnings.warn('time_h is still not monotonic after dropping — there are '
                                  'adjacent out-of-order frames. Inspect the .sav timing.')

    # Crop before the masks are built, not after: everything downstream — the cluster
    # search, the metrics, the spectra, the animation — then sees exactly the stretch asked
    # for, and none of it needs to know a crop happened.
    t0_h = 0.0
    if time_limit is not None:
        kept_from = cube_cont.shape[0]
        cubes = dict(cube_cont=cube_cont, cube_mag=cube_mag, cube_dop=cube_dop,
                     cube_dop_qsun=cube_dop_qsun, cube_mag_qsun=cube_mag_qsun)
        if normalize_mag_by_mu:
            cubes['cube_mu'] = cube_mu
        cubes, cadence_s, t0_h, kept = _crop_to_time_limit(cubes, cadence_s, time_limit)
        cube_cont, cube_mag, cube_dop = (cubes['cube_cont'], cubes['cube_mag'],
                                         cubes['cube_dop'])
        cube_dop_qsun, cube_mag_qsun = cubes['cube_dop_qsun'], cubes['cube_mag_qsun']
        if normalize_mag_by_mu:
            cube_mu = cubes['cube_mu']
        print(f'time_limit {time_limit} h -> kept {kept.size} of {kept_from} frames, '
              f'{t0_h:.2f}-{t0_h + np.sum(cadence_s) / 3600:.2f} h')

    result = masks_from_cubes(
        cube_cont, cube_mag, cube_dop, cube_dop_qsun, cube_mag_qsun,
        umbra_thresh=umbra_thresh, penumbra_thresh=penumbra_thresh,
        cluster_mode=cluster_mode, cadence_s=cadence_s,
        filter_mask=filter_mask, filter_both=filter_both, mag_filter=mag_filter,
        cluster_xlim=cluster_xlim, cluster_ylim=cluster_ylim,
    )
    # masks_from_cubes counts time from the first frame it was handed, which after a crop is
    # not the first frame of the cube. Put the offset back so `time_h` means the same thing
    # cropped or not — see `time_limit` above.
    result['time_h'] = result['time_h'] + t0_h
    result['time_limit'] = time_limit
    # Only loaded (and only meaningful) when normalize_mag_by_mu already read this cube once
    # to build cube_mag above — kept here so compute_metrics can average it per mask without
    # re-reading cube_mu.fits from disk a second time.
    result['cube_mu'] = cube_mu if normalize_mag_by_mu else None
    result['dropped_frames'] = dropped
    return result


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

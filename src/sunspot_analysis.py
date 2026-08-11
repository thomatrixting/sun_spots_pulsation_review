"""
Sunspot pulsation analysis utilities.

Typical workflow
----------------
0. verify_cadence(ds_dir)                               # confirm 720 s cadence from .sav
1. data = load_and_mask(ds_dir)                         # load cubes + build masks
2. plot_calibration_frame(data)                         # visual check, adjust thresholds if needed
2b. plot_histogram(data)                                 # histogram of entire cube (default: magnetogram)
    plot_histogram(data, cube='continuum')              # continuum intensity distribution
    plot_histogram(data, cube='dopplergram')            # Doppler velocity distribution
2c. plot_magnetogram_masks(data)                        # magnetogram + region overlays + colorbar
   # optionally reload with mag_filter to define a hot_spot region:
   # data = load_and_mask(ds_dir, mag_filter=lambda b: b > 500)
3. metrics = compute_metrics(data)                      # mean B / Doppler per region over time
4. save_metrics_csv(data, metrics, processed_dir)       # save time-series as CSV
5. plot_time_series(data, metrics)                      # value-vs-time panels
5b. plot_area(data, metrics)                            # area vs time
6. plot_ffts_separate(data, metrics)                    # 2×2 FFT subplots
7. plot_ffts_combined(data, metrics, xlim=(0, 0.1))     # overlaid with x-zoom
8. save_animation(data, metrics, save_path)             # HTML animation
"""

from __future__ import annotations

import datetime
import pathlib
import warnings
from collections.abc import Callable

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.io import fits
from matplotlib.animation import FuncAnimation, HTMLWriter
from scipy import ndimage
from scipy.io import readsav
from scipy.signal import find_peaks


# ── private helpers ───────────────────────────────────────────────────────────

def _load_cube(path: pathlib.Path, fill: float) -> np.ndarray:
    with fits.open(path) as hdul:
        arr = hdul[0].data.astype(float)  # type: ignore[union-attr]
    return np.where(np.abs(arr) > fill, np.nan, arr)


def _keep_central_cluster(binary_img: np.ndarray, mode: str) -> np.ndarray:
    if mode not in ('largest', 'central'):
        raise ValueError(f"cluster_mode must be 'largest' or 'central', got {mode!r}")
    labeled, n = ndimage.label(binary_img)  # type: ignore[misc]
    if n == 0:
        return binary_img
    if mode == 'largest':
        sizes = ndimage.sum(binary_img, labeled, range(1, n + 1))
        keep = int(np.argmax(sizes)) + 1
    else:
        cy, cx = np.array(binary_img.shape) / 2
        centroids = ndimage.center_of_mass(binary_img, labeled, range(1, n + 1))
        dists = [np.hypot(y - cy, x - cx) for y, x in centroids]
        keep = int(np.argmin(dists)) + 1
    return (labeled == keep).astype(binary_img.dtype)


#: Components smaller than this many pixels are discarded before a spot is chosen.
#: The penumbra threshold (0.90 I_qs) sits around the 8th percentile of a quiet-sun box, so
#: it catches the dark tail of granulation as well as the spot: a single NOAA 11536 frame
#: labels into ~160 components, ~150 of them under 50 px and totalling ~950 px of speckle.
#: At HMI's 0.5 arcsec/px, 50 px is a blob about 7 px across — well under a real pore
#: (2-5 Mm, i.e. 30-150 px), so this removes noise without touching solar structure. It
#: also stops the "largest" component percolating through the speckle field and merging
#: spots that are not actually connected.
MIN_CLUSTER_PX = 50


def select_spot_cluster(
    footprint: np.ndarray,
    mode: str = 'largest',
    track: bool = True,
    connectivity: int = 2,
    min_area: int = MIN_CLUSTER_PX,
) -> tuple[np.ndarray, dict]:
    """Keep one connected sunspot per frame out of a whole cube's thresholded footprint.

    Thresholding a continuum frame selects *every* dark pixel in the box — the target spot,
    the other members of the group, pores, and bad pixels. The area of that mask then moves
    for reasons that have nothing to do with the spot being measured: NOAA 11117's umbra
    area drifts 98% across its window, and every mean taken over the mask inherits that.
    Labelling the footprint into connected components and keeping one of them measures a
    sunspot instead of a box.

    Why the *combined* footprint (umbra | penumbra) rather than each mask separately, which
    is what `_keep_central_cluster` is used for in `masks_from_cubes`: a sunspot is one
    connected dark region with its umbra nested inside its penumbra, so the combined mask is
    the thing that has one blob per spot. Labelling umbra and penumbra independently can
    pick the largest umbra from one spot and the largest penumbra from another. Intersect
    afterwards instead — ``umbra & spot``, ``penumbra & spot``.

    Not k-means, deliberately: k-means clusters pixels in some feature space and has no
    notion of spatial connectedness, so it will happily merge two separate spots into one
    cluster and split one spot in half. Connected-component labelling is the operation that
    means "these pixels are the same spot".

    Parameters
    ----------
    footprint : ndarray, (n_t, ny, nx) bool
        The combined umbra|penumbra mask for every frame.
    mode : {'largest', 'central'}
        How the target is chosen in the first frame, and whenever tracking loses it:
        by area, or by centroid distance to the frame centre. 'central' is meaningful
        because `locate_ar_window` centres the cutout on the catalogue centroid.
    track : bool
        Follow the same spot from frame to frame by maximum pixel overlap with the previous
        selection, rather than re-running `mode` independently each time. Two comparable
        spots make plain 'largest' flip between them mid-window, which puts a step in every
        series — the exact artifact this function exists to remove.
    connectivity : {1, 2}
        1 = 4-connectivity, 2 = 8-connectivity (default), so diagonally touching penumbral
        pixels stay one blob.
    min_area : int
        Discard components below this many pixels before choosing — see `MIN_CLUSTER_PX`
        for why this is not optional in practice. 0 disables it.

    Returns
    -------
    (spot, info)
        `spot` is a bool cube of the same shape, True only inside the selected component,
        so ``spot`` is a subset of ``footprint`` by construction.
        `info` holds per-frame `n_clusters` (after the `min_area` cut), `area`, `fraction`
        (selected / total footprint), `centroid` (n_t, 2) and `switched`. **Look at
        `switched` and `fraction`.** Tracking losing the spot is the one way this makes
        things worse, and it is invisible in the masks themselves.

    Notes
    -----
    `fraction` is deliberately measured against the *whole* footprint, speckle included, so
    it stays an honest "how much of the dark area is this spot" rather than flattering
    itself by excluding what `min_area` already threw away.

    A `switched` frame is not automatically a bug. On NOAA 11536 the spot decays from 473
    to 18 px across a 4-day window, and the tracker re-picks once at the very end when
    there is essentially nothing left to track. Check *when* it happened before treating it
    as one.
    """
    if mode not in ('largest', 'central'):
        raise ValueError(f"mode must be 'largest' or 'central', got {mode!r}")

    footprint = np.asarray(footprint, dtype=bool)
    n_t = footprint.shape[0]
    structure = ndimage.generate_binary_structure(2, connectivity)
    centre = np.array(footprint.shape[1:]) / 2

    spot = np.zeros_like(footprint)
    info = dict(
        n_clusters=np.zeros(n_t, dtype=int),
        area=np.zeros(n_t, dtype=int),
        fraction=np.full(n_t, np.nan),
        centroid=np.full((n_t, 2), np.nan),
        switched=np.zeros(n_t, dtype=bool),
    )

    # The tracking reference is the last frame in which something was actually selected,
    # not literally t-1: a NaN gap frame selects nothing and must not break the chain.
    previous = None

    for t in range(n_t):
        frame = footprint[t]
        if not frame.any():
            continue

        labeled, n = ndimage.label(frame, structure=structure)
        if n == 0:
            continue
        labels = np.arange(1, n + 1)
        sizes = ndimage.sum_labels(frame, labeled, labels)

        if min_area:
            big = sizes >= min_area
            if not big.any():
                continue                        # nothing here but speckle
            labels, sizes = labels[big], sizes[big]
            # Blank the discarded components so overlap and centroids ignore them too.
            labeled = np.where(np.isin(labeled, labels), labeled, 0)
        info['n_clusters'][t] = len(labels)

        keep = None
        if track and previous is not None:
            # Overlap of every label with the previous selection, in one pass.
            overlap = np.bincount(labeled[previous].ravel(), minlength=n + 1)
            overlap[0] = 0                      # label 0 is background
            if overlap.max() > 0:
                keep = int(overlap.argmax())
            else:
                info['switched'][t] = True      # lost it — fall through to `mode`

        if keep is None:
            if mode == 'largest':
                keep = int(labels[np.argmax(sizes)])
            else:
                centroids = ndimage.center_of_mass(frame, labeled, labels)
                dists = [np.hypot(y - centre[0], x - centre[1]) for y, x in centroids]
                keep = int(labels[np.argmin(dists)])

        selected = labeled == keep
        spot[t] = selected
        info['area'][t] = int(selected.sum())
        info['fraction'][t] = info['area'][t] / frame.sum()
        info['centroid'][t] = ndimage.center_of_mass(selected)
        previous = selected

    return spot, info


def _fft(ts: np.ndarray, cadence_s: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (f_mhz, amplitude) skipping the DC component."""
    ts_c = np.array(ts, dtype=float)
    nans = ~np.isfinite(ts_c)
    if nans.any() and (~nans).sum() > 2:
        x = np.arange(len(ts_c))
        ts_c[nans] = np.interp(x[nans], x[~nans], ts_c[~nans])
    freqs = np.fft.rfftfreq(len(ts_c), d=cadence_s)
    amps = np.abs(np.fft.rfft(ts_c))
    return freqs[1:] * 1e3, amps[1:]


def _mean_series(cube: np.ndarray, mask: np.ndarray) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return np.nanmean(np.where(mask, cube, np.nan), axis=(1, 2))


def _savefig(fig: matplotlib.pyplot.figure, plots_dir: str | pathlib.Path | None, filename: str) -> None: # pyright: ignore[reportAttributeAccessIssue]
    if plots_dir is None:
        return
    plots_dir = pathlib.Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    out = plots_dir / filename
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f'Figure saved → {out}')


# ── phase 0: cadence verification ────────────────────────────────────────────

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


# ── phase 1: load + mask ──────────────────────────────────────────────────────

def load_and_mask(
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


def masks_from_cubes(
    cube_cont: np.ndarray,
    cube_mag: np.ndarray,
    cube_dop: np.ndarray,
    cube_dop_qsun: np.ndarray,
    cube_mag_qsun: np.ndarray,
    umbra_thresh: float = 30_000,
    penumbra_thresh: float = 50_000,
    cluster_mode: str = 'largest',
    cadence_s: float = 720.0,
    filter_mask: bool = True,
    filter_both: bool = False,
    mag_filter: Callable[[np.ndarray], np.ndarray] | None = None,
) -> dict:
    """
    Build region masks and the ``load_and_mask``-style data dict from
    already-loaded (n_t, ny, nx) cubes, instead of reading FITS files from
    disk. Used by ``load_and_mask`` internally, and directly by callers that
    have derived/reconstructed cubes (e.g. from a coefficient-inversion
    pipeline) rather than the on-disk ones.

    Parameters mirror ``load_and_mask`` — see its docstring for details.
    """
    n_t = cube_cont.shape[0]

    _cad = np.asarray(cadence_s, dtype=float)
    if _cad.ndim == 0:
        time_h = np.arange(n_t) * float(_cad) / 3600
        median_cadence = float(_cad)
    else:
        time_h = np.concatenate([[0.0], np.cumsum(_cad)]) / 3600
        median_cadence = float(np.median(_cad))

    umbra_raw    = (cube_cont < umbra_thresh)    & np.isfinite(cube_cont)
    penumbra_raw = (cube_cont < penumbra_thresh) & np.isfinite(cube_cont) & ~umbra_raw

    if filter_mask:
        umbra    = np.array([_keep_central_cluster(umbra_raw[t],    cluster_mode) for t in range(n_t)], dtype=bool)
        penumbra = np.array([_keep_central_cluster(penumbra_raw[t], cluster_mode) for t in range(n_t)], dtype=bool)
        del umbra_raw, penumbra_raw
    else:
        umbra = umbra_raw
        penumbra = penumbra_raw

    both = umbra | penumbra if filter_both else (cube_cont < penumbra_thresh) & np.isfinite(cube_cont)

    hot_spot = None
    if mag_filter is not None:
        hot_spot = mag_filter(cube_mag) & (umbra | penumbra)

    return dict(
        cube_cont=cube_cont, cube_mag=cube_mag, cube_dop=cube_dop,
        cube_dop_qsun=cube_dop_qsun, cube_mag_qsun=cube_mag_qsun,
        umbra=umbra, penumbra=penumbra, both=both, hot_spot=hot_spot,
        time_h=time_h, n_t=n_t, cadence_s=median_cadence,
        umbra_thresh=umbra_thresh, penumbra_thresh=penumbra_thresh,
    )


DEFAULT_HOTSPOT_G = 500.0


def build_regions(
    cube_cont: np.ndarray,
    cube_mag: np.ndarray | None = None,
    umbra_frac: float = 0.60,
    penumbra_frac: float = 0.90,
    qsun_percentile: int = 80,
    cluster_mode: str | None = 'largest',
    cluster_track: bool = True,
    cluster_min_area: int = MIN_CLUSTER_PX,
    mag_filter: Callable[[np.ndarray], np.ndarray] | None = None,
    hotspot_gauss: float = DEFAULT_HOTSPOT_G,
) -> dict:
    """Segment a continuum cube into umbra / penumbra / hot spot / quiet sun.

    This is the analysis-side counterpart to the corrections in
    ``notebooks/02A_data_procesing.ipynb``: 02A produces the three corrected cubes and stops,
    and everything here is re-derivable from them at any time. That split exists so that
    retuning a threshold or a tracker means re-running 03A only — 02A costs ~90 s per region
    because the limb-darkening and Doppler corrections re-read every per-frame FITS header,
    and none of that work depends on where the umbra boundary is drawn.

    Thresholds are fractions of each frame's **own** quiet-sun intensity, not absolute DN
    (which is what `masks_from_cubes` takes, for the DS0X path). An absolute cut makes the
    mask areas drift as the region rotates, and for a near-limb region the whole frame can
    fall under a fixed penumbra cut.

    Parameters
    ----------
    cube_cont : ndarray, (n_t, ny, nx)
        Limb-darkening-corrected continuum, as 02A writes it.
    cube_mag : ndarray, optional
        Plane-corrected magnetogram. Needed only for the hot spot; without it `hot_spot`
        comes back None.
    umbra_frac, penumbra_frac : float
        ``I < frac * I_qs``. Penumbra additionally excludes umbra.
    qsun_percentile : int
        Percentile of the frame's finite pixels used as its ``I_qs``.
    cluster_mode : {'largest', 'central', None}
        Restrict umbra/penumbra to one connected sunspot — see `select_spot_cluster`. None
        keeps every dark pixel in the box, other spots and pores included.
    mag_filter : callable, optional
        ``B -> bool`` for the hot spot. Defaults to ``|B| > hotspot_gauss``, on ``|B|``
        rather than signed B because getting the sign wrong yields an *empty* mask rather
        than an error.

    Returns
    -------
    dict
        ``umbra``, ``penumbra``, ``both``, ``hot_spot``, ``qsun`` bool cubes; ``i_qs``;
        ``cluster_info`` (or None); ``raw_area_px`` for the footprint before the cluster
        selection; and ``umbra_thresh`` / ``penumbra_thresh``, the median absolute DN the
        fractional cuts worked out to, which the plot legends quote.

    Notes
    -----
    Every step is NaN-safe, because a gap frame on 02A's uniform time grid is entirely NaN:
    the percentile guards on there being finite pixels, and a comparison against a NaN
    threshold is False, so a gap frame simply gets empty masks.

    **Quiet sun is the complement of the raw footprint, deliberately** — computed before the
    cluster selection narrows things down. Otherwise every dark pixel the selection
    discarded, a second sunspot's umbra included, would land in the quiet-sun mask and
    contaminate the quiet-sun reference that everything downstream subtracts.
    """
    cube_cont = np.asarray(cube_cont)
    n_t = cube_cont.shape[0]

    finite = np.isfinite(cube_cont)
    i_qs = np.array([np.percentile(cube_cont[t][finite[t]], qsun_percentile)
                     if finite[t].any() else np.nan for t in range(n_t)])

    with np.errstate(invalid='ignore'):
        raw_umbra = (cube_cont < (umbra_frac * i_qs)[:, None, None]) & finite
        raw_pen   = (cube_cont < (penumbra_frac * i_qs)[:, None, None]) & finite & ~raw_umbra
    raw_both = raw_umbra | raw_pen

    qsun = finite & ~raw_both

    cluster_info = None
    if cluster_mode:
        spot, cluster_info = select_spot_cluster(
            raw_both, mode=cluster_mode, track=cluster_track, min_area=cluster_min_area)
        umbra, penumbra, both = raw_umbra & spot, raw_pen & spot, spot
    else:
        umbra, penumbra, both = raw_umbra, raw_pen, raw_both

    hot_spot = None
    if cube_mag is not None:
        if mag_filter is None:
            def mag_filter(b):
                return np.abs(b) > hotspot_gauss
        with np.errstate(invalid='ignore'):
            hot_spot = mag_filter(cube_mag) & both

    return dict(
        umbra=umbra, penumbra=penumbra, both=both, hot_spot=hot_spot, qsun=qsun,
        i_qs=i_qs, cluster_info=cluster_info,
        raw_area_px={'umbra': raw_umbra.reshape(n_t, -1).sum(axis=1),
                     'penumbra': raw_pen.reshape(n_t, -1).sum(axis=1),
                     'both': raw_both.reshape(n_t, -1).sum(axis=1)},
        umbra_thresh=float(np.nanmedian(i_qs) * umbra_frac),
        penumbra_thresh=float(np.nanmedian(i_qs) * penumbra_frac),
        cluster_mode=cluster_mode, umbra_frac=umbra_frac, penumbra_frac=penumbra_frac,
    )


#: Bit values of the mask cube `write_masks_cube` produces, also written as BIT_* keywords.
BIT_UMBRA, BIT_PENUMBRA, BIT_HOTSPOT = 1, 2, 4


def write_masks_cube(data: dict, path, header=None, history=None):
    """Write the regions in `data` as one ``uint8`` bitmask cube, for DS9.

    A single integer cube rather than three float ones: DS9 renders integers far more
    cleanly, and one file keeps the overlapping hot spot alongside the regions it sits
    inside. The bit values and threshold fractions go in as keywords rather than as a
    convention, so a reader gets what was actually used instead of assuming defaults.

    This is an **export**, not an input. `load_noaa_region` rebuilds the masks from the
    cubes every time rather than reading this file, so it can never go stale against the
    thresholds currently set in the notebook.
    """
    from src.utilities import write_cube

    masks = (data['umbra'].astype(np.uint8) * BIT_UMBRA
             | data['penumbra'].astype(np.uint8) * BIT_PENUMBRA)
    if data.get('hot_spot') is not None:
        masks |= data['hot_spot'].astype(np.uint8) * BIT_HOTSPOT

    header = fits.Header() if header is None else header.copy()
    header['BUNIT']    = ('', 'bit flags, see BIT_* keywords')
    header['BIT_UMB']  = (BIT_UMBRA, 'bit value for umbra')
    header['BIT_PEN']  = (BIT_PENUMBRA, 'bit value for penumbra')
    header['BIT_HOT']  = (BIT_HOTSPOT, 'bit value for hot spot')
    header['UMB_FRAC'] = (data.get('umbra_frac', np.nan), 'umbra threshold / I_qs')
    header['PEN_FRAC'] = (data.get('penumbra_frac', np.nan), 'penumbra threshold / I_qs')
    header['CLUSTER']  = (data.get('cluster_mode') or 'none', 'connected-component selection')

    return write_cube(masks, path, header=header, timestamps=data.get('timestamps'),
                      history=list(history or []) + [
                          '03A: 0 = quiet sun; bits overlap (hot spot is inside the spot)',
                          '03A: 0 also covers dark pixels outside the selected spot - '
                          'they are NOT quiet sun',
                          '03A: a gap frame has every bit 0'])


def load_noaa_region(
    processed_dir: str | pathlib.Path,
    region: str = 'region_01',
    **region_kwargs,
) -> dict:
    """
    Build the ``load_and_mask``-style data dict from the three corrected cubes that
    ``notebooks/02A_data_procesing.ipynb`` writes for a NOAA region.

    This is the bridge between the NOAA pipeline and everything in this module: the
    result can be passed straight to ``compute_metrics``, ``plot_magnetogram_masks``,
    ``plot_time_series``, ``plot_area``, ``plot_ffts_*`` and ``save_animation`` exactly
    like a ``load_and_mask`` result for a DS0X directory.

    **Masks are rebuilt here, every time, by `build_regions`** — never read from disk, even
    when a mask cube happens to sit next to the data. That is the whole point of the split:
    what gets analysed is always what the current ``region_kwargs`` produce, so a stale mask
    file can never silently drive the analysis, and retuning a threshold costs one 03A run
    instead of a full 02A re-correction.

    The other differences from ``load_and_mask``:

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
    **region_kwargs : forwarded to `build_regions` — thresholds, cluster mode, mag_filter.

    Returns
    -------
    dict with the same keys as ``masks_from_cubes``, plus ``timestamps``, ``present``,
    ``i_qs``, ``cluster_info``, and ``c_mean`` / ``doppler_terms`` when 02A's per-frame
    diagnostics table is present beside the cubes.
    """
    from src.utilities import read_cube

    processed_dir = pathlib.Path(processed_dir)

    cube_cont, timestamps = read_cube(processed_dir / f'{region}_continuum_cube.fits')
    cube_mag, _ = read_cube(processed_dir / f'{region}_magnetogram_corrected_cube.fits')
    cube_dop, _ = read_cube(processed_dir / f'{region}_dopplergram_calibrated_cube.fits')

    n_t = len(timestamps)
    present = {name: np.isfinite(cube).any(axis=(1, 2))
               for name, cube in [('cont', cube_cont), ('mag', cube_mag), ('dop', cube_dop)]}

    regions = build_regions(cube_cont, cube_mag, **region_kwargs)

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
        timestamps=timestamps, present=present,
        c_mean=c_mean, doppler_terms=doppler_terms,
        **{k: regions[k] for k in (
            'umbra', 'penumbra', 'both', 'hot_spot', 'qsun', 'i_qs', 'cluster_info',
            'raw_area_px', 'umbra_thresh', 'penumbra_thresh', 'cluster_mode',
            'umbra_frac', 'penumbra_frac')},
    )


def plot_calibration_frame(
    data: dict,
    frame_idx: int | None = None,
    save: bool = False,
    plots_dir: str | pathlib.Path | None = None,
) -> None:
    """
    Show one continuum frame with umbra/penumbra masks overlaid for calibration.

    Parameters
    ----------
    data      : dict returned by load_and_mask()
    frame_idx : frame to display; None → middle frame of the cube
    save      : if True, save the figure to plots_dir
    plots_dir : directory for saved figures (required when save=True)
    """
    if frame_idx is None:
        frame_idx = data['n_t'] // 2

    cube_cont       = data['cube_cont']
    umbra           = data['umbra']
    penumbra        = data['penumbra']
    umbra_thresh    = data['umbra_thresh']
    penumbra_thresh = data['penumbra_thresh']

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(cube_cont[frame_idx], cmap='gray', origin='lower')
    ax.imshow(np.where(umbra[frame_idx],    1.0, np.nan), cmap='Greens',  alpha=0.5, origin='lower', vmin=0, vmax=1)
    ax.imshow(np.where(penumbra[frame_idx], 1.0, np.nan), cmap='Purples', alpha=0.5, origin='lower', vmin=0, vmax=1)
    ax.set_xlabel('X (px)'); ax.set_ylabel('Y (px)')
    ax.set_title(f'Frame {frame_idx} — continuum with region masks')
    ax.legend(handles=[
        mpatches.Patch(color='green',  alpha=0.6, label=f'Umbra  (cont < {umbra_thresh:,})'),
        mpatches.Patch(color='purple', alpha=0.6, label=f'Penumbra  ({umbra_thresh:,}–{penumbra_thresh:,})'),
    ], loc='upper right')
    plt.tight_layout()
    _savefig(fig, plots_dir if save else None, f'calibration_frame_{frame_idx:04d}.png')
    plt.show()
    plt.close(fig)


def plot_histogram(
    data: dict,
    cube: str = "magnetogram",
    save: bool = False,
    plots_dir: str | pathlib.Path | None = None,
) -> None:
    """
    Plot the value distribution of an entire data cube across all frames and pixels.

    Parameters
    ----------
    data      : dict returned by load_and_mask()
    cube      : 'continuum', 'magnetogram', or 'dopplergram'
    save      : if True, save the figure
    plots_dir : output directory when save=True
    """

    _meta = {
        "continuum": (
            "cube_cont",
            "Intensity (DN)",
            "Continuum",
            "",
        ),
        "magnetogram": (
            "cube_mag",
            "Magnetic field strength (G)",
            "Magnetogram",
            "G",
        ),
        "dopplergram": (
            "cube_dop",
            "Doppler velocity (m/s)",
            "Dopplergram",
            "m/s",
        ),
    }

    if cube not in _meta:
        raise ValueError(
            f"cube must be one of {list(_meta.keys())}, got {cube!r}"
        )

    cube_key, xlabel, title_name, unit = _meta[cube]

    cube_data = data[cube_key]
    both = data["both"]

    values = cube_data[both]
    values = values[np.isfinite(values)]

    p1, p5, p25, p50, p75, p95, p99 = np.nanpercentile(
        values, [1, 5, 25, 50, 75, 95, 99]
    )

    fig, ax = plt.subplots(figsize=(9, 4))

    ax.hist(
        values,
        bins=200,
        color="steelblue",
        edgecolor="none",
        alpha=0.85,
    )

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Pixel count (all frames)")
    ax.set_title(
        f"{title_name} distribution — sunspot interior "
        f"({both.sum():,} pixel-frames)"
    )

    # Zero line only makes sense for signed quantities
    if cube in ("magnetogram", "dopplergram"):
        ax.axvline(
            0,
            color="black",
            lw=1.0,
            ls="--",
            label="0",
        )

    for val, lbl, col in [
        (p5, "5th pct", "orange"),
        (p50, "median", "gray"),
        (p95, "95th pct", "red"),
    ]:
        suffix = f" {unit}" if unit else ""
        ax.axvline(
            val,
            color=col,
            lw=1.2,
            ls=":",
            label=f"{lbl}: {val:.2f}{suffix}",
        )

    ax.legend(fontsize=9)
    plt.tight_layout()

    _savefig(
        fig,
        plots_dir if save else None,
        f"{title_name.lower()}_histogram.png",
    )

    plt.show()
    plt.close(fig)

    suffix = f" {unit}" if unit else ""

    print(
        f"min={values.min():.2f}"
        f"  p1={p1:.2f}"
        f"  p5={p5:.2f}"
        f"  p25={p25:.2f}"
        f"  median={p50:.2f}"
        f"  p75={p75:.2f}"
        f"  p95={p95:.2f}"
        f"  p99={p99:.2f}"
        f"  max={values.max():.2f}"
        f"{suffix}"
    )


def plot_magnetogram_masks(
    data: dict,
    frame_idx: int | None = None,
    save: bool = False,
    plots_dir: str | pathlib.Path | None = None,
) -> None:
    """
    Show one magnetogram frame with sunspot and hot_spot region overlays and a colorbar.

    The colormap is centred at zero using the 99th percentile of |B| so that
    positive (blue) and negative (red) polarities are balanced.

    Parameters
    ----------
    data      : dict returned by load_and_mask()
    frame_idx : frame to display; None → middle frame of the cube
    save      : if True, save the figure to plots_dir
    plots_dir : directory for saved figures (required when save=True)
    """
    if frame_idx is None:
        frame_idx = data['n_t'] // 2

    cube_mag        = data['cube_mag']
    umbra           = data['umbra']
    penumbra        = data['penumbra']
    hot_spot        = data.get('hot_spot')
    penumbra_thresh = data['penumbra_thresh']

    sun_spot = umbra[frame_idx] | penumbra[frame_idx]

    frame = cube_mag[frame_idx]
    finite = frame[np.isfinite(frame)]
    vmax = np.nanpercentile(np.abs(finite), 99) if finite.size else 1.0

    fig, ax = plt.subplots(figsize=(7, 7))
    im = ax.imshow(frame, cmap='bwr', origin='lower', vmin=-vmax, vmax=vmax)
    fig.colorbar(im, ax=ax, label='B  (G)', fraction=0.046, pad=0.04)

    ax.imshow(
        np.where(sun_spot, 1.0, np.nan),
        cmap='Purples', alpha=0.4, origin='lower', vmin=0, vmax=1,
    )

    legend_handles = [
        mpatches.Patch(color='purple', alpha=0.5,
                       label=f'Sun spot  (cont < {penumbra_thresh:,} DN)'),
    ]

    if hot_spot is not None:
        ax.imshow(
            np.where(hot_spot[frame_idx], 1.0, np.nan),
            cmap='Greens', alpha=0.55, origin='lower', vmin=0, vmax=1,
        )
        legend_handles.append(
            mpatches.Patch(color='green', alpha=0.7, label='Hot spot  (mag_filter)')
        )

    ax.set_xlabel('X (px)')
    ax.set_ylabel('Y (px)')
    ax.set_title(f'Frame {frame_idx} — magnetogram with region masks')
    ax.legend(handles=legend_handles, loc='upper right')
    plt.tight_layout()
    _savefig(fig, plots_dir if save else None, f'magnetogram_masks_{frame_idx:04d}.png')
    plt.show()
    plt.close(fig)


# ── phase 2: analysis ─────────────────────────────────────────────────────────

def compute_metrics(data: dict) -> dict:
    """
    Compute per-frame mean magnetogram and Doppler velocity for each region.

    Umbra / penumbra / both use the boolean masks from load_and_mask().
    Quiet sun uses the spatial mean of the dedicated qsun cubes
    (``cube_dopplergram_qsun`` / ``cube_magnetogram_qsun``) — not a mask.

    **Gap frames.** Where ``data['present']`` says a frame has no continuum (a NaN slot on
    02A's uniform time grid), the areas are set to NaN. The mean series need no such
    handling — an all-NaN frame already averages to NaN — but an area would otherwise come
    back as a perfectly confident 0 px, which plots as the spot briefly vanishing.

    Parameters
    ----------
    data : dict from load_and_mask()

    Returns
    -------
    dict with 1-D arrays: mean_mag_{umb,pen,both,quiet}, mean_dop_{umb,pen,both,quiet},
    area_{umb,pen,both}.  If ``data['hot_spot']`` is not None, also includes
    mean_mag_hotspot, area_hotspot (magnetogram only — no Doppler for hot_spot).
    """
    cube_mag      = data['cube_mag']
    cube_dop      = data['cube_dop']
    cube_mag_qsun = data['cube_mag_qsun']
    cube_dop_qsun = data['cube_dop_qsun']
    umbra         = data['umbra']
    penumbra      = data['penumbra']
    both          = data['both']
    hot_spot      = data.get('hot_spot')
    n_qsun        = cube_mag_qsun.shape[0]

    result = dict(
        mean_mag_umb   = _mean_series(cube_mag, umbra),
        mean_mag_pen   = _mean_series(cube_mag, penumbra),
        mean_mag_both  = _mean_series(cube_mag, both),
        mean_mag_quiet = np.array([np.nanmean(cube_mag_qsun[t]) for t in range(n_qsun)]),
        mean_dop_umb   = _mean_series(cube_dop, umbra),
        mean_dop_pen   = _mean_series(cube_dop, penumbra),
        mean_dop_both  = _mean_series(cube_dop, both),
        mean_dop_quiet = np.array([np.nanmean(cube_dop_qsun[t]) for t in range(n_qsun)]),
        area_umb       = umbra.sum(axis=(1, 2)).astype(float),
        area_pen       = penumbra.sum(axis=(1, 2)).astype(float),
        area_both      = both.sum(axis=(1, 2)).astype(float),
    )

    if hot_spot is not None:
        result['mean_mag_hotspot'] = _mean_series(cube_mag, hot_spot)
        result['area_hotspot']     = hot_spot.sum(axis=(1, 2)).astype(float)

    # A frame with no continuum has empty masks by construction, so its areas mean
    # "no data", not "no spot".
    present = data.get('present')
    if present is not None:
        missing = ~np.asarray(present['cont'], dtype=bool)
        for key in ('area_umb', 'area_pen', 'area_both', 'area_hotspot'):
            if key in result:
                result[key][missing] = np.nan

    return result


def save_metrics_csv(
    data: dict,
    metrics: dict,
    processed_dir: str | pathlib.Path,
    raw_dopler: bool = False,
) -> pathlib.Path:
    """
    Save the per-frame time-series metrics to a CSV file.

    Columns: time_h, area_umb, area_pen, area_both,
             mean_mag_umb, mean_mag_pen, mean_mag_both, mean_mag_quiet,
             mean_dop_umb, mean_dop_pen, mean_dop_both, mean_dop_quiet

    Parameters
    ----------
    data          : dict from load_and_mask()
    metrics       : dict from compute_metrics()
    processed_dir : directory where the CSV will be written
                    (created if it does not exist)

    Returns
    -------
    Path to the saved CSV file.
    """
    out_dir = pathlib.Path(processed_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / 'metrics.csv'

    cols = {
        'time_h'        : data['time_h'],
        'area_umb'      : metrics['area_umb'],
        'area_pen'      : metrics['area_pen'],
        'area_both'     : metrics['area_both'],
        'mean_mag_umb'  : metrics['mean_mag_umb'],
        'mean_mag_pen'  : metrics['mean_mag_pen'],
        'mean_mag_both' : metrics['mean_mag_both'],
        'mean_mag_quiet': metrics['mean_mag_quiet'],
        'mean_dop_quiet': metrics['mean_dop_quiet'],
    }
    if 'area_hotspot' in metrics:
        cols['area_hotspot']     = metrics['area_hotspot']
        cols['mean_mag_hotspot'] = metrics['mean_mag_hotspot']
    
    if not raw_dopler:
        cols['mean_dop_umb']  = metrics['mean_dop_umb']
        cols['mean_dop_pen']  = metrics['mean_dop_pen']
        cols['mean_dop_both'] = metrics['mean_dop_both']
    else:
        cols['mean_dop_umb_uncorrected']  = metrics['mean_dop_umb']
        cols['mean_dop_pen_uncorrected']  = metrics['mean_dop_pen']
        cols['mean_dop_both_uncorrected'] = metrics['mean_dop_both']

    df = pd.DataFrame(cols)
    df.to_csv(csv_path, index=False, float_format='%.4f')
    print(f'Metrics saved → {csv_path}  ({len(df)} rows)')
    return csv_path


def add_mag_residuals(data: dict, metrics: dict, degree: int = 2) -> dict:
    """
    Fit a degree-``degree`` polynomial (default: parabola) vs. time to each
    region's mean magnetogram series and store the residual (raw − fit) in
    ``metrics`` as ``mean_mag_<region>_residual``.

    This removes slow systematic trends (e.g. foreshortening/mu-angle
    effects) from the magnetogram time series before pulsation analysis.
    Generalizes the fit done ad hoc in notebooks/03_data_analisis.ipynb.

    Parameters
    ----------
    data    : dict from load_and_mask() / masks_from_cubes()
    metrics : dict from compute_metrics(); updated in place
    degree  : polynomial degree for the fit (default 2, i.e. parabolic)

    Returns
    -------
    metrics, with the added ``_residual`` keys.
    """
    time_h = data['time_h']
    for region in ('umb', 'pen', 'both', 'hotspot'):
        key = f'mean_mag_{region}'
        if key not in metrics:
            continue
        series = metrics[key]
        coeffs = np.polyfit(time_h, series, degree)
        fit = np.polyval(coeffs, time_h)
        metrics[f'{key}_residual'] = series - fit
    return metrics


def plot_time_series(
    data: dict,
    metrics: dict,
    normalized: bool = False,
    save: bool = False,
    plots_dir: str | pathlib.Path | None = None,
    mag_residual: bool = False,
    subtract_quiet: bool = False,
) -> None:
    """
    Plot mean B and Doppler velocity vs time for umbra, penumbra, both, and quiet sun.

    Parameters
    ----------
    data       : dict from load_and_mask()
    metrics    : dict from compute_metrics()
    normalized : if True, min-max normalize each series before plotting
    save       : if True, save the figure to plots_dir
    plots_dir  : directory for saved figures (required when save=True)
    """
    time_h = data['time_h']

    def _norm(arr):
        lo, hi = np.nanmin(arr), np.nanmax(arr)
        return (arr - lo) / (hi - lo) if hi > lo else arr

    proc = _norm if normalized else (lambda x: x)
    ylabel_mag = 'Norm. mean B'        if normalized else 'Mean B  (G)'
    ylabel_dop = 'Norm. mean velocity' if normalized else 'Mean velocity  (m/s)'
    suffix     = '_normalized'         if normalized else ''
    suffix     += '_residual'          if mag_residual else ''
    suffix     += '_minus_quiet'       if subtract_quiet else ''

    if not mag_residual:
        series = [
            ('Umbra',     metrics['mean_mag_umb'],   metrics['mean_dop_umb'],   'red'),
            ('Penumbra',  metrics['mean_mag_pen'],   metrics['mean_dop_pen'],   'blue'),
            ('Both',      metrics['mean_mag_both'],  metrics['mean_dop_both'],  'purple'),
            ('Quiet Sun', metrics['mean_mag_quiet'], metrics['mean_dop_quiet'], 'black'),
        ]
    else:
         series = [
            ('Umbra',     metrics['mean_mag_umb_residual'],   metrics['mean_dop_umb'],   'red'),
            ('Penumbra',  metrics['mean_mag_pen_residual'],   metrics['mean_dop_pen'],   'blue'),
            ('Both',      metrics['mean_mag_both_residual'],  metrics['mean_dop_both'],  'purple'),
            ('Quiet Sun', None, metrics['mean_dop_quiet'], 'black'),
        ]

    hotspot_mag = metrics.get('mean_mag_hotspot')

    if subtract_quiet:
        mag_quiet = metrics['mean_mag_quiet']
        dop_quiet = metrics['mean_dop_quiet']
        series = [
            (name, mag - mag_quiet if mag is not None else None,
                   dop - dop_quiet if dop is not None else None, color)
            for name, mag, dop, color in series
            if name != 'Quiet Sun'
        ]
        ylabel_mag = ('Norm. mean B (rel. quiet)' if normalized else 'Mean B − quiet sun  (G)')
        ylabel_dop = ('Norm. mean velocity (rel. quiet)' if normalized else 'Mean velocity − quiet sun  (m/s)')
        if hotspot_mag is not None:
            hotspot_mag = hotspot_mag - mag_quiet

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    for name, mag, dop, color in series:
        if mag is not None:
            ax1.plot(time_h, proc(mag), color=color, lw=0.8, label=name)
        if dop is not None:
            ax2.plot(time_h, proc(dop), color=color, lw=0.8, label=name)

    if hotspot_mag is not None:
        ax1.plot(time_h, proc(hotspot_mag), color='darkorange', lw=0.8, label='Hot spot')

    ax1.set_ylabel(ylabel_mag); ax1.set_title('Mean magnetogram' + suffix.replace('_', ' '))
    ax1.grid(alpha=0.3); ax1.legend()
    ax2.set_ylabel(ylabel_dop); ax2.set_xlabel('Time  (h)')
    ax2.set_title('Mean dopplergram' + suffix.replace('_', ' '))
    ax2.grid(alpha=0.3); ax2.legend()

    plt.tight_layout()
    _savefig(fig, plots_dir if save else None, f'time_series{suffix}.png')
    plt.show()
    plt.close(fig)


def plot_area(
    data: dict,
    metrics: dict,
    save: bool = False,
    plots_dir: str | pathlib.Path | None = None,
) -> None:
    """
    Plot sunspot region area (pixel count) vs time.

    Shows umbra, penumbra, both, and hot_spot (when available) in a single panel.

    Parameters
    ----------
    data      : dict from load_and_mask()
    metrics   : dict from compute_metrics()
    save      : if True, save the figure to plots_dir
    plots_dir : directory for saved figures (required when save=True)
    """
    time_h = data['time_h']

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(time_h, metrics['area_pen'],  color='purple', lw=0.9, label='Penumbra')
    ax.plot(time_h, metrics['area_umb'],  color='orange', lw=0.9, label='Umbra')
    ax.plot(time_h, metrics['area_both'], color='steelblue', lw=0.9, label='Both')
    if 'area_hotspot' in metrics:
        ax.plot(time_h, metrics['area_hotspot'], color='red', lw=0.9, label='Hot spot')
    ax.set_xlabel('Time  (h)')
    ax.set_ylabel('Area  (pixels)')
    ax.set_title('Region area vs time')
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    _savefig(fig, plots_dir if save else None, 'area_vs_time.png')
    plt.show()
    plt.close(fig)


# ── phase 3: the 24 h oscillation ─────────────────────────────────────────────

#: Period of the oscillation `fit_diurnal` looks for, in hours.
DIURNAL_PERIOD_H = 24.0


def fit_diurnal(
    time_h: np.ndarray,
    series: np.ndarray,
    window: tuple[float, float] | None = None,
    period_h: float = DIURNAL_PERIOD_H,
    mask: np.ndarray | None = None,
    label: str = '',
) -> dict:
    """Fit ``c + a sin(2 pi t / P) + b cos(2 pi t / P)`` over a stretch of the series.

    The period is **fixed**, which is what makes this worth doing at all: with `P` held at
    24 h the model is linear in ``(c, a, b)``, so it is one `np.linalg.lstsq` with an exact
    answer — no initial guess, no convergence failure, no local minimum. Fitting
    ``A sin(2 pi t / P + phi) + c`` with `scipy.optimize.curve_fit` describes the same
    curve but has to be started somewhere and can fail; the amplitude and phase come out of
    the linear fit anyway as ``hypot(a, b)`` and ``atan2(b, a)``.

    Parameters
    ----------
    time_h : ndarray
        Elapsed hours, i.e. ``data['time_h']``. Zero is the first frame of the cube.
    series : ndarray
        The quantity to fit, same length. NaNs (gap frames, empty masks) are dropped.
    window : (float, float), optional
        ``(t_start, t_end)`` in hours, inclusive of both ends. Default: the whole series.
    period_h : float
        The period to fit. Changing it changes what the "amplitude" means, so it is
        recorded in the result.
    mask : ndarray of bool, optional
        Extra points to exclude. Pass a mask shared between two series so their fits are
        directly comparable — see the note on the instrumental control below.
    label : str
        Used only in the warning messages, so a marginal fit can be traced to its region.

    Returns
    -------
    dict
        ``intercept``, ``amplitude``, ``phase_rad``, ``t_max`` (hours after the window
        start at which the fitted curve peaks), the matching ``sigma_intercept`` /
        ``sigma_amplitude``, ``n`` points used, ``rms_residual``, ``period_h`` and
        ``window``. Every numeric field is NaN when there was too little data, rather than
        raising, so one thin window cannot abort a loop over regions.

    Notes
    -----
    **Read the uncertainty, not just the amplitude.** Over a window one period long the
    sinusoid completes exactly one cycle, and amplitude, phase and intercept are then only
    weakly separated — they trade against each other and `sigma_amplitude` is what says so.
    A window of 1.5 periods or more is much better constrained; below that this warns.

    **24 h is also the period of the instrument.** SDO's line-of-sight velocity is
    dominated by ``OBS_VR``, which varies diurnally with its geosynchronous orbit, so any
    residual of the ``v_SDO`` correction in `src/doppler_calibration.py` lands at exactly
    the period fitted here. Because this fit is linear, fitting ``umbra - quiet_sun`` gives
    coefficients *exactly* equal to the umbral fit's minus the quiet sun's, provided both
    used the same points — which is what `mask` is for. Comparing the two amplitudes is
    therefore the control: close together means the signal is umbral, an absolute amplitude
    much larger than the quiet-subtracted one means most of it is common to the whole box.
    """
    time_h = np.asarray(time_h, dtype=float)
    series = np.asarray(series, dtype=float)
    window = (float(time_h.min()), float(time_h.max())) if window is None else (
        float(window[0]), float(window[1]))

    inside = (time_h >= window[0]) & (time_h <= window[1]) & np.isfinite(series)
    if mask is not None:
        inside &= np.asarray(mask, dtype=bool)
    n = int(inside.sum())
    # Three free parameters, so four points is the minimum that leaves anything to check
    # the fit against. `n` is reported even when the fit is refused, so the table says how
    # close the window came rather than a bare zero.
    if n < 4:
        warnings.warn(f'fit_diurnal{f" [{label}]" if label else ""}: only {n} finite '
                      f'point(s) in {window[0]:g}-{window[1]:g} h — no fit', stacklevel=2)
        return dict(intercept=np.nan, amplitude=np.nan, phase_rad=np.nan, t_max=np.nan,
                    sigma_intercept=np.nan, sigma_amplitude=np.nan, n=n,
                    rms_residual=np.nan, period_h=period_h, window=window,
                    coefficients=np.full(3, np.nan))

    span = window[1] - window[0]
    if span < 1.5 * period_h:
        warnings.warn(
            f'fit_diurnal{f" [{label}]" if label else ""}: the {span:g} h window is only '
            f'{span / period_h:.2f} periods long. Amplitude, phase and intercept are '
            f'poorly separated over so little of a cycle — read sigma_amplitude before '
            f'reading amplitude.', stacklevel=2)

    t, y = time_h[inside], series[inside]
    omega = 2 * np.pi / period_h
    design = np.column_stack([np.ones(n), np.sin(omega * t), np.cos(omega * t)])

    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    c, a, b = coef
    residual = y - design @ coef
    amplitude = float(np.hypot(a, b))

    # Parameter covariance the textbook way: residual variance times inv(X'X). With only
    # as many points as parameters there is no residual left to estimate it from.
    dof = n - 3
    if dof > 0:
        var = float(residual @ residual) / dof
        cov = var * np.linalg.pinv(design.T @ design)
        sigma_c = float(np.sqrt(max(cov[0, 0], 0.0)))
        # sigma_A^2 = (a^2 var_a + b^2 var_b + 2 a b cov_ab) / A^2
        sigma_a2 = (a * a * cov[1, 1] + b * b * cov[2, 2] + 2 * a * b * cov[1, 2])
        sigma_amp = float(np.sqrt(max(sigma_a2, 0.0)) / amplitude) if amplitude > 0 else np.nan
    else:
        sigma_c = sigma_amp = np.nan

    # Where the fitted curve peaks. a sin + b cos peaks where omega t = atan2(a, b), which
    # is easier to read against the plot than a phase offset in radians.
    phase = float(np.arctan2(b, a))
    t_peak = float(np.arctan2(a, b) / omega)
    t_max = window[0] + (t_peak - window[0]) % period_h

    return dict(
        intercept=float(c), amplitude=amplitude, phase_rad=phase, t_max=t_max,
        sigma_intercept=sigma_c, sigma_amplitude=sigma_amp, n=n,
        rms_residual=float(np.sqrt(np.mean(residual ** 2))),
        period_h=period_h, window=window, coefficients=coef,
    )


def diurnal_curve(fit: dict, time_h: np.ndarray) -> np.ndarray:
    """Evaluate a `fit_diurnal` result at arbitrary times, for plotting over the data."""
    c, a, b = fit['coefficients']
    omega = 2 * np.pi / fit['period_h']
    t = np.asarray(time_h, dtype=float)
    return c + a * np.sin(omega * t) + b * np.cos(omega * t)


def plot_diurnal_fits(
    rows: list[dict],
    series_key: str = 'fit_rel',
    save: bool = False,
    plots_dir: str | pathlib.Path | None = None,
) -> None:
    """One panel per fitted row: the data, the window, and the fitted curve.

    This is the check that has to happen *before* reading anything off the amplitude
    scatter. A fit that latched onto a download gap, a segmentation step or a slow trend
    still produces a perfectly respectable-looking number; the only way to catch it is to
    look at the curve sitting on the points.

    Parameters
    ----------
    rows : list of dict
        As built by the fitting cell of 03A: each needs ``label``, ``time_h``, ``series``
        (or ``series_rel``), the fit under `series_key`, and ``window``.
    series_key : {'fit_rel', 'fit_abs'}
        Which of the two fits to draw. The series drawn alongside matches it.
    """
    rows = [r for r in rows if np.isfinite(r[series_key]['amplitude'])]
    if not rows:
        print('plot_diurnal_fits: nothing to draw — every fit failed')
        return

    value_key = 'series_rel' if series_key == 'fit_rel' else 'series'
    ylabel = ('Umbra − quiet sun  (m/s)' if series_key == 'fit_rel'
              else 'Umbra, absolute  (m/s)')

    n_cols = min(2, len(rows))
    n_rows = int(np.ceil(len(rows) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6.5 * n_cols, 3.2 * n_rows),
                             squeeze=False)

    for ax, row in zip(axes.ravel(), rows):
        fit = row[series_key]
        t, y = row['time_h'], row[value_key]
        t0, t1 = fit['window']

        # The whole series in grey for context, the fitted stretch on top of it — so a
        # window that sits on an unrepresentative piece of the record is obvious.
        ax.plot(t, y, color='0.8', lw=0.7, zorder=1)
        inside = (t >= t0) & (t <= t1)
        ax.plot(t[inside], y[inside], color='steelblue', lw=0.9, zorder=2, label='umbra')

        dense = np.linspace(t0, t1, 400)
        ax.plot(dense, diurnal_curve(fit, dense), color='crimson', lw=1.8, zorder=3,
                label=f'{fit["period_h"]:g} h fit')
        ax.axhline(fit['intercept'], color='crimson', lw=0.8, ls=':', zorder=3)
        ax.axvspan(t0, t1, color='gold', alpha=0.12, zorder=0)

        ax.set_title(f'{row["label"]}   A = {fit["amplitude"]:.1f} ± '
                     f'{fit["sigma_amplitude"]:.1f} m/s   (n = {fit["n"]})', fontsize=10)
        ax.set_xlabel('Time  (h)')
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc='upper right')

    for ax in axes.ravel()[len(rows):]:
        ax.set_visible(False)

    plt.tight_layout()
    _savefig(fig, plots_dir if save else None, f'diurnal_{series_key}.png')
    plt.show()
    plt.close(fig)


def plot_amplitude_vs_depression(
    rows: list[dict],
    depth_key: str = 'z_div',
    save: bool = False,
    plots_dir: str | pathlib.Path | None = None,
    normalize_area = False
) -> None:
    """Fitted 24 h amplitude against the Wilson depression reported for each region.

    Both fitted series are drawn: the absolute umbral velocity and the same thing with the
    quiet sun subtracted. That pair is the instrumental control — see `fit_diurnal`'s
    notes. If the two markers for a region sit on top of each other the oscillation is
    umbral; if the absolute one is far higher, most of that amplitude is common to the
    whole box and is more likely a residual of the diurnal ``v_SDO`` correction than a
    property of the sunspot.

    Pearson r is annotated per series, with the number of points it was computed from.
    With five points it is a description of this sample, not evidence of a relationship.
    """
    usable = [r for r in rows if np.isfinite(r['fit_rel']['amplitude'])
              and np.isfinite(r[depth_key])]
    if len(usable) < 2:
        print(f'plot_amplitude_vs_depression: only {len(usable)} usable point(s)')
        return

    depth_label = {'z_div': r'$z_{W,\mathrm{div}}$', 'z_press': r'$z_{W,\mathrm{press}}$'}
    fig, ax = plt.subplots(figsize=(8, 6))

    for key, color, marker, name in [
            ('fit_abs', 'darkorange', 'o', 'Umbra, absolute'),
            ('fit_rel', 'steelblue', 's', 'Umbra − quiet sun')]:
        x = np.array([r[depth_key] for r in usable], dtype=float)

        if normalize_area:
            y = np.array([r[key]['amplitude'] / r['area'] for r in usable], dtype=float)
        else:
            y = np.array([r[key]['amplitude'] for r in usable], dtype=float)
        e = np.array([r[key]['sigma_amplitude'] for r in usable], dtype=float)
        ax.errorbar(x, y, yerr=e, fmt=marker, color=color, ms=7, capsize=3, lw=0,
                    elinewidth=1, label=name)

        finite = np.isfinite(x) & np.isfinite(y)
        if finite.sum() > 2:
            r_p = float(np.corrcoef(x[finite], y[finite])[0, 1])
            ax.plot([], [], ' ', label=f'   r = {r_p:+.2f}  (n = {int(finite.sum())})')

    # Label each point once, next to the quiet-subtracted marker.
    for row in usable:
        ax.annotate(row['label'], (row[depth_key], row['fit_rel']['amplitude']),
                    textcoords='offset points', xytext=(7, 4), fontsize=7, color='0.3')

    period = usable[0]['fit_rel']['period_h']
    ax.set_xlabel(f'Wilson depression {depth_label.get(depth_key, depth_key)}  (km)')
    ax.set_ylabel(f'Fitted {period:g} h amplitude  (m/s)')
    ax.set_title(f'{period:g} h umbral Doppler amplitude vs Wilson depression')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    plt.tight_layout()
    _savefig(fig, plots_dir if save else None, f'amplitude_vs_{depth_key}.png')
    plt.show()
    plt.close(fig)


def _fft_regions(metrics: dict, cadence_s: float) -> list[tuple[str, np.ndarray, np.ndarray, str]]:
    """Return list of (name, f_mhz, amplitude, color) for the four regions."""
    return [
        ('Umbra',     *_fft(metrics['mean_dop_umb'],   cadence_s), 'green'),
        ('Penumbra',  *_fft(metrics['mean_dop_pen'],   cadence_s), 'purple'),
        ('Both',      *_fft(metrics['mean_dop_both'],  cadence_s), 'steelblue'),
        ('Quiet Sun', *_fft(metrics['mean_dop_quiet'], cadence_s), 'darkorange'),
    ]


def plot_ffts_separate(
    data: dict,
    metrics: dict,
    annotate_peaks: bool = True,
    print_table: bool = True,
    save: bool = False,
    plots_dir: str | pathlib.Path | None = None,
) -> None:
    """
    2×2 subplot FFT amplitude for each region with optional peak annotation.

    Parameters
    ----------
    data          : dict from load_and_mask()
    metrics       : dict from compute_metrics()
    annotate_peaks: draw vertical dashed lines and period labels on top-5 peaks
    print_table   : print the peak table to stdout
    save          : if True, save the figure and peak table CSV to plots_dir
    plots_dir     : directory for saved outputs (required when save=True)
    """
    cadence_s = data['cadence_s']
    regions = _fft_regions(metrics, cadence_s)
    f_nyq = 1e3 / (2 * cadence_s)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    for ax, (name, f_mhz, amp, color) in zip(axes.flat, regions):
        ax.plot(f_mhz, amp, color=color, lw=0.8)
        peaks, _ = find_peaks(amp, height=np.percentile(amp, 85), distance=5)
        top5 = peaks[np.argsort(amp[peaks])[::-1][:5]]
        if annotate_peaks:
            for idx in sorted(top5):
                period_min = 1000 / (f_mhz[idx] * 60)
                ax.axvline(f_mhz[idx], color=color, lw=0.7, ls='--', alpha=0.5)
                ax.text(f_mhz[idx], amp[idx] * 1.05,
                        f'{period_min:.0f} min', fontsize=7, color=color,
                        rotation=90, va='bottom', ha='center')
        ax.set_title(name, color=color, fontweight='bold')
        ax.set_ylabel('Amplitude (arb. units)')
        ax.set_xlabel('Frequency (mHz)')
        ax.grid(alpha=0.25)

    fig.suptitle(
        f'FFT Amplitude of Mean Doppler per Region\n'
        f'(cadence {cadence_s:.0f} s  |  Nyquist ≈ {f_nyq:.3f} mHz)',
        fontsize=12)
    plt.tight_layout()
    _savefig(fig, plots_dir if save else None, 'fft_separate.png')
    plt.show()
    plt.close(fig)

    _out_dir = pathlib.Path(plots_dir) if (save and plots_dir is not None) else None
    if print_table:
        _print_peak_table(regions, _out_dir)
    elif _out_dir is not None:
        _save_peak_table_csv(regions, _out_dir)


def plot_ffts_combined(
    data: dict,
    metrics: dict,
    xlim: tuple[float, float] | None = None,
    annotate_peaks: bool = True,
    print_table: bool = True,
    save: bool = False,
    plots_dir: str | pathlib.Path | None = None,
) -> None:
    """
    Overlay FFT amplitudes for all regions on a single axis.

    Parameters
    ----------
    data          : dict from load_and_mask()
    metrics       : dict from compute_metrics()
    xlim          : (f_min, f_max) in mHz to zoom the x-axis; None → full range
    annotate_peaks: draw vertical dashed lines for top-5 peaks per region
    print_table   : print the peak table to stdout
    save          : if True, save the figure and peak table CSV to plots_dir
    plots_dir     : directory for saved outputs (required when save=True)
    """
    cadence_s = data['cadence_s']
    regions = _fft_regions(metrics, cadence_s)
    f_nyq = 1e3 / (2 * cadence_s)

    fig, ax = plt.subplots(figsize=(14, 6))
    for name, f_mhz, amp, color in regions:
        ax.plot(f_mhz, amp, color=color, lw=0.8, label=name)
        if annotate_peaks:
            peaks, _ = find_peaks(amp, height=np.percentile(amp, 85), distance=5)
            top5 = peaks[np.argsort(amp[peaks])[::-1][:5]]
            for idx in sorted(top5):
                ax.axvline(float(f_mhz[idx]), color=color, lw=0.7, ls='--', alpha=0.5)

    ax.set_ylabel('Amplitude (arb. units)')
    ax.set_xlabel('Frequency (mHz)')
    ax.grid(alpha=0.25)
    ax.legend()
    if xlim is not None:
        ax.set_xlim(*xlim)

    ax.set_title(
        f'FFT Amplitude of Mean Doppler per Region\n'
        f'(cadence {cadence_s:.0f} s  |  Nyquist ≈ {f_nyq:.3f} mHz)',
        fontsize=12)
    plt.tight_layout()
    _savefig(fig, plots_dir if save else None, 'fft_combined.png')
    plt.show()
    plt.close(fig)

    _out_dir = pathlib.Path(plots_dir) if (save and plots_dir is not None) else None
    if print_table:
        _print_peak_table(regions, _out_dir)
    elif _out_dir is not None:
        _save_peak_table_csv(regions, _out_dir)


def _build_peak_rows(regions: list) -> list[dict]:
    rows = []
    for name, f_mhz, amp, _ in regions:
        peaks, _ = find_peaks(amp, height=np.percentile(amp, 85), distance=5)
        top5 = peaks[np.argsort(amp[peaks])[::-1][:5]]
        a_max = amp.max()
        for rank, idx in enumerate(top5, 1):
            rows.append({
                'region'       : name,
                'rank'         : rank,
                'freq_mhz'     : round(float(f_mhz[idx]), 4),
                'period_min'   : round(1000 / (f_mhz[idx] * 60), 1),
                'rel_amplitude': round(float(amp[idx] / a_max), 4),
            })
    return rows


def _save_peak_table_csv(regions: list, plots_dir: pathlib.Path) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)
    csv_path = plots_dir / 'fft_peaks.csv'
    pd.DataFrame(_build_peak_rows(regions)).to_csv(csv_path, index=False)
    print(f'FFT peak table saved → {csv_path}')


def _print_peak_table(regions: list, plots_dir: pathlib.Path | None = None) -> None:
    print(f'\n{"Region":<12}  {"Rank":>4}  {"Freq (mHz)":>11}  {"Period (min)":>13}  {"Rel. amplitude":>14}')
    print('─' * 60)
    rows = _build_peak_rows(regions)
    last_region = None
    for row in rows:
        if row['region'] != last_region and last_region is not None:
            print()
        last_region = row['region']
        print(f'{row["region"]:<12}  {row["rank"]:>4}  {row["freq_mhz"]:>11.4f}  '
              f'{row["period_min"]:>13.1f}  {row["rel_amplitude"]:>14.4f}')
    print()
    if plots_dir is not None:
        _save_peak_table_csv(regions, plots_dir)


# ── animation ─────────────────────────────────────────────────────────────────

def save_animation(
    data: dict,
    metrics: dict,
    save_path: str | pathlib.Path,
    step: int = 50,
    fps: int = 5,
    embed_limit_mb: float = 50.0,
    mag_symmetric_cbar: bool = True,
    embed_frames: bool = True,
) -> None:
    """
    Save a 3-channel (continuum / magnetogram / dopplergram) animation as HTML.

    Parameters
    ----------
    data               : dict from load_and_mask()
    metrics            : dict from compute_metrics()
    save_path          : output .html file path (parent dirs created if needed)
    step               : subsample every Nth frame to keep file size manageable
    fps                : frames per second
    embed_limit_mb     : matplotlib animation size limit in MB
    embed_frames       : if True (default) the frames are base64'd into the .html, so it
                         is a single self-contained file that can be moved or shared.
                         False writes them to a sibling ``<name>_frames/`` directory
                         instead — smaller, but the .html breaks if it is moved without
                         that directory. Raise ``step`` if an embedded file gets too big.
    mag_symmetric_cbar : if True, use a symmetric colorbar for the magnetogram
                         channel centred at zero (vmin = −vmax where vmax is the
                         99th percentile of |B| across all frames).  Default False
                         keeps the original [2nd, 98th] percentile limits.
    """
    save_path = pathlib.Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    matplotlib.rcParams['animation.embed_limit'] = embed_limit_mb

    cube_cont    = data['cube_cont']
    cube_mag     = data['cube_mag']
    cube_dop     = data['cube_dop']
    umbra        = data['umbra']
    penumbra     = data['penumbra']
    n_t          = data['n_t']
    cadence_s    = data['cadence_s']
    mean_mag_umb = metrics['mean_mag_umb']
    mean_mag_pen = metrics['mean_mag_pen']

    frames_idx = np.arange(0, n_t, step)
    cubes_ch   = [cube_cont,        cube_mag,           cube_dop]
    cmaps_ch   = ['gray',           'bwr',              'RdBu_r']
    labels_ch  = ['Continuum (DN)', 'Magnetogram (G)',  'Dopplergram (m/s)']
    _sample    = np.arange(0, n_t, max(1, n_t // 50))
    clims      = [np.nanpercentile(c[_sample], [2, 98]) for c in cubes_ch]

    if mag_symmetric_cbar:
        mag_vmax = float(np.nanpercentile(np.abs(cube_mag[_sample]), 99))
        clims[1] = np.array([-mag_vmax, mag_vmax])

    _LEGEND_HANDLES = [
        mpatches.Patch(color=(0.0, 0.85, 0.0, 0.75), label='Umbra'),
        mpatches.Patch(color=(0.55, 0.0, 1.0, 0.75), label='Penumbra'),
    ]

    def _make_filled_rgba(umb, pen):
        """Filled colour overlay for the continuum panel."""
        h, w = umb.shape
        rgba = np.zeros((h, w, 4), dtype=float)
        rgba[pen, 0] = 0.55; rgba[pen, 2] = 1.0;  rgba[pen, 3] = 0.40
        rgba[umb, 0] = 0.0;  rgba[umb, 1] = 0.85; rgba[umb, 3] = 0.45
        return rgba

    def _make_outline_rgba(umb, pen, thickness=2):
        """Border-only overlay for magnetogram and dopplergram panels."""
        h, w = umb.shape
        rgba = np.zeros((h, w, 4), dtype=float)
        pen_border = ndimage.binary_dilation(pen, iterations=thickness) & ~pen
        umb_border = ndimage.binary_dilation(umb, iterations=thickness) & ~umb
        rgba[pen_border, 0] = 0.55; rgba[pen_border, 2] = 1.0;  rgba[pen_border, 3] = 0.95
        rgba[umb_border, 0] = 0.0;  rgba[umb_border, 1] = 0.85; rgba[umb_border, 3] = 0.95
        return rgba

    fig, axes = plt.subplots(1, 3, figsize=(19, 6))
    fig.subplots_adjust(wspace=0.55, top=0.88, left=0.06, right=0.97)
    ims = []; overlays = []

    for idx, (ax, cube, cmap, clim, lbl) in enumerate(
            zip(axes, cubes_ch, cmaps_ch, clims, labels_ch)):
        im = ax.imshow(cube[0], origin='lower', cmap=cmap,
                       vmin=clim[0], vmax=clim[1], interpolation='nearest')
        fig.colorbar(im, ax=ax, label=lbl, fraction=0.046, pad=0.06)
        ov_data = (_make_filled_rgba if idx == 0 else _make_outline_rgba)(umbra[0], penumbra[0])
        ov = ax.imshow(ov_data, origin='lower', interpolation='nearest')
        ax.set_xlabel('X (px)')
        ax.set_ylabel('Y (px)')
        ax.text(0.02, 0.02, f'Cadence: {cadence_s:.0f} s', transform=ax.transAxes,
                color='white', fontsize=9, va='bottom',
                bbox=dict(boxstyle='round,pad=0.2', fc='black', alpha=0.6))
        ax.legend(handles=_LEGEND_HANDLES, loc='upper right', fontsize=7)
        ims.append(im); overlays.append(ov)

    suptitle = fig.suptitle('', fontsize=11)

    def _update(i):
        t = frames_idx[i]
        for j, (im, ov, cube) in enumerate(zip(ims, overlays, cubes_ch)):
            im.set_data(cube[t])
            ov_data = (_make_filled_rgba if j == 0 else _make_outline_rgba)(umbra[t], penumbra[t])
            ov.set_data(ov_data)
        elapsed_h = data['time_h'][t]
        b_u = f'{mean_mag_umb[t]:.0f}' if np.isfinite(mean_mag_umb[t]) else 'N/A'
        b_p = f'{mean_mag_pen[t]:.0f}' if np.isfinite(mean_mag_pen[t]) else 'N/A'
        suptitle.set_text(
            f'SDO/HMI  |  frame {t:04d}  |  t = {elapsed_h:.2f} h  |  '
            f'<B> umb={b_u} G   pen={b_p} G'
        )
        return ims + overlays

    anim = FuncAnimation(fig, _update, frames=len(frames_idx),
                         interval=1000 // fps, blit=False)
    plt.close(fig)
    # HTMLWriter defaults to embed_frames=False, which scatters the frames into a sibling
    # directory the .html then depends on. Pass the writer explicitly so the default here
    # is a single portable file, as embed_limit_mb always implied.
    anim.save(str(save_path),
              writer=HTMLWriter(fps=fps, embed_frames=embed_frames))
    size_mb = save_path.stat().st_size / 1e6
    print(f'Animation saved → {save_path}  ({len(frames_idx)} frames, {size_mb:.1f} MB'
          f'{"" if embed_frames else ", frames in a sibling directory"})')

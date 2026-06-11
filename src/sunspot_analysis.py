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

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.io import fits
from matplotlib.animation import FuncAnimation
from scipy import ndimage
from scipy.io import readsav
from scipy.signal import find_peaks


# ── private helpers ───────────────────────────────────────────────────────────

def _load_cube(path: pathlib.Path, fill: float) -> np.ndarray:
    with fits.open(path) as hdul:
        arr = hdul[0].data.astype(float)
    return np.where(np.abs(arr) > fill, np.nan, arr)


def _keep_central_cluster(binary_img: np.ndarray, mode: str) -> np.ndarray:
    labeled, n = ndimage.label(binary_img)
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
    return np.array([
        np.nanmean(cube[t][mask[t]]) if mask[t].any() else np.nan
        for t in range(cube.shape[0])
    ])


def _savefig(fig: plt.Figure, plots_dir: pathlib.Path | None, filename: str) -> None:
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
) -> float:
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
    mag_filter: 'callable | None' = None,
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
    cube_dop      = _load_cube(ds_dir / 'cube_dopplergram_corrected.fits', fill)
    cube_dop_qsun = _load_cube(ds_dir / 'cube_dopplergram_qsun.fits', fill)
    cube_mag_qsun = _load_cube(ds_dir / 'cube_magnetogram_qsun.fits', fill)

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
    both         = (cube_cont < penumbra_thresh) & np.isfinite(cube_cont)

    if filter_mask:
        umbra    = np.array([_keep_central_cluster(umbra_raw[t],    cluster_mode) for t in range(n_t)], dtype=bool)
        penumbra = np.array([_keep_central_cluster(penumbra_raw[t], cluster_mode) for t in range(n_t)], dtype=bool)
    else:
        umbra = umbra_raw
        penumbra = penumbra_raw

    hot_spot = None
    if mag_filter is not None:
        hot_spot = mag_filter(cube_mag) & both

    return dict(
        cube_cont=cube_cont, cube_mag=cube_mag, cube_dop=cube_dop,
        cube_dop_qsun=cube_dop_qsun, cube_mag_qsun=cube_mag_qsun,
        umbra=umbra, penumbra=penumbra, both=both, hot_spot=hot_spot,
        time_h=time_h, n_t=n_t, cadence_s=median_cadence,
        umbra_thresh=umbra_thresh, penumbra_thresh=penumbra_thresh,
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


def plot_histogram(
    data: dict,
    cube: str = 'magnetogram',
    save: bool = False,
    plots_dir: str | pathlib.Path | None = None,
) -> None:
    """
    Plot the value distribution of an entire data cube across all frames and pixels.

    Parameters
    ----------
    data      : dict returned by load_and_mask()
    cube      : which cube to histogram — 'continuum', 'magnetogram', or 'dopplergram'
    save      : if True, save the figure to plots_dir
    plots_dir : directory for saved figures (required when save=True)
    """
    _meta = {
        'continuum':   ('cube_cont', 'Intensity (DN)',              'Continuum'),
        'magnetogram': ('cube_mag',  'Magnetic field strength (G)', 'Magnetogram'),
        'dopplergram': ('cube_dop',  'Doppler velocity (m/s)',       'Dopplergram'),
    }
    if cube not in _meta:
        raise ValueError(f"cube must be 'continuum', 'magnetogram', or 'dopplergram', got {cube!r}")

    data_key, xlabel, title_label = _meta[cube]
    arr = data[data_key]

    # Percentiles on the 3D array — no flat copy needed
    p1, p5, p25, p50, p75, p95, p99 = np.nanpercentile(arr, [1, 5, 25, 50, 75, 95, 99])
    n_finite = int(np.isfinite(arr).sum())

    # Pre-compute bin counts from finite values, then discard the flat array.
    # Passing millions of raw points to ax.hist() stalls / OOMs the kernel.
    finite_vals = arr[np.isfinite(arr)]
    counts, edges = np.histogram(finite_vals, bins=300)
    vmin, vmax_val = float(edges[0]), float(edges[-1])
    del finite_vals

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.stairs(counts, edges, fill=True, color='steelblue', alpha=0.85)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Pixel count (all frames)')
    ax.set_title(f'{title_label} distribution — full cube  ({n_finite:,} pixel-frames)')

    if cube == 'magnetogram':
        ax.axvline(0, color='black', lw=1.0, ls='--', label='B = 0')
    for val, lbl, col in [(p5, '5th pct', 'orange'), (p50, 'median', 'gray'), (p95, '95th pct', 'red')]:
        ax.axvline(val, color=col, lw=1.2, ls=':', label=f'{lbl}: {val:.1f}')
    ax.legend(fontsize=9)
    plt.tight_layout()
    _savefig(fig, plots_dir if save else None, f'histogram_{cube}.png')
    plt.show()

    unit = 'G' if cube == 'magnetogram' else ('DN' if cube == 'continuum' else 'm/s')
    print(
        f'min={vmin:.1f}  p1={p1:.1f}  p5={p5:.1f}  p25={p25:.1f}  '
        f'median={p50:.1f}  p75={p75:.1f}  p95={p95:.1f}  p99={p99:.1f}  '
        f'max={vmax_val:.1f}  [{unit}]'
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
            cmap='Oranges', alpha=0.55, origin='lower', vmin=0, vmax=1,
        )
        legend_handles.append(
            mpatches.Patch(color='orange', alpha=0.7, label='Hot spot  (mag_filter)')
        )

    ax.set_xlabel('X (px)')
    ax.set_ylabel('Y (px)')
    ax.set_title(f'Frame {frame_idx} — magnetogram with region masks')
    ax.legend(handles=legend_handles, loc='upper right')
    plt.tight_layout()
    _savefig(fig, plots_dir if save else None, f'magnetogram_masks_{frame_idx:04d}.png')
    plt.show()


# ── phase 2: analysis ─────────────────────────────────────────────────────────

def compute_metrics(data: dict) -> dict:
    """
    Compute per-frame mean magnetogram and Doppler velocity for each region.

    Umbra / penumbra / both use the boolean masks from load_and_mask().
    Quiet sun uses the spatial mean of the dedicated qsun cubes
    (``cube_dopplergram_qsun`` / ``cube_magnetogram_qsun``) — not a mask.

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

    return result


def save_metrics_csv(
    data: dict,
    metrics: dict,
    processed_dir: str | pathlib.Path,
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
        'mean_dop_umb'  : metrics['mean_dop_umb'],
        'mean_dop_pen'  : metrics['mean_dop_pen'],
        'mean_dop_both' : metrics['mean_dop_both'],
        'mean_dop_quiet': metrics['mean_dop_quiet'],
    }
    if 'area_hotspot' in metrics:
        cols['area_hotspot']     = metrics['area_hotspot']
        cols['mean_mag_hotspot'] = metrics['mean_mag_hotspot']

    df = pd.DataFrame(cols)
    df.to_csv(csv_path, index=False, float_format='%.4f')
    print(f'Metrics saved → {csv_path}  ({len(df)} rows)')
    return csv_path


def plot_time_series(
    data: dict,
    metrics: dict,
    normalized: bool = False,
    save: bool = False,
    plots_dir: str | pathlib.Path | None = None,
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

    series = [
        ('Umbra',     metrics['mean_mag_umb'],   metrics['mean_dop_umb'],   'red'),
        ('Penumbra',  metrics['mean_mag_pen'],   metrics['mean_dop_pen'],   'blue'),
        ('Both',      metrics['mean_mag_both'],  metrics['mean_dop_both'],  'purple'),
        ('Quiet Sun', metrics['mean_mag_quiet'], metrics['mean_dop_quiet'], 'black'),
    ]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    for name, mag, dop, color in series:
        ax1.plot(time_h, proc(mag), color=color, lw=0.8, label=name)
        ax2.plot(time_h, proc(dop), color=color, lw=0.8, label=name)

    if 'mean_mag_hotspot' in metrics:
        ax1.plot(time_h, proc(metrics['mean_mag_hotspot']), color='darkorange', lw=0.8, label='Hot spot')

    ax1.set_ylabel(ylabel_mag); ax1.set_title('Mean magnetogram' + suffix.replace('_', ' '))
    ax1.grid(alpha=0.3); ax1.legend()
    ax2.set_ylabel(ylabel_dop); ax2.set_xlabel('Time  (h)')
    ax2.set_title('Mean dopplergram' + suffix.replace('_', ' '))
    ax2.grid(alpha=0.3); ax2.legend()

    plt.tight_layout()
    _savefig(fig, plots_dir if save else None, f'time_series{suffix}.png')
    plt.show()


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

    _out_dir = pathlib.Path(plots_dir) if save else None
    if print_table:
        _print_peak_table(regions, _out_dir)
    elif save:
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
                ax.axvline(f_mhz[idx], color=color, lw=0.7, ls='--', alpha=0.5)

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

    _out_dir = pathlib.Path(plots_dir) if save else None
    if print_table:
        _print_peak_table(regions, _out_dir)
    elif save:
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
    mag_symmetric_cbar: bool = False,
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
    clims      = [np.nanpercentile(c, [2, 98]) for c in cubes_ch]

    if mag_symmetric_cbar:
        mag_finite = cube_mag[np.isfinite(cube_mag)]
        mag_vmax = float(np.nanpercentile(np.abs(mag_finite), 99))
        clims[1] = [-mag_vmax, mag_vmax]

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
    anim.save(str(save_path), writer='html', fps=fps)
    print(f'Animation saved → {save_path}')

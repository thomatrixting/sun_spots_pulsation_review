"""Stage 3 — per-region analysis: masks and cubes in, time series and figures out.

Shared by `03A` (NOAA) and `03B` (DS0N): both loaders return the same `data` dict, so
every function here works on either line unchanged.

Spectral work lives in `spectra`, amplitude fitting in `oscillation`, cross-region
comparison in `comparison`, and the animation in `animation`.
"""

from __future__ import annotations

import pathlib

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .plotting import add_colorbar, overlay_mask, save_figure, symmetric_limits
from .utilities import mean_series as _mean_series, normalize



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
    data      : dict returned by load_ds0n_region()
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
    save_figure(fig, plots_dir, f'calibration_frame_{frame_idx:04d}.png', save)
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
    data      : dict returned by load_ds0n_region()
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

    save_figure(fig, plots_dir, f'{title_name.lower()}_histogram.png', save)

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
    data      : dict returned by load_ds0n_region()
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
    save_figure(fig, plots_dir, f'magnetogram_masks_{frame_idx:04d}.png', save)
    plt.show()
    plt.close(fig)


def compute_metrics(data: dict) -> dict:
    """
    Compute per-frame mean magnetogram and Doppler velocity for each region.

    Umbra / penumbra / both use the boolean masks from load_ds0n_region().
    Quiet sun uses the spatial mean of the dedicated qsun cubes
    (``cube_dopplergram_qsun`` / ``cube_magnetogram_qsun``) — not a mask.

    **Gap frames.** Where ``data['present']`` says a frame has no continuum (a NaN slot on
    02A's uniform time grid), the areas are set to NaN. The mean series need no such
    handling — an all-NaN frame already averages to NaN — but an area would otherwise come
    back as a perfectly confident 0 px, which plots as the spot briefly vanishing.

    Parameters
    ----------
    data : dict from load_ds0n_region()

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
        mean_mag_quiet = np.array([np.nanmean(cube_mag_qsun[t], dtype=np.float64) for t in range(n_qsun)]),
        mean_dop_umb   = _mean_series(cube_dop, umbra),
        mean_dop_pen   = _mean_series(cube_dop, penumbra),
        mean_dop_both  = _mean_series(cube_dop, both),
        mean_dop_quiet = np.array([np.nanmean(cube_dop_qsun[t], dtype=np.float64) for t in range(n_qsun)]),
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
    data          : dict from load_ds0n_region()
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
    data    : dict from load_ds0n_region() / masks_from_cubes()
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
    do_not_show_quiet: bool = False,
) -> None:
    """
    Plot mean B and Doppler velocity vs time for umbra, penumbra, both, and quiet sun.

    Parameters
    ----------
    data       : dict from load_ds0n_region()
    metrics    : dict from compute_metrics()
    normalized : if True, min-max normalize each series before plotting
    save       : if True, save the figure to plots_dir
    plots_dir  : directory for saved figures (required when save=True)
    do_not_show_quiet : if True, do not show the quiet sun series
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
         if not do_not_show_quiet:
            series = [
                ('Umbra',     metrics['mean_mag_umb_residual'],   metrics['mean_dop_umb'],   'red'),
                ('Penumbra',  metrics['mean_mag_pen_residual'],   metrics['mean_dop_pen'],   'blue'),
                ('Both',      metrics['mean_mag_both_residual'],  metrics['mean_dop_both'],  'purple'),
                ('Quiet Sun', None, metrics['mean_dop_quiet'], 'black'),
            ]
         else:
            series = [
                ('Umbra',     metrics['mean_mag_umb_residual'],   metrics['mean_dop_umb'],   'red'),
                ('Penumbra',  metrics['mean_mag_pen_residual'],   metrics['mean_dop_pen'],   'blue'),
                ('Both',      metrics['mean_mag_both_residual'],  metrics['mean_dop_both'],  'purple'),
                ('Quiet Sun', None, None, 'black'),
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
    save_figure(fig, plots_dir, f'time_series{suffix}.png', save)
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
    data      : dict from load_ds0n_region()
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
    save_figure(fig, plots_dir, 'area_vs_time.png', save)
    plt.show()
    plt.close(fig)


# ── heliocentric angle (mu) diagnostics ──────────────────────────────────────
# The open question these answer: a sunspot's magnetogram shows a slow trend across a
# multi-day window that looks like a parabola. Is that solar, or is it just the spot's
# changing foreshortening as it rotates across the disk? mu = cos(heliocentric angle) is
# the geometry, so if the magnetogram trend tracks mu the trend is a viewing effect.
#
# This existed only for line B, because DS0N ships a `cube_mu.fits`. Line A has no such
# file, but `limb_darkening.mu_from_map` already computes mu per pixel for exactly these
# frames — so `mu_cube_noaa` builds the equivalent and both lines get the diagnostic.

def mu_cube_ds0n(ds_dir, fill=1e6):
    """Read DS0N's shipped `cube_mu.fits`, masking its fill value to NaN."""
    from astropy.io import fits

    with fits.open(pathlib.Path(ds_dir) / 'cube_mu.fits') as hdul:
        cube_mu = hdul[0].data.astype(float)
    return np.where(np.abs(cube_mu) > fill, np.nan, cube_mu)


def mu_cube_noaa(raw_dir, grid, cadence_s, method='geometry',
                 series_glob='hmi.ic_*.continuum.fits'):
    """Build the mu cube for a NOAA region, on the same uniform grid as its cubes.

    Reads each frame's own header, like every other per-frame correction — mu at the box
    centre runs 0.78 -> 0.87 -> 0.85 across NOAA 11536's window, so a single map would be
    meaningless. Costs one map load per frame, so it is a step you run deliberately.
    """
    from .limb_darkening import mu_from_map
    from .utilities import apply_per_frame_correction, reindex_on_grid

    raw_dir = pathlib.Path(raw_dir)
    cube, times, _ = apply_per_frame_correction(
        raw_dir / 'region_01_continuum_cube.fits',
        raw_dir / 'region_01',
        lambda smap, _ts: (mu_from_map(smap, method=method), {}),
        series_glob=series_glob,
    )
    cube_mu, _present = reindex_on_grid(cube, times, grid, cadence_s)
    return cube_mu


def mu_series(cube_mu, data, regions=('umbra', 'penumbra', 'both')):
    """Per-frame mean of mu over each mask. NaN where the mask is empty that frame."""
    out = {}
    for name in regions:
        mask = data[name]
        out[name] = np.array([
            np.nanmean(cube_mu[t][mask[t]]) if mask[t].any() else np.nan
            for t in range(data['n_t'])
        ])
    return out


def plot_mu_means(cube_mu, data, save=False, plots_dir=None):
    """Mean mu per region against time, with a quadratic fit over each.

    Returns `(mu_means, mu_fits)`; the fits are what `plot_mu_vs_trend` compares the
    magnetogram trend against.
    """
    from .plotting import REGION_COLORS

    time_h = data['time_h']
    means = mu_series(cube_mu, data)
    fits_by_region = {}

    fig, ax = plt.subplots(figsize=(11, 5))
    for name, series in means.items():
        colour = REGION_COLORS.get(name, 'grey')
        ax.plot(time_h, series, color=colour, lw=0.9, label=name.capitalize())
        good = np.isfinite(series)
        if good.sum() > 3:
            coefficients = np.polyfit(time_h[good], series[good], 2)
            fits_by_region[name] = coefficients
            ax.plot(time_h, np.polyval(coefficients, time_h), color=colour, lw=1.4,
                    ls='--', alpha=0.7)

    ax.set_xlabel('Time (h)')
    ax.set_ylabel(r'Mean $\mu = \cos\theta$')
    ax.set_title(r'Heliocentric angle $\mu$ per region (dashed: quadratic fit)')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    save_figure(fig, plots_dir, 'mu_means.png', save)
    plt.show()
    plt.close(fig)

    for name, coefficients in fits_by_region.items():
        print(f'  {name:<10}: a={coefficients[0]:.6e}  b={coefficients[1]:.6e}  '
              f'c={coefficients[2]:.6e}')
    return means, fits_by_region


def plot_mu_vs_trend(data, metrics, mu_means, key='mean_mag', degree=2,
                     regions=('umbra', 'penumbra', 'both'), save=False, plots_dir=None):
    """Normalised mu trend against the normalised magnetogram trend, per region.

    Both are min-max scaled because they have no common units — which means this figure
    can say "same shape" or "opposite shape" and nothing about relative size. A magnetogram
    trend that tracks mu is geometry; one that runs against it, or that has a different
    curvature, is not explained by foreshortening alone.
    """
    from .plotting import REGION_COLORS

    suffix = {'umbra': 'umb', 'penumbra': 'pen', 'both': 'both'}
    time_h = data['time_h']

    fig, ax = plt.subplots(figsize=(11, 5))
    for name in regions:
        series = metrics.get(f'{key}_{suffix[name]}')
        mu = mu_means.get(name)
        if series is None or mu is None:
            continue
        colour = REGION_COLORS.get(name, 'grey')
        good = np.isfinite(series)
        if good.sum() <= degree + 1:
            continue
        trend = np.polyval(np.polyfit(time_h[good], series[good], degree), time_h)
        ax.plot(time_h, normalize(trend), color=colour, lw=1.5,
                label=f'{name.capitalize()} — {key} trend')
        ax.plot(time_h, normalize(mu), color=colour, lw=1.2, ls='--', alpha=0.7,
                label=f'{name.capitalize()} — $\\mu$')

    ax.set_xlabel('Time (h)')
    ax.set_ylabel('normalised (min-max)')
    ax.set_title(r'Does the magnetogram trend follow the geometry? '
                 r'(solid: fitted trend, dashed: $\mu$)')
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    save_figure(fig, plots_dir, f'mu_vs_{key}_trend.png', save)
    plt.show()
    plt.close(fig)


# ── region report ────────────────────────────────────────────────────────────

def describe_region(data, verbose=True):
    """Print what a loaded region actually contains, and return the same as a dict.

    This was ~60 lines repeated at the top of every per-region block in 03A and 03T. It is
    the same for both lines because both loaders return the same `data` dict.

    The things worth reading every time:

    - **gaps** — NaN slots where a series has no frame. Not measurements; anything derived
      from them is NaN by design.
    - **cont & mag together** — every magnetogram quantity is averaged over a mask built
      from the *continuum*, so it needs both series in the same slot. Where they never
      coincide, every mean B is NaN and the B panels break off even though both cubes have
      data. `mag_fill_slots` in `config.REGION_PARAMS` is the repair.
    - **cluster switches** — if the tracker lost the spot and re-picked, every series has a
      step at that frame and nothing else would show it. Not automatically a bug: a decaying
      spot legitimately vanishes out from under the tracker near the end of a long window,
      so check *when* it happened.
    """
    report = {}
    n_t = data['n_t']
    timestamps = data.get('timestamps')
    present = data.get('present') or {}

    if verbose:
        print(f"  cube shape (n_t, ny, nx) : {data['cube_cont'].shape}")
        print(f"  time span                : {data['time_h'][-1]:.1f} h "
              f"({n_t} frames @ {data['cadence_s']:.0f} s)")
        if data.get('umbra_thresh') is not None:
            print(f"  segmentation             : umbra < {data['umbra_thresh']:,.0f}, "
                  f"penumbra < {data['penumbra_thresh']:,.0f}")

    report['gaps'] = {k: int((~v).sum()) for k, v in present.items()}
    if verbose and present:
        if any(report['gaps'].values()):
            print(f"  NaN (missing) frames     : {report['gaps']}")
            for name, flags in present.items():
                missing = np.flatnonzero(~flags)
                for i in missing[:5]:
                    print(f'      {name}: frame {i} = {timestamps[i]:%Y-%m-%d %H:%M}')
                if len(missing) > 5:
                    print(f'      {name}: ... and {len(missing) - 5} more')
        else:
            print('  NaN (missing) frames     : none')

    if 'cont' in present and 'mag' in present:
        both_present = present['cont'] & present['mag']
        n_both, n_cont = int(both_present.sum()), int(present['cont'].sum())
        report['cont_and_mag'] = (n_both, n_cont)
        if verbose:
            print(f'  cont & mag together      : {n_both} of {n_cont} continuum frames')
        if n_both < n_cont:
            missing = np.flatnonzero(present['cont'] & ~present['mag'])
            runs = (np.split(missing, np.flatnonzero(np.diff(missing) > 1) + 1)
                    if len(missing) else [])
            longest = max(runs, key=len) if runs else []
            if verbose and len(longest) > 5:
                print(f'  WARNING: {len(longest)} consecutive frames from '
                      f'{timestamps[longest[0]]:%Y-%m-%d %H:%M} have a continuum but no '
                      f'magnetogram — every mean B is NaN there and the B panels break off.')
                print('           Set mag_fill_slots = 1 for this region in '
                      'config.REGION_PARAMS, or re-download it on one clock.')

    if data.get('mag_filled') is not None and np.any(data['mag_filled']):
        report['mag_filled'] = int(np.sum(data['mag_filled']))
        if verbose:
            print(f"  magnetogram slots filled : {report['mag_filled']} "
                  f'(copied from a neighbouring frame — see mag_fill_slots)')

    info = data.get('cluster_info')
    if info is not None:
        with_spot = info['area'] > 0
        switched = np.flatnonzero(info['switched'])
        report['cluster_switches'] = len(switched)
        if verbose:
            print(f"  cluster                  : mode={data.get('cluster_mode')}  "
                  f"{info['n_clusters'][with_spot].min()}-{info['n_clusters'][with_spot].max()} "
                  f"components/frame, keeping "
                  f"{100 * np.nanmean(info['fraction']):.0f}% of the dark pixels")
            if len(switched):
                print(f'  WARNING: tracker re-picked the spot in {len(switched)} frame(s), '
                      f'first at {timestamps[switched[0]]:%Y-%m-%d %H:%M} '
                      f'(frame {switched[0]})')

    with_data = present.get('cont', np.ones(n_t, dtype=bool))
    raw = data.get('raw_area_px') or {}
    report['areas'] = {}
    for name, key in [('umbra', 'umbra'), ('penumbra', 'penumbra'), ('hot_spot', None)]:
        if data.get(name) is None:
            continue
        area = data[name].sum(axis=(1, 2))[with_data]
        report['areas'][name] = (float(area.mean()), int(area.min()), int(area.max()))
        if verbose:
            before = (f'  (raw {raw[key][with_data].mean():.0f})'
                      if key and key in raw else '')
            print(f'  {name:9s} area px        : mean={area.mean():.0f}  '
                  f'min={area.min()}  max={area.max()}{before}')
    return report

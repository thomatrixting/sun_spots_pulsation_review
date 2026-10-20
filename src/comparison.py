"""Stage 4 — consolidated analysis across regions and across datasets.

Everything here takes *rows* — one per region or per Löptien table entry — rather than a
single region's `data` dict. `04_data_comparison.ipynb` is the only driver; this used to
sit at the tail of `03A`, which meant the per-region notebook could not be run for one
region without also re-running the cross-region conclusions.
"""

from __future__ import annotations

import pathlib

import matplotlib.pyplot as plt
import numpy as np

from .oscillation import diurnal_curve
from .plotting import save_figure

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
    save_figure(fig, plots_dir, f'diurnal_{series_key}.png', save)
    plt.show()
    plt.close(fig)


def plot_amplitude_vs_depression(rows, depth_key='z_div', save=False, plots_dir=None,
                                 normalize_area=False, include_quiet=True,
                                 fit_var='amplitude'):
    """Fitted amplitude against the Wilson depression reported for each region.

    Both fitted series are drawn: the absolute umbral velocity and the same thing with the
    quiet sun subtracted. That pair is the instrumental control — if the two markers for a
    region sit on top of each other the oscillation is umbral; if the absolute one is far
    higher, most of that amplitude is common to the whole box and is more likely a residual
    of the diurnal `v_SDO` correction than a property of the sunspot.

    Pearson r is annotated per series with the number of points behind it. With five points
    it describes this sample; it is not evidence of a relationship.

    Parameters
    ----------
    normalize_area : divide the amplitude by the spot area. Note that plotting amplitude
        against 1/area was found to fit the sample better than the area-normalised version,
        which over-fitted it — so this is a comparison to make deliberately, not a default.
    include_quiet : draw the quiet-sun-subtracted series as well as the absolute one
    fit_var : which fitted quantity to put on the x axis ('amplitude', 'intercept', ...)

    This merges the `src` version with the extended copy that `03A` used to define locally
    and shadow it with; `normalize_area` and `fit_var` came from that copy.
    """
    usable = [r for r in rows if np.isfinite(r['fit_rel'][fit_var])
              and np.isfinite(r[depth_key])]
    if len(usable) < 2:
        print(f'plot_amplitude_vs_depression: only {len(usable)} usable point(s)')
        return

    depth_label = {'z_div': r'$z_{W,\mathrm{div}}$', 'z_press': r'$z_{W,\mathrm{press}}$'}
    fig, ax = plt.subplots(figsize=(8, 6))

    series = [('fit_abs', 'darkorange', 'o', 'Umbra, absolute')]
    if include_quiet:
        series.append(('fit_rel', 'steelblue', 's', 'Umbra − quiet sun'))

    def _x(row, key):
        value = row[key][fit_var]
        return value / row['area_mm2'] if normalize_area else value

    def _err(row, key):
        sigma = row[key].get(f'sigma_{fit_var}', np.nan)
        return sigma / row['area_mm2'] if normalize_area else sigma

    for key, colour, marker, name in series:
        x = np.array([_x(r, key) for r in usable], dtype=float)
        y = np.array([r[depth_key] for r in usable], dtype=float)
        e = np.array([_err(r, key) for r in usable], dtype=float)

        ax.errorbar(x, y, xerr=e, fmt=marker, color=colour, ms=7, capsize=3, lw=0,
                    elinewidth=1, label=name)

        finite = np.isfinite(x) & np.isfinite(y)
        if finite.sum() > 2:
            r_p = float(np.corrcoef(x[finite], y[finite])[0, 1])
            ax.plot([], [], ' ', label=f'   r = {r_p:+.2f}  (n = {int(finite.sum())})')

    annotate_key = 'fit_rel' if include_quiet else 'fit_abs'
    for row in usable:
        ax.annotate(row['label'], (_x(row, annotate_key), row[depth_key]),
                    textcoords='offset points', xytext=(7, 4), fontsize=9, color='0.3')

    period = usable[0]['fit_rel']['period_h']
    ax.set_ylabel(f'Wilson depression {depth_label.get(depth_key, depth_key)}  (km)')
    ax.set_xlabel(f'Fitted {period:g} h {fit_var} (m/s)'
                  + (' / area (Mm²)' if normalize_area else ''))
    ax.set_title(f'{period:g} h umbral Doppler {fit_var} vs Wilson depression')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    plt.tight_layout()
    suffix = '_per_area' if normalize_area else ''
    save_figure(fig, plots_dir, f'{fit_var}_vs_{depth_key}{suffix}.png', save)
    plt.show()
    plt.close(fig)


def plot_amplitude_vs_area(rows, fit_key='fit_rel', fit_var='amplitude', inverse=False,
                           save=False, plots_dir=None):
    """Fitted amplitude against spot area, or against 1/area.

    `inverse=True` is the version that actually described the sample: plotting amplitude
    against 1/area fitted better than normalising the amplitude by area, which over-fitted.
    """
    usable = [r for r in rows
              if np.isfinite(r[fit_key][fit_var]) and np.isfinite(r.get('area_mm2', np.nan))]
    if len(usable) < 2:
        print(f'plot_amplitude_vs_area: only {len(usable)} usable point(s)')
        return

    x = np.array([1 / r['area_mm2'] if inverse else r['area_mm2'] for r in usable])
    y = np.array([r[fit_key][fit_var] for r in usable])
    e = np.array([r[fit_key].get(f'sigma_{fit_var}', np.nan) for r in usable])

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.errorbar(x, y, yerr=e, fmt='o', color='steelblue', ms=7, capsize=3, lw=0,
                elinewidth=1)
    for row, xi, yi in zip(usable, x, y):
        ax.annotate(row['label'], (xi, yi), textcoords='offset points', xytext=(7, 4),
                    fontsize=9, color='0.3')

    finite = np.isfinite(x) & np.isfinite(y)
    if finite.sum() > 2:
        r_p = float(np.corrcoef(x[finite], y[finite])[0, 1])
        ax.set_title(f'{fit_var} vs {"1/area" if inverse else "area"}   '
                     f'r = {r_p:+.2f}  (n = {int(finite.sum())})')
    ax.set_xlabel('1 / area  (Mm⁻²)' if inverse else 'Spot area  (Mm²)')
    ax.set_ylabel(f'Fitted {fit_var} (m/s)')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    save_figure(fig, plots_dir, f'{fit_var}_vs_{"inv_" if inverse else ""}area.png', save)
    plt.show()
    plt.close(fig)


def plot_amplitude_vs_predictors(frame, x_keys, y_key, yerr_key=None, ax=None,
                                 annotate_key=None, xlabel='', ylabel='amplitude (m/s)',
                                 title='', save=False, plots_dir=None, filename=None):
    """One amplitude against several candidate predictors, drawn on a single axis.

    The predictors share an axis on purpose: `area_umb` and `area_both` are both areas in
    pixels, `mean_mag_umb` and `mean_mag_hotspot` both fields in gauss, so putting them side
    by side is what lets the two be compared rather than merely both plotted. Pearson r and
    the number of points behind it go in the legend per series — with a dozen points r
    describes the sample and nothing more, which is why n is never hidden.

    `ax=None` makes its own figure; pass an axis to compose a row of panels.
    """
    own_figure = ax is None
    if own_figure:
        _, ax = plt.subplots(figsize=(7, 5))

    markers = ('o', 's', '^', 'D')
    y = frame[y_key].to_numpy(dtype=float)
    yerr = (frame[yerr_key].to_numpy(dtype=float)
            if yerr_key is not None and yerr_key in frame.columns else None)

    for (key, marker) in zip(x_keys, markers):
        x = frame[key].to_numpy(dtype=float)
        ok = np.isfinite(x) & np.isfinite(y)
        label = key
        if ok.sum() > 2:
            r = float(np.corrcoef(x[ok], y[ok])[0, 1])
            label = f'{key}   r = {r:+.2f} (n = {int(ok.sum())})'
        ax.errorbar(x, y, yerr=yerr, fmt=marker, ms=6, lw=0, capsize=3, elinewidth=1,
                    alpha=0.85, label=label)

        if annotate_key is not None:
            for name, xi, yi in zip(frame[annotate_key], x, y):
                ax.annotate(str(name), (xi, yi), textcoords='offset points', xytext=(6, 3),
                            fontsize=7, color='0.45')

    ax.set_xlabel(xlabel or ' / '.join(x_keys))
    ax.set_ylabel(ylabel)
    ax.set_title(title or y_key)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)

    if own_figure:
        plt.tight_layout()
        save_figure(ax.figure, plots_dir, filename or f'{y_key}_vs_predictors.png', save)
        plt.show()
        plt.close(ax.figure)
    return ax


def r_squared(x, y):
    """R² of the least-squares line y = a·x + b over the finite points, and how many.

    n comes back with it because with four points in a category R² describes those four
    points and nothing else.
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    n = int(ok.sum())
    if n < 3:
        return float('nan'), n
    a, b = np.polyfit(x[ok], y[ok], 1)
    ss_res = float(np.sum((y[ok] - (a * x[ok] + b)) ** 2))
    ss_tot = float(np.sum((y[ok] - y[ok].mean()) ** 2))
    return (1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')), n


def plane_fit(x1, x2, y):
    """Least-squares plane y = a·x1 + b·x2 + c over the finite points.

    Returns ((a, b, c), R², n). The R² is that of the two predictors together, so it can be
    read against `r_squared` of either one alone: the gain is what the second one adds.
    """
    x1, x2, y = (np.asarray(v, dtype=float) for v in (x1, x2, y))
    ok = np.isfinite(x1) & np.isfinite(x2) & np.isfinite(y)
    n = int(ok.sum())
    if n < 4:
        return (np.nan, np.nan, np.nan), float('nan'), n
    design = np.column_stack([x1[ok], x2[ok], np.ones(n)])
    coeffs, *_ = np.linalg.lstsq(design, y[ok], rcond=None)
    ss_res = float(np.sum((y[ok] - design @ coeffs) ** 2))
    ss_tot = float(np.sum((y[ok] - y[ok].mean()) ** 2))
    return tuple(coeffs), (1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')), n


def kmeans_1d(values, k, iters=100):
    """Lloyd's algorithm on a single axis, seeded from the k quantiles.

    Deterministic by construction — no random restarts, so re-running the notebook cannot
    silently renumber the clusters. Returns (labels, centres) with the labels ordered by
    centre, so cluster 0 is always the smallest value.
    """
    x = np.asarray(values, dtype=float)
    centres = np.quantile(x, np.linspace(0, 1, 2 * k + 1)[1::2])
    for _ in range(iters):
        labels = np.argmin(np.abs(x[:, None] - centres[None, :]), axis=1)
        moved = np.array([x[labels == j].mean() if np.any(labels == j) else centres[j]
                          for j in range(k)])
        if np.allclose(moved, centres):
            break
        centres = moved
    order = np.argsort(centres)
    remap = np.empty(k, dtype=int)
    remap[order] = np.arange(k)
    return remap[labels], centres[order]


# An ordinal ramp, not a categorical palette: clusters of a 1-D variable are ordered bins, so
# one hue light -> dark says "larger value" the way unrelated hues cannot. The light end still
# clears contrast on a white ground; marker shape repeats the level for anyone who cannot
# separate the steps by lightness alone.
BLUE_ORDINAL = ['#86b6ef', '#6da7ec', '#5598e7', '#3987e5', '#2a78d6',
                '#256abf', '#1c5cab', '#184f95', '#104281', '#0d366b']
CLUSTER_MARKERS = ('o', 's', '^', 'D', 'v', 'P', 'X')


def cluster_styles(k):
    """{cluster: (colour, marker)} for k ordered clusters, spread over `BLUE_ORDINAL`."""
    steps = np.linspace(0, len(BLUE_ORDINAL) - 1, k).round().astype(int)
    return {j: (BLUE_ORDINAL[s], CLUSTER_MARKERS[j % len(CLUSTER_MARKERS)])
            for j, s in enumerate(steps)}


def plot_amplitude_by_cluster(panels, cluster_of, centres, x_key='area_umb',
                              xlabel='mean umbral area (px)', suptitle='',
                              save=False, plots_dir=None, filename='amplitude_by_cluster.png'):
    """Amplitude vs `x_key`, one panel per entry of `panels`, coloured by cluster, with the
    R² of the line fitted *within* each cluster and over all points.

    `panels` holds (frame, y_key, yerr_key, annotate, title); each frame needs a `dataset`
    column, which `cluster_of` (a Series dataset -> cluster) maps to its cluster. Returns
    the table of R² per (panel, cluster), cluster None being all points.
    """
    import pandas as pd

    styles = cluster_styles(len(centres))
    rows = []
    fig, axes = plt.subplots(1, len(panels), figsize=(6.3 * len(panels), 5.5), sharey=True,
                             squeeze=False)
    for ax, (frame, y_key, yerr_key, annotate, title) in zip(axes.ravel(), panels):
        level = frame['dataset'].map(cluster_of)
        x_all = frame[x_key].to_numpy(dtype=float)
        y_all = frame[y_key].to_numpy(dtype=float)

        for j, (colour, marker) in styles.items():
            sel = (level == j).to_numpy()
            if not sel.any():
                continue
            yerr = (frame[yerr_key].to_numpy(dtype=float)[sel]
                    if yerr_key is not None and yerr_key in frame.columns else None)
            # R² within the cluster: how much of the spread the predictor explains among
            # spots oscillating at the same period.
            r2, n_fit = r_squared(x_all[sel], y_all[sel])
            rows.append({'panel': y_key, 'cluster': j, 'centre_h': round(centres[j], 2),
                         'n': n_fit, 'r2': r2})
            ax.errorbar(x_all[sel], y_all[sel], yerr=yerr, fmt=marker, ms=6, lw=0,
                        capsize=3, elinewidth=1, alpha=0.85, color=colour,
                        markeredgecolor='0.25', markeredgewidth=0.6,
                        label=f'{j}  {centres[j]:.2f} h  (n = {n_fit})'
                              + (f'  R² = {r2:.2f}' if np.isfinite(r2) else '  R² = -'))

        ok = np.isfinite(x_all) & np.isfinite(y_all)
        r = float(np.corrcoef(x_all[ok], y_all[ok])[0, 1]) if ok.sum() > 2 else float('nan')
        r2_all, _ = r_squared(x_all, y_all)
        rows.append({'panel': y_key, 'cluster': None, 'centre_h': np.nan,
                     'n': int(ok.sum()), 'r2': r2_all})

        if annotate:
            for name, xi, yi in zip(frame['dataset'], x_all, y_all):
                ax.annotate(str(name), (xi, yi), textcoords='offset points', xytext=(6, 3),
                            fontsize=7, color='0.45')

        ax.set_xlabel(xlabel)
        ax.set_title(f'{title}\nall points: r = {r:+.2f}, R² = {r2_all:.2f} '
                     f'(n = {int(ok.sum())})', fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, title='cluster (centre period)', title_fontsize=8)

    axes.ravel()[0].set_ylabel('amplitude (m/s)')
    fig.suptitle(suptitle, fontsize=13)
    plt.tight_layout()
    save_figure(fig, plots_dir, filename, save)
    plt.show()
    plt.close(fig)
    return pd.DataFrame(rows)


def correlation_heatmap(frame, title='Correlations', save=False, plots_dir=None,
                        filename='correlation_heatmap.png'):
    """Pearson correlation matrix of a DataFrame's numeric columns.

    A screening tool on a handful of datasets: it says which pairs are worth a scatter
    plot, and nothing more. Every coefficient here rests on as many points as there are
    rows, which is usually under a dozen.
    """
    import seaborn as sns

    corr = frame.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(1.1 * len(corr) + 3, 0.9 * len(corr) + 2))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r', vmin=-1, vmax=1,
                square=True, cbar_kws={'shrink': 0.8}, ax=ax)
    ax.set_title(f'{title}  (n = {len(frame)})')
    plt.tight_layout()
    save_figure(fig, plots_dir, filename, save)
    plt.show()
    plt.close(fig)
    return corr


def region_summary_rows(all_metrics):
    """One row per region: means and spans of the quantities worth comparing across regions.

    `all_metrics` maps a region name to its `(data, metrics)` pair, which is what the
    "all regions" loop in 04 builds.
    """
    rows = []
    for name, (data, metrics) in all_metrics.items():
        row = {'region': name, 'n_frames': data['n_t'],
               'span_h': float(data['time_h'][-1]),
               'cadence_s': float(data['cadence_s'])}
        for key in ('area_umb', 'area_pen', 'mean_dop_umb', 'mean_dop_quiet',
                    'mean_mag_umb', 'mean_mag_hotspot'):
            values = metrics.get(key)
            if values is not None:
                row[f'{key}_mean'] = float(np.nanmean(values))
        umb, quiet = metrics.get('mean_dop_umb'), metrics.get('mean_dop_quiet')
        if umb is not None and quiet is not None:
            row['dop_umb_minus_quiet_mean'] = float(np.nanmean(umb - quiet))
        rows.append(row)
    return rows

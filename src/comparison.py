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

"""Measuring an oscillation's amplitude by fitting, rather than by spectrum.

Three estimators, deliberately kept side by side because they disagree in informative
ways and 04 compares them:

* `fit_diurnal` — least squares of `c + a.sin(2.pi.t/P) + b.cos(2.pi.t/P)` at a *fixed*
  period. The most constrained, and the one the Wilson-depression analysis uses.
* `sine_amplitude` — the same fit, but at a period found from the spectrum first.
* `peak_to_peak_amplitude` — detrend, then max minus min. Cheap, and contaminated by
  trend, gaps and outliers; kept because it is what the early 04 cells used and dropping
  it silently would make old and new numbers incomparable.

Detrending and filtering helpers are in `spectra`, since they are shared with the
spectral estimators.
"""

from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import numpy as np

from .plotting import REGION_COLORS, save_figure
from .spectra import band_amplitude, detrend_poly, moving_average, notch_filter

#: Period of the oscillation `fit_diurnal` looks for, in hours.
DIURNAL_PERIOD_H = 24.0


def fit_diurnal(
    time_h: np.ndarray,
    series: np.ndarray,
    window: tuple[float, float] | None = None,
    period_h: float = DIURNAL_PERIOD_H,
    mask: np.ndarray | None = None,
    label: str = '',
    warn_short_window: bool = True,
    trend_deg: int = 0,
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
    trend_deg : int
        Degree of a polynomial trend fitted *jointly* with the sinusoid: 0 (default) is
        the intercept alone and reproduces the model in the summary line above, 1 adds a
        slope, 2 a curvature. The trend is in ``(t - window_start)``, so ``intercept``
        keeps meaning the level at the start of the window whichever degree is used.

        **Use 1 for a series carrying the spot's own drift.** As the spot rotates across
        the disk its line-of-sight projection changes, so a mean Doppler series sits on a
        slow ramp that has nothing to do with the oscillation. Fitting that ramp *with*
        the sinusoid rather than subtracting it first is what keeps the two from
        contaminating each other: over a window that is not a whole number of cycles, a
        line fitted to data that still contains the sinusoid absorbs part of it, and the
        amplitude comes out low. See `sine_amplitude`, which pre-detrends when
        ``trend_deg=0`` and hands the job to this fit when it is 1 or more.
    warn_short_window : bool
        Whether to warn when the window is under 1.5 periods. Set False only where the
        window length is the deliberate design — `windowed_amplitudes` fits one period per
        window by construction and would otherwise emit the same warning a hundred times.
        The caveat does not go away with the warning: read `sigma_amplitude`.

    Returns
    -------
    dict
        ``intercept``, ``amplitude``, ``phase_rad``, ``t_max`` (hours after the window
        start at which the fitted curve peaks), the matching ``sigma_intercept`` /
        ``sigma_amplitude``, ``n`` points used, ``rms_residual``, ``period_h``,
        ``window``, ``trend_deg`` and ``trend`` (the fitted trend coefficients, lowest
        order first, empty when ``trend_deg`` is 0; ``trend[0]`` is the slope in units
        per hour). Every numeric field is NaN when there was too little data, rather than
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
    # Intercept, sin and cos, plus one column per trend degree; one more point than that
    # is the minimum that leaves anything to check the fit against. `n` is reported even
    # when the fit is refused, so the table says how close the window came rather than a
    # bare zero.
    n_params = 3 + int(trend_deg)
    if n < n_params + 1:
        warnings.warn(f'fit_diurnal{f" [{label}]" if label else ""}: only {n} finite '
                      f'point(s) in {window[0]:g}-{window[1]:g} h — no fit', stacklevel=2)
        return dict(intercept=np.nan, amplitude=np.nan, phase_rad=np.nan, t_max=np.nan,
                    sigma_intercept=np.nan, sigma_amplitude=np.nan, n=n,
                    rms_residual=np.nan, period_h=period_h, window=window,
                    coefficients=np.full(n_params, np.nan),
                    trend_deg=int(trend_deg), trend=np.full(int(trend_deg), np.nan))

    span = window[1] - window[0]
    if warn_short_window and span < 1.5 * period_h:
        warnings.warn(
            f'fit_diurnal{f" [{label}]" if label else ""}: the {span:g} h window is only '
            f'{span / period_h:.2f} periods long. Amplitude, phase and intercept are '
            f'poorly separated over so little of a cycle — read sigma_amplitude before '
            f'reading amplitude.', stacklevel=2)

    t, y = time_h[inside], series[inside]
    omega = 2 * np.pi / period_h
    # Trend columns come last so `coefficients[:3]` is still (c, a, b) whatever the
    # degree, and they are powers of the time *since the window start* rather than of
    # `t` itself: a 200 h offset raised to a power is what makes the normal equations
    # ill-conditioned, and it would also stop `intercept` meaning the starting level.
    columns = [np.ones(n), np.sin(omega * t), np.cos(omega * t)]
    columns += [(t - window[0]) ** k for k in range(1, int(trend_deg) + 1)]
    design = np.column_stack(columns)

    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    c, a, b = coef[:3]
    residual = y - design @ coef
    amplitude = float(np.hypot(a, b))

    # Parameter covariance the textbook way: residual variance times inv(X'X). With only
    # as many points as parameters there is no residual left to estimate it from.
    dof = n - n_params
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
        trend_deg=int(trend_deg), trend=coef[3:],
    )


def diurnal_curve(fit: dict, time_h: np.ndarray) -> np.ndarray:
    """Evaluate a `fit_diurnal` result at arbitrary times, for plotting over the data."""
    c, a, b = fit['coefficients'][:3]
    omega = 2 * np.pi / fit['period_h']
    t = np.asarray(time_h, dtype=float)
    curve = c + a * np.sin(omega * t) + b * np.cos(omega * t)
    # Fits made before `trend_deg` existed have neither key; they are pure sinusoids.
    for k, coefficient in enumerate(fit.get('trend', ()), start=1):
        curve = curve + coefficient * (t - fit['window'][0]) ** k
    return curve


# ── amplitude estimators ─────────────────────────────────────────────────────
# Three ways to answer "how big is the oscillation", kept side by side because they
# disagree in informative ways and 04 compares them directly.

def sine_amplitude(time_h, series, period_h, label='', detrend_deg=1, trend_deg=0):
    """`fit_diurnal` on the same prepared series the spectrum sees.

    Returns the full fit dict: `amplitude` is the semi-amplitude A of
    `A sin(2.pi.t/P + phi)` and `sigma_amplitude` its uncertainty — with only ~2 cycles in
    the window that sigma is what says whether the amplitude can be read at all.

    The slow drift a rotating spot puts on a mean Doppler series can be taken out two
    ways, and `trend_deg` chooses between them. At 0 (the default, and what every number
    written before this existed used) `detrend_deg` subtracts a line *first* and the fit
    sees the residual. At 1 or more the pre-detrend is skipped and the trend is fitted
    together with the sinusoid, which is the better-conditioned of the two: over a window
    that is not a whole number of cycles a line fitted to data that still contains the
    sinusoid absorbs part of it, and the amplitude comes out low.
    """
    from .spectra import resample_uniform

    # Fitting the trend and pre-removing it would take it out twice — harmlessly for the
    # amplitude, but `intercept` and `trend` would then describe the residual rather than
    # the series, which is exactly what someone reading a drift rate off them wants.
    prepare_deg = None if trend_deg else detrend_deg
    t, y = resample_uniform(time_h, series, detrend_deg=prepare_deg)
    if t.size < 8 or not np.isfinite(period_h):
        return dict(amplitude=np.nan, sigma_amplitude=np.nan, rms_residual=np.nan,
                    period_h=period_h, coefficients=np.full(3 + trend_deg, np.nan),
                    trend_deg=trend_deg, trend=np.full(trend_deg, np.nan))
    return fit_diurnal(t, y, period_h=float(period_h), label=label, trend_deg=trend_deg)


def peak_to_peak_amplitude(time_h, series, detrend_deg=2):
    """Detrend, then max minus min. Half of it, to be comparable with a semi-amplitude.

    The crudest estimator, and the one the early 04 cells used. It is contaminated by any
    surviving trend, by gaps, and by a single outlier — which is exactly why the other two
    exist. Kept so old and new numbers stay comparable rather than silently incomparable.
    """
    residual = detrend_poly(time_h, series, detrend_deg)
    finite = residual[np.isfinite(residual)]
    if finite.size < 2:
        return np.nan
    return float(finite.max() - finite.min()) / 2


def amplitude_summary(time_h, series, band_h=(16.0, 36.0), period_h=None, label='',
                      trend_deg=0):
    """All three estimators for one series, as one dict.

    `period_h` fixes the sinusoidal fit's period; None takes the period the band-limited
    spectrum found, which is the honest default when the period is not known a priori.

    `trend_deg` goes to `sine_amplitude`: 1 fits the rotational drift jointly with the
    sinusoid instead of subtracting a line beforehand, and reports it as `fit_slope`.
    The key set never changes — at the default 0, `fit_slope` is NaN rather than missing —
    so a caller stacking these into a DataFrame gets the same columns either way. Only
    the `fit_*` values change: the FFT and peak-to-peak estimators do their own detrending
    and are left alone, which is what keeps the three comparable.
    """
    spectral = band_amplitude(time_h, series, band_h=band_h)
    fit_period = period_h if period_h is not None else spectral['period_h']
    fit = sine_amplitude(time_h, series, fit_period, label=label, trend_deg=trend_deg)
    trend = np.asarray(fit.get('trend', ()), dtype=float)
    # Always present, NaN when no trend was fitted. Making a key's *existence* depend on an
    # argument is what turns "you forgot trend_deg" into a KeyError three lines later
    # instead of a NaN you can see in the table.
    return {
        'fit_slope': float(trend[0]) if trend.size else np.nan,
        'fit_trend_deg': int(trend_deg),
        'fft_amplitude': spectral['amplitude'],
        'fft_period_h': spectral['period_h'],
        'n_cycles': spectral['n_cycles'],
        'fit_amplitude': fit['amplitude'],
        'fit_sigma': fit.get('sigma_amplitude', np.nan),
        'fit_period_h': fit_period,
        'peak_to_peak': peak_to_peak_amplitude(time_h, series),
    }


def windowed_amplitudes(time_h, series, window_h=DIURNAL_PERIOD_H,
                        period_h=DIURNAL_PERIOD_H, min_fraction=0.5, label=''):
    """Amplitude per consecutive `window_h` window, rather than one number per series.

    One row per window buys two things a whole-series amplitude cannot give. It says whether
    the oscillation is steady or comes and goes — a single fit over 200 h reports the average
    of both cases and cannot tell them apart — and it pairs each amplitude with the spot as it
    was *during that window*, so an amplitude can be set against an area or a field strength
    measured at the same time instead of against a 200 h mean taken while the spot crossed
    half the disk.

    **One window is one cycle.** At the default 24 h window and 24 h period the sinusoid
    completes exactly one turn, and amplitude, phase and offset are only weakly separated —
    the reason `fit_diurnal` normally warns here, and the reason `sigma_amplitude` comes back
    with every row. `minmax_amplitude` is returned alongside precisely because it fails
    differently: it needs no period at all, but a single outlier moves it.

    Parameters
    ----------
    window_h : length of each window, in hours. Windows are consecutive and do not overlap,
        starting at the first finite sample.
    period_h : the period fitted inside each window.
    min_fraction : drop a trailing window that covers less than this much of `window_h`.
        The last stretch of a series is rarely a whole window, and a fit over a few hours of
        a 24 h sinusoid is not an amplitude.

    Returns
    -------
    list of dict — `t_start`, `t_end`, `n`, `amplitude`, `sigma_amplitude`,
    `minmax_amplitude`. Windows with too few finite points are skipped.
    """
    time_h = np.asarray(time_h, dtype=float)
    series = np.asarray(series, dtype=float)
    finite = np.isfinite(time_h) & np.isfinite(series)
    if finite.sum() < 4:
        return []

    start, stop = float(time_h[finite].min()), float(time_h[finite].max())
    rows = []
    for edge in np.arange(start, stop, window_h):
        window = (edge, edge + window_h)
        if (min(window[1], stop) - window[0]) < min_fraction * window_h:
            continue

        inside = finite & (time_h >= window[0]) & (time_h <= window[1])
        if inside.sum() < 4:
            continue

        fit = fit_diurnal(time_h, series, window=window, period_h=period_h,
                          label=f'{label} {window[0]:.0f}-{window[1]:.0f} h',
                          warn_short_window=False)
        rows.append({
            't_start': window[0],
            't_end': window[1],
            'n': int(inside.sum()),
            'amplitude': fit['amplitude'],
            'sigma_amplitude': fit['sigma_amplitude'],
            'minmax_amplitude': peak_to_peak_amplitude(time_h[inside], series[inside]),
        })
    return rows


# ── detrending views ─────────────────────────────────────────────────────────

def plot_moving_average(data, metrics, window_min=1422, key='mean_dop',
                        regions=(('umb', 'Umbra'), ('pen', 'Penumbra'),
                                 ('both', 'Both'), ('quiet', 'Quiet Sun')),
                        save=False, plots_dir=None):
    """Each region's series smoothed with a centred rolling mean.

    A window of one full period averages that period away, which is the point: whatever
    is left is either a slower trend or something the dominant oscillation was hiding.
    The default 1422 min is close to 24 h.
    """
    time_h = data['time_h']
    cadence_s = data['cadence_s']
    colours = {'umb': REGION_COLORS['umbra'], 'pen': REGION_COLORS['penumbra'],
               'both': REGION_COLORS['both'], 'quiet': REGION_COLORS['quiet']}

    fig, ax = plt.subplots(figsize=(12, 5))
    overall = {}
    window = None
    for suffix, label in regions:
        series = metrics.get(f'{key}_{suffix}')
        if series is None:
            continue
        grid_h, smoothed, window = moving_average(series, cadence_s, window_min,
                                                  time_h=time_h)
        overall[label] = float(np.nanmean(smoothed))
        axis_h = grid_h if grid_h is not None else time_h
        ax.plot(axis_h, smoothed, color=colours.get(suffix, 'grey'), lw=1.2, label=label)

    ax.set_xlabel('Time (h)')
    ax.set_ylabel(f'Mean velocity, {window_min:.0f} min window  (m/s)')
    ax.set_title(f'{key} — {window_min:.0f} min moving average '
                 f'(window = {window} frames)')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    save_figure(fig, plots_dir, f'moving_average_{window_min:.0f}min.png', save)
    plt.show()
    plt.close(fig)

    print('Mean per region after smoothing:')
    for label, value in overall.items():
        print(f'  {label:<10}: {value:.1f} m/s')
    return overall


def plot_filtered(data, metrics, period_min=1442, key='mean_dop', width_mhz=0.0002,
                  regions=(('umb', 'Umbra'), ('pen', 'Penumbra'),
                           ('both', 'Both'), ('quiet', 'Quiet Sun')),
                  save=False, plots_dir=None):
    """Each region's series with one period notched out of it.

    Pair it with `spectra.plot_spectrum_before_after` on a single region: this shows what
    the series looks like afterwards, that shows what the filter actually took.
    """
    time_h = data['time_h']
    cadence_s = data['cadence_s']
    colours = {'umb': REGION_COLORS['umbra'], 'pen': REGION_COLORS['penumbra'],
               'both': REGION_COLORS['both'], 'quiet': REGION_COLORS['quiet']}

    fig, ax = plt.subplots(figsize=(12, 5))
    filtered_by_region = {}
    n_bins = 0
    for suffix, label in regions:
        series = metrics.get(f'{key}_{suffix}')
        if series is None:
            continue
        grid_h, filtered, n_bins = notch_filter(series, cadence_s, period_min,
                                                width_mhz=width_mhz, time_h=time_h)
        filtered_by_region[label] = filtered
        axis_h = grid_h if grid_h is not None else time_h
        ax.plot(axis_h[:len(filtered)], filtered, color=colours.get(suffix, 'grey'),
                lw=0.9, label=label)

    ax.set_xlabel('Time (h)')
    ax.set_ylabel('Mean velocity (m/s)')
    ax.set_title(f'{key} with the {period_min:.0f} min period removed '
                 f'({n_bins} bin(s) zeroed)')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    save_figure(fig, plots_dir, f'filtered_{period_min:.0f}min.png', save)
    plt.show()
    plt.close(fig)
    return filtered_by_region


def plot_diurnal_fit(time_h, series, fit, label='', save=False, plots_dir=None):
    """One region's series with its fitted sinusoid overlaid — the fit's own sanity check."""
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(time_h, series, color='grey', lw=0.7, alpha=0.7, label='series')
    ax.plot(time_h, diurnal_curve(fit, time_h), color='darkorange', lw=1.6,
            label=f'fit: A={fit["amplitude"]:.1f} ± {fit.get("sigma_amplitude", np.nan):.1f} m/s, '
                  f'P={fit.get("period_h", np.nan):.1f} h')
    ax.set_xlabel('Time (h)')
    ax.set_ylabel('Mean velocity (m/s)')
    ax.set_title(f'Diurnal fit  {label}')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    save_figure(fig, plots_dir, f'diurnal_fit_{label or "region"}.png'.replace(' ', '_'), save)
    plt.show()
    plt.close(fig)

"""Signal conditioning and spectral estimation — one implementation, both lines.

This module replaces four independent Fourier implementations that had grown up in
parallel:

* `_fft` + `plot_ffts_*` in the old `sunspot_analysis.py` — raw `|rfft|`, no window
* `_psd`, redefined once per region (five times) inside `03A` — bare `periodogram`
* `plot_psd_separate/combined` + peak tables, prototyped in `03_data_analisis`
* `fft_amplitude` in `04` — Hann window, zero-padding, coherent-gain-corrected
  amplitude, peak searched inside a period band. The most careful of the four, and the
  one that only existed in a notebook.

Two conventions, chosen so old numbers stay reproducible:

* **`window=None` is the default** for `fft_spectrum` and `psd_spectrum`, which
  reproduces what `03A` and `03B` computed before. Pass `window='hann'` to suppress
  spectral leakage — worth doing, but it changes the numbers.
* **Frequencies are in mHz** everywhere except `band_amplitude`, which works in
  cycles/hour because its band is stated in hours. Both report `period_min`/`period_h`
  alongside, so nothing downstream has to do the conversion itself.

A note on the notch filter: the prototype had two disagreeing implementations —
`remove_period` zeroed a `width`-wide band while its companion before/after plot zeroed
only the single nearest bin. `notch_filter` here uses the band, and the before/after plot
calls it rather than re-implementing it.
"""

from __future__ import annotations

import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, periodogram

from .plotting import save_figure

# The four regions every per-region figure shows, with the colours the FFT plots have
# always used: (label, metrics-key suffix, colour).
DEFAULT_REGIONS = (
    ('Umbra', 'umb', 'green'),
    ('Penumbra', 'pen', 'purple'),
    ('Both', 'both', 'steelblue'),
    ('Quiet Sun', 'quiet', 'darkorange'),
)


# ── conditioning ──────────────────────────────────────────────────────────────

def interpolate_gaps(series):
    """Linearly fill NaNs so a transform can run. Returns a copy.

    A gap is a real absence of data, but `rfft` has no way to represent one: leaving the
    NaN in poisons the entire spectrum, so the choice is between interpolating and
    dropping the frame. Dropping breaks the uniform cadence every estimator here assumes,
    so interpolation is the lesser evil — and it is why gap-heavy regions deserve a look
    at `present` before their spectra are believed.
    """
    values = np.array(series, dtype=float)
    nans = ~np.isfinite(values)
    if nans.any() and (~nans).sum() > 2:
        x = np.arange(len(values))
        values[nans] = np.interp(x[nans], x[~nans], values[~nans])
    return values


def detrend_poly(time, series, degree=1):
    """Subtract a least-squares polynomial. degree=0 removes the mean, 1 a line, 2 a parabola.

    The slow drift is exactly what makes `max - min` not the amplitude of anything, and
    for the magnetogram a parabola is the right shape — see `analysis.add_mag_residuals`.
    """
    time = np.asarray(time, dtype=float)
    values = np.asarray(series, dtype=float)
    good = np.isfinite(values) & np.isfinite(time)
    if good.sum() < degree + 2:
        return values - np.nanmean(values)
    coefficients = np.polyfit(time[good], values[good], degree)
    return values - np.polyval(coefficients, time)


def resample_uniform(time_h, series, detrend_deg=None):
    """Put a series on a uniform time grid, interpolating gaps, optionally detrended.

    Returns `(grid_h, values)`, both empty if there is too little finite data to bother.
    Every transform below assumes uniform sampling; this is what guarantees it when the
    caller only has a `time_h` column and no promise about its spacing.
    """
    t = np.asarray(time_h, dtype=float)
    y = np.asarray(series, dtype=float)
    good = np.isfinite(y) & np.isfinite(t)
    if good.sum() < 4:
        return np.array([]), np.array([])

    dt = float(np.median(np.diff(t)))
    grid = np.arange(t[0], t[-1] + 0.5 * dt, dt)
    values = np.interp(grid, t[good], y[good])
    if detrend_deg is not None:
        values = detrend_poly(grid, values, detrend_deg)
    return grid, values


def moving_average(series, cadence_s, window_min):
    """Centred rolling mean over a window given in minutes.

    Used to strip the dominant ~24 h component and see what is underneath it. The window
    is converted to a whole number of frames, so the effective width is whatever that
    rounds to — the return value says which.

    Returns `(smoothed, window_frames)`.
    """
    cadence_min = cadence_s / 60
    window = max(1, round(window_min / cadence_min))
    # min_periods matters: with a 120-frame window, a series carrying scattered gaps has a
    # NaN inside almost every window, and the default (min_periods=window) would return NaN
    # nearly everywhere. Half a window of real data is enough for a mean.
    smoothed = pd.Series(np.asarray(series, dtype=float)).rolling(
        window, center=True, min_periods=max(1, window // 2)).mean().to_numpy()
    return smoothed, window


def notch_filter(series, cadence_s, period_min, width_mhz=0.0002):
    """Zero every frequency bin within `width_mhz` of 1/period_min, then transform back.

    A blunt instrument: it removes the tone *and* whatever real signal shares those bins,
    and zeroing a band rings in the time domain. It answers one narrow question — "what
    is left once this period is gone" — and the answer should always be checked with
    `plot_spectrum_before_after`.

    `width_mhz` is a half-width. The default 0.0002 mHz is about 5 bins on a 3-day,
    720 s series.
    """
    values = interpolate_gaps(series)
    freq_mhz = np.fft.rfftfreq(len(values), d=cadence_s) * 1e3
    spectrum = np.fft.rfft(values)

    target_mhz = 1e3 / (period_min * 60)
    band = np.abs(freq_mhz - target_mhz) < width_mhz
    spectrum[band] = 0
    return np.fft.irfft(spectrum, n=len(values)), int(band.sum())


# ── estimation ────────────────────────────────────────────────────────────────

def _apply_window(values, window):
    """Return (windowed_values, coherent_gain). `window=None` is a rectangular window."""
    if window is None:
        return values, float(len(values))
    if window == 'hann':
        w = np.hanning(len(values))
    elif window == 'hamming':
        w = np.hamming(len(values))
    else:
        raise ValueError(f"window must be None, 'hann' or 'hamming', got {window!r}")
    return values * w, float(w.sum())


def fft_spectrum(series, cadence_s, window=None, pad_factor=1, detrend_deg=None,
                 scale='raw'):
    """Amplitude spectrum of a uniformly sampled series. Returns `(freq_mhz, amplitude)`.

    The DC bin is dropped: it is the mean, it dwarfs everything else on a velocity
    series, and it is never the answer to a question about an oscillation.

    Parameters
    ----------
    window : None (default, rectangular — reproduces the pre-refactor numbers) or 'hann'
    pad_factor : zero-padding multiple. It interpolates the spectrum so a peak can be
        read off more precisely; it adds no actual resolution, which is set by the length
        of the series.
    scale : 'raw' gives `|X_k|` (what the old `_fft` returned, arbitrary units);
        'amplitude' gives `2|X_k| / sum(w)`, corrected for the window's coherent gain, so
        a pure sinusoid of semi-amplitude A reads back as A in the series' own units.
    """
    values = interpolate_gaps(series)
    if detrend_deg is not None:
        values = detrend_poly(np.arange(len(values)), values, detrend_deg)

    n = len(values)
    windowed, gain = _apply_window(values, window)
    n_fft = int(pad_factor) * n

    freq_mhz = np.fft.rfftfreq(n_fft, d=cadence_s) * 1e3
    amplitude = np.abs(np.fft.rfft(windowed, n=n_fft))
    if scale == 'amplitude':
        amplitude = 2 * amplitude / gain
    elif scale != 'raw':
        raise ValueError(f"scale must be 'raw' or 'amplitude', got {scale!r}")
    return freq_mhz[1:], amplitude[1:]


def psd_spectrum(series, cadence_s, window=None, detrend_deg=None):
    """One-sided power spectral density. Returns `(freq_mhz, power)` in units^2/Hz.

    The density normalisation is what makes two series of different length comparable —
    an amplitude spectrum's height depends on how many samples went into it, a PSD's does
    not. Use it to ask "where is the power", and `fft_spectrum(scale='amplitude')` to ask
    "how many m/s is that peak".
    """
    values = interpolate_gaps(series)
    detrend = 'constant'
    if detrend_deg is not None:
        values = detrend_poly(np.arange(len(values)), values, detrend_deg)
        detrend = False          # already done, and more thoroughly

    freq_hz, power = periodogram(values, fs=1.0 / cadence_s, window=window or 'boxcar',
                                 scaling='density', return_onesided=True, detrend=detrend)
    return freq_hz[1:] * 1e3, power[1:]


def spectral_peaks(freq_mhz, power, n=5, height_pct=85, min_distance=5):
    """The `n` tallest peaks, as a DataFrame with freq, period and relative height.

    `height_pct` is a percentile of the spectrum itself rather than an absolute floor, so
    the same call works on an amplitude spectrum and on a PSD, whose scales differ by
    orders of magnitude.
    """
    freq_mhz = np.asarray(freq_mhz)
    power = np.asarray(power)
    if power.size == 0 or not np.isfinite(power).any():
        return pd.DataFrame(columns=['rank', 'freq_mhz', 'period_min', 'relative'])

    peaks, _ = find_peaks(power, height=np.percentile(power, height_pct),
                          distance=min_distance)
    if peaks.size == 0:
        return pd.DataFrame(columns=['rank', 'freq_mhz', 'period_min', 'relative'])

    top = peaks[np.argsort(power[peaks])[::-1][:n]]
    peak_max = power.max()
    return pd.DataFrame([{
        'rank': rank,
        'freq_mhz': round(float(freq_mhz[i]), 4),
        'period_min': round(1000 / (freq_mhz[i] * 60), 1),
        'relative': round(float(power[i] / peak_max), 4),
    } for rank, i in enumerate(top, 1)])


def band_amplitude(time_h, series, band_h=(16.0, 36.0), pad_factor=8, detrend_deg=1,
                   window='hann'):
    """Semi-amplitude and period of the tallest peak within a band of periods, in hours.

    This is the careful amplitude estimator: Hann window so leakage does not split the
    peak between neighbouring bins, zero-padding so the peak can be located between them,
    and the coherent-gain correction that puts the result back in m/s rather than in FFT
    counts.

    Restricting the search to a band is what makes it usable on a series whose largest
    feature is a trend: the default (16, 36) h brackets the ~24 h signal without letting
    the fit run off to the length of the window.

    Returns a dict with `amplitude`, `period_h`, `n_cycles` (how many periods the window
    actually covers — below ~1.5 the amplitude is barely constrained), and the
    `freq_per_h`/`amplitude_spectrum` arrays for plotting.
    """
    empty = dict(amplitude=np.nan, period_h=np.nan, n_cycles=np.nan,
                 freq_per_h=np.array([]), amplitude_spectrum=np.array([]))

    t, y = resample_uniform(time_h, series, detrend_deg=detrend_deg)
    if t.size < 8:
        return empty

    n = t.size
    windowed, gain = _apply_window(y, window)
    n_fft = int(pad_factor) * n

    freq_per_h = np.fft.rfftfreq(n_fft, d=t[1] - t[0])      # cycles per hour
    amplitude = 2 * np.abs(np.fft.rfft(windowed, n=n_fft)) / gain

    in_band = (freq_per_h >= 1 / band_h[1]) & (freq_per_h <= 1 / band_h[0])
    if not in_band.any():
        return empty

    k = int(np.argmax(np.where(in_band, amplitude, -np.inf)))
    period_h = 1 / freq_per_h[k]
    return dict(amplitude=float(amplitude[k]), period_h=float(period_h),
                n_cycles=float((t[-1] - t[0]) / period_h),
                freq_per_h=freq_per_h, amplitude_spectrum=amplitude)


# ── per-region helpers ────────────────────────────────────────────────────────

def region_spectra(metrics, cadence_s, kind='fft', key='mean_dop', regions=DEFAULT_REGIONS,
                   **kwargs):
    """Spectra for the four regions at once.

    Returns a list of `(label, freq_mhz, power, colour)`, which is what every plot below
    consumes. `key` picks the quantity: 'mean_dop' for velocity, 'mean_mag' for field,
    'mean_mag_residual' for the parabola-subtracted field (see `analysis.add_mag_residuals`).
    """
    estimate = {'fft': fft_spectrum, 'psd': psd_spectrum}.get(kind)
    if estimate is None:
        raise ValueError(f"kind must be 'fft' or 'psd', got {kind!r}")

    out = []
    for label, suffix, colour in regions:
        series_key = f'{key}_{suffix}' if not key.endswith(suffix) else key
        if series_key not in metrics:
            continue
        freq, power = estimate(metrics[series_key], cadence_s, **kwargs)
        out.append((label, freq, power, colour))
    return out


def peak_table(spectra, n=5, height_pct=85, min_distance=5):
    """`spectral_peaks` for every region, concatenated with a `region` column."""
    frames = []
    for label, freq, power, _ in spectra:
        peaks = spectral_peaks(freq, power, n=n, height_pct=height_pct,
                               min_distance=min_distance)
        peaks.insert(0, 'region', label)
        frames.append(peaks)
    if not frames:
        return pd.DataFrame(columns=['region', 'rank', 'freq_mhz', 'period_min', 'relative'])
    return pd.concat(frames, ignore_index=True)


def print_peak_table(peaks, value_label='Rel. amplitude'):
    """Print a peak table with a blank line between regions."""
    print(f'\n{"Region":<12}  {"Rank":>4}  {"Freq (mHz)":>11}  {"Period (min)":>13}  '
          f'{value_label:>14}')
    print('─' * 60)
    last = None
    for _, row in peaks.iterrows():
        if row['region'] != last and last is not None:
            print()
        last = row['region']
        print(f'{row["region"]:<12}  {int(row["rank"]):>4}  {row["freq_mhz"]:>11.4f}  '
              f'{row["period_min"]:>13.1f}  {row["relative"]:>14.4f}')
    print()


def save_peak_table(peaks, plots_dir, filename='fft_peaks.csv'):
    """Write a peak table next to the figures. No-op if `plots_dir` is None."""
    if plots_dir is None:
        return None
    plots_dir = pathlib.Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    path = plots_dir / filename
    peaks.to_csv(path, index=False)
    print(f'Peak table saved → {path}')
    return path


# ── plots ─────────────────────────────────────────────────────────────────────

def _spectrum_labels(kind):
    return ('Power spectral density  ((m/s)²/Hz)', 'Rel. power', 'PSD') if kind == 'psd' \
        else ('Amplitude (arb. units)', 'Rel. amplitude', 'FFT amplitude')


def plot_spectra_separate(spectra, cadence_s, kind='fft', title_extra='',
                          annotate_peaks=True, print_table=True, log_y=False,
                          xlim=None, save=False, plots_dir=None, filename=None):
    """One panel per region, in a 2x2 grid."""
    y_label, value_label, kind_label = _spectrum_labels(kind)
    f_nyq = 1e3 / (2 * cadence_s)
    peaks = peak_table(spectra)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    for ax, (label, freq, power, colour) in zip(axes.flat, spectra):
        ax.plot(freq, power, color=colour, lw=0.8)
        if annotate_peaks:
            for _, row in peaks[peaks['region'] == label].iterrows():
                ax.axvline(row['freq_mhz'], color=colour, lw=0.7, ls='--', alpha=0.5)
                ax.text(row['freq_mhz'], power.max() * 0.95,
                        f'{row["period_min"]:.0f} min', fontsize=7, color=colour,
                        rotation=90, va='top', ha='center')
        ax.set_title(label, color=colour, fontweight='bold')
        ax.set_ylabel(y_label)
        ax.set_xlabel('Frequency (mHz)')
        if log_y:
            ax.set_yscale('log')
        if xlim is not None:
            ax.set_xlim(*xlim)
        ax.grid(alpha=0.25)

    fig.suptitle(f'{kind_label} per region{title_extra}\n'
                 f'(cadence {cadence_s:.0f} s  |  Nyquist ≈ {f_nyq:.3f} mHz)', fontsize=12)
    plt.tight_layout()
    save_figure(fig, plots_dir, filename or f'{kind}_separate.png', save)
    plt.show()
    plt.close(fig)

    if print_table:
        print_peak_table(peaks, value_label)
    if save:
        save_peak_table(peaks, plots_dir, f'{kind}_peaks.csv')
    return peaks


def plot_spectra_combined(spectra, cadence_s, kind='fft', title_extra='', xlim=None,
                          annotate_peaks=True, print_table=True, log_y=False,
                          save=False, plots_dir=None, filename=None):
    """All regions overlaid on one axis, so their peaks can be compared directly."""
    y_label, value_label, kind_label = _spectrum_labels(kind)
    f_nyq = 1e3 / (2 * cadence_s)
    peaks = peak_table(spectra)

    fig, ax = plt.subplots(figsize=(14, 6))
    for label, freq, power, colour in spectra:
        ax.plot(freq, power, color=colour, lw=0.8, label=label)
        if annotate_peaks:
            for _, row in peaks[peaks['region'] == label].iterrows():
                ax.axvline(row['freq_mhz'], color=colour, lw=0.7, ls='--', alpha=0.5)

    ax.set_ylabel(y_label)
    ax.set_xlabel('Frequency (mHz)')
    if log_y:
        ax.set_yscale('log')
    if xlim is not None:
        ax.set_xlim(*xlim)
    ax.grid(alpha=0.25)
    ax.legend()
    ax.set_title(f'{kind_label} per region{title_extra}\n'
                 f'(cadence {cadence_s:.0f} s  |  Nyquist ≈ {f_nyq:.3f} mHz)', fontsize=12)
    plt.tight_layout()
    save_figure(fig, plots_dir, filename or f'{kind}_combined.png', save)
    plt.show()
    plt.close(fig)

    if print_table:
        print_peak_table(peaks, value_label)
    if save:
        save_peak_table(peaks, plots_dir, f'{kind}_peaks.csv')
    return peaks


def plot_spectra_compare(spectra_a, spectra_b, cadence_s, label_a='Dopplergram',
                         label_b='Magnetogram', kind='psd', normalize=False, xlim=None,
                         log_y=False, save=False, plots_dir=None, filename=None):
    """Two quantities' spectra per region, overlaid — one panel per region.

    The question it answers: does the field oscillate at the same period as the velocity,
    and if so is it in phase or against it? Comparing against the *parabola-residual*
    magnetogram (`key='mean_mag_residual'`) rather than the raw one is the sharper test,
    because the raw magnetogram's slow trend dominates its own spectrum.

    `normalize` scales each curve by its own maximum, which is the only way to see two
    quantities with different units on one axis — at the cost of every statement about
    relative size.
    """
    y_label, _, kind_label = _spectrum_labels(kind)
    by_label_b = {label: (freq, power) for label, freq, power, _ in spectra_b}

    n = len(spectra_a)
    fig, axes = plt.subplots(n, 1, figsize=(12, 3 * n), sharex=True, squeeze=False)
    for ax, (label, freq_a, power_a, colour) in zip(axes.flat, spectra_a):
        pair = by_label_b.get(label)
        pa = power_a / power_a.max() if (normalize and power_a.size and power_a.max() > 0) else power_a
        ax.plot(freq_a, pa, color=colour, lw=0.9, label=f'{label_a} — {label}')
        if pair is not None:
            freq_b, power_b = pair
            pb = power_b / power_b.max() if (normalize and power_b.size and power_b.max() > 0) else power_b
            ax.plot(freq_b, pb, color='black', lw=0.8, alpha=0.65,
                    label=f'{label_b} — {label}')
        ax.set_ylabel('normalised' if normalize else y_label)
        if log_y:
            ax.set_yscale('log')
        if xlim is not None:
            ax.set_xlim(*xlim)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)

    axes.flat[-1].set_xlabel('Frequency (mHz)')
    fig.suptitle(f'{kind_label}: {label_a} vs {label_b}'
                 f'{" (each normalised to its own peak)" if normalize else ""}', fontsize=12)
    plt.tight_layout()
    save_figure(fig, plots_dir, filename or f'{kind}_compare.png', save)
    plt.show()
    plt.close(fig)


def plot_spectrum_before_after(series, cadence_s, period_min, label='', width_mhz=0.0002,
                               kind='psd', xlim=None, save=False, plots_dir=None):
    """Spectrum before and after notching out one period, plus the filtered series.

    This is the check that makes `notch_filter` usable rather than merely available: it
    shows what was removed and, just as importantly, how much of the neighbourhood went
    with it.
    """
    estimate = {'fft': fft_spectrum, 'psd': psd_spectrum}[kind]
    y_label, _, kind_label = _spectrum_labels(kind)

    filtered, n_bins = notch_filter(series, cadence_s, period_min, width_mhz=width_mhz)
    freq_before, power_before = estimate(series, cadence_s)
    freq_after, power_after = estimate(filtered, cadence_s)
    target_mhz = 1e3 / (period_min * 60)

    fig, (ax_spec, ax_time) = plt.subplots(2, 1, figsize=(12, 7))
    ax_spec.plot(freq_before, power_before, color='steelblue', lw=0.8, label='before')
    ax_spec.plot(freq_after, power_after, color='crimson', lw=0.8, label='after')
    ax_spec.axvline(target_mhz, color='black', ls='--', lw=0.8,
                    label=f'{period_min:.0f} min ({n_bins} bin(s) zeroed)')
    ax_spec.set_xlabel('Frequency (mHz)')
    ax_spec.set_ylabel(y_label)
    ax_spec.set_yscale('log')
    if xlim is not None:
        ax_spec.set_xlim(*xlim)
    ax_spec.grid(alpha=0.25)
    ax_spec.legend(fontsize=8)
    ax_spec.set_title(f'{kind_label} before and after the notch  {label}')

    time_h = np.arange(len(filtered)) * cadence_s / 3600
    ax_time.plot(time_h, interpolate_gaps(series), color='steelblue', lw=0.7, alpha=0.6,
                 label='before')
    ax_time.plot(time_h, filtered, color='crimson', lw=0.9, label='after')
    ax_time.set_xlabel('Time (h)')
    ax_time.set_ylabel('Mean velocity (m/s)')
    ax_time.grid(alpha=0.25)
    ax_time.legend(fontsize=8)

    plt.tight_layout()
    save_figure(fig, plots_dir, f'notch_{period_min:.0f}min.png', save)
    plt.show()
    plt.close(fig)
    return filtered


# ── the pre-refactor entry points, kept so 03B reads the way 03T did ─────────

def plot_ffts_separate(data, metrics, annotate_peaks=True, print_table=True,
                       save=False, plots_dir=None, key='mean_dop', **kwargs):
    """2x2 FFT amplitude per region. Thin wrapper over `plot_spectra_separate`."""
    spectra = region_spectra(metrics, data['cadence_s'], kind='fft', key=key, **kwargs)
    return plot_spectra_separate(spectra, data['cadence_s'], kind='fft',
                                 annotate_peaks=annotate_peaks, print_table=print_table,
                                 save=save, plots_dir=plots_dir, filename='fft_separate.png')


def plot_ffts_combined(data, metrics, xlim=None, annotate_peaks=True, print_table=True,
                       save=False, plots_dir=None, key='mean_dop', **kwargs):
    """FFT amplitudes overlaid. Thin wrapper over `plot_spectra_combined`."""
    spectra = region_spectra(metrics, data['cadence_s'], kind='fft', key=key, **kwargs)
    return plot_spectra_combined(spectra, data['cadence_s'], kind='fft', xlim=xlim,
                                 annotate_peaks=annotate_peaks, print_table=print_table,
                                 save=save, plots_dir=plots_dir, filename='fft_combined.png')

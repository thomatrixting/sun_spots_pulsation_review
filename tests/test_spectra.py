"""Verification for src/spectra.py.

Run directly:  python tests/test_spectra.py

Two kinds of check:

1. **Correctness** on a synthetic signal whose period and amplitude are known, so the
   estimators are pinned to a right answer rather than to each other.
2. **Equivalence** to the four implementations `spectra.py` replaces. Those reference
   implementations are transcribed here verbatim from where they used to live
   (`sunspot_analysis._fft`, `03_data_analisis`'s `_psd` and `remove_period`,
   `04_data_comparison`'s `fft_amplitude`), so that the merge is provably not a change
   in numbers.
"""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.spectra import (  # noqa: E402
    band_amplitude,
    detrend_poly,
    fft_spectrum,
    interpolate_gaps,
    moving_average,
    notch_filter,
    psd_spectrum,
    spectral_peaks,
)

CADENCE_S = 720.0
PERIOD_H = 24.0
AMPLITUDE = 40.0          # m/s semi-amplitude
N_FRAMES = 481            # 4 days at 720 s, the length of a real NOAA cube


def _series(n=N_FRAMES, period_h=PERIOD_H, amplitude=AMPLITUDE, noise=0.0, seed=0):
    """A clean sinusoid on a real cube's time axis, optionally with noise."""
    time_h = np.arange(n) * CADENCE_S / 3600
    values = amplitude * np.sin(2 * np.pi * time_h / period_h)
    if noise:
        values = values + np.random.default_rng(seed).normal(0, noise, n)
    return time_h, values


# ── reference implementations, as they were before the merge ─────────────────

def _ref_fft(ts, cadence_s):
    """sunspot_analysis._fft, verbatim."""
    ts_c = np.array(ts, dtype=float)
    nans = ~np.isfinite(ts_c)
    if nans.any() and (~nans).sum() > 2:
        x = np.arange(len(ts_c))
        ts_c[nans] = np.interp(x[nans], x[~nans], ts_c[~nans])
    freqs = np.fft.rfftfreq(len(ts_c), d=cadence_s)
    amps = np.abs(np.fft.rfft(ts_c))
    return freqs[1:] * 1e3, amps[1:]


def _ref_psd(ts, cadence_s):
    """03_data_analisis Step 6c `_psd`, verbatim."""
    from scipy.signal import periodogram
    ts_c = np.array(ts, dtype=float)
    nans = ~np.isfinite(ts_c)
    if nans.any() and (~nans).sum() > 2:
        x = np.arange(len(ts_c))
        ts_c[nans] = np.interp(x[nans], x[~nans], ts_c[~nans])
    freqs_hz, power = periodogram(ts_c, fs=1.0 / cadence_s, scaling='density',
                                  return_onesided=True)
    return freqs_hz[1:] * 1e3, power[1:]


def _ref_prepare(time_h, series, detrend_deg=1):
    """04_data_comparison `_prepare`, verbatim."""
    t = np.asarray(time_h, dtype=float)
    y = np.asarray(series, dtype=float)
    good = np.isfinite(y)
    if good.sum() < 4:
        return np.array([]), np.array([])
    dt = float(np.median(np.diff(t)))
    grid = np.arange(t[0], t[-1] + 0.5 * dt, dt)
    y_u = np.interp(grid, t[good], y[good])
    return grid, y_u - np.polyval(np.polyfit(grid, y_u, detrend_deg), grid)


def _ref_fft_amplitude(time_h, series, band=(16.0, 36.0), pad=8):
    """04_data_comparison `fft_amplitude`, verbatim."""
    nan = dict(amplitude=np.nan, period_h=np.nan, n_cycles=np.nan)
    t, y = _ref_prepare(time_h, series)
    if t.size < 8:
        return nan
    n = t.size
    w = np.hanning(n)
    spec = np.fft.rfft(y * w, n=pad * n)
    freq = np.fft.rfftfreq(pad * n, d=t[1] - t[0])
    amp = 2 * np.abs(spec) / w.sum()
    inband = (freq >= 1 / band[1]) & (freq <= 1 / band[0])
    if not inband.any():
        return nan
    k = int(np.argmax(np.where(inband, amp, -np.inf)))
    period = 1 / freq[k]
    return dict(amplitude=float(amp[k]), period_h=float(period),
                n_cycles=float((t[-1] - t[0]) / period))


def _ref_remove_period(signal, dt_min, period_min, width=0.0002):
    """03_data_analisis `remove_period`, verbatim (frequencies in cycles/minute)."""
    import pandas as pd
    from scipy.fft import irfft, rfft, rfftfreq
    signal = np.asarray(signal)
    mask = np.isnan(signal)
    if np.any(mask):
        signal = pd.Series(signal).interpolate(limit_direction='both').to_numpy()
    freqs = rfftfreq(len(signal), d=dt_min)
    spectrum = rfft(signal)
    band = np.abs(freqs - 1.0 / period_min) < width
    spectrum[band] = 0
    return irfft(spectrum, n=len(signal))


# ── correctness ───────────────────────────────────────────────────────────────

def test_fft_recovers_the_injected_period():
    _, values = _series()
    freq_mhz, amp = fft_spectrum(values, CADENCE_S)
    peak_mhz = freq_mhz[np.argmax(amp)]
    period_h = 1e3 / (peak_mhz * 3600)
    assert abs(period_h - PERIOD_H) < 1.0, f'got {period_h:.2f} h, want {PERIOD_H} h'
    return f'peak at {peak_mhz:.5f} mHz = {period_h:.2f} h (injected {PERIOD_H} h)'


def test_amplitude_scaling_returns_physical_units():
    _, values = _series()
    _, amp = fft_spectrum(values, CADENCE_S, window='hann', scale='amplitude')
    peak = amp.max()
    assert abs(peak - AMPLITUDE) / AMPLITUDE < 0.05, f'got {peak:.2f}, want {AMPLITUDE}'
    return f'peak amplitude {peak:.2f} m/s (injected {AMPLITUDE} m/s)'


def test_band_amplitude_recovers_period_and_amplitude():
    time_h, values = _series(noise=3.0)
    result = band_amplitude(time_h, values)
    assert abs(result['period_h'] - PERIOD_H) < 1.0
    assert abs(result['amplitude'] - AMPLITUDE) / AMPLITUDE < 0.10
    return (f'period {result["period_h"]:.2f} h, amplitude {result["amplitude"]:.2f} m/s, '
            f'{result["n_cycles"]:.1f} cycles')


def test_band_amplitude_ignores_a_tone_outside_the_band():
    """A strong 6 h tone must not be mistaken for the 24 h signal."""
    time_h, values = _series()
    values = values + 200 * np.sin(2 * np.pi * time_h / 6.0)
    result = band_amplitude(time_h, values, band_h=(16.0, 36.0))
    assert abs(result['period_h'] - PERIOD_H) < 1.0, f'got {result["period_h"]:.2f} h'
    assert abs(result['amplitude'] - AMPLITUDE) / AMPLITUDE < 0.10
    return (f'a 200 m/s 6 h tone left the 24 h reading at '
            f'{result["amplitude"]:.2f} m/s, period {result["period_h"]:.2f} h')

def test_psd_recovers_the_injected_period():
    _, values = _series()
    freq_mhz, power = psd_spectrum(values, CADENCE_S)
    period_h = 1e3 / (freq_mhz[np.argmax(power)] * 3600)
    assert abs(period_h - PERIOD_H) < 1.0
    return f'PSD peak at {period_h:.2f} h'


def test_notch_removes_the_tone_and_keeps_the_rest():
    time_h, values = _series()
    other = 15.0 * np.sin(2 * np.pi * time_h / 4.0)
    filtered, n_bins = notch_filter(values + other, CADENCE_S, PERIOD_H * 60,
                                    width_mhz=0.002)
    # the 24 h component is gone
    _, amp_after = fft_spectrum(filtered, CADENCE_S, window='hann', scale='amplitude')
    freq_after, _ = fft_spectrum(filtered, CADENCE_S, window='hann', scale='amplitude')
    target_mhz = 1e3 / (PERIOD_H * 3600)
    near = np.abs(freq_after - target_mhz) < 0.003
    assert amp_after[near].max() < 0.05 * AMPLITUDE, 'the 24 h tone survived the notch'
    # the 4 h component is untouched
    four_h_mhz = 1e3 / (4.0 * 3600)
    near4 = np.abs(freq_after - four_h_mhz) < 0.003
    assert amp_after[near4].max() > 0.8 * 15.0, 'the notch removed the 4 h tone too'
    return (f'{n_bins} bin(s) zeroed; 24 h residual '
            f'{amp_after[near].max():.2f} m/s, 4 h kept at {amp_after[near4].max():.1f} m/s')


def test_interpolate_gaps_fills_nans_only():
    values = np.array([1.0, np.nan, 3.0, 4.0])
    out = interpolate_gaps(values)
    assert np.allclose(out, [1.0, 2.0, 3.0, 4.0])
    assert np.isfinite(out).all()
    return f'{values.tolist()} -> {out.tolist()}'


def test_detrend_removes_a_line_but_not_the_oscillation():
    time_h, values = _series()
    trended = values + 500 + 12 * time_h
    out = detrend_poly(time_h, trended, degree=1)
    assert abs(np.mean(out)) < 1e-6
    assert abs(out.std() - values.std()) / values.std() < 0.02
    return f'std kept at {out.std():.2f} vs {values.std():.2f}, mean {np.mean(out):.2e}'


def test_moving_average_smooths_out_the_period():
    """A full-period window averages the tone away wherever the window is complete.

    Only the interior is checked. `min_periods` lets the edges return a mean over a
    *partial* window, which is deliberate — it keeps a gappy series from going all-NaN —
    but a partial window covers a fraction of a period and so does not cancel it. Edge
    values of a rolling mean are not to be read as signal.
    """
    _, values = _series()
    smoothed, window = moving_average(values, CADENCE_S, PERIOD_H * 60)
    half = window // 2
    interior = smoothed[half:-half]
    assert np.isfinite(interior).all(), 'the interior should be fully defined'
    assert np.nanmax(np.abs(interior)) < 0.1 * AMPLITUDE, 'a full-period window left the tone'
    edge = np.nanmax(np.abs(smoothed[:half]))
    return (f'window {window} frames; interior residual '
            f'{np.nanmax(np.abs(interior)):.3f} m/s, partial-window edge {edge:.1f} m/s')


def test_spectral_peaks_finds_the_tone():
    time_h, values = _series(noise=2.0)
    freq_mhz, amp = fft_spectrum(values, CADENCE_S)
    peaks = spectral_peaks(freq_mhz, amp, n=5)
    assert len(peaks) > 0
    top = peaks.iloc[0]
    assert abs(top['period_min'] / 60 - PERIOD_H) < 1.5
    return f'top peak {top["period_min"] / 60:.2f} h, relative {top["relative"]:.3f}'


# ── equivalence to what came before ───────────────────────────────────────────

def test_fft_spectrum_matches_the_old_fft():
    _, values = _series(noise=5.0)
    values[10:14] = np.nan                      # gaps, to exercise the interpolation too
    f_ref, a_ref = _ref_fft(values, CADENCE_S)
    f_new, a_new = fft_spectrum(values, CADENCE_S)
    assert np.array_equal(f_ref, f_new), 'frequency axis differs'
    assert np.array_equal(a_ref, a_new), 'amplitudes differ'
    return f'{len(a_new)} bins identical to sunspot_analysis._fft'


def test_psd_spectrum_matches_the_prototype():
    _, values = _series(noise=5.0)
    values[10:14] = np.nan
    f_ref, p_ref = _ref_psd(values, CADENCE_S)
    f_new, p_new = psd_spectrum(values, CADENCE_S)
    assert np.array_equal(f_ref, f_new), 'frequency axis differs'
    assert np.allclose(p_ref, p_new, rtol=0, atol=0), 'power differs'
    return f'{len(p_new)} bins identical to 03_data_analisis._psd'


def test_band_amplitude_matches_04s_fft_amplitude():
    time_h, values = _series(noise=4.0)
    values[100:104] = np.nan
    ref = _ref_fft_amplitude(time_h, values)
    new = band_amplitude(time_h, values)
    for key in ('amplitude', 'period_h', 'n_cycles'):
        assert np.isclose(ref[key], new[key], rtol=0, atol=0), \
            f'{key}: {ref[key]!r} vs {new[key]!r}'
    return (f'amplitude {new["amplitude"]:.6f} m/s and period {new["period_h"]:.6f} h '
            f'identical to 04.fft_amplitude')


def test_notch_filter_matches_the_prototype_band():
    _, values = _series(noise=3.0)
    cadence_min = CADENCE_S / 60
    period_min = PERIOD_H * 60
    # The prototype works in cycles/minute; spectra.notch_filter works in mHz. Same
    # half-width expressed in each unit: 1 cycle/min = 1e3/60 mHz.
    width_cpm = 0.0002
    width_mhz = width_cpm * 1e3 / 60
    ref = _ref_remove_period(values, cadence_min, period_min, width=width_cpm)
    new, n_bins = notch_filter(values, CADENCE_S, period_min, width_mhz=width_mhz)
    assert np.allclose(ref, new, rtol=1e-12, atol=1e-9), \
        f'max diff {np.abs(ref - new).max():.3e}'
    return f'{n_bins} bin(s) zeroed; identical to 03_data_analisis.remove_period'


def _run():
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    passed = failed = 0
    for test in tests:
        try:
            note = test()
            print(f'PASS  {test.__name__}: {note}')
            passed += 1
        except AssertionError as exc:
            print(f'FAIL  {test.__name__}: {exc}')
            failed += 1
    print(f'\n{passed} passed, {failed} failed')
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(_run())

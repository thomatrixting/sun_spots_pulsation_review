"""
HMI dopplergram/magnetogram calibration coefficients (``hmi.coefficients``).

JSOC corrects the raw line-of-sight velocities V_lcp/V_rcp with a per-time
cubic polynomial::

    y = c0 + c1*x + c2*x**2 + c3*x**3

whose COEFF0..3 coefficients are published in the ``hmi.coefficients`` DRMS
series, one row per ~12h validity window (keyed by T_REC, valid over
[T_START, T_STOP)).  This module fetches those coefficients, matches them to
per-frame observation times, and applies/inverts the cubic vectorized over an
entire (n_t, ny, nx) cube (no sympy, no per-pixel Python loop).

It also implements the linear V_lcp/V_rcp <-> V, B relation:

    V = (V_lcp + V_rcp) / 2
    B = (V_lcp - V_rcp) * K
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from astropy.time import Time

K = 1.0 / (2.0 * 4.67e-5 * 0.000061733433 * 2.5 * 299792458.0)

_COEFF_KEYS = ['T_REC', 'T_START', 'T_STOP', 'COEFF0', 'COEFF1', 'COEFF2', 'COEFF3']


def fetch_coefficients(t_start: Time, t_stop: Time, pad_hours: float = 24.0) -> pd.DataFrame:
    """
    Fetch ``hmi.coefficients`` rows covering [t_start, t_stop], padded on
    each side so the very first/last cube frames still fall inside a
    matched [T_START, T_STOP) window.

    Returns a DataFrame with columns T_REC, T_START, T_STOP (as astropy
    Time-parsable strings converted to pandas Timestamps) and COEFF0..3,
    sorted by T_START.
    """
    import drms

    client = drms.Client()
    query_start = (t_start - pad_hours / 24.0).strftime('%Y.%m.%d_%H:%M:%S_TAI')
    span_days = (t_stop - t_start).jd + 2 * pad_hours / 24.0
    query = f'hmi.coefficients[{query_start}/{span_days:.3f}d]'
    df = client.query(query, key=_COEFF_KEYS)
    if df.empty:
        raise ValueError(f'No hmi.coefficients rows returned for query {query!r}')

    for col in ('T_REC', 'T_START', 'T_STOP'):
        df[col] = pd.to_datetime(
            df[col].str.replace('_TAI', '', regex=False).str.replace('.', '-', n=2, regex=False),
            format='%Y-%m-%d_%H:%M:%S',
        )
    return df.sort_values('T_START').reset_index(drop=True)


def match_coefficients_to_times(
    obs_times: Time, coeff_df: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Match each frame's observation time to the hmi.coefficients row whose
    [T_START, T_STOP) window contains it.

    Parameters
    ----------
    obs_times : astropy.time.Time, shape (n_t,)
    coeff_df  : DataFrame from fetch_coefficients()

    Returns
    -------
    c0, c1, c2, c3 : np.ndarray, shape (n_t,)
    """
    frame_ts = pd.DataFrame({'t_obs': pd.to_datetime(obs_times.utc.datetime)}).sort_values('t_obs')

    matched = pd.merge_asof(
        frame_ts, coeff_df, left_on='t_obs', right_on='T_START', direction='backward',
    )
    # merge_asof requires sorted input; restore original frame order afterwards.
    matched = matched.sort_index()

    out_of_window = matched['t_obs'] >= matched['T_STOP']
    if out_of_window.any():
        raise ValueError(
            f'{out_of_window.sum()} frame(s) fall outside their matched '
            f'[T_START, T_STOP) coefficient window — widen pad_hours in fetch_coefficients().'
        )
    if matched['T_REC'].isna().any():
        raise ValueError('Some frames could not be matched to a coefficient row.')

    return (
        matched['COEFF0'].to_numpy(dtype=float),
        matched['COEFF1'].to_numpy(dtype=float),
        matched['COEFF2'].to_numpy(dtype=float),
        matched['COEFF3'].to_numpy(dtype=float),
    )


def _broadcast(c: np.ndarray) -> np.ndarray:
    """(n_t,) -> (n_t, 1, 1) for broadcasting against (n_t, ny, nx) cubes."""
    return np.asarray(c, dtype=float)[:, None, None]


def apply_cubic(x: np.ndarray, c0: np.ndarray, c1: np.ndarray, c2: np.ndarray, c3: np.ndarray) -> np.ndarray:
    """
    Forward evaluation y = c0 + c1*x + c2*x**2 + c3*x**3, per-frame coefficients.

    x is expected in the same units as the raw dopplergram cubes (m/s).
    """
    c0, c1, c2, c3 = (_broadcast(c) for c in (c0, c1, c2, c3))
    return c0 + c1 * x + c2 * x**2 + c3 * x**3


def _cbrt(v: np.ndarray) -> np.ndarray:
    return np.sign(v) * np.abs(v) ** (1.0 / 3.0)


def invert_cubic(y: np.ndarray, c0: np.ndarray, c1: np.ndarray, c2: np.ndarray, c3: np.ndarray) -> np.ndarray:
    """
    Invert y = c0 + c1*x + c2*x**2 + c3*x**3 for x, vectorized over the whole
    cube via Cardano's formula on the depressed cubic.

    Per-frame coefficients (c0..c3, shape (n_t,)) are broadcast against y of
    shape (n_t, ny, nx). When the cubic has three real roots, the root
    closest to ``y`` itself is selected — a generic, assumption-free
    tie-breaker (no near-identity slope is assumed) that picks the
    physically plausible solution of the same order of magnitude as the
    measured value, rather than a wildly different root.
    """
    a, b, c = c3, c2, c1
    a_b, b_b, c_b = (_broadcast(v) for v in (a, b, c))
    d_b = _broadcast(c0) - y

    p = (3 * a_b * c_b - b_b**2) / (3 * a_b**2)
    p = np.broadcast_to(p, y.shape)
    q = (2 * b_b**3 - 9 * a_b * b_b * c_b + 27 * a_b**2 * d_b) / (27 * a_b**3)

    disc = (q / 2) ** 2 + (p / 3) ** 3

    t = np.empty_like(y, dtype=float)

    one_real = disc > 0
    if np.any(one_real):
        sqrt_disc = np.sqrt(disc[one_real])
        u = _cbrt(-q[one_real] / 2 + sqrt_disc)
        v = _cbrt(-q[one_real] / 2 - sqrt_disc)
        t[one_real] = u + v

    three_real = ~one_real
    if np.any(three_real):
        p3 = p[three_real]
        q3 = q[three_real]
        r = 2 * np.sqrt(-p3 / 3)
        arg = np.clip((3 * q3) / (2 * p3) * np.sqrt(-3 / p3), -1.0, 1.0)
        theta = np.arccos(arg)
        candidates = np.stack(
            [r * np.cos((theta - 2 * np.pi * k) / 3) for k in range(3)], axis=0
        )
        # broadcast b/(3a) shift + pick root closest to y itself
        shift = (b_b / (3 * a_b))
        shift3 = np.broadcast_to(shift, y.shape)[three_real]
        x_candidates = candidates - shift3[None, ...]
        y3 = y[three_real]
        best = np.argmin(np.abs(x_candidates - y3[None, ...]), axis=0)
        t[three_real] = np.take_along_axis(candidates, best[None, ...], axis=0)[0]

    return t - b_b / (3 * a_b)


def vlcp_rcp_from_v_b(V: np.ndarray, B: np.ndarray, k: float = K) -> tuple[np.ndarray, np.ndarray]:
    """Given dopplergram V and magnetogram B, recover V_lcp, V_rcp."""
    V_lcp = V + B / (2.0 * k)
    V_rcp = V - B / (2.0 * k)
    return V_lcp, V_rcp


def v_b_from_vlcp_rcp(V_lcp: np.ndarray, V_rcp: np.ndarray, k: float = K) -> tuple[np.ndarray, np.ndarray]:
    """Given V_lcp, V_rcp, compute dopplergram V and magnetogram B."""
    V = (V_lcp + V_rcp) / 2.0
    B = (V_lcp - V_rcp) * k
    return V, B

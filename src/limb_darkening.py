"""
Limb-darkening correction for SDO/HMI continuum images, following Castellanos Durán &
Kleint (2020), ApJ 904, 96, Sect. 2.1 ("Flare Sample and Data Reduction") — the fifth
cleaning step in ``How to clean data from (Duran_2021).md``.

The continuum intensity of the quiet Sun is not flat across the disk: it falls towards the
limb because a slanted line of sight probes shallower, cooler layers. The paper corrects
this "to second order"::

    I_corrected = I_observed / C                                             (Eq. 1)
    C(mu)       = 1 - u - v + u mu + v mu^2                                  (Eq. 2)
    Theta       = arcsin( sqrt((x - x_c)^2 + (y - y_c)^2) / R_sun ),  mu = cos(Theta)

with the wavelength-dependent coefficients of Allen (1976) at the Fe I 6173.3 A line HMI
observes::

    u = 0.836      v = -0.204

``C`` is a divisor, so the correction only ever brightens, and most so near the limb. It is
1 at disk centre by construction (``1 - u - v + u + v``) and ``1 - u - v = 0.368`` at the
extreme limb, so it is bounded in ``[0.368, 1]`` and can never change the sign of an
intensity.

Why this matters here, measured on NOAA 11536 (2012-07-31, 484x139 px box, 72 h)::

    time              mu at box centre   mean C   1/C     C across the box
    20120731_000000        0.783          0.895   1.118   0.840..0.935  (11.3%)
    20120801_120000        0.870          0.941   1.063   0.918..0.953  ( 3.8%)
    20120802_060000        0.873          0.942   1.061   0.922..0.951  ( 3.2%)
    20120803_000000        0.848          0.930   1.075   0.896..0.951  ( 6.1%)

Two separate effects, and thresholding each frame against its own quiet-Sun percentile
(what ``02A_data_procesing.ipynb`` does) only removes the first:

1. The frame-mean level drifts ~5% over the window, and *non-monotonically* — the region
   rotates in from the east limb, passes closest to disk centre around 2012-08-02, and
   heads back out. A smooth 5% hump on a 3-day baseline is exactly the shape a slow
   pulsation search would report as a detection.
2. Within a single frame ``C`` spans 3-11%. A per-frame scalar threshold cannot touch this,
   so the umbra/penumbra cut sits systematically tighter on the limbward side of the box —
   by an amount that itself changes across the window.

Applies to the **continuum only**. Limb darkening is an intensity effect; the magnetogram
and Dopplergram are not divided by ``C``. Their zero-point corrections live in
``src/doppler_calibration.py``, whose structure this module deliberately mirrors: analytic
pieces first, then a cube-level driver that matches per-frame files to the cube by
timestamp.
"""

from __future__ import annotations

import numpy as np

# ── Constants from the paper ─────────────────────────────────────────────────

#: Eq. 2 — limb-darkening coefficients at 6173.3 A (Allen 1976), the line HMI observes.
#: These are wavelength-dependent; using them at another wavelength is simply wrong.
U_LAMBDA = 0.836
V_LAMBDA = -0.204

#: Eq. 3 — conversion from HMI's DN/s to physical intensity, derived by the authors from
#: the Neckel (1994) atlas: disk-centre intensity is 0.315e7 erg/s/cm^2/sr/A against a
#: measured 60000 +/- 300 DN/s.
#:
#: Recorded, deliberately not applied. Every quantity downstream (the umbra/penumbra
#: thresholds, the quiet-Sun percentile) is a *ratio* of intensities, so a global scale
#: factor cancels out of all of them while making every plotted number harder to compare
#: against the raw frames and against DS9. Multiply by this only when an absolute
#: radiometric intensity is actually wanted.
DN_TO_CGS = 52.5  # [erg s^-1 cm^-2 sr^-1 A^-1] / [DN s^-1]

#: How `mu_from_map` can obtain cos(Theta). See its docstring for the measured difference.
MU_METHODS = ('geometry', 'paper')


# ── Eq. 2: the limb-darkening function ───────────────────────────────────────

def limb_darkening_function(mu, u_lambda: float = U_LAMBDA, v_lambda: float = V_LAMBDA):
    """Eq. 2 — the second-order limb-darkening factor ``C(mu)``.

    ``C = 1 - u - v + u mu + v mu^2``

    A pure function of ``mu``: no map, no units, no I/O, so it can be checked against its
    two analytic fixed points without any data on disk. With the default coefficients::

        C(1) = 1                     exactly, by construction — disk centre is the reference
        C(0) = 1 - u - v = 0.368     the extreme limb

    Parameters
    ----------
    mu : array_like
        ``cos(Theta)`` in [0, 1]. NaN (an off-disk pixel) propagates to NaN, which is the
        honest answer for a pixel with no surface.

    Returns
    -------
    ndarray
        ``C``, same shape as ``mu``.
    """
    floor = 1.0 - u_lambda - v_lambda
    if floor <= 0:
        raise ValueError(
            f'Limb-darkening coefficients u={u_lambda}, v={v_lambda} give C(mu=0) = '
            f'1-u-v = {floor:.3f} <= 0, so Eq. 1 would divide by zero or flip the sign of '
            f'the intensity near the limb. Check the coefficients are for the observed '
            f'wavelength (6173.3 A for HMI: u={U_LAMBDA}, v={V_LAMBDA}).')

    mu = np.asarray(mu, dtype=float)
    return 1.0 - u_lambda - v_lambda + u_lambda * mu + v_lambda * mu**2


# ── cos(Theta) from a map ────────────────────────────────────────────────────

def mu_from_map(smap, method: str = 'geometry') -> np.ndarray:
    """Per-pixel ``mu = cos(Theta)`` for one map, NaN off the disk.

    Two methods, which differ by more than the paper's formulation suggests:

    ``'geometry'`` (default)
        Reuses `src.doppler_calibration.frame_geometry`, which derives ``mu`` from the
        Heliocentric ``z`` coordinate through sunpy's own transforms and therefore accounts
        for the observer being a finite distance away.

    ``'paper'``
        Eq. 2 literally: ``Theta = arcsin(hypot(Tx, Ty) / R_sun)``, i.e.
        ``mu = sqrt(1 - (r/R_sun)^2)``. This is the observer-at-infinity approximation.

    On the first frame of NOAA 11536 (box spanning ``mu`` 0.68-0.86) the two differ by up to
    **2.5e-3** in ``mu``, which through ``dC/dmu = u + 2 v mu ~ 0.43`` is a ~0.1% intensity
    error. That is 25x larger than the ~1e-4 quoted in `frame_geometry`'s docstring, because
    that figure is for a region near disk centre and the approximation degrades towards the
    limb — exactly where this correction does its work. ``'geometry'`` costs 0.105 s per
    frame against 0.034 s for ``'paper'``, i.e. 38 s instead of 12 s for a 361-frame cube,
    which is not a trade worth making. ``'paper'`` is kept as the fast path and as the
    independent cross-check the tests use.

    Returns
    -------
    ndarray
        ``mu``, shape ``smap.data.shape``, NaN outside the solar disk.
    """
    if method not in MU_METHODS:
        raise ValueError(f'method must be one of {MU_METHODS}, got {method!r}')

    if method == 'geometry':
        from src.doppler_calibration import frame_geometry
        return np.asarray(frame_geometry(smap).mu, dtype=float)

    import astropy.units as u
    import sunpy.map

    # Only pixel_to_world, no frame transform — that is where the 3x saving comes from.
    coords = sunpy.map.all_coordinates_from_map(smap)
    r = np.hypot(coords.Tx.to_value(u.arcsec), coords.Ty.to_value(u.arcsec))
    r_sun = smap.rsun_obs.to_value(u.arcsec)

    with np.errstate(invalid='ignore'):
        mu = np.sqrt(1.0 - (r / r_sun) ** 2)
    mu[r > r_sun] = np.nan  # off-disk: no surface, so no mu
    return mu


# ── Eq. 1: apply it ──────────────────────────────────────────────────────────

def correct_limb_darkening(smap, method: str = 'geometry', u_lambda: float = U_LAMBDA,
                           v_lambda: float = V_LAMBDA, mu: np.ndarray | None = None):
    """Eq. 1 — divide one continuum frame by its limb-darkening factor.

    Returns ``(corrected, C, mu)`` rather than just the quotient, for the same reason
    `calibrate_dopplergram` returns its ``terms`` dict: a correction whose size cannot be
    inspected is indistinguishable from solar signal when it goes wrong. ``mean(C)`` per
    frame is the diagnostic 02A plots.

    Parameters
    ----------
    smap : sunpy.map.Map
        An HMI continuum intensity map (BUNIT ``DN/s``).
    method : {'geometry', 'paper'}
        How to obtain ``mu`` — see `mu_from_map`.
    mu : ndarray, optional
        A precomputed ``mu`` for this frame, skipping the geometry entirely.

    Returns
    -------
    (corrected, C, mu)
        ``corrected`` is float32 in the same units as the input; ``C`` and ``mu`` are float64
        arrays of the same shape, NaN off-disk.
    """
    if mu is None:
        mu = mu_from_map(smap, method=method)

    c = limb_darkening_function(mu, u_lambda=u_lambda, v_lambda=v_lambda)
    corrected = (np.asarray(smap.data, dtype=np.float64) / c).astype(np.float32)
    return corrected, c, mu


# ── Cube-level application ───────────────────────────────────────────────────

def limb_darkening_cube(region_dir, region: str = 'region_01', method: str = 'geometry',
                        u_lambda: float = U_LAMBDA, v_lambda: float = V_LAMBDA,
                        series_glob: str = 'hmi.ic_*.continuum.fits'):
    """Apply the limb-darkening correction to every frame of a continuum cube.

    **The correction is per-frame, not per-cube.** A tracked cutout follows the region as it
    rotates, so ``mu`` at the box centre runs 0.78 -> 0.87 -> 0.85 across NOAA 11536's 72 h
    window. One ``C`` map applied cube-wide would leave most of the effect in place *and*
    inject a spurious 5% trend of its own.

    `make_cube` keeps only the reference frame's header, so the geometry has to be read back
    from the surviving per-file frames in ``region_dir/region``, matched to the cube **by
    timestamp** via its TIMESTAMPS extension. That is the same join `calibrate_cube` uses,
    and for the same reason: a frame missing from the middle of the window must not be able
    to silently shift the corrections out of step.

    Parameters
    ----------
    region_dir : path
        An AR directory containing ``<region>_continuum_cube.fits`` and ``<region>/``.
    region : str
        Region prefix within that directory.
    method : {'geometry', 'paper'}
        How to obtain ``mu`` — see `mu_from_map`.
    series_glob : str
        Frame filename pattern. The default matches both cadences (``hmi.ic_720s.*`` and
        ``hmi.ic_45s.*``), since NOAA 11117 is downloaded at 45 s.

    Returns
    -------
    (cube, timestamps, c_means)
        ``cube`` is the corrected intensity as float32, same shape as the input.
        ``c_means`` is an ``(n_t,)`` array of the per-frame spatial mean of ``C`` — plot it
        to see the correction grow as the region approaches the limb. If it comes back flat,
        the per-frame headers are not being read and the correction is wrong.
    """
    import pathlib

    from src.utilities import apply_per_frame_correction

    region_dir = pathlib.Path(region_dir)

    def correct(smap, _timestamp):
        corrected, c, _ = correct_limb_darkening(
            smap, method=method, u_lambda=u_lambda, v_lambda=v_lambda)
        return corrected, {'c': np.nanmean(c)}

    cube, timestamps, diag = apply_per_frame_correction(
        region_dir / f'{region}_continuum_cube.fits',
        region_dir / region,
        correct,
        series_glob=series_glob,
        diag_names=('c',),
    )
    return cube, timestamps, diag['c']

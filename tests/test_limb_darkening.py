"""Verification for src/limb_darkening.py.

Run directly:  python tests/test_limb_darkening.py

The analytic checks pin Eq. 2 to values stated in, or directly derivable from, Castellanos
Durán & Kleint (2020). The real-frame checks need HMI continuum frames and are skipped with
a notice if they aren't on disk.
"""

import glob
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.limb_darkening import (  # noqa: E402
    U_LAMBDA,
    V_LAMBDA,
    correct_limb_darkening,
    limb_darkening_function,
    mu_from_map,
)

REAL_FRAMES = 'data/raw/NOAA_11536_2012-07-31/region_01'
CONTINUUM_GLOB = f'{REAL_FRAMES}/hmi.ic_*.continuum.fits'


def test_disk_centre_is_unity():
    """Eq. 2 is normalised so disk centre is the reference: C(1) = 1 - u - v + u + v = 1."""
    c = float(limb_darkening_function(1.0))
    assert abs(c - 1.0) < 1e-12, f'C(mu=1) is {c}, expected exactly 1'
    return f'C(mu=1) = {c:.12f}'


def test_limb_value():
    """At the extreme limb only the constant term survives: C(0) = 1 - u - v."""
    c = float(limb_darkening_function(0.0))
    expected = 1 - U_LAMBDA - V_LAMBDA
    assert abs(c - expected) < 1e-12, f'C(mu=0) is {c}, expected {expected}'
    assert abs(c - 0.368) < 5e-4, f'C(mu=0) = {c}, expected ~0.368 for the 6173.3 A pair'
    return f'C(mu=0) = {c:.4f} = 1 - u - v'


def test_monotonic_and_bounded():
    """C must rise monotonically to 1 and never approach zero, or Eq. 1 misbehaves.

    A divisor that changed sign, or dipped near zero, would flip or explode the corrected
    intensity — which is why `limb_darkening_function` refuses coefficients with 1-u-v <= 0.
    """
    mu = np.linspace(0, 1, 1001)
    c = limb_darkening_function(mu)
    assert np.all(np.diff(c) > 0), 'C(mu) is not strictly increasing'
    assert c.min() > 0.36 and c.max() <= 1.0 + 1e-12, f'C out of bounds: {c.min()}..{c.max()}'
    return f'C rises monotonically over {c.min():.4f}..{c.max():.4f}'


def test_bad_coefficients_are_refused():
    """A wrong-wavelength pair must fail loudly, not divide by ~0 near the limb."""
    try:
        limb_darkening_function(0.5, u_lambda=1.5, v_lambda=0.2)
    except ValueError as exc:
        assert '1-u-v' in str(exc)
        return 'coefficients giving C(0) <= 0 are rejected'
    raise AssertionError('expected a ValueError for coefficients with 1 - u - v <= 0')


def test_nan_off_disk_propagates():
    """An off-disk pixel has no surface, so mu is NaN and the correction must stay NaN."""
    c = limb_darkening_function(np.array([np.nan, 1.0]))
    assert np.isnan(c[0]) and c[1] == 1.0
    return 'NaN mu -> NaN C'


def test_mu_methods_agree_on_real_frame():
    """'paper' and 'geometry' must agree to well inside the effect being corrected.

    They are not identical: 'paper' is the observer-at-infinity approximation, and the two
    differ by ~2.5e-3 in mu over this box (mu 0.68-0.86). Through dC/dmu ~ 0.43 that is a
    ~0.1% intensity error, two orders below the 3-11% limb darkening itself — but large
    enough that 'geometry' is the default. A much larger difference means one of the two
    geometries regressed.
    """
    import sunpy.map

    files = sorted(glob.glob(CONTINUUM_GLOB))
    if not files:
        return None
    smap = sunpy.map.Map(files[0])

    mu_paper = mu_from_map(smap, method='paper')
    mu_geom = mu_from_map(smap, method='geometry')
    diff = np.nanmax(np.abs(mu_paper - mu_geom))
    assert diff < 1e-2, f'the two mu methods differ by {diff}, expected ~2.5e-3'

    c_diff = np.nanmax(np.abs(limb_darkening_function(mu_paper)
                              - limb_darkening_function(mu_geom)))
    return (f'mu {np.nanmin(mu_geom):.4f}..{np.nanmax(mu_geom):.4f}, '
            f'methods differ by {diff:.1e} in mu, {c_diff:.1e} in C')


def test_correction_only_brightens():
    """Eq. 1 divides by C <= 1, so the corrected intensity can only go up."""
    import sunpy.map

    files = sorted(glob.glob(CONTINUUM_GLOB))
    if not files:
        return None
    smap = sunpy.map.Map(files[0])
    corrected, c, mu = correct_limb_darkening(smap)

    on_disk = np.isfinite(mu) & np.isfinite(smap.data)
    assert np.all(c[on_disk] <= 1.0 + 1e-12), 'C exceeds 1 somewhere on disk'
    positive = on_disk & (smap.data > 0)
    assert np.all(corrected[positive] >= smap.data[positive]), 'correction dimmed a pixel'

    boost = np.nanmean(corrected[on_disk]) / np.nanmean(smap.data[on_disk])
    return f'mean C = {np.nanmean(c):.4f}, mean intensity boost {boost:.4f}x'


#: Frame timestamp -> mean C over the box, derived straight from the FITS headers
#: (CRPIX/CRVAL/RSUN_OBS) with a standalone script rather than through this module, so a WCS
#: or coefficient regression shows up here even though every check above stays
#: self-consistent. Keyed by timestamp, not by index: the download window grows as more days
#: are fetched, and index 0/mid/last then point at different frames.
MEASURED_MEAN_C = {
    '20120731_000000': 0.8946,
    '20120801_120000': 0.9412,
    '20120802_060000': 0.9424,
    '20120803_000000': 0.9303,
}


def test_mean_c_matches_measured_values():
    """Pin the whole chain — WCS -> mu -> Eq. 2 — to numbers measured independently."""
    import sunpy.map

    by_stamp = {stamp: path for path in sorted(glob.glob(CONTINUUM_GLOB))
                for stamp in MEASURED_MEAN_C if f'.{stamp}_TAI.' in path}
    if len(by_stamp) < len(MEASURED_MEAN_C):
        return None

    measured = []
    for stamp, want in MEASURED_MEAN_C.items():
        _, c, _ = correct_limb_darkening(sunpy.map.Map(by_stamp[stamp]))
        got = float(np.nanmean(c))
        measured.append(f'{stamp[4:]}={got:.4f}')
        assert abs(got - want) < 5e-3, f'{stamp}: mean C {got:.4f}, expected {want:.4f}'
    return 'mean C at ' + ', '.join(measured)


def test_correction_flattens_the_rotation_trend():
    """The point of the correction: the frame-mean intensity trend must shrink.

    NOAA 11536 rotates from mu~0.78 in towards disk centre and back out, so the raw
    frame-mean continuum carries a ~5% geometric hump over the window. After dividing by C
    that hump should largely be gone; what is left is real evolution of the spot.
    """
    import sunpy.map

    files = sorted(glob.glob(CONTINUUM_GLOB))
    if len(files) < 361:
        return None

    sample = files[::40]  # ~10 frames spanning the window
    raw, corrected = [], []
    for path in sample:
        smap = sunpy.map.Map(path)
        fixed, _, mu = correct_limb_darkening(smap)
        on_disk = np.isfinite(mu)
        raw.append(np.nanmean(smap.data[on_disk]))
        corrected.append(np.nanmean(fixed[on_disk]))

    raw_drift = 100 * np.ptp(raw) / np.mean(raw)
    corr_drift = 100 * np.ptp(corrected) / np.mean(corrected)
    assert corr_drift < raw_drift, (
        f'correction did not flatten the trend: {raw_drift:.2f}% -> {corr_drift:.2f}%')
    return f'frame-mean drift {raw_drift:.2f}% -> {corr_drift:.2f}% over the window'


if __name__ == '__main__':
    import warnings

    warnings.filterwarnings('ignore')
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    failures, skipped = 0, 0
    for test in tests:
        try:
            detail = test()
        except AssertionError as exc:
            print(f'FAIL  {test.__name__}: {exc}')
            failures += 1
        except Exception as exc:
            print(f'ERROR {test.__name__}: {exc!r}')
            failures += 1
        else:
            if detail is None:
                print(f'SKIP  {test.__name__} (no data at {REAL_FRAMES})')
                skipped += 1
            else:
                print(f'PASS  {test.__name__}: {detail}')
    print(f'\n{len(tests) - failures - skipped} passed, {skipped} skipped, {failures} failed')
    sys.exit(1 if failures else 0)

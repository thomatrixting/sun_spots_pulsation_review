"""Verification for src/doppler_calibration.py.

Run directly:  python tests/test_doppler_calibration.py

The analytic checks pin each formula to a value stated in, or directly derivable from,
Castellanos Durán et al. (2021). The closure check needs real HMI frames and is skipped
with a notice if they aren't on disk.
"""

import glob
import pathlib
import sys
import types

import astropy.units as u
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.doppler_calibration import (  # noqa: E402
    GRAVITATIONAL_REDSHIFT,
    FrameGeometry,
    convective_blueshift,
    frame_geometry,
    large_scale_flow_los,
    sdo_los_velocity_keywords,
)

REAL_FRAMES = 'data/raw/NOAA_11536_2012-07-31/region_01'


def _geometry(lat_deg, lon_deg, b0_deg=0.0, theta_rho_arcsec=0.0, psi_deg=0.0, mu=1.0):
    """A FrameGeometry for one hand-specified point, for the analytic checks."""
    return FrameGeometry(
        theta_rho=np.atleast_1d(theta_rho_arcsec) * u.arcsec,
        psi=np.atleast_1d(psi_deg) * u.deg,
        mu=np.atleast_1d(mu),
        lat=np.atleast_1d(lat_deg) * u.deg,
        lon=np.atleast_1d(lon_deg) * u.deg,
        b0=b0_deg * u.deg,
    )


def test_clv_at_disk_centre():
    """Eq. 5 must reproduce the paper's quoted disk-centre convective blueshift."""
    value = convective_blueshift(1.0).to_value(u.m / u.s)
    assert abs(value - (-278.0)) < 1.0, f'CLV at mu=1 is {value}, expected -278'
    # The curve is not monotonic - it deepens before turning over towards the limb.
    assert convective_blueshift(0.7) < convective_blueshift(1.0)
    assert convective_blueshift(0.3) > convective_blueshift(0.7)
    return f'CLV(mu=1) = {value:.1f} m/s  (paper: -275)'


def test_lsf_zero_at_disk_centre():
    """Rotation is perpendicular to the LOS on the central meridian, so Eq. 2 vanishes."""
    v = large_scale_flow_los(None, geometry=_geometry(0, 0)).to_value(u.m / u.s)[0]
    assert abs(v) < 1e-6, f'LSF at disk centre is {v}, expected 0'
    return f'LSF(0N, 0E) = {v:.3e} m/s'


def test_lsf_west_limb_sign_and_size():
    """At the equatorial west limb the surface recedes: a redshift of about +1.8 km/s.

    This is the check that catches a sign flip in the Carrington/synodic bracket, which
    reads as if it were wrong (v_rot is a residual, not a bulk velocity).
    """
    v = large_scale_flow_los(None, geometry=_geometry(0, 89)).to_value(u.m / u.s)[0]
    assert 1700 < v < 1900, f'LSF at the west limb is {v}, expected ~+1811'
    east = large_scale_flow_los(None, geometry=_geometry(0, -89)).to_value(u.m / u.s)[0]
    assert abs(east + v) < 1e-6, 'east and west limb should be antisymmetric'
    return f'LSF(0N, 89W) = {v:.1f} m/s, LSF(0N, 89E) = {east:.1f} m/s'


def test_sdo_reduces_to_obs_vr_at_disk_centre():
    """At theta_rho = 0 only the radial term survives, so Eq. 1 collapses to OBS_VR."""
    from sunpy.util import MetaDict

    fake = types.SimpleNamespace(
        meta=MetaDict({'OBS_VR': 2056.4, 'OBS_VW': 29116.2, 'OBS_VN': 4334.9}))
    v = sdo_los_velocity_keywords(fake, geometry=_geometry(0, 0, theta_rho_arcsec=0.0))
    value = v.to_value(u.m / u.s)[0]
    assert abs(value - 2056.4) < 1e-6, f'Eq. 1 at disk centre is {value}, expected OBS_VR'
    return f'Eq.1 at disk centre = {value:.1f} m/s = OBS_VR'


def test_gravitational_redshift():
    assert abs(GRAVITATIONAL_REDSHIFT.to_value(u.m / u.s) - 636.03) < 1e-9
    return f'v_gravity = {GRAVITATIONAL_REDSHIFT:.2f}'


def test_geometry_against_real_frame():
    """theta_rho must equal hypot(Tx, Ty), and mu must match the spherical identity."""
    import sunpy.map

    files = sorted(glob.glob(f'{REAL_FRAMES}/hmi.v_720s.*.fits'))
    if not files:
        return None
    smap = sunpy.map.Map(files[0])
    g = frame_geometry(smap)

    coords = sunpy.map.all_coordinates_from_map(smap)
    rho = np.hypot(coords.Tx, coords.Ty)
    diff = np.nanmax(np.abs(g.theta_rho - rho)).to_value(u.arcsec)
    assert diff < 1e-3, f'theta_rho differs from hypot(Tx,Ty) by {diff} arcsec'

    b0, lat, lon = g.b0.to_value(u.rad), g.lat.to_value(u.rad), g.lon.to_value(u.rad)
    mu_identity = np.sin(b0) * np.sin(lat) + np.cos(b0) * np.cos(lat) * np.cos(lon)
    mu_diff = np.nanmax(np.abs(g.mu - mu_identity))
    assert mu_diff < 1e-3, f'mu differs from the spherical identity by {mu_diff}'
    return f'theta_rho match {diff * 1000:.2f} mas, mu identity match {mu_diff:.1e}'


def test_closure_on_real_data():
    """The four terms must account for the measured quiet-Sun velocity.

    Baseline on NOAA 11536 frame 0: predicted 1580.1 vs measured 1367.2 m/s, i.e. the
    terms explain the measurement to ~213 m/s while individually ranging over
    800-2100 m/s. A materially worse residual means a sign or a formula regressed.
    """
    import sunpy.map
    from astropy.io import fits

    from src.doppler_calibration import calibrate_dopplergram

    v_files = sorted(glob.glob(f'{REAL_FRAMES}/hmi.v_720s.*.fits'))
    b_files = sorted(glob.glob(f'{REAL_FRAMES}/hmi.m_720s.*.magnetogram.fits'))
    if not v_files or not b_files:
        return None

    smap = sunpy.map.Map(v_files[0])
    B = fits.getdata(b_files[0], ext=1).astype(float)
    if B.shape != smap.data.shape:
        return None

    corrected, terms = calibrate_dopplergram(smap)
    quiet = (np.abs(B) < 100) & np.isfinite(smap.data)

    predicted = sum(np.broadcast_to(t.to_value(u.m / u.s), smap.data.shape)[quiet].mean()
                    for t in terms.values())
    measured = smap.data[quiet].mean()
    residual = measured - predicted
    assert abs(residual) < 500, f'closure residual {residual:.1f} m/s exceeds 500'

    # The corrected map is just the measurement minus the terms.
    assert abs(np.nanmean(corrected.to_value(u.m / u.s)[quiet]) - residual) < 1e-3
    breakdown = '  '.join(f'{k}={np.mean(np.broadcast_to(v.to_value(u.m/u.s), smap.data.shape)[quiet]):.0f}'
                          for k, v in terms.items())
    return f'predicted {predicted:.1f} vs measured {measured:.1f} -> residual {residual:.1f} m/s\n    {breakdown}'


def test_corrections_flatten_instrumental_drift():
    """The point of the whole exercise: the corrections must remove the drift.

    Over a few hours the raw frame-mean velocity swings by hundreds of m/s, almost all
    of it SDO's orbit. After correction that swing should mostly be gone — and gone
    because it was modelled, not because a plane was fitted to it. Measured baseline
    over 10 consecutive NOAA 11536 frames: 360 m/s peak-to-peak raw vs 13 m/s corrected.
    """
    import sunpy.map

    from src.doppler_calibration import calibrate_dopplergram

    files = sorted(glob.glob(f'{REAL_FRAMES}/hmi.v_720s.*.fits'))[:10]
    if len(files) < 10:
        return None

    raw, corrected = [], []
    for path in files:
        smap = sunpy.map.Map(path)
        raw.append(np.nanmean(smap.data))
        values, _ = calibrate_dopplergram(smap)
        corrected.append(np.nanmean(values.to_value(u.m / u.s)))

    raw_ptp, corrected_ptp = np.ptp(raw), np.ptp(corrected)
    assert corrected_ptp < raw_ptp / 5, (
        f'corrections only reduced the drift from {raw_ptp:.1f} to {corrected_ptp:.1f} m/s')
    return f'frame-mean drift {raw_ptp:.1f} -> {corrected_ptp:.1f} m/s ({raw_ptp / corrected_ptp:.0f}x flatter)'


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

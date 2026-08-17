"""
Doppler zero-point calibration for SDO/HMI, following Castellanos Durán, Lagg & Solanki
(2021), A&A 652, L1 — "How rare are counter Evershed flows?", Sect. 2 ("Observational
data and analysis").

HMI's ``v_LOS`` is not an absolute velocity. Four effects set its zero level, and all
four must be removed before a number in m/s means anything physical::

    v_corrected = v_LOS - v_SDO - v_LSF - v_CLV - v_gravity

1. ``v_SDO``     the observatory's own line-of-sight velocity relative to the Sun.
                 Two independent methods, as in the paper:
                 `sdo_los_velocity_keywords` (Eq. 1, from the OBS_V* header keywords,
                 following Schuck et al. 2016) and `sdo_los_velocity_quiet_sun`
                 (the average quiet Sun within +/-15" of disk centre).
2. ``v_LSF``     large-scale flows: solar differential rotation and meridional
                 circulation (Eqs. 2-4).
3. ``v_CLV``     the centre-to-limb variation of the convective blueshift (Eq. 5).
4. ``v_gravity`` the gravitational redshift (Eq. 6), a constant.

Everything is expressed with `astropy.units` and all geometry comes from sunpy's own
coordinate machinery rather than hand-rolled trigonometry, so the frames and their
conventions are the documented ones rather than this module's opinion.

Sign convention, verified end to end against real data (see `docstring of
`calibrate_dopplergram`): all four terms are *subtracted* from the measured velocity.

Relationship to the empirical plane fit in ``02A_data_procesing.ipynb``: these
corrections and that plane fit remove overlapping things. Applying the physical
corrections gives an absolute velocity scale; fitting a plane to the residual afterwards
throws that scale away again. Do one or the other deliberately, not both by accident.
"""

from __future__ import annotations

from dataclasses import dataclass

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord

# ── Constants from the paper ─────────────────────────────────────────────────

#: Eq. 5 — centre-to-limb variation of the convective blueshift, in *increasing*
#: powers of mu, m/s. Reconstructed by the authors from Löhner-Böttcher &
#: Schlichenmaier (2013) and shifted to match the LARS disk-centre measurement of the
#: Fe I 6173.3 A line, which at HMI's resolving power (~81 000) is -275 m/s.
#: Sanity check: evaluating at mu = 1 gives -278 m/s.
CLV_COEFFICIENTS = (131.0, -1179.0, -2029.0, 9112.0, -10409.0, 4096.0)

#: Eq. 3 — differential rotation, v_rot(Theta) = (a + b sin^2 Theta + c sin^4 Theta) cos Theta.
#: Hathaway & Rightmire (2011). NOTE these are *residuals relative to the Carrington
#: frame* (35.6 m/s at the equator, not the ~2 km/s of the bulk rotation) — see
#: `large_scale_flow_los` for why that matters.
A_ROTATION = 35.6   # m/s
B_ROTATION = -208.6  # m/s
C_ROTATION = -420.6  # m/s

#: Eq. 4 — meridional flow, v_mer(Theta) = (d sin Theta + e sin^3 Theta) cos Theta.
D_MERIDIONAL = 29.7   # m/s
E_MERIDIONAL = -17.7  # m/s

#: Carrington rotation, 14.184 deg/day, expressed as a linear velocity at the equator.
V_CARRINGTON = 1994.2 * u.m / u.s

#: Mean orbital angular velocity of the observer (0.986 deg/day), which converts the
#: sidereal rotation to the synodic rotation seen from the Sun-Earth line. At the
#: equator this is 138.63 m/s.
V_SYNODIC = 138.63 * u.m / u.s

#: Eq. 6 — gravitational redshift, dlambda_G = lambda G M_sun / (R_sun c^2).
GRAVITATIONAL_REDSHIFT = 636.03 * u.m / u.s

#: Solar radius used to turn the Heliocentric z coordinate into mu.
_R_SUN = 696_000.0 * u.km


# ── Geometry ─────────────────────────────────────────────────────────────────

@dataclass
class FrameGeometry:
    """Per-pixel geometry for one frame, in the coordinate systems the paper uses.

    Attributes
    ----------
    theta_rho : Quantity
        Angular distance from disk centre, the ``theta_rho`` of Eq. 1. This is
        helioprojective-radial (Thompson 2006), and equals ``hypot(Tx, Ty)``.
    psi : Quantity
        Position angle, counter-clockwise from solar north — the ``psi`` of Eq. 1.
    mu : ndarray
        ``cos(heliocentric angle)``, for Eq. 5. NaN off-disk.
    lat, lon : Quantity
        Stonyhurst heliographic latitude ``Theta`` and longitude ``Phi``, for Eqs. 2-4.
        NaN off-disk.
    b0 : Quantity
        Solar B-angle (heliographic latitude of the observer) for this frame.
    """

    theta_rho: u.Quantity
    psi: u.Quantity
    mu: np.ndarray
    lat: u.Quantity
    lon: u.Quantity
    b0: u.Quantity


def frame_geometry(smap) -> FrameGeometry:
    """Compute the per-pixel geometry of a sunpy Map.

    Costs roughly 0.15 s for a 500x150 px cutout, dominated by the coordinate
    transforms, so compute it once per frame and pass it to the individual correction
    functions rather than letting each recompute it.

    Pixels off the solar disk transform to NaN in the heliographic and heliocentric
    frames, so ``mu``, ``lat`` and ``lon`` are NaN there and every correction derived
    from them is NaN too. That is the honest answer for a pixel with no surface.
    """
    import sunpy.map
    from sunpy.coordinates import frames

    coords = sunpy.map.all_coordinates_from_map(smap)
    observer, obstime = smap.observer_coordinate, smap.date

    # Helioprojective-radial (Thompson 2006) is exactly the (theta_rho, psi) system
    # Eq. 1 is written in. sunpy stores the latitude-like component as `delta`, which
    # is measured from the anti-Sun direction, hence the +90 deg.
    hpr = coords.transform_to(frames.HelioprojectiveRadial(observer=observer, obstime=obstime))
    theta_rho = hpr.delta + 90 * u.deg
    psi = hpr.psi

    hgs = coords.transform_to(frames.HeliographicStonyhurst)

    # mu from the Heliocentric z coordinate. The often-quoted identity
    #     mu = sin(B0) sin(Theta) + cos(B0) cos(Theta) cos(Phi)
    # agrees with this to ~1e-4 (it assumes an observer at infinity); the difference is
    # worth ~0.01 m/s through Eq. 5, but there is no reason to take the approximation.
    hcc = coords.transform_to(frames.Heliocentric(observer=observer, obstime=obstime))
    mu = (hcc.z / _R_SUN).decompose().value

    return FrameGeometry(
        theta_rho=theta_rho,
        psi=psi,
        mu=mu,
        lat=hgs.lat,
        lon=hgs.lon,
        b0=observer.lat,
    )


# ── Effect 1: observatory line-of-sight velocity ─────────────────────────────

def sdo_los_velocity_keywords(smap, geometry: FrameGeometry | None = None) -> u.Quantity:
    """Eq. 1 — the observatory's LOS velocity, from the HMI header keywords.

    ``v_SDO = OBS_VW sin(theta_rho) sin(psi)
              - OBS_VN sin(theta_rho) cos(psi)
              + OBS_VR cos(theta_rho)``

    where OBS_VR is the observatory's radial velocity away from the Sun, OBS_VW its
    velocity westward along Earth's orbit, and OBS_VN its velocity northward along the
    solar rotation axis (Schuck et al. 2016; Thompson 2006 for the coordinate
    conversion).

    All three keywords survive JSOC's ``im_patch`` cutout, so this works directly on the
    AR cutouts. The result is dominated by OBS_VR (a few km/s, varying diurnally with
    SDO's geosynchronous orbit) but is *not* constant across a frame: the OBS_VW term
    contributes a gradient of tens of m/s over a few hundred arcsec.
    """
    g = geometry if geometry is not None else frame_geometry(smap)
    meta = smap.meta

    try:
        obs_vw = meta['obs_vw'] * u.m / u.s
        obs_vn = meta['obs_vn'] * u.m / u.s
        obs_vr = meta['obs_vr'] * u.m / u.s
    except KeyError as exc:
        raise KeyError(
            f'{exc} missing from the map header — Eq. 1 needs OBS_VR, OBS_VW and OBS_VN. '
            f'Use sdo_los_velocity_quiet_sun instead, or check that this really is an '
            f'HMI file.') from exc

    sin_theta, cos_theta = np.sin(g.theta_rho), np.cos(g.theta_rho)
    return (obs_vw * sin_theta * np.sin(g.psi)
            - obs_vn * sin_theta * np.cos(g.psi)
            + obs_vr * cos_theta).to(u.m / u.s)


def sdo_los_velocity_quiet_sun(v_map, b_map, radius: u.Quantity = 15 * u.arcsec,
                               b_threshold: u.Quantity = 500 * u.G) -> tuple[u.Quantity, int]:
    """The alternative method — average quiet-Sun velocity near disk centre.

    The paper's second way of getting the observatory velocity: take the mean HMI
    velocity in a region of ``+/-radius`` around disk centre, masking strong magnetic
    concentrations (``|B_LOS| > b_threshold``) so that magnetised plasma flows don't
    bias the average. Unlike Eq. 1 this needs no header keywords, which makes it a
    genuinely independent cross-check (the paper's Fig. 1a compares the two).

    ``v_map`` and ``b_map`` must be a Dopplergram/magnetogram pair **covering disk
    centre and sampled on the same grid**. The AR cutouts do not contain disk centre —
    use `download_disk_center` to fetch the data this needs.

    Returns
    -------
    (velocity, n_pixels)
        The scalar mean velocity and how many pixels went into it. Check the count:
        a small number means the mask ate the region and the value is noise.
    """
    import sunpy.map

    if v_map.data.shape != b_map.data.shape:
        raise ValueError(f'Dopplergram {v_map.data.shape} and magnetogram '
                         f'{b_map.data.shape} are not on the same grid')

    coords = sunpy.map.all_coordinates_from_map(v_map)
    within = np.hypot(coords.Tx, coords.Ty) <= radius
    quiet = np.abs(b_map.data) <= b_threshold.to_value(u.G)
    usable = within & quiet & np.isfinite(v_map.data) & np.isfinite(b_map.data)

    if not usable.any():
        raise ValueError(
            f'No usable pixels within {radius} of disk centre with |B| <= {b_threshold}. '
            f'Does this map actually cover disk centre? '
            f'Tx range {np.nanmin(coords.Tx):.0f}..{np.nanmax(coords.Tx):.0f}')

    return np.nanmean(v_map.data[usable]) * u.m / u.s, int(usable.sum())


# ── Effect 2: large-scale flows ──────────────────────────────────────────────

def large_scale_flow_los(smap, geometry: FrameGeometry | None = None,
                         include_meridional: bool = True) -> u.Quantity:
    """Eqs. 2-4 — the LOS projection of solar differential rotation and meridional flow.

    ``v_LSF|_LOS = v_mer(Theta) [sin B0 cos Theta - cos B0 cos Phi sin Theta]
                   - [v_rot(Theta) - v_Carrington + v_synodic] cos B0 sin Phi``

    with ``v_rot`` from Eq. 3 and ``v_mer`` from Eq. 4.

    The rotation bracket looks wrong at first glance and is not: Eq. 3's ``v_rot`` is a
    *residual relative to the Carrington frame* (35.6 m/s at the equator), so the
    bracket evaluates to about -1820 m/s there, and the leading minus sign turns that
    into the correct +1820 m/s redshift at the equatorial west limb. Do not "fix" the
    signs to make ``v_rot`` look like a bulk rotation velocity.

    The paper notes the meridional contribution is tiny; ``include_meridional`` exists so
    that can be confirmed on real data rather than taken on faith.

    This deliberately does not use ``sunpy.physics.differential_rotation``, which offers
    the howard/snodgrass/allen/rigid profiles — none of them the Hathaway & Rightmire
    (2011) coefficients this paper uses.
    """
    g = geometry if geometry is not None else frame_geometry(smap)

    theta, phi, b0 = g.lat, g.lon, g.b0
    sin_theta, cos_theta = np.sin(theta), np.cos(theta)
    sin_b0, cos_b0 = np.sin(b0), np.cos(b0)

    v_rot = (A_ROTATION + B_ROTATION * sin_theta**2
             + C_ROTATION * sin_theta**4) * cos_theta * u.m / u.s
    rotation = -(v_rot - V_CARRINGTON + V_SYNODIC) * cos_b0 * np.sin(phi)

    if include_meridional:
        v_mer = (D_MERIDIONAL * sin_theta + E_MERIDIONAL * sin_theta**3) * cos_theta * u.m / u.s
        meridional = v_mer * (sin_b0 * cos_theta - cos_b0 * np.cos(phi) * sin_theta)
    else:
        meridional = 0 * u.m / u.s

    return (meridional + rotation).to(u.m / u.s)


# ── Effect 3: convective blueshift ───────────────────────────────────────────

def convective_blueshift(mu) -> u.Quantity:
    """Eq. 5 — centre-to-limb variation of the convective blueshift.

    ``v_CLV(mu) = 131 - 1179 mu - 2029 mu^2 + 9112 mu^3 - 10409 mu^4 + 4096 mu^5``  [m/s]

    Evaluates to -278 m/s at disk centre, matching the paper's quoted -275 m/s for the
    Fe I 6173.3 A line at HMI's resolving power. The curve is not monotonic: it deepens
    to about -370 m/s around mu ~ 0.7 before turning over towards the limb.
    """
    mu_value = np.asarray(u.Quantity(mu, u.dimensionless_unscaled).value
                          if isinstance(mu, u.Quantity) else mu, dtype=float)
    return np.polynomial.polynomial.polyval(mu_value, CLV_COEFFICIENTS) * u.m / u.s


# ── Combined ─────────────────────────────────────────────────────────────────

#: The correction terms `calibrate_dopplergram` knows about, in the order applied.
ALL_TERMS = ('sdo', 'lsf', 'clv', 'gravity')


def calibrate_dopplergram(smap, sdo_method: str = 'keywords',
                          v_sdo: u.Quantity | None = None,
                          include: tuple[str, ...] = ALL_TERMS,
                          geometry: FrameGeometry | None = None,
                          include_meridional: bool = True):
    """Remove the four zero-point effects from one HMI Dopplergram.

    ``v_corrected = v_LOS - v_SDO - v_LSF - v_CLV - v_gravity``

    That sign convention is not a guess. Summing the four terms over the quiet pixels
    (|B| < 100 G) of a real ``hmi.v_720s`` frame of NOAA 11536 predicts +1580 m/s
    against a measured +1367 m/s, i.e. the four terms account for the measured velocity
    to within ~213 m/s while the individual terms range over 800-2100 m/s. The residual
    is the known HMI absolute-Doppler bias plus real flows inside the box.

    Parameters
    ----------
    smap : sunpy.map.Map
        An HMI Dopplergram (BUNIT m/s).
    sdo_method : {'keywords', 'quiet_sun'}
        How to obtain the observatory velocity. ``'quiet_sun'`` requires ``v_sdo`` to be
        supplied, since it is measured from separate disk-centre data — see
        `sdo_los_velocity_quiet_sun` and `download_disk_center`.
    v_sdo : Quantity, optional
        A precomputed observatory velocity, overriding ``sdo_method``.
    include : tuple of str
        Which of ``ALL_TERMS`` to apply. Useful for isolating one effect.
    geometry : FrameGeometry, optional
        Reuse a geometry computed earlier for this same frame.
    include_meridional : bool
        Passed to `large_scale_flow_los`.

    Returns
    -------
    (corrected, terms)
        ``corrected`` is a Quantity array in m/s. ``terms`` maps each applied term to
        its Quantity, so the size of each contribution stays inspectable instead of
        collapsing into one opaque number.
    """
    unknown = set(include) - set(ALL_TERMS)
    if unknown:
        raise ValueError(f'Unknown correction term(s) {sorted(unknown)}; known: {ALL_TERMS}')

    g = geometry if geometry is not None else frame_geometry(smap)
    terms: dict[str, u.Quantity] = {}

    if 'sdo' in include:
        if v_sdo is not None:
            terms['sdo'] = u.Quantity(v_sdo, u.m / u.s)
        elif sdo_method == 'keywords':
            terms['sdo'] = sdo_los_velocity_keywords(smap, geometry=g)
        elif sdo_method == 'quiet_sun':
            raise ValueError(
                "sdo_method='quiet_sun' needs v_sdo, because the quiet-Sun method is "
                "measured from disk-centre data that an AR cutout does not contain. "
                "Compute it with sdo_los_velocity_quiet_sun and pass it as v_sdo.")
        else:
            raise ValueError(f"sdo_method must be 'keywords' or 'quiet_sun', got {sdo_method!r}")

    if 'lsf' in include:
        terms['lsf'] = large_scale_flow_los(smap, geometry=g, include_meridional=include_meridional)
    if 'clv' in include:
        terms['clv'] = convective_blueshift(g.mu)
    if 'gravity' in include:
        terms['gravity'] = GRAVITATIONAL_REDSHIFT

    corrected = u.Quantity(smap.data, u.m / u.s, copy=True)
    for term in terms.values():
        corrected = corrected - term
    return corrected, terms


# ── Disk-centre reference data for the quiet-Sun method ──────────────────────

#: Series needed by the quiet-Sun method: a Dopplergram and a magnetogram to mask on.
DISK_CENTER_SERIES = ('hmi.V_720s', 'hmi.M_720s')


def download_disk_center(base_dir, time_start, time_end, notify_email,
                         radius: u.Quantity = 15 * u.arcsec, sample=None,
                         series=DISK_CENTER_SERIES):
    """Fetch the disk-centre cutouts the quiet-Sun method needs.

    An AR cutout never contains disk centre, so `sdo_los_velocity_quiet_sun` has no data
    to work with unless it is fetched separately. This grabs a ``+/-radius`` box at
    helioprojective (0, 0) — about 60x60 px at HMI's plate scale, negligible next to the
    AR cutouts themselves.

    ``tracking=False`` is essential: the box must stay pinned at disk centre rather than
    rotating away with the surface.

    Files land in ``base_dir/region_01/``, reusing `download_regions` and therefore also
    its resume behaviour and partial-failure reporting.
    """
    import pathlib

    from sunpy.coordinates import frames

    from src.download import download_regions

    base_dir = pathlib.Path(base_dir)
    # a.jsoc.Cutout ignores the observer and assumes SDO, but the frame still needs an
    # obstime because sunpy passes it through as JSOC's t_ref.
    frame = frames.Helioprojective(obstime=time_start, observer='earth')
    bottom_left = SkyCoord(-radius, -radius, frame=frame)
    top_right = SkyCoord(radius, radius, frame=frame)

    summaries = {}
    for one_series in series:
        summaries[one_series] = download_regions(
            [(bottom_left, top_right)], base_dir, time_start, time_end, notify_email,
            one_series, sample=sample, tracking=False)
    return summaries


def disk_center_velocity_series(disk_center_dir, radius: u.Quantity = 15 * u.arcsec,
                                b_threshold: u.Quantity = 500 * u.G,
                                region: str = 'region_01'):
    """Run the quiet-Sun method over every frame of a disk-centre download.

    Pairs the Dopplergram and magnetogram frames **by timestamp** — they are downloaded
    independently and individual files do fail, so pairing by sort order would silently
    mismatch them after the first gap.

    Returns
    -------
    dict mapping timestamp -> (velocity, n_pixels)
        Feed the velocities to `calibrate_cube(sdo_method='quiet_sun')`, or plot them
        against `sdo_los_velocity_keywords` to reproduce the paper's Fig. 1a.
    """
    import pathlib

    import sunpy.map

    from src.utilities import parse_frame_timestamp

    frame_dir = pathlib.Path(disk_center_dir) / region
    v_frames = {parse_frame_timestamp(p): p for p in frame_dir.glob('hmi.v_720s.*.fits')}
    b_frames = {parse_frame_timestamp(p): p for p in frame_dir.glob('hmi.m_720s.*.magnetogram.fits')}
    v_frames.pop(None, None)
    b_frames.pop(None, None)

    common = sorted(set(v_frames) & set(b_frames))
    if not common:
        raise ValueError(
            f'No matching Dopplergram/magnetogram pair in {frame_dir} '
            f'({len(v_frames)} v frames, {len(b_frames)} m frames). '
            f'Run download_disk_center first.')

    unpaired = (len(v_frames) - len(common)) + (len(b_frames) - len(common))
    if unpaired:
        print(f'disk_center_velocity_series: {len(common)} paired frames, '
              f'{unpaired} unpaired frame(s) ignored')

    out = {}
    for timestamp in common:
        v_map = sunpy.map.Map(str(v_frames[timestamp]))
        b_map = sunpy.map.Map(str(b_frames[timestamp]))
        out[timestamp] = sdo_los_velocity_quiet_sun(
            v_map, b_map, radius=radius, b_threshold=b_threshold)
    return out


# ── Cube-level application ───────────────────────────────────────────────────

def calibrate_cube(region_dir, sdo_method: str = 'keywords', v_sdo_by_time=None,
                   include: tuple[str, ...] = ALL_TERMS, region: str = 'region_01',
                   include_meridional: bool = True, series_glob: str = 'hmi.v_*.fits'):
    """Apply the calibration to every frame of a Dopplergram cube.

    The corrections are per-frame, not per-cube: a tracked box sweeps across the sky, and
    B0 and the OBS_V* keywords change with every frame. `make_cube` keeps only the
    reference frame's header, so the geometry is read from the surviving per-file frames
    in ``region_dir/region``, matched to the cube **by timestamp** via the cube's
    TIMESTAMPS extension. That reuse of the timestamp join is deliberate: a frame missing
    from the middle of the window cannot silently shift the corrections out of step.

    Parameters
    ----------
    region_dir : path
        An AR directory containing ``<region>_dopplergram_cube.fits`` and ``<region>/``.
    sdo_method : {'keywords', 'quiet_sun'}
        'quiet_sun' requires ``v_sdo_by_time`` from `disk_center_velocity_series`.
    v_sdo_by_time : dict, optional
        timestamp -> velocity (or (velocity, n_pixels), as returned by
        `disk_center_velocity_series`).
    series_glob : str
        Frame filename pattern. The default deliberately matches every cadence
        (``hmi.v_720s.*.fits`` and ``hmi.v_45s.*.Dopplergram.fits``): NOAA 11117 has to be
        downloaded from the 45 s series because JSOC's 720 s series 500-errors across its
        window, and a 720 s-only pattern silently finds no frames and reports every frame
        of the cube as missing.

    Returns
    -------
    (cube, timestamps, term_means)
        ``cube`` is the corrected velocity in m/s as float32, same shape as the input.
        ``term_means`` maps each term to an (n_t,) array of its spatial mean, which is
        what you plot to see how each effect evolves over the window.
    """
    import pathlib

    from src.utilities import apply_per_frame_correction

    region_dir = pathlib.Path(region_dir)

    def correct(smap, timestamp):
        v_sdo = None
        if v_sdo_by_time is not None:
            entry = v_sdo_by_time[timestamp]
            v_sdo = entry[0] if isinstance(entry, tuple) else entry

        corrected, terms = calibrate_dopplergram(
            smap, sdo_method=sdo_method, v_sdo=v_sdo, include=include,
            include_meridional=include_meridional)

        return (corrected.to_value(u.m / u.s),
                {name: np.nanmean(term.to_value(u.m / u.s)) for name, term in terms.items()})

    return apply_per_frame_correction(
        region_dir / f'{region}_dopplergram_cube.fits',
        region_dir / region,
        correct,
        series_glob=series_glob,
        diag_names=include,
    )

"""Stage 2 — correction: turn raw cubes into corrected ones on a uniform clock.

`02A_data_processing.ipynb` is the driver; these four functions used to be defined in its
cells and existed nowhere else, which meant nothing else could reuse them and nothing
could test them.

    from src import config, processing
    result = processing.process_region(region_dir, **config.PROCESSING)
    out_dir, written = processing.write_region(region_dir, result)

**This stage builds no regions and computes no metrics.** Umbra, penumbra, hot spot and
the quiet-sun mask are all built in `03A` by `segmentation.build_regions`, from the cubes
written here. The reason is cost: the corrections re-read every per-frame FITS header,
~90 s per region, and none of that work depends on where an umbra boundary is drawn.
Keeping the regions out means retuning a threshold costs one 03A run rather than a full
re-correction.

The one place a mask is still needed is the magnetogram plane fit, which has to know which
pixels are quiet sun. That uses its own pinned `plane_qsun_*` cut rather than 03A's
thresholds, so the cube on disk does not silently change meaning when a threshold is
retuned downstream.
"""

from __future__ import annotations

import pathlib

import numpy as np
from astropy.io import fits

from .doppler_calibration import calibrate_cube
from .limb_darkening import U_LAMBDA, V_LAMBDA, limb_darkening_cube
from .utilities import (crop_to_common_window, read_cube, regular_time_grid,
                        reindex_on_grid, reindex_series_on_grid, write_cube)

_SERIES = [('cont', 'continuum'), ('mag', 'magnetogram'), ('dop', 'dopplergram')]


def load_aligned(region_dir, calibrate=True, limb_darken=True, mu_method='geometry',
                 crop_to_data=False, v_sdo_by_time=None, verbose=True):
    """Load the three cubes, correct them, and place them on one uniform time grid.

    The three series are downloaded independently and individual files do fail, so their
    frame counts differ. Two joins were considered and only one of them is safe:

    - *Intersection* of the timestamps (what 02A originally did). It keeps only frames all
      three series have, which throws away good frames from the others and, worse, leaves
      holes in the time axis. NOAA 11536's dopplergram is missing 2012-08-01 09:48 and
      2012-08-02 21:48, so the intersection has two 1440 s jumps in an otherwise 720 s
      series — and every FFT downstream assumes a single cadence, so all of them come out
      quietly wrong.
    - A *uniform grid* covering the union of the timestamps, which is what happens here.
      Every series gets one frame per slot; where a series has no frame the slot is NaN.
      The time axis is uniform by construction rather than inherited from whatever
      happened to download, and a missing mid-window frame stays visible as a gap instead
      of shifting every later frame out of step.

    Both corrections are applied *before* the join and per frame, because both depend on
    where the tracked box is pointing — see `utilities.apply_per_frame_correction`.

    Parameters
    ----------
    crop_to_data : bool
        Trim all three cubes to the window in which every frame has data. Off by default
        because it is a repair for a bad download, not part of the pipeline: on a healthy
        region it finds nothing to trim and only costs a pass over the cubes.

    Returns
    -------
    dict with keys
        cubes         {'cont', 'mag', 'dop'}, each (n_slots, ny, nx) float32
        grid          list of datetimes, one per slot, evenly spaced
        cadence_s     the grid spacing
        present       {'cont', 'mag', 'dop'} bool arrays — True where the series has data
        doppler_terms per-term spatial means on the grid, or None
        c_means       per-frame spatial mean of the limb-darkening factor C, or None
        crop_offset   (row0, col0) of the crop in the continuum's original pixels, or None
    """
    region_dir = pathlib.Path(region_dir)
    cubes, times = {}, {}
    for name, fname in _SERIES:
        data, ts = read_cube(region_dir / f'region_01_{fname}_cube.fits')
        cubes[name], times[name] = data.astype(np.float32), ts

    c_means = None
    if limb_darken:
        corrected, cont_times, c_means = limb_darkening_cube(
            region_dir, method=mu_method, u_lambda=U_LAMBDA, v_lambda=V_LAMBDA)
        if cont_times != times['cont']:
            raise ValueError('limb_darkening_cube returned different timestamps than the cube')
        cubes['cont'] = corrected

    term_means = None
    if calibrate:
        corrected, dop_times, term_means = calibrate_cube(region_dir, v_sdo_by_time=v_sdo_by_time)
        if dop_times != times['dop']:
            raise ValueError('calibrate_cube returned different timestamps than the cube')
        cubes['dop'] = corrected

    grid, cadence_s = regular_time_grid([times['cont'], times['mag'], times['dop']])

    present = {}
    for name in cubes:
        cubes[name], present[name] = reindex_on_grid(cubes[name], times[name], grid, cadence_s)

    # The per-frame diagnostics have to move onto the same grid as the cubes they describe,
    # or they end up plotted against the wrong times.
    if term_means is not None:
        term_means = {k: reindex_series_on_grid(v, times['dop'], grid, cadence_s)
                      for k, v in term_means.items()}
    if c_means is not None:
        c_means = reindex_series_on_grid(c_means, times['cont'], grid, cadence_s)

    if verbose:
        _report_gaps(grid, cadence_s, present)

    crop_offset = None
    if crop_to_data:
        cubes, offsets = crop_to_common_window(cubes, present=present)
        crop_offset = offsets['cont']
    else:
        # Everything downstream indexes the magnetogram and dopplergram with masks built
        # from the continuum, so a shape mismatch has to stop here with an explanation
        # rather than as a broadcasting error 60 lines later.
        shapes = {name: cube.shape[1:] for name, cube in cubes.items()}
        if len(set(shapes.values())) > 1:
            raise ValueError(
                f'{region_dir.name}: the three series are on different pixel grids '
                f'({shapes}) — they were downloaded under different boxes. Either '
                f're-download the region with one consistent box, or give this region '
                f"crop_to_data=True in config.REGION_PARAMS to trim all three to their "
                f'common data window.')

    return dict(cubes=cubes, grid=grid, cadence_s=cadence_s, present=present,
                doppler_terms=term_means, c_means=c_means, crop_offset=crop_offset)


def _report_gaps(grid, cadence_s, present):
    """Print where each series has no data. A gap is a fact about the download, not a bug."""
    gaps = {k: np.flatnonzero(~v) for k, v in present.items()}
    n_gaps = {k: len(v) for k, v in gaps.items()}
    print(f'  {len(grid)} slots on a uniform {cadence_s:.0f} s grid, '
          f'{grid[0]:%Y-%m-%d %H:%M} .. {grid[-1]:%Y-%m-%d %H:%M}')
    if not any(n_gaps.values()):
        print('  no gaps — all three series cover every slot')
        return
    print(f'  NaN frames (no data): {n_gaps}')
    for name, idx in gaps.items():
        for i in idx[:5]:
            print(f'      {name}: slot {i} = {grid[i]:%Y-%m-%d %H:%M}')
        if len(idx) > 5:
            print(f'      {name}: ... and {len(idx) - 5} more')


def remove_quiet_sun_plane(frame, qsun_mask, xn, yn):
    """Fit a plane to the quiet-sun pixels and subtract it from the whole frame.

    Subtracting a scalar quiet-sun mean only removes the spatially uniform term — for the
    dopplergram that is mostly the SDO orbital velocity. What survives is the
    line-of-sight solar-rotation gradient across the box, and because the umbra sits off
    to one side of the box its mean picks up a residual that drifts as the region rotates.
    Removing a plane instead takes out that gradient to first order.

    This is the *empirical* alternative to the physical corrections in
    `doppler_calibration`. It always applies to the magnetogram (which those corrections
    do not address) but only to the dopplergram when `residual_plane_fit` is set.

    A NaN (gap) frame has no finite quiet-sun pixels, so it returns unchanged with NaN
    coefficients rather than raising.

    Returns (corrected_frame, coefficients) with coefficients = (offset, d/dx, d/dy) in
    the normalized coordinates xn, yn.
    """
    valid = qsun_mask & np.isfinite(frame)
    if valid.sum() < 3:
        return frame, np.array([np.nan, np.nan, np.nan])
    design = np.column_stack([np.ones(valid.sum()), xn[valid], yn[valid]])
    coef, *_ = np.linalg.lstsq(design, frame[valid].astype(np.float64), rcond=None)
    plane = coef[0] + coef[1] * xn + coef[2] * yn
    return (frame - plane).astype(np.float32), coef


def process_region(region_dir, crop_to_data=False, v_sdo_by_time=None,
                   calibrate_doppler=True, residual_plane_fit=False,
                   correct_limb_darkening=True, mu_method='geometry',
                   plane_qsun_frac=0.90, plane_qsun_percentile=80, verbose=True):
    """Correct one region's three cubes. Builds no regions and computes no metrics.

    Every step is NaN-safe, because a gap slot on the uniform grid is an all-NaN frame:
    the I_qs percentile and the plane fit both guard on there being finite pixels.
    `present` is carried through so 03A can tell a gap apart from a frame where nothing
    was detected.

    The settings used are returned under `'settings'`, so `write_region` records in each
    file's header what was actually done rather than what a notebook global happened to
    say at write time.
    """
    settings = dict(calibrate_doppler=calibrate_doppler, residual_plane_fit=residual_plane_fit,
                    correct_limb_darkening=correct_limb_darkening, mu_method=mu_method,
                    plane_qsun_frac=plane_qsun_frac,
                    plane_qsun_percentile=plane_qsun_percentile)

    aligned = load_aligned(region_dir, calibrate=calibrate_doppler,
                           limb_darken=correct_limb_darkening, mu_method=mu_method,
                           crop_to_data=crop_to_data, v_sdo_by_time=v_sdo_by_time,
                           verbose=verbose)
    cubes = aligned['cubes']
    timestamps = aligned['grid']
    cube_cont, cube_mag, cube_dop = cubes['cont'], cubes['mag'], cubes['dop']
    n_t, ny, nx = cube_cont.shape

    # Normalized pixel coordinates for the plane fit, so the design matrix stays well
    # conditioned regardless of box size.
    yy, xx = np.mgrid[0:ny, 0:nx]
    xn = ((xx - nx / 2) / (nx / 2)).astype(np.float64)
    yn = ((yy - ny / 2) / (ny / 2)).astype(np.float64)

    # Quiet sun for the plane fit only: everything not dark. A NaN threshold compares
    # False everywhere, so a gap frame contributes no pixels and its plane comes back NaN.
    finite = np.isfinite(cube_cont)
    i_qs = np.array([np.percentile(cube_cont[t][finite[t]], plane_qsun_percentile)
                     if finite[t].any() else np.nan for t in range(n_t)])
    with np.errstate(invalid='ignore'):
        plane_qsun = finite & ~(cube_cont < (plane_qsun_frac * i_qs)[:, None, None])

    plane_coefs = {'mag': np.full((n_t, 3), np.nan), 'dop': np.full((n_t, 3), np.nan)}
    cube_mag = cube_mag.copy()          # crop_to_common_window returns views
    cube_dop = cube_dop.copy() if residual_plane_fit else cube_dop
    for t in range(n_t):
        cube_mag[t], plane_coefs['mag'][t] = remove_quiet_sun_plane(
            cube_mag[t], plane_qsun[t], xn, yn)
        if residual_plane_fit:
            cube_dop[t], plane_coefs['dop'][t] = remove_quiet_sun_plane(
                cube_dop[t], plane_qsun[t], xn, yn)

    return dict(
        cubes={'cont': cube_cont, 'mag': cube_mag, 'dop': cube_dop},
        timestamps=timestamps, i_qs=i_qs, plane_coefs=plane_coefs,
        doppler_terms=aligned['doppler_terms'], c_means=aligned['c_means'],
        present=aligned['present'], cadence_s=aligned['cadence_s'],
        crop_offset=aligned['crop_offset'],
        plane_qsun_px=plane_qsun.reshape(n_t, -1).sum(axis=1),
        settings=settings,
    )


def write_region(region_dir, result, processed_dir=None):
    """Write one region's three corrected cubes plus a per-frame diagnostics table.

    Four files, no masks. The regions are 03A's job and are rebuilt from these cubes every
    time it runs, so there is deliberately no mask file here to go stale against whatever
    thresholds 03A currently uses. 03A exports its own mask cube for DS9 when you want one.

    Everything goes out through `utilities.write_cube` so it comes back through `read_cube`
    unchanged, and so DS9 sees the same structure as the raw cubes: a 3D primary HDU
    carrying the reference frame's spatial WCS, plus a TIMESTAMPS extension.
    """
    from . import config

    region_dir = pathlib.Path(region_dir)
    processed_dir = pathlib.Path(processed_dir) if processed_dir else config.PROCESSED_DIR
    out_dir = processed_dir / region_dir.name

    settings = result['settings']
    timestamps = result['timestamps']
    present = result['present']
    cadence_s = result['cadence_s']
    n_gaps = int((~(present['cont'] & present['mag'] & present['dop'])).sum())

    # Reuse the raw continuum cube's header so the corrected cubes keep the spatial WCS,
    # and DS9 puts them on the same footprint as the frames they came from.
    src_header = fits.getheader(region_dir / 'region_01_continuum_cube.fits')
    for key in ('NAXIS', 'NAXIS1', 'NAXIS2', 'NAXIS3', 'NFRAMES', 'BITPIX'):
        src_header.pop(key, None)
    src_header['CADENCE'] = (cadence_s, '[s] uniform grid spacing along axis 3')
    src_header['NGAPS'] = (n_gaps, 'frames with no data in at least one series')

    # A crop moves the origin, so the inherited CRPIX would put the cube in the wrong place
    # on the sky. FITS pixel coordinates are 1-based and the crop offset is 0-based, but
    # both refer to the same axis, so the shift is just a subtraction.
    crop_offset = result.get('crop_offset')
    if crop_offset is not None:
        row0, col0 = crop_offset
        src_header['CRPIX1'] = src_header['CRPIX1'] - col0
        src_header['CRPIX2'] = src_header['CRPIX2'] - row0
        src_header['CROPROW'] = (row0, 'first row of the crop in the original cube')
        src_header['CROPCOL'] = (col0, 'first column of the crop in the original cube')

    provenance = [
        f'02A: {len(timestamps)} frames on a uniform {cadence_s:.0f} s grid, '
        f'{n_gaps} NaN (missing) frame(s)',
        '02A: corrections only - umbra/penumbra/hot spot are built in 03A from these cubes',
    ]
    if crop_offset is not None:
        ny, nx = result['cubes']['cont'].shape[1:]
        provenance.append(f'02A: cropped to the common data window {ny}x{nx} at '
                          f'row {crop_offset[0]}, col {crop_offset[1]}; CRPIX shifted to match')

    doppler_history = (
        ['02A: dopplergram calibrated - Castellanos Duran 2021 sdo+lsf+clv+gravity']
        if settings['calibrate_doppler'] else ['02A: dopplergram NOT calibrated'])
    if settings['residual_plane_fit']:
        doppler_history.append('02A: residual quiet-sun plane also subtracted (absolute scale lost)')

    if settings['correct_limb_darkening']:
        continuum_history = [
            '02A: continuum limb-darkening corrected - Castellanos Duran & Kleint 2020 '
            f'Eq.1-2 (u={U_LAMBDA}, v={V_LAMBDA}, mu from {settings["mu_method"]})',
            '02A: intensity left in DN/s - Eq.3 DN->cgs factor NOT applied']
    else:
        continuum_history = ['02A: continuum NOT limb-darkening corrected']

    cont_header = src_header.copy()
    cont_header['LDCORR'] = (settings['correct_limb_darkening'],
                             'limb darkening divided out (Duran 2020 Eq.1)')
    if settings['correct_limb_darkening']:
        cont_header['LD_U'] = (U_LAMBDA, 'limb-darkening coefficient u at 6173.3 A')
        cont_header['LD_V'] = (V_LAMBDA, 'limb-darkening coefficient v at 6173.3 A')
        cont_header['LD_MU'] = (settings['mu_method'], 'how cos(Theta) was obtained')

    mag_header = src_header.copy()
    mag_header['PLQSFRAC'] = (settings['plane_qsun_frac'],
                              'I/I_qs above which a pixel is in the plane fit')
    mag_header['PLQSPCT'] = (settings['plane_qsun_percentile'],
                             'percentile used as I_qs for the fit')

    written = {}
    written['continuum'] = write_cube(
        result['cubes']['cont'], out_dir / 'region_01_continuum_cube.fits',
        header=cont_header, timestamps=timestamps,
        history=provenance + continuum_history)

    written['magnetogram'] = write_cube(
        result['cubes']['mag'], out_dir / 'region_01_magnetogram_corrected_cube.fits',
        header=mag_header, timestamps=timestamps,
        history=provenance + [
            f'02A: quiet-sun plane subtracted, fitted over '
            f'I >= {settings["plane_qsun_frac"]} I_qs '
            f'(I_qs = p{settings["plane_qsun_percentile"]})'])

    written['dopplergram'] = write_cube(
        result['cubes']['dop'], out_dir / 'region_01_dopplergram_calibrated_cube.fits',
        header=src_header, timestamps=timestamps, history=provenance + doppler_history)

    written['frames'] = write_frames_table(out_dir, result)
    return out_dir, written


def write_frames_table(out_dir, result):
    """Per-frame correction diagnostics, as a FITS binary table.

    Not analysis output — these say how big each correction was, which is the only way to
    tell a correction that went wrong from solar signal. The Doppler term means in
    particular used to be computed and thrown away at the end of 02A's driver cell.
    """
    out_dir = pathlib.Path(out_dir)
    timestamps = result['timestamps']
    present = result['present']
    n_t = len(timestamps)

    terms = result['doppler_terms'] or {}
    c_means = result['c_means']
    if c_means is None:
        c_means = np.full(n_t, np.nan)
    grad = np.hypot(result['plane_coefs']['mag'][:, 1], result['plane_coefs']['mag'][:, 2])

    cols = [
        fits.Column(name='T_OBS', format='23A',
                    array=np.array([t.isoformat() for t in timestamps])),
        fits.Column(name='PRESENT_CONT', format='L', array=present['cont']),
        fits.Column(name='PRESENT_MAG', format='L', array=present['mag']),
        fits.Column(name='PRESENT_DOP', format='L', array=present['dop']),
        fits.Column(name='I_QS', format='E', array=result['i_qs'], unit='DN/s'),
        fits.Column(name='C_MEAN', format='E', array=c_means),
        fits.Column(name='PLANE_QSUN_PX', format='J', array=result['plane_qsun_px']),
        fits.Column(name='MAG_PLANE_GRAD', format='E', array=grad, unit='G'),
    ]
    for name, key in [('V_SDO', 'sdo'), ('V_LSF', 'lsf'),
                      ('V_CLV', 'clv'), ('V_GRAVITY', 'gravity')]:
        cols.append(fits.Column(name=name, format='E', unit='m/s',
                                array=terms.get(key, np.full(n_t, np.nan))))

    frames_path = out_dir / 'region_01_frames.fits'
    frames_path.parent.mkdir(parents=True, exist_ok=True)
    fits.BinTableHDU.from_columns(cols, name='FRAMES').writeto(frames_path, overwrite=True)
    return str(frames_path)


def summarize(region_name, result):
    """Print how big each correction was for one region. Diagnostics, not analysis."""
    present = result['present']
    terms = result['doppler_terms'] or {}
    c_means = result['c_means']
    settings = result['settings']

    n_t = len(result['timestamps'])
    n_gaps = int((~(present['cont'] & present['mag'] & present['dop'])).sum())
    print(f'{region_name}')
    print(f'  frames                : {n_t} on a {result["cadence_s"]:.0f} s grid, {n_gaps} gap(s)')
    print(f'  I_qs        (DN/s)    : {np.nanmin(result["i_qs"]):,.0f} .. '
          f'{np.nanmax(result["i_qs"]):,.0f}')
    if settings['correct_limb_darkening'] and c_means is not None:
        print(f'  <C>  limb darkening   : {np.nanmin(c_means):.4f} .. {np.nanmax(c_means):.4f}'
              f'   (flat means the per-frame headers are not being read)')
    if settings['calibrate_doppler'] and terms:
        for key, label in [('sdo', 'v_SDO'), ('lsf', 'v_LSF'),
                           ('clv', 'v_CLV'), ('gravity', 'v_gravity')]:
            if key in terms:
                v = terms[key]
                print(f'  {label:20s}  : {np.nanmin(v):9.1f} .. {np.nanmax(v):9.1f} m/s')
    grad = np.hypot(result['plane_coefs']['mag'][:, 1], result['plane_coefs']['mag'][:, 2])
    print(f'  magnetogram plane     : |grad| {np.nanmin(grad):.1f} .. {np.nanmax(grad):.1f} G '
          f'over the box, fitted on {int(np.nanmedian(result["plane_qsun_px"])):,} px (median)')

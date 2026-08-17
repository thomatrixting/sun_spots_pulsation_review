"""Shared plumbing: reading and writing cubes, and putting frames on a common clock.

Used by every stage. Anything that talks to the network lives in `download`; anything
that draws lives in `plotting`.

The time-grid functions are the heart of it. The three HMI series for one region do not
arrive on the same timestamps — frames go missing, and NOAA 11117 runs its continuum and
magnetogram 360 s out of phase for a whole day. Intersecting the timestamps was the
original approach and it was wrong: dropping a frame leaves an uneven cadence, which
breaks every downstream Fourier transform silently. Instead `regular_time_grid` lays one
uniform clock over the union of the series and `reindex_on_grid` drops each frame into
its slot, leaving NaN where an observation genuinely does not exist.
"""

import os
import pathlib
import re
import warnings
from datetime import datetime

import numpy as np
from astropy.io import fits

_FRAME_TS_RE = re.compile(r'\.(\d{8})_(\d{6})_TAI\.')


def parse_frame_timestamp(filename):
    """Extract the embedded YYYYMMDD_HHMMSS_TAI timestamp from a JSOC cutout filename.

    Returns a naive datetime, or None if the filename doesn't match the expected pattern.
    """
    m = _FRAME_TS_RE.search(str(filename))
    if not m:
        return None
    return datetime.strptime(m.group(1) + m.group(2), '%Y%m%d%H%M%S')


def _fit_to_shape(data, target_shape):
    """Center-crop and/or NaN-pad a 2D array to target_shape, keeping its center fixed."""
    out = np.full(target_shape, np.nan, dtype=np.float32)
    src_slices, dst_slices = [], []
    for src_n, dst_n in zip(data.shape, target_shape):
        n = min(src_n, dst_n)
        src_start = (src_n - n) // 2
        dst_start = (dst_n - n) // 2
        src_slices.append(slice(src_start, src_start + n))
        dst_slices.append(slice(dst_start, dst_start + n))
    out[tuple(dst_slices)] = data[tuple(src_slices)]
    return out


def make_cube(pattern, output_path, overwrite=False):

    """
    Combine a series of FITS files matching the given pattern into a single 3D cube,
    stacked by pixel index and carrying a per-frame timestamp table.

    These are *tracked* JSOC cutouts: JSOC has already registered every frame so the
    active region sits at the same pixel location throughout. Stacking is therefore a
    plain pixel-index stack — the frames must NOT be re-registered to any single frame's
    sky WCS, because each tracked frame legitimately covers a different patch of sky and
    reprojecting them onto one common footprint undoes exactly the alignment tracking
    provided (the spot wanders across the cube, then snaps into place).

    A fixed-arcsec tracked cutout has a constant pixel shape, so a shape mismatch means
    something is wrong — most likely frames on disk left over from an earlier download
    under a different box. Mismatched frames are center-cropped/NaN-padded onto the
    majority grid (which at least preserves the tracking alignment) and reported loudly,
    but the real fix is to re-download the region with one consistent box.

    Timestamps are parsed from the JSOC filenames and written to a `TIMESTAMPS` binary
    table extension, so downstream code can align series by time instead of re-globbing
    the frame directory and assuming index correspondence. Read both back with
    `read_cube`.

    Parameters:
    - pattern: A glob pattern to match the input FITS files (e.g., "data/*.fits").
    - output_path: The path where the output cube FITS file will be saved.
    - overwrite: If True, overwrite the output file if it already exists.

    Returns:
    - The path to the created cube FITS file.
    """
    import glob
    from collections import Counter

    from astropy.io import fits

    # Find all files matching the pattern
    file_list = sorted(glob.glob(str(pattern)))
    if not file_list:
        raise ValueError(f"No files found matching pattern: {pattern}")

    if not overwrite and os.path.exists(output_path):
        print(f"Omiting execution: file already created at {output_path}")
        return output_path

    timestamps = [parse_frame_timestamp(f) for f in file_list]
    unparsed = [f for f, ts in zip(file_list, timestamps) if ts is None]
    if unparsed:
        raise ValueError(
            f"make_cube: {len(unparsed)} file(s) have no parseable YYYYMMDD_HHMMSS_TAI "
            f"timestamp, e.g. {unparsed[0]} — cannot build a time axis for {pattern}")

    # Read every frame's shape to find the majority grid, then use one such frame's
    # header as the cube header — cheap header-only reads.
    shapes = []
    for file in file_list:
        with fits.open(file) as hdu:
            shapes.append(hdu[1].data.shape)
    target_shape = Counter(shapes).most_common(1)[0][0]
    ref_index = shapes.index(target_shape)
    with fits.open(file_list[ref_index]) as hdu:
        header = hdu[1].header.copy()

    # float32 matches HMI's real precision and halves peak memory, which matters for
    # multi-day cubes (a 2-day 720s cube of a 600px box is ~1.3 GB in float64).
    cube = np.full((len(file_list), *target_shape), np.nan, dtype=np.float32)

    mismatched = []
    for i, (file, shape) in enumerate(zip(file_list, shapes)):
        with fits.open(file) as hdu:
            data = hdu[1].data
        if shape == target_shape:
            cube[i] = data
        else:
            cube[i] = _fit_to_shape(data, target_shape)
            mismatched.append((file, shape))
    if mismatched:
        print(f"make_cube: WARNING — {len(mismatched)}/{len(file_list)} frame(s) do not match "
              f"the majority {target_shape} grid and were center-cropped/padded onto it. "
              f"A tracked cutout has a fixed pixel size, so these are almost certainly "
              f"left over from a download under a different box — re-download this region.")
        for file, shape in mismatched[:5]:
            print(f"  {shape} {file}")
        if len(mismatched) > 5:
            print(f"  ... and {len(mismatched) - 5} more")

    return write_cube(cube, output_path, header=header, timestamps=timestamps,
                      overwrite=overwrite)


def apply_per_frame_correction(cube_path, frame_dir, correct, series_glob,
                               diag_names=()):
    """Apply a per-frame correction to a cube, matching frames to their originals by time.

    Both physical corrections need each frame's *own* header: the OBS_V* keywords change
    every frame, and mu at the box centre runs 0.78 -> 0.87 -> 0.85 across NOAA 11536's
    72 h window, so one correction map applied cube-wide would leave most of the effect in
    place and inject a spurious trend of its own. `make_cube` keeps only the reference
    frame's header, so the geometry has to be read back from the surviving per-file frames
    and matched to the cube **by timestamp** — never by position, so that a frame missing
    from the middle of the window cannot silently shift every later correction out of step.

    `doppler_calibration.calibrate_cube` and `limb_darkening.limb_darkening_cube` are both
    this function plus a `correct`; they used to be two copies of the same 70 lines.

    Parameters
    ----------
    cube_path : path to the cube to correct (read with `read_cube`)
    frame_dir : directory holding the per-frame FITS the cube was built from
    correct   : callable (sunpy_map, timestamp) -> (values_2d, {diag_name: scalar})
    series_glob : filename pattern for the per-frame files. Patterns here deliberately
        match every cadence, since NOAA 11117 has to come from the 45 s series — a
        720 s-only pattern silently finds nothing and reports every frame as missing.
    diag_names : names of the per-frame diagnostics `correct` returns

    Returns
    -------
    (cube, timestamps, diag_means) with `cube` float32 and the same shape as the input,
    and `diag_means` mapping each name to an (n_t,) array.
    """
    import sunpy.map

    frame_dir = pathlib.Path(frame_dir)
    cube, timestamps = read_cube(cube_path)

    frame_files = {parse_frame_timestamp(p): p for p in frame_dir.glob(series_glob)}
    frame_files.pop(None, None)

    missing = [t for t in timestamps if t not in frame_files]
    if missing:
        raise ValueError(
            f'{len(missing)} cube frame(s) have no matching file in {frame_dir} '
            f'(first: {missing[0]}) — the per-frame headers are needed for the geometry. '
            f'Was the frame directory cleaned after the cube was built?')

    out = np.empty_like(cube, dtype=np.float32)
    diag_means = {name: np.full(len(timestamps), np.nan) for name in diag_names}

    for i, timestamp in enumerate(timestamps):
        smap = sunpy.map.Map(str(frame_files[timestamp]))
        values, diagnostics = correct(smap, timestamp)

        # A frame make_cube had to pad or crop is a different shape from the cube; match
        # its handling so the two stay aligned.
        out[i] = (values if values.shape == cube.shape[1:]
                  else _fit_to_shape(values, cube.shape[1:]))
        for name, value in diagnostics.items():
            diag_means[name][i] = value

    return out, timestamps, diag_means


def write_cube(cube, output_path, header=None, timestamps=None, history=None,
               overwrite=True):
    """Write a 3D cube plus its time axis, in the format `read_cube` expects.

    Used both by `make_cube` (stacking downloaded frames) and by the processing notebooks
    to save derived cubes — corrected dopplergrams, masks — so that everything downstream
    reads through one code path and DS9 sees a consistent set of files.

    Parameters
    ----------
    cube : ndarray, shape (n_t, ny, nx)
        Written as-is, so pass float32 for data and uint8 for masks; DS9 renders an
        integer mask far more cleanly than a float one.
    header : fits.Header, optional
        Spatial header, normally copied from one of the source frames. Its WCS describes
        the two image axes only — it says nothing meaningful about axis 3, which is why
        the timestamps go in their own extension rather than into a CTYPE3.
    timestamps : sequence of datetime, optional
        One per frame. Omitting them produces a cube `read_cube` will refuse, which is
        deliberate: a cube with no time axis cannot be safely joined to another series.
    history : sequence of str, optional
        HISTORY cards describing how the cube was produced, so a file opened months later
        is self-describing about which corrections it already has.
    """
    from astropy.io import fits

    header = fits.Header() if header is None else header.copy()
    # BLANK/BSCALE/BZERO describe how the *source* frames were stored, and carrying them
    # onto a derived cube is actively harmful: on an integer cube (a uint8 mask) astropy
    # applies the inherited BLANK as a missing-data sentinel on read and hands back floats
    # with NaNs, and on a float cube it just emits a VerifyWarning on every single write.
    for key in ('BLANK', 'BSCALE', 'BZERO'):
        header.pop(key, None)
    if timestamps is not None:
        header['NFRAMES'] = (len(cube), 'number of frames along axis 3')
    for line in (history or []):
        header.add_history(line)

    hdus = [fits.PrimaryHDU(cube, header=header)]
    if timestamps is not None:
        if len(timestamps) != len(cube):
            raise ValueError(f'{len(cube)} frames but {len(timestamps)} timestamps')
        time_col = fits.Column(name='T_OBS', format='23A',
                               array=np.array([t.isoformat() for t in timestamps]))
        hdus.append(fits.BinTableHDU.from_columns([time_col], name='TIMESTAMPS'))

    output_path = str(output_path)
    pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    # output_verify='silentfix' silently fixes non-standard header values inherited from
    # the HMI headers (e.g. the string 'nan' in CRDER/CSYSER cards).
    fits.HDUList(hdus).writeto(output_path, overwrite=overwrite, output_verify='silentfix')
    return output_path


def read_cube(path):
    """Read a cube written by `make_cube`, returning (data, timestamps).

    timestamps is a list of naive datetimes (TAI, as embedded in the JSOC filenames),
    one per frame along axis 0. Raises if the cube predates the TIMESTAMPS extension —
    rebuild it with `make_cube` rather than falling back to re-globbing the frame
    directory, which cannot detect a missing mid-window frame.
    """
    from astropy.io import fits

    with fits.open(path) as hdul:
        data = hdul[0].data
        if 'TIMESTAMPS' not in hdul:
            raise ValueError(
                f"{path} has no TIMESTAMPS extension — it was built by an older make_cube. "
                f"Rebuild it so frames can be aligned by time.")
        timestamps = [datetime.fromisoformat(t) for t in hdul['TIMESTAMPS'].data['T_OBS']]
    if len(timestamps) != len(data):
        raise ValueError(f"{path}: {len(data)} frames but {len(timestamps)} timestamps")
    return data, timestamps


def regular_time_grid(time_lists, cadence_s=None, tolerance_s=None):
    """Build one evenly-spaced time axis covering every timestamp in `time_lists`.

    The three HMI series of a region are downloaded independently and individual JSOC files
    do fail, so their frame counts differ. Joining them on the *intersection* of their
    timestamps throws away good frames from the other series and, worse, leaves holes in the
    time axis: NOAA 11536's Dopplergram is missing 2012-08-01 09:48 and 2012-08-02 21:48, so
    the intersection has two 1440 s jumps in an otherwise 720 s series. Anything that assumes
    a single cadence — every FFT in src/sunspot_analysis.py — is then quietly wrong.

    This returns a grid that is uniform *by construction* rather than one inherited from
    whatever happened to download, so a missing frame becomes a NaN frame at the right time
    (see `reindex_on_grid`) instead of a shortened axis.

    Parameters
    ----------
    time_lists : sequence of sequences of datetime
        One list per series. Order and duplicates don't matter.
    cadence_s : float, optional
        Grid spacing. Default is the *finest* of the per-series median spacings — see the
        note below on why it is neither the pooled median nor a per-series maximum.
    tolerance_s : float, optional
        How far a real timestamp may sit from its grid slot. Default `cadence_s / 2`, i.e.
        each timestamp claims its nearest slot and nothing else.

    Notes
    -----
    The cadence and the validation are both **per series**, not over the pooled timestamps,
    because the three series do not necessarily share a clock. NOAA 11117 has to be fetched
    from the 45 s series (JSOC's 720 s series 500-errors across its window) with
    ``a.Sample``, and the records that come back sit on grids offset from each other by
    45 s: continuum and Dopplergram every 360 s, magnetogram every 720 s, and mostly
    45 s later. Pooling those gives alternating 45 s and 315 s intervals and a median of
    315 s, which describes none of the three.

    Per series it is unambiguous: medians of 360, 720 and 360 s. The grid takes the
    **finest** of them, since a grid coarser than the fastest series would put two of that
    series' frames in one slot; the slower series simply leaves every other slot empty.
    Two frames from *different* series landing in one slot is not a collision — it is the
    entire point of a shared grid.

    Returns
    -------
    (grid, cadence_s)
        `grid` is a list of datetimes from the earliest to the latest timestamp inclusive.
    """
    from datetime import timedelta

    series = [sorted(set(times)) for times in time_lists if len(times)]
    pooled = sorted({t for times in time_lists for t in times})
    if not pooled:
        raise ValueError('regular_time_grid: no timestamps given')
    if len(pooled) == 1:
        return list(pooled), float(cadence_s or 0.0)

    if cadence_s is None:
        medians = [float(np.median(np.diff([t.timestamp() for t in times])))
                   for times in series if len(times) > 1]
        # No series has two timestamps of its own, so there is no per-series spacing to
        # measure; the pooled spacing is all there is.
        cadence_s = min(medians) if medians else float(
            np.median(np.diff([t.timestamp() for t in pooled])))
    cadence_s = float(cadence_s)
    if cadence_s <= 0:
        raise ValueError(f'regular_time_grid: cadence must be positive, got {cadence_s}')
    if tolerance_s is None:
        tolerance_s = cadence_s / 2

    span = (pooled[-1] - pooled[0]).total_seconds()
    n = int(round(span / cadence_s)) + 1
    grid = [pooled[0] + timedelta(seconds=i * cadence_s) for i in range(n)]

    # Verify before returning, so a wrong cadence surfaces here rather than as a subtly
    # mis-slotted cube 200 lines downstream. One series at a time: `_slot_indices` refuses
    # two timestamps in one slot, which is right within a series (the cadence is wrong) and
    # wrong across them (they are the same instant observed by two instruments).
    for times in series:
        _slot_indices(times, grid[0], cadence_s, tolerance_s, len(grid))
    return grid, cadence_s


def _slot_indices(times, grid_start, cadence_s, tolerance_s, n_slots):
    """Map timestamps onto grid slots, raising rather than mangling the axis.

    A timestamp that doesn't fit its nearest slot, or two timestamps landing in the same
    one, means the assumed cadence is wrong. Both are refused: silently snapping them is
    precisely the failure `regular_time_grid` exists to prevent.
    """
    seen, indices = {}, []
    for t in times:
        offset = (t - grid_start).total_seconds()
        k = int(round(offset / cadence_s))
        drift = abs(offset - k * cadence_s)
        if not 0 <= k < n_slots:
            raise ValueError(
                f'timestamp {t} maps to grid slot {k}, outside 0..{n_slots - 1} — the '
                f'assumed cadence of {cadence_s:g} s does not describe this series')
        if drift > tolerance_s:
            raise ValueError(
                f'timestamp {t} is {drift:.1f} s from its nearest grid slot, more than the '
                f'{tolerance_s:.1f} s tolerance — the series is not on a {cadence_s:g} s '
                f'cadence. Pass an explicit cadence_s, or a larger tolerance_s if the '
                f'jitter is real.')
        if k in seen:
            raise ValueError(
                f'timestamps {seen[k]} and {t} both map to grid slot {k} — the assumed '
                f'cadence of {cadence_s:g} s is too coarse for this series')
        seen[k] = t
        indices.append(k)
    return indices


def reindex_on_grid(cube, times, grid, cadence_s, tolerance_s=None):
    """Place a cube's frames onto `grid`, leaving NaN frames where the series has no data.

    This is the "keep the times consistent" half of the join: the output always has one
    frame per grid slot, so the three series stay index-aligned with each other and with the
    time axis, and a missing mid-window frame stays visible as a gap instead of pulling every
    later frame one slot out of step.

    Parameters
    ----------
    cube : ndarray, shape (n_t, ny, nx)
    times : sequence of datetime
        One per frame of `cube`.
    grid, cadence_s : as returned by `regular_time_grid`.

    Returns
    -------
    (out, present)
        `out` is float32, shape `(len(grid), ny, nx)`, NaN in unfilled slots.
        `present` is a `(len(grid),)` bool array — True where this series has a real frame.
    """
    if len(times) != len(cube):
        raise ValueError(f'{len(cube)} frames but {len(times)} timestamps')
    if tolerance_s is None:
        tolerance_s = cadence_s / 2

    indices = _slot_indices(times, grid[0], cadence_s, tolerance_s, len(grid))

    out = np.full((len(grid), *cube.shape[1:]), np.nan, dtype=np.float32)
    present = np.zeros(len(grid), dtype=bool)
    for i, k in enumerate(indices):
        out[k] = cube[i]
        present[k] = True
    return out, present


def crop_to_common_window(cubes, present=None, verbose=True):
    """Put cubes on one spatial grid and trim to the window where all of them have data.

    Two separate things put NaN borders on a cube, and this removes both:

    - **The box changed mid-window.** A tracked cutout is supposed to have a constant pixel
      size, but if a region is re-downloaded under a different box — or JSOC returns a
      smaller patch for part of the window, which is what NOAA 11117 does towards the end —
      `make_cube` center-crops/NaN-pads the odd frames onto the majority grid. Those frames
      then carry a NaN frame of padding that no segmentation threshold will ever select, so
      the mask areas step down for exactly as long as the smaller box lasted.
    - **The three series were downloaded under different boxes.** NOAA 11117's magnetogram
      is 402x402 against 433x433 for its continuum and Dopplergram, so the cubes cannot even
      be indexed against each other.

    The window is the intersection, over every frame of every cube, of the bounding box of
    that frame's finite pixels — the largest rectangle in which nothing is padding. Frames
    are matched to the smallest common shape by center-cropping first, mirroring
    `_fit_to_shape`'s centered padding so the two undo each other.

    Using each frame's *bounding box* rather than its finite pixels means an isolated bad
    pixel in the middle of a frame cannot shrink the window; only missing edges can.

    Parameters
    ----------
    cubes : dict of str -> ndarray, each (n_t, ny, nx)
        All must have the same n_t; the spatial shapes may differ.
    present : dict of str -> bool array, optional
        Which frames of each cube hold real data. Gap frames are entirely NaN and would
        collapse the window to nothing, so they are skipped.
    verbose : bool
        Print what was trimmed. Worth leaving on — a crop that eats most of the box means
        the region was downloaded under two very different geometries.

    Returns
    -------
    (cropped, offsets)
        `cropped` maps each name to its trimmed cube, all now the same shape.
        `offsets` maps each name to the ``(row0, col0)`` of the window in *that cube's own
        original pixel coordinates*, so a caller can shift CRPIX1/CRPIX2 and keep the WCS
        pointing at the same sky.
    """
    shapes = {name: cube.shape[1:] for name, cube in cubes.items()}
    target = (min(s[0] for s in shapes.values()), min(s[1] for s in shapes.values()))

    # Center-crop onto the common shape, and remember by how much, so the offsets come back
    # in each cube's own coordinates rather than in the intermediate grid's.
    centered, center_offset = {}, {}
    for name, cube in cubes.items():
        ny, nx = shapes[name]
        r0, c0 = (ny - target[0]) // 2, (nx - target[1]) // 2
        center_offset[name] = (r0, c0)
        centered[name] = (cube if (ny, nx) == target
                          else cube[:, r0:r0 + target[0], c0:c0 + target[1]])

    row_lo, row_hi = 0, target[0] - 1
    col_lo, col_hi = 0, target[1] - 1
    for name, cube in centered.items():
        rows = np.isfinite(cube).any(axis=2)          # (n_t, ny)
        cols = np.isfinite(cube).any(axis=1)          # (n_t, nx)
        usable = rows.any(axis=1)
        if present is not None and name in present:
            usable &= np.asarray(present[name], dtype=bool)
        if not usable.any():
            raise ValueError(f'crop_to_common_window: every frame of {name!r} is all-NaN')
        rows, cols = rows[usable], cols[usable]
        row_lo = max(row_lo, int(rows.argmax(axis=1).max()))
        row_hi = min(row_hi, int((target[0] - 1 - rows[:, ::-1].argmax(axis=1)).min()))
        col_lo = max(col_lo, int(cols.argmax(axis=1).max()))
        col_hi = min(col_hi, int((target[1] - 1 - cols[:, ::-1].argmax(axis=1)).min()))

    if row_hi < row_lo or col_hi < col_lo:
        raise ValueError(
            f'crop_to_common_window: the frames share no common data window '
            f'(rows {row_lo}..{row_hi}, cols {col_lo}..{col_hi}). The boxes these cubes '
            f'were downloaded under do not overlap — re-download the region.')

    window = (slice(row_lo, row_hi + 1), slice(col_lo, col_hi + 1))
    cropped = {name: cube[:, window[0], window[1]] for name, cube in centered.items()}
    offsets = {name: (center_offset[name][0] + row_lo, center_offset[name][1] + col_lo)
               for name in cubes}

    if verbose:
        shape_list = ', '.join(f'{n}={s[0]}x{s[1]}' for n, s in shapes.items())
        new = next(iter(cropped.values())).shape[1:]
        kept = 100 * (new[0] * new[1]) / (target[0] * target[1])
        print(f'  cropped to the common data window: {shape_list} -> {new[0]}x{new[1]} '
              f'({kept:.0f}% of the smallest input box)')

    return cropped, offsets


def reindex_series_on_grid(values, times, grid, cadence_s, tolerance_s=None):
    """`reindex_on_grid` for a 1-D per-frame series (e.g. a correction term's mean).

    Kept separate rather than folded in with a shape check, because a 1-D array of length
    n_t and a cube of n_t frames want visibly different call sites.
    """
    values = np.asarray(values, dtype=float)
    if len(times) != len(values):
        raise ValueError(f'{len(values)} values but {len(times)} timestamps')
    if tolerance_s is None:
        tolerance_s = cadence_s / 2

    indices = _slot_indices(times, grid[0], cadence_s, tolerance_s, len(grid))
    out = np.full(len(grid), np.nan)
    out[indices] = values
    return out


def normalize(values):
    """Min-max scale to [0, 1], ignoring NaNs. All-NaN or flat input returns NaNs/zeros.

    For comparing the *shape* of two series with different units or offsets — a
    magnetogram trend against a Doppler one, say. It destroys amplitude information, so
    it is for looking, never for measuring.
    """
    values = np.asarray(values, dtype=float)
    lo, hi = np.nanmin(values), np.nanmax(values)
    if not np.isfinite(lo) or not np.isfinite(hi):
        return np.full_like(values, np.nan)
    if hi == lo:
        return np.zeros_like(values)
    return (values - lo) / (hi - lo)


def mean_series(cube, mask):
    """Spatial mean of `cube` over `mask`, per frame. NaN where the mask is empty.

    The RuntimeWarning for an all-NaN slice is suppressed on purpose: a gap frame, or a
    frame where the spot was not detected, legitimately has nothing to average, and NaN
    is the correct answer rather than something to be warned about once per frame.
    """
    import numpy as _np
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return _np.nanmean(_np.where(mask, cube, _np.nan), axis=(1, 2))

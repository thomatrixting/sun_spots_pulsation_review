# Source - https://stackoverflow.com/a/54529216
# Posted by Mc Missile, modified by community. See post 'Timeline' for change history
# Retrieved 2026-05-17, License - CC BY-SA 4.0

import os
import pathlib
import re
from datetime import datetime

import numpy as np
import pandas as pd
import astropy.units as u
from astropy.coordinates import SkyCoord
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


def locate_ar_window(noaa_number, date, n_days=4, padding=30 * u.arcsec,
                     min_half_extent=60 * u.arcsec, max_half_extent=250 * u.arcsec,
                     prefer_frm='NOAA SWPC Observer', statistic='median'):
    """
    Query the HEK for NOAA `noaa_number` over an n_days window anchored at `date`, and
    return a square cutout box sized to the sunspot group's typical reported extent across
    that window but positioned at `date`'s own reported center — for use with
    a.jsoc.Cutout(tracking=True).

    Position is left entirely to JSOC's rotation tracking; only the box's *size* accounts
    for the AR growing/shrinking across the window (real apparent position sweeps hundreds
    of arcsec/day from solar rotation alone, so combining raw day-to-day positions would be
    wrong — only the reported half-extent per row is meaningful to combine).

    Sizing notes, all learned the hard way from the records HEK actually returns:

    - **Provider matters more than statistics.** A single NOAA number returns rows from
      several `frm_name` providers describing different things. 'NOAA SWPC Observer' is the
      *sunspot group* (half-width ~70-140 arcsec); 'HMI SHARP' is the whole magnetic active
      region complex including plage (half-width ~250-400 arcsec). Mixing them and taking a
      max — or even a median — sizes the box off whichever provider happens to dominate,
      which is how NOAA 11363 ended up with a 736 arcsec box around a ~110 arcsec spot.
      Rows are therefore filtered to `prefer_frm` first, falling back to all rows (with a
      warning) when that provider is absent for the window.
    - **The box is square.** SWPC reports an essentially degenerate latitude extent
      (half-height 3-9 arcsec), so its height carries no information; the half-width is the
      only usable scale. A square box also keeps ARs comparable and gives the quiet-sun
      plane fit an isotropic footprint.
    - **Median, then clamp** to [min_half_extent, max_half_extent]. The bounding boxes are
      coarse and a single bad row should not set the size for the whole window.

    Oversizing is not harmless: a large box spans a large line-of-sight solar-rotation
    velocity gradient, which contaminates any quiet-sun reference computed over it.

    Undersizing is worse, though, because it silently truncates the thing being measured.
    SWPC's reported extent is not always generous: NOAA 11536's group actually grows to
    ~220 arcsec across and sits ~35 arcsec off the reported centroid, so median sizing
    clips it on every frame. Use `statistic='max'` (and more `padding`) for such regions —
    always after checking the resulting box against the frames, not on faith.

    Extents are measured from each row's own `hpc_x`/`hpc_y` centroid rather than from raw
    `hpc_bbox` min/max, because the reported bbox is not guaranteed to be centered on the
    reported centroid — and the centroid is what the box gets positioned on.

    Parameters:
    - noaa_number: NOAA AR catalog number
    - date: anchor day, 'YYYY-MM-DD' (box position and reference frame come from this day)
    - n_days: number of days the window should span (window is date 00:00 through
      date + n_days - 1 23:59:59)
    - padding: margin added to the box on each side
    - min_half_extent: floor on the box half-size, so a degenerate bbox can't collapse it
    - max_half_extent: ceiling on the box half-size, before padding
    - prefer_frm: HEK `frm_name` to size and position from; None uses every row
    - statistic: 'median' (default, robust to one bad row) or 'max' (largest reported
      extent in the window) — how the per-row half-widths are combined

    Returns:
    - (bottom_left, top_right, time_start, time_end): SkyCoord corners (in the anchor
      day's frame) and ISO time strings spanning the full n_days window.
    """
    from sunpy.net import Fido, attrs as a

    time_start = f'{date} 00:00:00'
    end_date = (pd.Timestamp(date) + pd.Timedelta(days=n_days - 1)).date()
    time_end = f'{end_date} 23:59:59'

    result = Fido.search(a.Time(time_start, time_end),
                          a.hek.EventType('AR'), a.hek.AR.NOAANum == noaa_number)
    rows = result['hek']
    if len(rows) == 0:
        raise ValueError(f"No HEK record found for NOAA {noaa_number} in {time_start}..{time_end}")

    if prefer_frm is not None:
        preferred = [row for row in rows if row['frm_name'] == prefer_frm]
        if preferred:
            rows = preferred
        else:
            print(f"WARNING: no '{prefer_frm}' HEK rows for NOAA {noaa_number} in the window; "
                  f"sizing from all {len(rows)} row(s) — check the resulting box.")

    anchor_date = pd.Timestamp(date).date()
    anchor_idx = next(
        (i for i, row in enumerate(rows)
         if pd.Timestamp(row['event_starttime'].iso).date() == anchor_date),
        None,
    )
    if anchor_idx is None:
        anchor_idx = 0
        print(f"WARNING: no HEK row exactly on {date} for NOAA {noaa_number}; "
              f"using earliest row in window as anchor ({rows[0]['event_starttime']}).")
    anchor_row = rows[anchor_idx]

    # Half-extents measured from each row's own centroid, so size and position agree.
    half_widths = [np.abs(row['hpc_bbox'].Tx.to_value(u.arcsec) - row['hpc_x']).max()
                   for row in rows]
    combine = {'median': np.median, 'max': np.max}.get(statistic)
    if combine is None:
        raise ValueError(f"statistic must be 'median' or 'max', got {statistic!r}")
    raw_half_w = combine(half_widths)
    half = min(max(raw_half_w * u.arcsec, min_half_extent), max_half_extent) + padding

    frame = anchor_row['hpc_bbox'].frame[0]
    cx, cy = anchor_row['hpc_x'] * u.arcsec, anchor_row['hpc_y'] * u.arcsec
    bottom_left = SkyCoord(cx - half, cy - half, frame=frame)
    top_right   = SkyCoord(cx + half, cy + half, frame=frame)
    print(f"NOAA {noaa_number}: box {2 * half:.0f} square, centered {cx:.0f},{cy:.0f} — from "
          f"{len(rows)} {prefer_frm or 'HEK'} row(s), {statistic} half-width {raw_half_w:.0f} arcsec")
    return bottom_left, top_right, time_start, time_end


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


def ds9_box_to_hpc(x_c_ds9, y_c_ds9, w_px, h_px, hmi_map):
    """
    Convert a DS9 box region (1-indexed pixels) to Helioprojective corner coordinates.

    Parameters:
    - x_c_ds9, y_c_ds9: DS9 pixel center (1-indexed)
    - w_px, h_px: box width and height in pixels
    - hmi_map: sunpy.map.Map with HMI WCS

    Returns:
    - (bottom_left, top_right): SkyCoord pair for a.jsoc.Cutout (SW and NE corners in HPC)
    """
    # CROTA2≈180°: Tx ∝ −pixel_x, Ty ∝ −pixel_y
    # SW corner (min Tx, min Ty) = max pixel_x, max pixel_y; DS9 is 1-indexed → subtract 1
    x_bl = x_c_ds9 + w_px / 2 - 1
    y_bl = y_c_ds9 + h_px / 2 - 1
    x_tr = x_c_ds9 - w_px / 2 - 1
    y_tr = y_c_ds9 - h_px / 2 - 1
    return hmi_map.wcs.pixel_to_world(x_bl, y_bl), hmi_map.wcs.pixel_to_world(x_tr, y_tr)


def plot_regions_on_map(hmi_map_rot, regions, cmap=None, norm=None):
    """
    Draw a list of HPC box regions as labelled quadrangles on a rotated HMI map.

    Works for both continuum and magnetogram maps:
    - Continuum (BUNIT != Gauss): gray colormap with percentile clipping
    - Magnetogram (BUNIT == Gauss): RdBu_r colormap with ±500 G symmetric norm

    Parameters:
    - hmi_map_rot: North-up sunpy.map.Map (already rotated)
    - regions: list of (bottom_left, top_right) SkyCoord pairs
    - cmap: override colormap (auto-detected if None)
    - norm: override matplotlib norm (auto-detected if None)
    """
    import matplotlib.pyplot as plt
    import astropy.units as u

    is_magnetogram = 'gauss' in hmi_map_rot.meta.get('bunit', '').lower()

    if cmap is None:
        cmap = 'RdBu_r' if is_magnetogram else 'gray'
    if norm is None and is_magnetogram:
        norm = plt.Normalize(vmin=-500, vmax=500)

    title = 'HMI Magnetogram — todas las regiones' if is_magnetogram else 'HMI Continuo — todas las regiones'

    colors = ['red', 'cyan', 'yellow', 'lime', 'magenta', 'orange', 'deepskyblue', 'white']
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection=hmi_map_rot)

    if norm is not None:
        hmi_map_rot.plot(axes=ax, cmap=cmap, norm=norm)
    else:
        hmi_map_rot.plot(axes=ax, cmap=cmap, clip_interval=(1, 99.9) * u.percent)

    hmi_map_rot.draw_grid(axes=ax, color='white', alpha=0.3, lw=0.5)
    for i, (bl, tr) in enumerate(regions):
        hmi_map_rot.draw_quadrangle(bl, top_right=tr,
                                    edgecolor=colors[i % len(colors)],
                                    linewidth=2, label=f'Region {i + 1}')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_title(title)
    plt.tight_layout()
    plt.show()
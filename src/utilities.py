# Source - https://stackoverflow.com/a/54529216
# Posted by Mc Missile, modified by community. See post 'Timeline' for change history
# Retrieved 2026-05-17, License - CC BY-SA 4.0

import os
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


def locate_ar_window(noaa_number, date, n_days=4, padding=30 * u.arcsec):
    """
    Query the HEK for NOAA `noaa_number` over an n_days window anchored at `date`, and
    return a cutout box sized to the AR's largest reported extent across that window but
    positioned at `date`'s own reported center — for use with a.jsoc.Cutout(tracking=True).

    Position is left entirely to JSOC's rotation tracking; only the box's *size* accounts
    for the AR growing/shrinking across the window (real apparent position sweeps hundreds
    of arcsec/day from solar rotation alone, so unioning raw day-to-day positions would be
    wrong — only the reported half-width/half-height per day is meaningful to union).

    Parameters:
    - noaa_number: NOAA AR catalog number
    - date: anchor day, 'YYYY-MM-DD' (box position and reference frame come from this day)
    - n_days: number of days the window should span (window is date 00:00 through
      date + n_days - 1 23:59:59)
    - padding: margin added to the box on each side

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

    half_widths, half_heights = zip(*(
        ((row['hpc_bbox'].Tx.max() - row['hpc_bbox'].Tx.min()) / 2,
         (row['hpc_bbox'].Ty.max() - row['hpc_bbox'].Ty.min()) / 2)
        for row in rows
    ))
    half_w = max(half_widths) + padding
    half_h = max(half_heights) + padding

    frame = anchor_row['hpc_bbox'].frame[0]
    cx, cy = anchor_row['hpc_x'] * u.arcsec, anchor_row['hpc_y'] * u.arcsec
    bottom_left = SkyCoord(cx - half_w, cy - half_h, frame=frame)
    top_right   = SkyCoord(cx + half_w, cy + half_h, frame=frame)
    return bottom_left, top_right, time_start, time_end


def make_cube(pattern, output_path, overwrite=False):

    """
    Combine a series of FITS files matching the given pattern into a single 3D cube.

    Frames don't always share the same pixel shape (e.g. cutout requests made under a
    different box definition across separate download runs). The majority shape is used
    as the cube's grid; any frame with a different shape is reprojected onto it using its
    own WCS (already correct in these files) via sunpy/astropy, landing as real data over
    its original footprint and NaN elsewhere in that frame.

    Parameters:
    - pattern: A glob pattern to match the input FITS files (e.g., "data/*.fits").
    - output_path: The path where the output cube FITS file will be saved.
    - overwrite: If True, overwrite the output file if it already exists.

    Returns:
    - The path to the created cube FITS file.
    """
    import glob
    from collections import Counter

    import sunpy.map
    from astropy.io import fits

    # Find all files matching the pattern
    file_list = sorted(glob.glob(str(pattern)))
    if not file_list:
        raise ValueError(f"No files found matching pattern: {pattern}")

    if not overwrite and os.path.exists(output_path):
        print(f"Omiting execution: file already created at {output_path}")
        return output_path

    # Read every frame's shape to find the majority grid, then use one such frame as
    # the reference WCS/header — cheap header-only reads, no reprojection needed yet.
    shapes = []
    for file in file_list:
        with fits.open(file) as hdu:
            shapes.append(hdu[1].data.shape)
    target_shape = Counter(shapes).most_common(1)[0][0]
    ref_index = shapes.index(target_shape)
    with fits.open(file_list[ref_index]) as hdu:
        header = hdu[1].header.copy()
    ref_wcs = sunpy.map.Map(file_list[ref_index]).wcs

    # Initialize an empty cube, NaN-filled so any reprojected frame's off-footprint
    # region reads as missing data rather than 0.
    cube = np.full((len(file_list), *target_shape), np.nan)

    # Fill the cube with data from each file, reprojecting any mismatched-shape frame
    # onto the reference grid.
    mismatched = 0
    for i, (file, shape) in enumerate(zip(file_list, shapes)):
        if shape == target_shape:
            with fits.open(file) as hdu:
                cube[i] = hdu[1].data
        else:
            cube[i] = sunpy.map.Map(file).reproject_to(ref_wcs).data
            mismatched += 1
    if mismatched:
        print(f"make_cube: reprojected {mismatched}/{len(file_list)} mismatched-shape "
              f"frame(s) onto the majority {target_shape} grid ({pattern})")

    # Write the cube to a new FITS file
    # output_verify='silentfix' silently fixes non-standard header values (e.g. string 'nan' in CRDER/CSYSER cards)
    hdu_new = fits.PrimaryHDU(cube, header=header)
    hdu_new.writeto(output_path, overwrite=overwrite, output_verify='silentfix')
    return output_path


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
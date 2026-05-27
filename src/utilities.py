# Source - https://stackoverflow.com/a/54529216
# Posted by Mc Missile, modified by community. See post 'Timeline' for change history
# Retrieved 2026-05-17, License - CC BY-SA 4.0

import os
from astropy.io import fits
import numpy as np

def make_cube(pattern, output_path, overwrite=False):

    """
    Combine a series of FITS files matching the given pattern into a single 3D cube.

    Parameters:
    - pattern: A glob pattern to match the input FITS files (e.g., "data/*.fits").
    - output_path: The path where the output cube FITS file will be saved.
    - overwrite: If True, overwrite the output file if it already exists.

    Returns:
    - The path to the created cube FITS file.
    """
    import glob
    from astropy.io import fits

    # Find all files matching the pattern
    file_list = sorted(glob.glob(str(pattern)))
    if not file_list:
        raise ValueError(f"No files found matching pattern: {pattern}")

    # Read the first file to get dimensions and header
    with fits.open(file_list[0]) as hdu:
        data_shape = hdu[1].data.shape
        header = hdu[1].header.copy()

    # Initialize an empty cube
    cube = np.zeros((len(file_list), *data_shape))

    if not overwrite and os.path.exists(output_path):
        print(f"Omiting execution: file already created at {output_path}")
        return
    # Fill the cube with data from each file
    for i, file in enumerate(file_list):
        with fits.open(file) as hdu:
            cube[i] = hdu[1].data

    # Write the cube to a new FITS file
    # output_verify='silentfix' silently fixes non-standard header values (e.g. string 'nan' in CRDER/CSYSER cards)
    hdu_new = fits.PrimaryHDU(cube, header=header)
    hdu_new.writeto(output_path, overwrite=overwrite, output_verify='silentfix')


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


def plot_regions_on_map(hmi_map_rot, regions):
    """
    Draw a list of HPC box regions as labelled quadrangles on a rotated HMI map.

    Parameters:
    - hmi_map_rot: North-up sunpy.map.Map (already rotated)
    - regions: list of (bottom_left, top_right) SkyCoord pairs
    """
    import matplotlib.pyplot as plt
    import astropy.units as u

    colors = ['red', 'cyan', 'yellow', 'lime', 'magenta', 'orange', 'deepskyblue', 'white']
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection=hmi_map_rot)
    hmi_map_rot.plot(axes=ax, cmap='gray', clip_interval=(1, 99.9) * u.percent)
    hmi_map_rot.draw_grid(axes=ax, color='white', alpha=0.3, lw=0.5)
    for i, (bl, tr) in enumerate(regions):
        hmi_map_rot.draw_quadrangle(bl, top_right=tr,
                                    edgecolor=colors[i % len(colors)],
                                    linewidth=2, label=f'Region {i + 1}')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_title('HMI Continuo — todas las regiones')
    plt.tight_layout()
    plt.show()
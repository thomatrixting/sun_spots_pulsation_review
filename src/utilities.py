# Source - https://stackoverflow.com/a/54529216
# Posted by Mc Missile, modified by community. See post 'Timeline' for change history
# Retrieved 2026-05-17, License - CC BY-SA 4.0

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

    # Fill the cube with data from each file
    for i, file in enumerate(file_list):
        with fits.open(file) as hdu:
            cube[i] = hdu[1].data

    # Write the cube to a new FITS file
    # output_verify='silentfix' silently fixes non-standard header values (e.g. string 'nan' in CRDER/CSYSER cards)
    hdu_new = fits.PrimaryHDU(cube, header=header)
    hdu_new.writeto(output_path, overwrite=overwrite, output_verify='silentfix')
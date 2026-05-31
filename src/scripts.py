"""
Download SDO/HMI cutouts and build FITS cubes for all sunspot regions.

Run standalone from the project root:
    python src/scripts.py

Or import individual functions into a notebook:
    from src.scripts import download_regions, make_cubes
"""

import pathlib
import re
import sys

from sunpy.net import Fido, attrs as a

# Allow both `python src/scripts.py` and `from src.scripts import ...`
_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.utilities import ds9_box_to_hpc, make_cube  # noqa: E402

# Maps each JSOC series to the glob pattern used to find its files and the
# label used in the output cube filename.
_SERIES_META = {
    'hmi.Ic_45s': {'glob': 'hmi.ic_45s.*.continuum.fits',  'label': 'continuum'},
    'hmi.M_45s':  {'glob': 'hmi.m_45s.*.magnetogram.fits', 'label': 'magnetogram'},
    'hmi.V_45s':  {'glob': 'hmi.v_45s.*.fits',             'label': 'dopplergram'},
}


# ── Single-region helper ─────────────────────────────────────────────────────

def download_single_region(result, output_dir):
    """Fetch files from a Fido search result into output_dir."""
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    files = Fido.fetch(result, path=output_dir)
    print(files)
    return files


# ── Unified download and cube builder ────────────────────────────────────────

def download_regions(regions_hpc, base_dir, time_start, time_end, notify_email,
                     series, region_indexes=None):
    """Search and download HMI cutouts for every region.

    Parameters
    ----------
    series : str
        JSOC series name — e.g. 'hmi.Ic_45s', 'hmi.M_45s', 'hmi.V_45s'.
    region_indexes : list[int] or None
        1-based region numbers to download. None downloads all regions.
    """
    base_dir = pathlib.Path(base_dir)
    for i, (bl, tr) in enumerate(regions_hpc):
        if region_indexes is not None and (i + 1) not in region_indexes:
            print(f"Region {i+1:02d}: skipped")
            continue
        region_dir = base_dir / f'region_{i+1:02d}'
        region_dir.mkdir(parents=True, exist_ok=True)
        cutout = a.jsoc.Cutout(bl, top_right=tr, tracking=True)
        result = Fido.search(
            a.Time(time_start, time_end),
            a.jsoc.Series(series),
            a.jsoc.Notify(notify_email),
            cutout,
        )
        print(f"Region {i+1:02d}: {len(result[0])} frames found")
        files = Fido.fetch(result, path=region_dir)
        print(f"  -> {len(files)} files saved to {region_dir}")


def make_cubes(regions_hpc, base_dir, series, region_indexes=None):
    """Build one FITS cube per region from downloaded frames.

    Parameters
    ----------
    series : str
        JSOC series name — e.g. 'hmi.Ic_45s', 'hmi.M_45s', 'hmi.V_45s'.
    region_indexes : list[int] or None
        1-based region numbers to process. None processes all regions.
    """
    if series not in _SERIES_META:
        raise ValueError(f"Unknown series '{series}'. Known: {list(_SERIES_META)}")
    meta = _SERIES_META[series]
    base_dir = pathlib.Path(base_dir)
    for i in range(1, len(regions_hpc) + 1):
        if region_indexes is not None and i not in region_indexes:
            continue
        region_dir = base_dir / f'region_{i:02d}'
        make_cube(
            region_dir / meta['glob'],
            base_dir / f'region_{i:02d}_{meta["label"]}_cube.fits',
            overwrite=True,
        )


# ── Standalone entry point ───────────────────────────────────────────────────

if __name__ == "__main__":
    import sunpy.map

    PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
    DATA_DIR   = PROJECT_ROOT / 'data' / 'raw' / '2026-04-01'
    BASE_DIR   = PROJECT_ROOT / 'data' / 'raw' / 'regions'
    DS9_FILE   = PROJECT_ROOT / 'data' / 'raw' / 'ds9.txt'
    TIME_START = '2026-04-01T12:00:00'
    TIME_END   = '2026-04-02T24:00:00'
    NOTIFY     = 'thomas.quamtum@gmail.com'

    # Load reference HMI continuum frame for WCS conversions
    fits_files = sorted(DATA_DIR.glob('hmi.ic_45s.*.continuum.fits'))
    if not fits_files:
        raise FileNotFoundError(f"No continuum FITS found in {DATA_DIR}")
    hmi_map = sunpy.map.Map(str(fits_files[0]))

    # Parse DS9 box regions (format: box(x, y, w, h, angle))
    regions_raw = []
    for line in DS9_FILE.read_text().strip().splitlines():
        m = re.match(r'box\(([^)]+)\)', line)
        if m:
            x_c, y_c, w, h, _ = map(float, m.group(1).split(','))
            if w > 0 and h > 0:
                regions_raw.append((x_c, y_c, w, h))
    print(f"{len(regions_raw)} regions parsed from {DS9_FILE}")

    regions_hpc = [ds9_box_to_hpc(x, y, w, h, hmi_map) for x, y, w, h in regions_raw]

    #download_regions(regions_hpc, BASE_DIR, TIME_START, TIME_END, NOTIFY, 'hmi.Ic_45s')
    #make_cubes(regions_hpc, BASE_DIR, 'hmi.Ic_45s')

    #download_regions(regions_hpc, BASE_DIR, TIME_START, TIME_END, NOTIFY, 'hmi.M_45s', region_indexes=[4])
    #make_cubes(regions_hpc, BASE_DIR, 'hmi.M_45s')

    download_regions(regions_hpc, BASE_DIR, TIME_START, TIME_END, NOTIFY, 'hmi.V_45s', region_indexes=[4])
    make_cubes(regions_hpc, BASE_DIR, 'hmi.V_45s', region_indexes=[4])

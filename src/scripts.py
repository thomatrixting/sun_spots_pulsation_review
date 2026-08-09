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
from datetime import timedelta

import astropy.units as u
import pandas as pd
from sunpy.net import Fido, attrs as a

# Allow both `python src/scripts.py` and `from src.scripts import ...`
_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.utilities import ds9_box_to_hpc, make_cube, parse_frame_timestamp  # noqa: E402

# Maps each JSOC series to the glob pattern used to find its files, the label used in
# the output cube filename, and its native cadence (informational only).
_SERIES_META = {
    'hmi.Ic_45s':  {'glob': 'hmi.ic_45s.*.continuum.fits',   'label': 'continuum',   'cadence_seconds': 45},
    'hmi.M_45s':   {'glob': 'hmi.m_45s.*.magnetogram.fits',  'label': 'magnetogram', 'cadence_seconds': 45},
    'hmi.V_45s':   {'glob': 'hmi.v_45s.*.fits',              'label': 'dopplergram', 'cadence_seconds': 45},
    'hmi.Ic_720s': {'glob': 'hmi.ic_720s.*.continuum.fits',  'label': 'continuum',   'cadence_seconds': 720},
    'hmi.M_720s':  {'glob': 'hmi.m_720s.*.magnetogram.fits', 'label': 'magnetogram', 'cadence_seconds': 720},
    'hmi.V_720s':  {'glob': 'hmi.v_720s.*.fits',             'label': 'dopplergram', 'cadence_seconds': 720},
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
                     series, region_indexes=None, sample=None):
    """Search and download HMI cutouts for every region.

    Skips regions/series whose requested range is already fully covered by files on
    disk, and resumes (rather than re-fetching from scratch) if it's partially covered.
    Assumes existing files are contiguous from the original start of the range — no gap
    detection/backfill.

    Parameters
    ----------
    series : str
        JSOC series name — e.g. 'hmi.Ic_45s', 'hmi.M_45s', 'hmi.V_45s'.
    region_indexes : list[int] or None
        1-based region numbers to download. None downloads all regions.
    sample : astropy.units.Quantity or None
        If given, passed as a.Sample(sample) to have JSOC downsample server-side
        (e.g. 360*u.s to get every 8th frame of a 45s series).

    Returns
    -------
    dict mapping 1-based region index -> {'status': 'ok'|'skipped'|'failed', ...}
    """
    base_dir = pathlib.Path(base_dir)
    meta = _SERIES_META[series]
    requested_end = pd.Timestamp(time_end).to_pydatetime()

    summary = {}
    for i, (bl, tr) in enumerate(regions_hpc):
        if region_indexes is not None and (i + 1) not in region_indexes:
            print(f"Region {i+1:02d}: skipped")
            continue
        region_dir = base_dir / f'region_{i+1:02d}'
        region_dir.mkdir(parents=True, exist_ok=True)

        existing = sorted(region_dir.glob(meta['glob']))
        existing_timestamps = [ts for ts in (parse_frame_timestamp(p) for p in existing) if ts is not None]
        latest = max(existing_timestamps) if existing_timestamps else None

        if latest is not None and latest >= requested_end:
            print(f"Region {i+1:02d} [{series}]: already covers requested range through {latest} — skipping")
            summary[i + 1] = {'status': 'skipped', 'files': existing}
            continue

        if latest is None:
            fetch_start_dt = None
            fetch_start = time_start
        else:
            # JSOC's time-range query rounds an in-between start back down to the nearest
            # existing record (verified live: start = latest + 1s still returned `latest`
            # itself) — nudge by a full cadence step (the actual sampling interval, native
            # or a.Sample-downsampled) so the resume request can't re-include it.
            step_seconds = sample.to_value(u.s) if sample is not None else meta['cadence_seconds']
            fetch_start_dt = latest + timedelta(seconds=step_seconds)
            fetch_start = fetch_start_dt.isoformat()

        if fetch_start_dt is not None and fetch_start_dt >= requested_end:
            print(f"Region {i+1:02d} [{series}]: already covers requested range through {latest} — skipping")
            summary[i + 1] = {'status': 'skipped', 'files': existing}
            continue

        try:
            cutout = a.jsoc.Cutout(bl, top_right=tr, tracking=True)
            query_args = [a.Time(fetch_start, time_end), a.jsoc.Series(series), a.jsoc.Notify(notify_email), cutout]
            if sample is not None:
                query_args.append(a.Sample(sample))
            result = Fido.search(*query_args)

            resumed = ' (resuming)' if latest is not None else ''
            print(f"Region {i+1:02d} [{series}]: {len(result[0])} new frames found{resumed}")
            files = Fido.fetch(result, path=region_dir)
            print(f"  -> {len(files)} files saved to {region_dir}")
            summary[i + 1] = {'status': 'ok', 'files': files}
        except Exception as exc:
            print(f"Region {i+1:02d} [{series}]: FAILED — {exc!r}")
            summary[i + 1] = {'status': 'failed', 'error': str(exc)}

    return summary


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

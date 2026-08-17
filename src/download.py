"""Stage 1 — acquisition: find an active region, fetch its cutouts, build cubes.

Everything that talks to HEK or JSOC lives here. `01A_download_data.ipynb` is the only
driver; the parameters it used to hardcode (which ARs, which series, per-AR overrides)
are in `config`.

    from src import config, download
    for ar in config.ACTIVE_REGIONS:
        bl, tr, t0, t1 = download.locate_ar_window(ar['noaa'], ar['date'],
                                                   n_days=config.N_DAYS,
                                                   **config.BOX_OVERRIDE.get(ar['noaa'], {}))

Nothing here is imported at module level from sunpy — the Fido machinery is slow to
import and only two functions need it.
"""

from __future__ import annotations

import pathlib
import re
from datetime import timedelta

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord

from .utilities import make_cube, parse_frame_timestamp

# Each JSOC series -> the glob that finds its files, the label used in the output cube
# filename, and its native cadence.
SERIES_META = {
    'hmi.Ic_45s':  {'glob': 'hmi.ic_45s.*.continuum.fits',   'label': 'continuum',   'cadence_seconds': 45},
    'hmi.M_45s':   {'glob': 'hmi.m_45s.*.magnetogram.fits',  'label': 'magnetogram', 'cadence_seconds': 45},
    'hmi.V_45s':   {'glob': 'hmi.v_45s.*.fits',              'label': 'dopplergram', 'cadence_seconds': 45},
    'hmi.Ic_720s': {'glob': 'hmi.ic_720s.*.continuum.fits',  'label': 'continuum',   'cadence_seconds': 720},
    'hmi.M_720s':  {'glob': 'hmi.m_720s.*.magnetogram.fits', 'label': 'magnetogram', 'cadence_seconds': 720},
    'hmi.V_720s':  {'glob': 'hmi.v_720s.*.fits',             'label': 'dopplergram', 'cadence_seconds': 720},
}

# Kept under the old private name too: `doppler_calibration` reads it.
_SERIES_META = SERIES_META


# ── finding a region ────────────────────────────────────────────────────────

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


def parse_ds9_boxes(ds9_file):
    """Parse `box(x, y, w, h, angle)` lines from a DS9 region file.

    Returns a list of (x_centre, y_centre, width, height) in 1-indexed pixels, skipping
    degenerate boxes. Pair with `ds9_box_to_hpc` to get JSOC cutout corners.
    """
    boxes = []
    for line in pathlib.Path(ds9_file).read_text().strip().splitlines():
        m = re.match(r'box\(([^)]+)\)', line)
        if not m:
            continue
        x_c, y_c, w, h, _ = map(float, m.group(1).split(','))
        if w > 0 and h > 0:
            boxes.append((x_c, y_c, w, h))
    return boxes


# ── downloading ───────────────────────────────────────────────────────────

def download_single_region(result, output_dir):
    """Fetch files from a Fido search result into output_dir."""
    from sunpy.net import Fido

    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    files = Fido.fetch(result, path=output_dir)
    print(files)
    return files


# ── Unified download and cube builder ────────────────────────────────────────

def download_regions(regions_hpc, base_dir, time_start, time_end, notify_email,
                     series, region_indexes=None, sample=None, tracking=True):
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
    tracking : bool
        Whether JSOC should rotate the cutout box with the solar surface. True for
        following an active region. False pins the box to fixed helioprojective
        coordinates, which is what a disk-centre reference region needs — see
        src.doppler_calibration.download_disk_center.

    Returns
    -------
    dict mapping 1-based region index -> {'status': 'ok'|'skipped'|'partial'|'failed', ...}
    'partial' means some individual files still failed after one retry; their URLs are
    under 'failed_urls'. Treat it as seriously as 'failed' — a missing mid-window frame
    silently shifts every later frame out of step with the other series.
    """
    from sunpy.net import Fido, attrs as a

    base_dir = pathlib.Path(base_dir)
    meta = SERIES_META[series]
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
            cutout = a.jsoc.Cutout(bl, top_right=tr, tracking=tracking)
            query_args = [a.Time(fetch_start, time_end), a.jsoc.Series(series), a.jsoc.Notify(notify_email), cutout]
            if sample is not None:
                query_args.append(a.Sample(sample))
            result = Fido.search(*query_args)

            n_found = len(result[0]) if len(result) else 0
            resumed = ' (resuming)' if latest is not None else ''
            print(f"Region {i+1:02d} [{series}]: {n_found} new frames found{resumed}")
            if n_found == 0:
                summary[i + 1] = {'status': 'ok', 'files': []}
                continue

            files = Fido.fetch(result, path=region_dir)
            # parfive reports per-file failures on .errors instead of raising; left unchecked
            # these become silent *mid-window* gaps that later misalign the three series
            # against each other. Retry once (re-fetching a Results retries only its errors,
            # and returns every path, old successes included).
            if files.errors:
                print(f"  {len(files.errors)} file(s) failed — retrying")
                files = Fido.fetch(files)

            print(f"  -> {len(files)} files saved to {region_dir}")
            if files.errors:
                failed_urls = [err.url for err in files.errors]
                print(f"  PARTIAL: {len(failed_urls)} file(s) still missing after retry")
                summary[i + 1] = {'status': 'partial', 'files': files, 'failed_urls': failed_urls}
            else:
                summary[i + 1] = {'status': 'ok', 'files': files}
        except Exception as exc:
            print(f"Region {i+1:02d} [{series}]: FAILED — {exc!r}")
            summary[i + 1] = {'status': 'failed', 'error': str(exc)}

    return summary


def make_cubes(regions_hpc, base_dir, series, region_indexes=None, overwrite=True):
    """Build one FITS cube per region from downloaded frames.

    Parameters
    ----------
    series : str
        JSOC series name — e.g. 'hmi.Ic_45s', 'hmi.M_45s', 'hmi.V_45s'.
    region_indexes : list[int] or None
        1-based region numbers to process. None processes all regions.
    overwrite : bool
        Rebuild cubes that already exist. False skips them, which makes re-running a
        notebook cheap once the frames on disk have stopped changing.
    """
    if series not in SERIES_META:
        raise ValueError(f"Unknown series '{series}'. Known: {list(SERIES_META)}")
    meta = SERIES_META[series]
    base_dir = pathlib.Path(base_dir)
    for i in range(1, len(regions_hpc) + 1):
        if region_indexes is not None and i not in region_indexes:
            continue
        region_dir = base_dir / f'region_{i:02d}'
        make_cube(
            region_dir / meta['glob'],
            base_dir / f'region_{i:02d}_{meta["label"]}_cube.fits',
            overwrite=overwrite,
        )


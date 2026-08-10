import sys
import pathlib

# Ensure the notebook can import the local src package from the project root.
project_root = pathlib.Path.cwd()
if not (project_root / 'src').exists():
    project_root = project_root.parent
sys.path.insert(0, str(project_root.resolve()))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sunpy.net import Fido, attrs as a
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
import sunpy.map


from src.utilities import make_cube


# for now just dowloading 11536 and do tests with it

active_regions = [
    #{'noaa': 11039, 'date': '2010-01-01'},
    #{'noaa': 11041, 'date': '2010-01-26'},
    {'noaa': 11106, 'date': '2010-09-16'},
    # 11117's two catalog dates (2010-10-27, 2010-10-28) are one continuous AR passage —
    # a multi-day window anchored at the earlier date already covers the later one, so
    # they're merged into a single entry instead of being processed twice.
    {'noaa': 11117, 'date': '2010-10-27'},
    {'noaa': 11363, 'date': '2011-12-06'},
    {'noaa': 11536, 'date': '2012-07-31'},
]

from src.utilities import locate_ar_window

N_DAYS = 4  # minimum continuous days of data to fetch per AR

# 11039/11041 predate HMI science data (starts ~2010-05-01) — genuinely nothing to fetch.
# Dropped here, before any HEK or JSOC work, rather than after the download loop: leaving
# them in cost a HEK query plus three JSOC export requests each, every run, to rediscover
# that there is no data.
# 11117 stays: it's handled via series_override below (45s instead of the broken 720s window).
exclude_noaa = [11039, 11041]
active_regions = [ar for ar in active_regions if ar['noaa'] not in exclude_noaa]

# Per-AR overrides for the box geometry — keyword arguments forwarded straight to
# locate_ar_window. Comment an entry out to turn that override off. (This lives here rather
# than beside series_override/sample_override further down only because it has to be defined
# before the loop that uses it.)
#
# 11536: SWPC under-reports this group badly. Measured from the frames already on disk, it
# grows from 165" to 220" across and sits ~35" left of the reported centroid, so the default
# median sizing (200" square) clips it on the left in *every* frame. 'max' alone (244")
# still leaves ~0" margin at the start, so the padding goes up too.
box_override = {
    11536: {'statistic': 'max', 'padding': 60 * u.arcsec},   # -> 304" square
}

for ar in active_regions:
    ar['bl'], ar['tr'], ar['time_start'], ar['time_end'] = locate_ar_window(
        ar['noaa'], ar['date'], n_days=N_DAYS, **box_override.get(ar['noaa'], {}))
    print(f"NOAA {ar['noaa']} ({ar['date']}): window {ar['time_start']} .. {ar['time_end']}  "
          f"bl={ar['bl'].Tx:.1f},{ar['bl'].Ty:.1f}  tr={ar['tr'].Tx:.1f},{ar['tr'].Ty:.1f}")


from src.scripts import download_regions, make_cubes

notify_email = 'thomas.quamtum@gmail.com'  # must be a JSOC-registered export email
series_list  = ['hmi.Ic_720s', 'hmi.V_720s', 'hmi.M_720s']

# JSOC's *_720s series 500-errors for any query touching 2010-10-26..29 (a gap/bug in their
# 720s index for that window) even though the underlying 45s data is fine. NOAA 11117's
# whole N_DAYS window falls in/around that gap, so fetch it at 45s cadence instead.
series_override = {
    11117: ['hmi.Ic_45s', 'hmi.V_45s', 'hmi.M_45s'],
}

# 45s cadence over N_DAYS days is far more data than everyone else's 720s — downsample
# NOAA 11117 server-side to ~360s (every 8th 45s frame) to keep its volume comparable.
sample_override = {
    11117: 720 * u.s,
}

dry_run = False

all_summaries = []  # (noaa, date, series, summary) for the consolidated report below

for ar in active_regions:
    region_base = pathlib.Path('../data/raw') / f"NOAA_{ar['noaa']}_{ar['date']}"
    regions_hpc = [(ar['bl'], ar['tr'])]
    ar_series   = series_override.get(ar['noaa'], series_list)
    ar_sample   = sample_override.get(ar['noaa'])

    if dry_run:
        print(f"[dry run] NOAA {ar['noaa']} ({ar['date']}) -> {region_base} "
              f"({ar['time_start']} .. {ar['time_end']}, {ar_series}, sample={ar_sample})")
        continue

    for series in ar_series:
        summary = download_regions(regions_hpc, region_base, ar['time_start'], ar['time_end'],
                                    notify_email, series, sample=ar_sample)
        all_summaries.append((ar['noaa'], ar['date'], series, summary))

# Consolidated report — useful after a long multi-AR, multi-day run.
# 'partial' matters as much as 'failed': a missing *mid-window* frame is not a smaller
# dataset, it knocks that series out of step with the other two from that point on.
# Re-run this cell to resume; only clear the region and start over if the box changed.
by_status = {}
for noaa, date, series, summary in all_summaries:
    for region, info in summary.items():
        by_status.setdefault(info['status'], []).append((noaa, date, series, region, info))

for status in ('failed', 'partial'):
    entries = by_status.get(status, [])
    if entries:
        print(f"\n{status.upper()} downloads:")
        for noaa, date, series, region, info in entries:
            detail = info.get('error') or f"{len(info.get('failed_urls', []))} file(s) missing"
            print(f"  NOAA {noaa} ({date}) region_{region:02d} [{series}]: {detail}")

if all_summaries and not (by_status.get('failed') or by_status.get('partial')):
    print(f"\nAll {len(all_summaries)} region/series downloads complete, no missing files.")

if True:
    for ar in active_regions:
        region_base = pathlib.Path('../data/raw') / f"NOAA_{ar['noaa']}_{ar['date']}"
        regions_hpc = [(ar['bl'], ar['tr'])]
        for series in series_override.get(ar['noaa'], series_list):
            make_cubes(regions_hpc, region_base, series)
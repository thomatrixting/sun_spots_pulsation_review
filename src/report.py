"""Stage 5 — the one-page DS0N summary, built from what `03B` already wrote to disk.

This exists because the summary used to be hand-written HTML with absolute `file:///`
paths typed into it per dataset (`anim_sumary_arch.html`, `sumary_DS01.html`). Two things
went wrong with that and both are structural: the paths rotted the moment the project
moved — `sumary_DS01.html` still points at a `/mnt/ubuntu/home/...` that no longer exists
— and a peak table cannot be hand-maintained at all, because the numbers change every time
a threshold is retuned and `03B` re-runs.

So the page is generated from the directory instead: whatever `03B` last wrote is what it
shows, and it uses paths relative to itself, so the whole `sebastian_sun_spots` folder can
be copied anywhere and still open.
"""

from __future__ import annotations

import html
import pathlib
import re

import pandas as pd

from .config import DS0N_PROCESSED_DIR

#: The page itself, written into `processed_dir`. This is the file that was previously
#: maintained by hand, and it keeps its name so existing bookmarks still open it.
SUMMARY_PAGE = 'anim_sumary_arch.html'

#: What each dataset section shows, in order. Relative to the dataset directory.
ANIMATION = 'anim.html'

#: (caption, filename) of the figures shown under the animation, in order. Both come from
#: step 4; a dataset whose step 4 was run before `area_vs_time.png` existed simply shows
#: the figures it has, which is why each one is tested for separately below.
FIGURES = (
    ('Mean field and mean velocity vs time', 'plots/time_series.png'),
    ('Region area vs time', 'plots/area_vs_time.png'),
)
TIME_SERIES = FIGURES[0][1]

#: (label, filename) of the peak tables `spectra.plot_spectra_compare` saves.
PEAK_TABLES = (
    ('Dopplergram', 'plots/psd_peaks_dopplergram.csv'),
    ('Magnetogram residual', 'plots/psd_peaks_magnetogram_residual.csv'),
)

#: Only these regions, only this many peaks each — the quiet sun is deliberately absent
#: here for the same reason it is absent from the figure above it.
PEAK_REGIONS = ('Umbra', 'Penumbra')
N_PEAKS = 2


def has_output(ds_dir: pathlib.Path) -> bool:
    """Whether `03B` actually produced anything for this dataset.

    An empty variant directory is not a variant. `DS01_RemPol` and `DS06_A` exist but hold
    nothing, and treating them as real is what would make `DS01` and `DS06` disappear from
    the page in favour of two blank sections.
    """
    return (ds_dir / ANIMATION).exists() or (ds_dir / TIME_SERIES).exists()


#: A dataset as delivered, `DS00`, and a per-spot variant of one, `DS00_A`.
BASE = re.compile(r'^DS\d+$')
VARIANT = re.compile(r'^(DS\d+)_([A-Z])$')


def datasets_to_show(processed_dir: pathlib.Path, has_data=None) -> list[pathlib.Path]:
    """The dataset directories worth showing, in order.

    **A lettered variant supersedes its base.** Once a region has been split into `DS00_A` /
    `DS00_B` — one per sunspot — the un-suffixed `DS00` is the older whole-box run, and
    showing it alongside invites reading the two as different spots when they are the same
    data segmented differently. A base with no lettered variant is shown as it is.

    Only a single letter counts as a variant, which is what keeps `DS01_RemPol` from
    displacing `DS01`: it is a processing experiment, not a second spot. Directories like
    it are left out of the list entirely rather than shown as if they were another region.

    `has_data` decides what makes a directory usable, because that differs by caller: the
    summary page needs an animation or a figure, `04` needs a `metrics.csv`. Defaults to
    `has_output`.
    """
    has_data = has_output if has_data is None else has_data
    directories = sorted(d for d in processed_dir.iterdir() if d.is_dir())

    bases: dict[str, list[pathlib.Path]] = {}
    for directory in directories:
        match = VARIANT.match(directory.name)
        if match:
            bases.setdefault(match.group(1), []).append(directory)
        elif BASE.match(directory.name):
            bases.setdefault(directory.name, []).append(directory)
        # Anything else — `DS01_RemPol`, a stray `plots/` — is neither a dataset nor a spot
        # within one, and is left out rather than shown as if it were another region.

    chosen = []
    for base in sorted(bases):
        group = [d for d in bases[base] if has_data(d)]
        lettered = [d for d in group if VARIANT.match(d.name)]
        chosen.extend(lettered if lettered else [d for d in group if d.name == base])
    return chosen


def peak_rows(ds_dir: pathlib.Path) -> list[dict]:
    """The first `N_PEAKS` peaks of each region in `PEAK_REGIONS`, both quantities.

    Returns [] when the dataset has no peak tables — a section is still worth showing for
    its animation and figure, so a missing table is a gap in the page, not an error.
    """
    rows = []
    for quantity, filename in PEAK_TABLES:
        path = ds_dir / filename
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        for region in PEAK_REGIONS:
            selection = frame[frame['region'] == region].nsmallest(N_PEAKS, 'rank')
            for _, row in selection.iterrows():
                rows.append({
                    'quantity': quantity,
                    'region': region,
                    'rank': int(row['rank']),
                    'period_h': float(row['period_min']) / 60,
                    'period_min': float(row['period_min']),
                    'freq_mhz': float(row['freq_mhz']),
                    'relative': float(row['relative']),
                })
    return rows


def _peak_table_html(rows: list[dict]) -> str:
    if not rows:
        return '<p class="missing">No peak table — re-run Step 5 for this dataset.</p>'

    body = '\n'.join(
        '                    <tr>'
        f'<td>{html.escape(r["quantity"])}</td>'
        f'<td>{html.escape(r["region"])}</td>'
        f'<td class="num">{r["rank"]}</td>'
        f'<td class="num">{r["period_h"]:.2f}</td>'
        f'<td class="num">{r["period_min"]:.1f}</td>'
        f'<td class="num">{r["freq_mhz"]:.4f}</td>'
        f'<td class="num">{r["relative"]:.4f}</td>'
        '</tr>'
        for r in rows
    )
    return f"""<table class="peaks">
                <thead>
                    <tr><th>Quantity</th><th>Region</th><th>Rank</th><th>Period (h)</th>
                        <th>Period (min)</th><th>Freq (mHz)</th><th>Rel. power</th></tr>
                </thead>
                <tbody>
{body}
                </tbody>
            </table>"""


def _section_html(ds_dir: pathlib.Path) -> str:
    name = html.escape(ds_dir.name)
    parts = [f'        <div class="section">\n            <h2>Dataset {name}</h2>']

    if (ds_dir / ANIMATION).exists():
        parts.append(f'            <iframe class="animation-frame" '
                     f'src="{name}/{ANIMATION}"></iframe>')
    for caption, filename in FIGURES:
        if not (ds_dir / filename).exists():
            continue
        parts.append('            <figure class="plot">\n'
                     f'                <img class="plot-image" src="{name}/{filename}" '
                     f'alt="{name} — {html.escape(caption)}">\n'
                     f'                <figcaption>{html.escape(caption)}</figcaption>\n'
                     '            </figure>')

    parts.append(f'            <h3>Strongest {N_PEAKS} peaks per region</h3>')
    parts.append(f'            {_peak_table_html(peak_rows(ds_dir))}')
    parts.append('        </div>')
    return '\n'.join(parts)


STYLE = """        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: #f5f5f7;
            color: #333;
            margin: 0;
            padding: 40px 20px;
        }

        h1 {
            text-align: center;
            color: #1d1d1f;
            margin-bottom: 50px;
            font-size: 2.5rem;
        }

        .container {
            max-width: 2000px;
            margin: 0 auto;
        }

        .section {
            margin-bottom: 80px;
            background: #ffffff;
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
            padding: 30px;
        }

        .section h2 {
            font-size: 1.8rem;
            margin-top: 0;
            margin-bottom: 20px;
            color: #0066cc;
            border-bottom: 2px solid #f5f5f7;
            padding-bottom: 10px;
        }

        .section h3 {
            font-size: 1.1rem;
            color: #1d1d1f;
            margin: 28px 0 12px;
        }

        .animation-frame {
            width: 100%;
            height: 85vh;
            border: none;
            border-radius: 8px;
            background-color: #fff;
            margin-bottom: 25px;
        }

        figure.plot {
            margin: 0 0 25px;
        }

        figure.plot:last-of-type {
            margin-bottom: 0;
        }

        figure.plot figcaption {
            color: #6e6e73;
            font-size: 0.9rem;
            margin-top: 8px;
        }

        .plot-image {
            display: block;
            width: 100%;
            height: auto;
            max-width: 100%;
            border-radius: 8px;
            border: 1px solid #e5e5e7;
        }

        table.peaks {
            border-collapse: collapse;
            font-size: 0.95rem;
            min-width: 640px;
        }

        table.peaks th, table.peaks td {
            border-bottom: 1px solid #e5e5e7;
            padding: 7px 14px;
            text-align: left;
        }

        table.peaks th {
            background: #f5f5f7;
            font-weight: 600;
        }

        table.peaks td.num {
            text-align: right;
            font-variant-numeric: tabular-nums;
        }

        .missing {
            color: #8a8a8e;
            font-style: italic;
        }"""


def build_ds0n_summary(processed_dir=None, out_path=None,
                       title='Sun Spot Pulsation Review — DS0N summary') -> pathlib.Path:
    """Write the one-page summary and return its path.

    Every link is relative to the page, which sits in `processed_dir` alongside the
    dataset folders, so the folder stays self-contained when it is copied or shared.
    """
    processed_dir = pathlib.Path(processed_dir or DS0N_PROCESSED_DIR)
    out_path = pathlib.Path(out_path or processed_dir / SUMMARY_PAGE)

    shown = datasets_to_show(processed_dir)
    sections = '\n\n'.join(_section_html(d) for d in shown)

    out_path.write_text(f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{html.escape(title)}</title>
    <style>
{STYLE}
    </style>
</head>
<body>

    <div class="container">
        <h1>{html.escape(title)}</h1>

{sections}
    </div>

</body>
</html>
""", encoding='utf-8')

    print(f'Summary saved → {out_path}  ({len(shown)} datasets: '
          f'{", ".join(d.name for d in shown)})')
    return out_path

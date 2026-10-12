"""Resumable runs of the DS0N spot tracker: `scripts/track_spots.py` and the metadata
that lets it skip what is already computed.

`tracking.process_region` (what `03C` calls) redoes a region from scratch every time.
Here every output folder carries a ``tracking_meta.json`` saying how it was made, and a
run only redoes what that metadata no longer matches:

* **Region** ``<out>/DSNN/``: the full `TrackingParams`, whether the per-spot pipeline ran,
  and the mask fingerprint of each kept spot. If the params match and every kept spot is
  up to date, the region is skipped without even loading its cubes.
* **Spot** ``<out>/DSNN_X/``: the params of the run that produced it, its mask
  fingerprint and the pipeline settings. Written only after the pipeline succeeds, so a
  spot that crashed half way has no metadata and is redone next time.

When a region is re-tracked (other params), a spot's pipeline is still skipped if its
masks came out identical — same fingerprint, same thresholds, same pipeline settings —
because its outputs would be identical too. `recompute=True` ignores all metadata.
"""

from __future__ import annotations

import dataclasses
import datetime
import hashlib
import json
import pathlib
import re
import subprocess
import traceback

import numpy as np
import pandas as pd

from . import config, report, tracking
from .config import DS0N_DEFAULTS
from .tracking import TrackingParams

META_FILE = 'tracking_meta.json'
# Bump when `tracking.run_spot_pipeline` changes what it writes, so old spots are redone.
PIPELINE_VERSION = 1
PIPELINE = {'version': PIPELINE_VERSION, 'use_absolute_mag': True}
REGION = re.compile(r'^DS\d+$')
SUMMARY_TITLE = 'Sun Spot Pulsation Review — DS0N automatic spot tracking'


# ── params and metadata ─────────────────────────────────────────────────────

def load_params(path=None, overrides: dict | None = None) -> TrackingParams:
    """`TrackingParams` from a JSON file (missing keys keep their defaults), then
    `overrides` on top. An unknown key is an error, not silently ignored."""
    values = json.loads(pathlib.Path(path).read_text()) if path else {}
    values.update({k: v for k, v in (overrides or {}).items()})
    known = {f.name for f in dataclasses.fields(TrackingParams)}
    unknown = set(values) - known
    if unknown:
        raise ValueError(f'unknown tracking parameter(s): {sorted(unknown)}')
    return TrackingParams(**values)


def params_dict(params: TrackingParams) -> dict:
    # Through JSON so it compares equal to what a meta file reads back.
    return json.loads(json.dumps(dataclasses.asdict(params)))


def fingerprint(umbra: np.ndarray, penumbra: np.ndarray) -> str:
    """sha1 of a spot's (frame, pixel) mask arrays, as `tracking.mask_arrays` builds them."""
    h = hashlib.sha1()
    for a in (umbra, penumbra):
        a = np.ascontiguousarray(a, dtype=np.int32)
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def _code_version() -> str | None:
    try:
        out = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], cwd=config.PROJECT_ROOT,
                             capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except Exception:                                # noqa: BLE001 — informational only
        return None


def _now() -> str:
    return datetime.datetime.now().isoformat(timespec='seconds')


def read_meta(folder) -> dict | None:
    path = pathlib.Path(folder) / META_FILE
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def write_meta(folder, meta: dict) -> None:
    (pathlib.Path(folder) / META_FILE).write_text(json.dumps(meta, indent=2) + '\n')


def spot_up_to_date(spot_dir, fp: str, params: TrackingParams) -> bool:
    """The folder already holds this exact spot, made with the same thresholds and pipeline."""
    meta = read_meta(spot_dir)
    if meta is None or not (pathlib.Path(spot_dir) / 'metrics.csv').exists():
        return False
    p = meta.get('params', {})
    return (meta.get('mask_fingerprint') == fp
            and p.get('umbra_thresh') == params.umbra_thresh
            and p.get('penumbra_thresh') == params.penumbra_thresh
            and meta.get('pipeline') == PIPELINE)


def region_up_to_date(region_dir, out_root, params: TrackingParams, run_pipeline: bool) -> bool:
    region_dir, out_root = pathlib.Path(region_dir), pathlib.Path(out_root)
    meta = read_meta(region_dir)
    if meta is None or meta.get('params') != params_dict(params):
        return False
    if not all((region_dir / f).exists()
               for f in ('spots.csv', 'masks.npz', report.REGION_ANIMATION)):
        return False
    if not run_pipeline:
        return True
    if not meta.get('run_pipeline'):
        return False
    return all(spot_up_to_date(out_root / d, fp, params)
               for d, fp in meta.get('kept_spots', {}).items())


def _spot_meta(ds_name: str, spot_label: str, fp: str, params: TrackingParams,
               computed_at: str, code_version: str | None, **extra) -> dict:
    return {'dataset': ds_name, 'spot': spot_label, 'mask_fingerprint': fp,
            'params': params_dict(params), 'pipeline': PIPELINE,
            'computed_at': computed_at, 'code_version': code_version, **extra}


def _region_meta(ds_name: str, params: TrackingParams, run_pipeline: bool, kept: dict,
                 code_version: str | None, **extra) -> dict:
    return {'dataset': ds_name, 'params': params_dict(params), 'run_pipeline': run_pipeline,
            'kept_spots': kept, 'created_at': _now(), 'code_version': code_version, **extra}


def _leftover_dirs(out_root: pathlib.Path, ds_name: str, kept: set[str]) -> list[str]:
    """Spot folders of this region on disk that the current tracking did not keep."""
    out = []
    for d in sorted(out_root.glob(f'{ds_name}_*')):
        m = report.VARIANT.match(d.name)
        if d.is_dir() and m and m.group(1) == ds_name and d.name not in kept:
            out.append(d.name)
    return out


# ── running ─────────────────────────────────────────────────────────────────

def run_region(ds_dir, out_root, params: TrackingParams, run_pipeline: bool = True,
               recompute: bool = False) -> dict:
    """`tracking.process_region`, but skipping whatever the metadata says is current.

    Returns a status dict: ``status`` ('up to date' or 'tracked'), and the spots whose
    pipeline ``ran``, was ``reused`` or ``failed``, plus ``leftover`` folders.
    """
    from .loaders import load_ds0n_region, verify_cadence

    ds_dir, out_root = pathlib.Path(ds_dir), pathlib.Path(out_root)
    ds_out = out_root / ds_dir.name
    if not recompute and region_up_to_date(ds_out, out_root, params, run_pipeline):
        print(f'{ds_dir.name}: up to date, skipped')
        return {'status': 'up to date', 'ran': [], 'reused': [], 'failed': {}, 'leftover': []}

    _, gaps = verify_cadence(ds_dir, expected_s=DS0N_DEFAULTS['cadence_s'])
    data = load_ds0n_region(ds_dir, umbra_thresh=params.umbra_thresh,
                            penumbra_thresh=params.penumbra_thresh, cadence_s=gaps,
                            filter_mask=False, mag_filter=None, normalize_mag_by_mu=True,
                            drop_out_of_order=True)
    result = tracking.track_spots(data['cube_cont'], data['time_h'], ds_dir.name, params)
    tracking.plot_tracking_overview(data['cube_cont'], result, data['time_h'],
                                    plots_dir=ds_out / 'plots')
    tracking.save_region_animation(data, result, ds_out / report.REGION_ANIMATION)

    code_version = _code_version()
    kept, ran, reused, failed = {}, [], [], {}
    for s in result.kept():
        d_name = tracking.dir_name(ds_dir.name, s)
        label = tracking.display_name(ds_dir.name, s)
        spot_dir = out_root / d_name
        fp = fingerprint(*tracking.mask_arrays(s))
        kept[d_name] = fp
        if not run_pipeline:
            continue
        if not recompute and spot_up_to_date(spot_dir, fp, params):
            meta = read_meta(spot_dir)
            write_meta(spot_dir, {**meta, 'params': params_dict(params), 'confirmed_at': _now()})
            reused.append(label)
            print(f'── {label}: same masks as before, pipeline skipped')
            continue
        print(f'── {label}  ({len(s.frames)} frames)')
        try:
            tracking.run_spot_pipeline(tracking.spot_data(data, result, s.name), spot_dir)
        except Exception as exc:                     # noqa: BLE001 — record, keep going
            failed[label] = f'{type(exc).__name__}: {exc}'
            traceback.print_exc()
            continue
        write_meta(spot_dir, _spot_meta(ds_dir.name, label, fp, params, _now(), code_version))
        ran.append(label)

    result.table['pipeline_error'] = result.table['spot'].map(failed).fillna('')
    tracking.save_tracking_outputs(result, ds_out)
    write_meta(ds_out, _region_meta(ds_dir.name, params, run_pipeline, kept, code_version))

    leftover = _leftover_dirs(out_root, ds_dir.name, set(kept))
    if leftover:
        print(f'{ds_dir.name}: folders from earlier runs, not part of this one (kept on '
              f'disk, left out of the summary): {leftover}')
    return {'status': 'tracked', 'ran': ran, 'reused': reused, 'failed': failed,
            'leftover': leftover}


def collect_spots(out_root) -> pd.DataFrame:
    """Every region's `spots.csv` under `out_root`, this run's or not, in one table."""
    out_root = pathlib.Path(out_root)
    tables = [pd.read_csv(d / 'spots.csv').assign(dataset=d.name)
              for d in sorted(out_root.iterdir(), key=lambda p: (len(p.name), p.name))
              if d.is_dir() and REGION.match(d.name) and (d / 'spots.csv').exists()]
    return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()


def run(out_root, params: TrackingParams, datasets: list[str] | None = None,
        n_datasets: int | None = None, run_pipeline: bool = True, recompute: bool = False,
        summary: bool = True, title: str = SUMMARY_TITLE) -> dict:
    """Every raw DS0N region (or `datasets`, or the first `n_datasets` that load), then
    `spots_all.csv` and the summary page over everything in `out_root`."""
    out_root = pathlib.Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    statuses, skipped = {}, {}
    for ds_dir in config.ds0n_regions(datasets):
        if n_datasets is not None and len(statuses) >= n_datasets:
            break
        print(f'\n=================== {ds_dir.name} ===================')
        try:
            statuses[ds_dir.name] = run_region(ds_dir, out_root, params, run_pipeline, recompute)
        except Exception as exc:                     # noqa: BLE001 — a broken delivery is skipped
            skipped[ds_dir.name] = f'{type(exc).__name__}: {exc}'
            print(f'SKIPPED {ds_dir.name}: {skipped[ds_dir.name]}')

    all_spots = collect_spots(out_root)
    if not all_spots.empty:
        all_spots.to_csv(out_root / 'spots_all.csv', index=False)

    print('\n=================== run summary ===================')
    rows = [{'dataset': n, 'status': s['status'], 'pipeline_ran': len(s['ran']),
             'pipeline_reused': len(s['reused']), 'failed': len(s['failed'])}
            for n, s in statuses.items()]
    if rows:
        print(pd.DataFrame(rows).to_string(index=False))
    for n, s in statuses.items():
        for spot, err in s['failed'].items():
            print(f'FAILED {spot}: {err}')
    if skipped:
        print(f'could not load: {skipped}')
    if not all_spots.empty:
        print(f"{int(all_spots.kept.sum())} kept spots of {len(all_spots)} tracked in "
              f"{all_spots.dataset.nunique()} regions in {out_root}")

    page = report.build_ds0n_summary(processed_dir=out_root, title=title) if summary else None
    return {'statuses': statuses, 'skipped': skipped, 'spots': all_spots, 'summary': page}


# ── one-time backfill ───────────────────────────────────────────────────────

def stamp_existing(out_root, params: TrackingParams, run_pipeline: bool = True,
                   note: str = 'backfilled: outputs made by notebook 03C before metadata existed') -> dict:
    """Write metadata for outputs made before it existed, declaring they came from `params`.

    Only what can be checked is stamped: a kept spot gets metadata if its folder holds
    `metrics.csv` and `anim.html` written during its region's run (between the region
    animation and `spots.csv`, the order `process_region` writes them in) and its row
    has no pipeline error. The fingerprint comes from the region's `masks.npz`.
    """
    out_root = pathlib.Path(out_root)
    code_version = _code_version()
    stamped, refused = [], {}
    regions = sorted((d for d in out_root.iterdir() if d.is_dir() and REGION.match(d.name)),
                     key=lambda p: (len(p.name), p.name))
    for region in regions:
        spots_csv, masks_npz = region / 'spots.csv', region / 'masks.npz'
        anim = region / report.REGION_ANIMATION
        if not (spots_csv.exists() and masks_npz.exists() and anim.exists()):
            refused[region.name] = 'no spots.csv / masks.npz / region animation'
            continue
        t_from, t_to = anim.stat().st_mtime, spots_csv.stat().st_mtime
        table = pd.read_csv(spots_csv, keep_default_na=False)
        masks = np.load(masks_npz)
        kept = {}
        for row in table[table.kept.astype(str) == 'True'].itertuples():
            fp = fingerprint(masks[f'{row.dir}_umbra'], masks[f'{row.dir}_penumbra'])
            kept[row.dir] = fp
            if not run_pipeline:
                continue
            spot_dir = out_root / row.dir
            metrics, spot_anim = spot_dir / 'metrics.csv', spot_dir / 'anim.html'
            if row.pipeline_error:
                refused[row.spot] = f'pipeline error: {row.pipeline_error}'
            elif not (metrics.exists() and spot_anim.exists()):
                refused[row.spot] = 'metrics.csv or anim.html missing'
            elif not all(t_from <= f.stat().st_mtime <= t_to for f in (metrics, spot_anim)):
                refused[row.spot] = 'outputs not written by the run that made spots.csv'
            else:
                made = datetime.datetime.fromtimestamp(metrics.stat().st_mtime)
                write_meta(spot_dir, _spot_meta(region.name, row.spot, fp, params,
                                                made.isoformat(timespec='seconds'),
                                                code_version, note=note))
                stamped.append(row.spot)
        write_meta(region, _region_meta(region.name, params, run_pipeline, kept,
                                        code_version, note=note))
    print(f'stamped {len(stamped)} spots in {len(regions)} regions')
    for k, why in refused.items():
        print(f'  not stamped {k}: {why}')
    return {'stamped': stamped, 'refused': refused}

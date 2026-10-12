"""Track the spots of the DS0N regions and run the per-spot pipeline — `03C` as a script.

Already computed regions and spots are skipped: each output folder holds a
``tracking_meta.json`` with the parameters that produced it (see `src/tracking_runner.py`).

Examples (from the project root, in the `sun-spots` env):

    python scripts/track_spots.py                         # everything, params from tracking_params.json
    python scripts/track_spots.py --datasets DS00 DS05    # only these
    python scripts/track_spots.py --n-datasets 10 --min-spot-px 300
    python scripts/track_spots.py --params my.json --out-root data/processed/sebastian_sun_spots/automatic_test2
    python scripts/track_spots.py --datasets DS00 --recompute   # ignore metadata, redo everything
    python scripts/track_spots.py --no-pipeline           # tracking + overview only (fast, for tuning)
"""

from __future__ import annotations

import argparse
import dataclasses
import pathlib
import sys

import matplotlib

matplotlib.use('Agg')

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import config, tracking_runner  # noqa: E402
from src.tracking import TrackingParams  # noqa: E402

DEFAULT_PARAMS = pathlib.Path(__file__).resolve().parent / 'tracking_params.json'


def _value(annotation: str):
    if annotation == 'int':
        return int
    if 'None' in annotation:
        return lambda s: None if s.lower() == 'none' else float(s)
    return float


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--params', type=pathlib.Path, default=DEFAULT_PARAMS,
                    help=f'JSON with the tracking parameters (default: {DEFAULT_PARAMS.name})')
    ap.add_argument('--datasets', nargs='+', metavar='DSNN', help='only these datasets')
    ap.add_argument('--n-datasets', type=int, help='stop after this many datasets that load')
    ap.add_argument('--out-root', type=pathlib.Path,
                    default=config.DS0N_AUTO_DIR,
                    help='output folder (default: data/processed/sebastian_sun_spots/automatic)')
    ap.add_argument('--no-pipeline', action='store_true',
                    help='track and write the overview only, no per-spot analysis')
    ap.add_argument('--recompute', action='store_true',
                    help='ignore the saved metadata and recompute everything selected')
    ap.add_argument('--no-summary', action='store_true', help='do not rebuild the summary page')
    ap.add_argument('--stamp-existing', action='store_true',
                    help='one-time: write metadata for outputs made before it existed, '
                         'declaring they came from --params; computes nothing')

    overrides = ap.add_argument_group('parameter overrides (on top of --params)')
    for f in dataclasses.fields(TrackingParams):
        overrides.add_argument(f'--{f.name.replace("_", "-")}', dest=f'p_{f.name}',
                               type=_value(str(f.type)), metavar='X')
    args = ap.parse_args(argv)

    params = tracking_runner.load_params(
        args.params, {k[2:]: v for k, v in vars(args).items()
                      if k.startswith('p_') and v is not None})
    print(params)
    print(f'output -> {args.out_root}')

    if args.stamp_existing:
        tracking_runner.stamp_existing(args.out_root, params, run_pipeline=not args.no_pipeline)
        return 0

    out = tracking_runner.run(args.out_root, params, datasets=args.datasets,
                              n_datasets=args.n_datasets, run_pipeline=not args.no_pipeline,
                              recompute=args.recompute, summary=not args.no_summary)
    if out['summary'] is not None:
        print(f'summary -> {out["summary"]}')
    failed = any(s['failed'] for s in out['statuses'].values())
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())

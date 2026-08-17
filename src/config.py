"""Paths, dataset registries and per-region parameters.

Everything that used to be a hardcoded value in a notebook cell lives here, so that
`03A` and `03B` can run the *same* step sequence and differ only by what
`params_for()` hands them.

Nothing in this module imports from the rest of `src`, so any module may import it
without a cycle.

Two datasets feed the same analysis:

* **NOAA** (line A) — active regions queried, downloaded and corrected here.
  `data/raw/NOAA_<noaa>_<date>/` -> `data/processed/NOAA_<noaa>_<date>/`
* **DS0N** (line B) — the `sebastian_sun_spots` set, delivered already processed.
  `data/raw/sebastian_sun_spots/DS00..DS11/`

They keep separate loaders (`loaders.load_noaa_region` / `loaders.load_ds0n_region`)
because their thresholds mean different things: NOAA cuts are *fractions* of each
frame's own quiet-sun intensity, DS0N cuts are absolute DN.
"""

from __future__ import annotations

import pathlib

import astropy.units as u
import numpy as np

# ── paths ─────────────────────────────────────────────────────────────────────
# Resolved from this file rather than from the working directory, so a notebook can
# be run from `notebooks/` or from the repo root and get the same answer.
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
RAW_DIR = DATA_DIR / 'raw'
PROCESSED_DIR = DATA_DIR / 'processed'
DS0N_RAW_DIR = RAW_DIR / 'sebastian_sun_spots'
DS0N_PROCESSED_DIR = PROCESSED_DIR / 'sebastian_sun_spots'


def project_root() -> pathlib.Path:
    """The repo root. Use it to bootstrap `sys.path` at the top of a notebook."""
    return PROJECT_ROOT


# ── line A: which active regions, and how to fetch them ───────────────────────
# 11039 and 11041 are in the Löptien table but predate HMI science data (~2010-05-01),
# so they have no cubes and are left commented out rather than failing at download.
#
# 11117's two catalog dates (2010-10-27, 2010-10-28) are one continuous AR passage —
# a multi-day window anchored at the earlier date already covers the later one, so they
# are a single entry here and two rows in FIT_WINDOWS.
ACTIVE_REGIONS = [
    # {'noaa': 11039, 'date': '2010-01-01'},
    # {'noaa': 11041, 'date': '2010-01-26'},
    {'noaa': 11106, 'date': '2010-09-16'},
    {'noaa': 11117, 'date': '2010-10-27'},
    {'noaa': 11363, 'date': '2011-12-06'},
    {'noaa': 11536, 'date': '2012-07-31'},
    {'noaa': 13131, 'date': '2022-10-29'},
]

N_DAYS = 4  # minimum continuous days of data to fetch per AR

# Per-AR overrides for the box geometry, forwarded straight to `download.locate_ar_window`.
#
# 11536: SWPC under-reports this group badly. Measured from the frames already on disk it
# grows from 165" to 220" across and sits ~35" left of the reported centroid, so the default
# median sizing (200" square) clips it on the left in *every* frame. 'max' alone (244") still
# leaves ~0" margin at the start, so the padding goes up too.  -> 304" square
BOX_OVERRIDE = {
    11536: {'statistic': 'max', 'padding': 60 * u.arcsec},
}

NOTIFY_EMAIL = 'thomas.quamtum@gmail.com'  # must be a JSOC-registered export email
SERIES_LIST = ['hmi.Ic_720s', 'hmi.V_720s', 'hmi.M_720s']

# NOAA 11117's 720 s data is broken, so fetch the 45 s series downsampled to 720 s instead.
SERIES_OVERRIDE = {
    11117: ['hmi.Ic_45s', 'hmi.V_45s', 'hmi.M_45s'],
}
SAMPLE_OVERRIDE = {
    11117: 720 * u.s,
}


def series_for(noaa: int) -> list[str]:
    """The JSOC series to download for one AR."""
    return SERIES_OVERRIDE.get(noaa, SERIES_LIST)


def sample_for(noaa: int):
    """The `a.Sample` cadence for one AR, or None to take the series' native cadence."""
    return SAMPLE_OVERRIDE.get(noaa)


# ── stage 2: how to correct the cubes ─────────────────────────────────────────
# These are NOT analysis knobs. They decide what gets written to data/processed, so
# changing one means re-running 02A; the umbra/penumbra thresholds live in REGION_PARAMS
# below and cost only a 03A re-run.
#
# calibrate_doppler applies the four physical corrections of Castellanos Durán et al.
# (2021) Sect. 2 — observatory velocity, large-scale flows, convective blueshift CLV,
# gravitational redshift — putting velocities on an absolute m/s scale.
#
# residual_plane_fit additionally fits and subtracts a plane over the quiet-sun pixels.
# Off by default on purpose: it flattens whatever gradient is left, which also throws away
# the absolute scale the physical corrections just established. Turn it on to inspect what
# the physical model failed to remove, and do not read absolute velocities off the result.
#
# correct_limb_darkening divides the continuum by C(mu) = 1 - u - v + u.mu + v.mu^2
# (Castellanos Durán & Kleint 2020 Eqs. 1-2). Continuum ONLY — limb darkening is an
# intensity effect, so the magnetogram and dopplergram are not divided by C. Eq. 3's
# DN -> cgs factor is deliberately not applied: everything downstream is a *ratio* of
# intensities, so a global scale cancels while making every number harder to compare
# against the raw frames and against DS9.
#
# mu_method: 'geometry' goes through sunpy's transforms (finite observer distance);
# 'paper' is Eq. 2 literally (observer at infinity, ~3x faster, ~0.1% difference near the
# limb).
#
# plane_qsun_*: which pixels the magnetogram plane is fitted over — everything brighter
# than plane_qsun_frac x I_qs, where I_qs is that frame's own plane_qsun_percentile of the
# continuum. Pinned independently of the analysis thresholds so the cube on disk does not
# silently change meaning when a threshold is retuned downstream.
PROCESSING = dict(
    calibrate_doppler=True,
    residual_plane_fit=False,
    correct_limb_darkening=True,
    mu_method='geometry',
    plane_qsun_frac=0.90,
    plane_qsun_percentile=80,
)


# ── line B: which delivered datasets ──────────────────────────────────────────
# DS00..DS11 exist on disk. The old notebooks each used a different subset (04 used
# 00-09 in one cell and a hand-picked five in another); this is the one list.
#
# Only DS00..DS09 actually load. DS10's cube_continuum.fits is truncated — astropy
# refuses it with "buffer is too small for requested array" — and DS11 has no continuum
# cube at all. `ds0n_regions()` filters on the file existing, which catches DS11 but not
# DS10, so DS10 is excluded here explicitly until the file is re-delivered.
DS0N_IDS = [f'DS{i:02d}' for i in range(10)]
DS0N_IDS_ALL = [f'DS{i:02d}' for i in range(12)]
DS0N_BROKEN = {
    'DS10': 'cube_continuum.fits is truncated (buffer is too small for requested array)',
    'DS11': 'no cube_continuum.fits',
}


# ── segmentation defaults ─────────────────────────────────────────────────────
# Kept here rather than in `segmentation` so `REGION_PARAMS` below can reference them
# without importing anything.
# Components smaller than this many pixels are discarded before a spot is chosen.
# The penumbra threshold (0.90 I_qs) sits around the 8th percentile of a quiet-sun box, so
# it catches the dark tail of granulation as well as the spot: a single NOAA 11536 frame
# labels into ~160 components, ~150 of them under 50 px and totalling ~950 px of speckle.
# At HMI's 0.5 arcsec/px, 50 px is a blob about 7 px across — well under a real pore
# (2-5 Mm, i.e. 30-150 px), so this removes noise without touching solar structure. It also
# stops the "largest" component percolating through the speckle field and merging spots
# that are not actually connected.
MIN_CLUSTER_PX = 50

# Hot spot = |B| above this, inside the sunspot. The default is on |B| rather than signed B
# on purpose: the DS0N notebooks used `b > 500` or `b < -500` chosen per region, and getting
# the sign wrong yields an *empty* mask rather than an error — NOAA 11536's umbra averages
# about -400 G, so `b > 500` would select nothing there. Override `mag_filter` in
# REGION_PARAMS only to isolate one polarity of a bipolar group deliberately.
DEFAULT_HOTSPOT_G = 500.0

DIURNAL_PERIOD_H = 24.0

# Line A (NOAA). Thresholds are fractions of each frame's *own* quiet-sun continuum
# intensity, not absolute DN: an absolute cut makes the mask areas drift as the region
# rotates, and for a near-limb region the whole frame can fall under a fixed penumbra cut.
#
# cluster_mode restricts umbra and penumbra to ONE connected sunspot. Without it the masks
# are every dark pixel in the box — other spots, pores, bad pixels — and NOAA 11117's umbra
# area drifted 98% across its window as a result.
#   'largest' by area     'central' nearest the box centre     None no selection at all
#
# cluster_track follows the same spot between frames by maximum overlap instead of
# re-picking each time; otherwise two comparable spots make the selection flip mid-window,
# putting a step into every series.
#
# cluster_min_area drops components below that many pixels before choosing. Not cosmetic:
# penumbra_frac = 0.90 sits near the 8th percentile of a quiet-sun box, so the threshold
# catches the dark tail of granulation — one NOAA 11536 frame labels into ~160 components,
# ~150 of them speckle under 50 px.
NOAA_DEFAULTS = dict(
    umbra_frac=0.60,        # I < 0.60 I_qs            -> umbra
    penumbra_frac=0.90,     # I < 0.90 I_qs, not umbra -> penumbra
    qsun_percentile=80,     # percentile of finite pixels used as the frame's I_qs
    cluster_mode='largest',
    cluster_track=True,
    cluster_min_area=MIN_CLUSTER_PX,
    hotspot_gauss=DEFAULT_HOTSPOT_G,
)

# Line B (DS0N). Absolute DN cuts, as the delivered cubes were segmented originally.
DS0N_DEFAULTS = dict(
    umbra_thresh=30_000,
    penumbra_thresh=50_000,
    cadence_s=720.0,
    filter_mask=False,
    mag_filter=lambda b: b > 500,
)


def _valid_region_11536() -> np.ndarray:
    """NOAA 11536's box contains a second spot in its left half; keep only the right.

    Tied to that region's 139x484 box, which is why it is a per-region override and not
    a general option.
    """
    valid = np.ones((139, 484), dtype=bool)
    valid[:, 0:242] = False
    return valid


# Per-region deviations from the defaults. Anything absent uses the defaults untouched.
#
# `frame_idx` picks which frame the Step 2 calibration figures show; it is a viewing
# choice, not a science one, and does not affect any number.
#
# `mag_fill_slots`: magnetogram frames that never share a grid slot with a continuum frame
# yield NaN for every mean B, because the masks come from the continuum. 1 fills each empty
# slot from the nearest magnetogram frame; 0 leaves the NaNs, which is the honest default —
# filling fabricates a coincidence the observations do not have.
REGION_PARAMS: dict[int | str, dict] = {
    11106: dict(frame_idx=None),
    11117: dict(
        frame_idx=400,
        # From 2010-10-30 the continuum runs at 720 s on even slots and the magnetogram at
        # 720 s on odd slots, 360 s apart, so they never coincide and every B panel breaks
        # off there. The real fix is re-downloading that day on one clock.
        mag_fill_slots=0,
        crop_to_data=True,   # the download box is bad for this region; 02A crops it
    ),
    11363: dict(frame_idx=200, frame_idx_masks=411),
    11536: dict(custom_valid_region=_valid_region_11536, frame_idx=0, frame_idx_masks=411),
    13131: dict(frame_idx=200, frame_idx_masks=411),
}

# Per-DS deviations for line B. Empty so far — every DS0N dataset uses the same cuts.
DS0N_PARAMS: dict[str, dict] = {}

# Loading a region from its *raw* cubes instead of 02A's corrected ones, to see what the
# corrections removed. This is a comparison view, never an analysis configuration:
#
#   - the raw dopplergram has the whole calibration unapplied. v_SDO alone is ~3 km/s and
#     *diurnal*, i.e. sitting exactly on the period this project measures, so an amplitude
#     read off a raw cube is measuring the spacecraft, not the sunspot.
#   - the raw magnetogram still carries the quiet-sun plane.
#
# It used to live in 03A as two flags in NOAA 11106's block. Keeping it out of REGION_PARAMS
# is deliberate: parameters there flow into `04` as well, and 11106's fitted amplitude would
# silently have become an uncalibrated number.
RAW_COMPARE = dict(raw_magnetogram=True, raw_dopplergram=True)

# Keys that steer the notebook rather than the loader, and so must not be forwarded to
# `load_noaa_region` / `load_ds0n_region`.
_VIEW_KEYS = frozenset({'frame_idx', 'frame_idx_masks', 'crop_to_data'})


def region_noaa(region_dir) -> int | None:
    """NOAA number from a `NOAA_<number>_<date>` directory name, or None.

    Accepts a path or a bare name. Used to be defined identically in both 02A and 03A.
    """
    name = getattr(region_dir, 'name', str(region_dir))
    parts = name.split('_')
    return int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else None


def region_key(region_dir) -> int | str:
    """The `REGION_PARAMS` key for a region directory: a NOAA number, or a DS id."""
    noaa = region_noaa(region_dir)
    if noaa is not None:
        return noaa
    return getattr(region_dir, 'name', str(region_dir))


def params_for(region_dir, *, line: str | None = None, **overrides) -> dict:
    """Every parameter for one region: defaults, then its entry, then `overrides`.

    `line` is 'A'/'noaa' or 'B'/'ds0n'; inferred from the directory name when omitted.
    Callables stored in `REGION_PARAMS` are invoked here, so a per-region mask is built
    only for the region that needs it.

    Split the result with `loader_kwargs()` / `view_params()` before passing it on.
    """
    key = region_key(region_dir)
    if line is None:
        line = 'A' if isinstance(key, int) else 'B'
    line = line.upper()

    if line in ('A', 'NOAA'):
        params = dict(NOAA_DEFAULTS)
        params.update(REGION_PARAMS.get(key, {}))
    elif line in ('B', 'DS0N'):
        params = dict(DS0N_DEFAULTS)
        params.update(DS0N_PARAMS.get(key, {}))
    else:
        raise ValueError(f"line must be 'A'/'noaa' or 'B'/'ds0n', got {line!r}")

    params.update(overrides)
    return {k: (v() if callable(v) and k == 'custom_valid_region' else v)
            for k, v in params.items()}


def loader_kwargs(params: dict) -> dict:
    """The subset of `params_for()` that a loader accepts."""
    return {k: v for k, v in params.items() if k not in _VIEW_KEYS}


def view_params(params: dict) -> dict:
    """The subset of `params_for()` that steers the notebook's figures, with defaults."""
    frame_idx = params.get('frame_idx')
    return {
        'frame_idx': frame_idx,
        'frame_idx_masks': params.get('frame_idx_masks', frame_idx),
        'crop_to_data': params.get('crop_to_data', False),
    }


# ── the Löptien reference data, for the consolidated analysis ─────────────────
# Measured with Hinode by an independent method — this is what the fitted amplitudes get
# compared against, so it is transcribed as-is and not to be edited.
#
# 13131 is NOT from the Löptien catalog: its z_div/z_press of 800 km is the Romero (2020)
# value, carried here so the region can join the same scatter plot. Its theta/area/B_av are
# unknown, hence NaN.
LOPTIEN_TABLE = [
    # noaa,  date,          theta_deg, area_Mm2, B_av,   z_div, z_press
    (11039, '2010-01-01', 29, 284, 2471, 659, 366),
    (11041, '2010-01-26', 20, 170, 2068, 638, 305),
    (11106, '2010-09-16', 27, 195, 2349, 636, 381),
    (11117, '2010-10-27', 25, 522, 2093, 561, 291),
    (11117, '2010-10-28', 35, 223, 2090, 625, 313),
    (11363, '2011-12-06', 25, 1268, 2287, 524, 326),
    (11536, '2012-07-31', 34, 124, 2181, 577, 346),
    (13131, '2022-10-29', np.nan, np.nan, np.nan, 800, 800),
]

FIT_PERIOD_H = DIURNAL_PERIOD_H

# (noaa, date) -> (t_start, t_end) in hours from the FIRST FRAME of that region's cube,
# which is the anchor date at 00:00.
#
# 11117 appears twice because the Löptien table does: 2010-10-27 and 2010-10-28 are one
# continuous AR passage in a single cube, so its two rows are the 0-24 h and 24-48 h
# stretches of the same series and give two independent data points.
#
# A window longer than ~36 h constrains the fit considerably better than one period does.
FIT_WINDOWS = {
    (11106, '2010-09-16'): (0, 24),
    (11117, '2010-10-27'): (0, 24),
    (11117, '2010-10-28'): (24, 48),
    (11363, '2011-12-06'): (0, 24),
    (11536, '2012-07-31'): (0, 24),
    (13131, '2022-10-29'): (0, 24),
}


def processed_regions(pattern: str = 'NOAA_*') -> list[pathlib.Path]:
    """Processed region directories that actually have a continuum cube, sorted."""
    return sorted(p for p in PROCESSED_DIR.glob(pattern)
                  if (p / 'region_01_continuum_cube.fits').exists())


def ds0n_regions(ids: list[str] | None = None) -> list[pathlib.Path]:
    """DS0N raw directories that actually have a continuum cube, sorted."""
    ids = DS0N_IDS if ids is None else ids
    return [d for d in (DS0N_RAW_DIR / i for i in ids)
            if (d / 'cube_continuum.fits').exists()]

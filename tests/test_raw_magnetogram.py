"""Verification for `load_noaa_region(raw_magnetogram=True)` and `raw_dopplergram=True`
in src/loaders.py.

Run directly:  python tests/test_raw_magnetogram.py

The raw cubes are loaded to see the data *without* 02A's corrections — the field before the
quiet-sun plane subtraction, the velocity before `src/doppler_calibration.py`. The raw and
processed cubes do not share a time axis: `data/raw/` holds the frames that actually
downloaded, `data/processed/` holds 02A's uniform grid with a NaN frame per missing slot.
NOAA 11106 is 452 raw frames against 480 grid slots, so indexing one against the other
raises `operands could not be broadcast together`. These tests pin the reindex that fixes
it, and the guard for the other way the two can disagree — a spatial crop.
"""

import pathlib
import sys
import tempfile
from datetime import datetime, timedelta

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.loaders import load_noaa_region  # noqa: E402
from src.utilities import write_cube  # noqa: E402

CADENCE = 720.0
T0 = datetime(2010, 9, 16, 0, 12)
NY = NX = 12


def _frame(value, ny=NY, nx=NX):
    """One frame: a dark 3x3 spot on a bright background, so build_regions has a spot."""
    f = np.full((ny, nx), 100.0, dtype=np.float32)
    f[4:7, 4:7] = value
    return f


def _write_region(tmp, missing_mag_slots=(), mag_shape=None, n_slots=6,
                  missing_dop_slots=(), dop_shape=None):
    """Write a synthetic region: processed cubes on a full grid, raw magnetogram and
    dopplergram cubes missing the given slots. Returns (processed_dir, raw_dir, grid)."""
    grid = [T0 + timedelta(seconds=k * CADENCE) for k in range(n_slots)]

    processed = tmp / 'processed' / 'NOAA_99999_2010-09-16'
    raw = tmp / 'raw' / 'NOAA_99999_2010-09-16'
    processed.mkdir(parents=True)
    raw.mkdir(parents=True)

    cont = np.stack([_frame(50.0) for _ in grid])
    dop = np.stack([np.zeros((NY, NX), np.float32) for _ in grid])
    write_cube(cont, processed / 'region_01_continuum_cube.fits', timestamps=grid)
    write_cube(dop, processed / 'region_01_dopplergram_calibrated_cube.fits', timestamps=grid)
    write_cube(np.zeros_like(cont), processed / 'region_01_magnetogram_corrected_cube.fits',
               timestamps=grid)

    # The raw magnetogram: only the slots that "downloaded", each frame tagged with its slot
    # number inside the spot so the reindex can be checked slot by slot.
    ny, nx = mag_shape if mag_shape else (NY, NX)
    kept = [k for k in range(n_slots) if k not in missing_mag_slots]
    mag = np.stack([_frame(1000.0 + k, ny, nx) for k in kept])
    write_cube(mag, raw / 'region_01_magnetogram_cube.fits',
               timestamps=[grid[k] for k in kept])

    # The raw dopplergram, tagged the same way. The processed one is all zeros, so any
    # test can tell which of the two it was handed.
    ny, nx = dop_shape if dop_shape else (NY, NX)
    kept_dop = [k for k in range(n_slots) if k not in missing_dop_slots]
    raw_dop = np.stack([_frame(2000.0 + k, ny, nx) for k in kept_dop])
    write_cube(raw_dop, raw / 'region_01_dopplergram_cube.fits',
               timestamps=[grid[k] for k in kept_dop])
    return processed, raw, grid


def test_raw_magnetogram_is_placed_on_the_processed_grid():
    """A raw cube shorter than the grid must load, with its frames in the right slots."""
    with tempfile.TemporaryDirectory() as td:
        processed, raw, grid = _write_region(pathlib.Path(td), missing_mag_slots=(2, 4))

        data = load_noaa_region(processed, raw_magnetogram=True, raw_dir=raw)

        assert data['cube_mag'].shape[0] == len(grid), \
            f"cube_mag has {data['cube_mag'].shape[0]} frames, grid has {len(grid)}"
        assert data['cube_mag'].shape == data['cube_cont'].shape, \
            f"cube_mag {data['cube_mag'].shape} != cube_cont {data['cube_cont'].shape}"

        # The missing slots are NaN, not a frame borrowed from a neighbour.
        assert not np.isfinite(data['cube_mag'][2]).any(), 'slot 2 should be all NaN'
        assert not np.isfinite(data['cube_mag'][4]).any(), 'slot 4 should be all NaN'
        assert list(np.flatnonzero(~data['present']['mag'])) == [2, 4], \
            f"present['mag'] gaps are {np.flatnonzero(~data['present']['mag'])}"

        # Every present slot holds *its own* frame, not the one that followed it in the
        # raw cube — the failure a plain truncation would produce silently.
        for k in (0, 1, 3, 5):
            got = data['cube_mag'][k, 5, 5]
            assert got == 1000.0 + k, f'slot {k} holds frame {got - 1000:.0f}'
        return f'{len(grid)} slots, raw cube of 4 frames placed at 0,1,3,5'


def test_raw_magnetogram_matches_the_corrected_one_when_no_frames_are_missing():
    """With no gaps the reindex must be a no-op, so it cannot reorder a healthy cube."""
    with tempfile.TemporaryDirectory() as td:
        processed, raw, grid = _write_region(pathlib.Path(td))

        data = load_noaa_region(processed, raw_magnetogram=True, raw_dir=raw)

        assert data['present']['mag'].all(), 'no slot should be missing'
        expected = [1000.0 + k for k in range(len(grid))]
        got = list(data['cube_mag'][:, 5, 5])
        assert got == expected, f'frames came back as {got}'
        return f'{len(grid)} frames unchanged'


def test_spatially_mismatched_raw_magnetogram_is_refused_clearly():
    """02A can crop a region; the raw cube is uncropped. That must not be a broadcast error."""
    with tempfile.TemporaryDirectory() as td:
        processed, raw, _ = _write_region(pathlib.Path(td), mag_shape=(NY + 4, NX + 4))

        try:
            load_noaa_region(processed, raw_magnetogram=True, raw_dir=raw)
        except ValueError as exc:
            msg = str(exc)
            assert 'raw' in msg.lower() and str(NY + 4) in msg, \
                f'error does not say which shapes disagree: {msg}'
            return f'refused with: {msg[:70]}...'
        raise AssertionError('a 16x16 raw magnetogram against a 12x12 continuum was accepted')


def test_raw_dir_is_required():
    """raw_magnetogram=True without raw_dir must say so rather than fall through."""
    with tempfile.TemporaryDirectory() as td:
        processed, _, _ = _write_region(pathlib.Path(td))
        try:
            load_noaa_region(processed, raw_magnetogram=True)
        except ValueError as exc:
            return f'refused with: {str(exc)[:70]}'
        raise AssertionError('raw_magnetogram=True with no raw_dir was accepted')


def test_raw_dopplergram_is_placed_on_the_processed_grid():
    """The Doppler cube takes the same route as the magnetogram: gaps stay NaN, frames
    stay in their own slot, and the corrected magnetogram is still what gets loaded."""
    with tempfile.TemporaryDirectory() as td:
        processed, raw, grid = _write_region(pathlib.Path(td), missing_dop_slots=(1, 3))

        data = load_noaa_region(processed, raw_dopplergram=True, raw_dir=raw)

        assert data['cube_dop'].shape == data['cube_cont'].shape, \
            f"cube_dop {data['cube_dop'].shape} != cube_cont {data['cube_cont'].shape}"
        assert not np.isfinite(data['cube_dop'][1]).any(), 'slot 1 should be all NaN'
        assert not np.isfinite(data['cube_dop'][3]).any(), 'slot 3 should be all NaN'
        assert list(np.flatnonzero(~data['present']['dop'])) == [1, 3], \
            f"present['dop'] gaps are {np.flatnonzero(~data['present']['dop'])}"

        for k in (0, 2, 4, 5):
            got = data['cube_dop'][k, 5, 5]
            assert got == 2000.0 + k, f'slot {k} holds frame {got - 2000:.0f}'

        # raw_dopplergram must not drag the magnetogram along with it.
        assert np.nanmax(np.abs(data['cube_mag'])) == 0.0, \
            'cube_mag is not the corrected (all-zero) cube'
        return f'{len(grid)} slots, raw Doppler cube of 4 frames placed at 0,2,4,5'


def test_calibrated_dopplergram_is_the_default():
    """Without the flag the calibrated cube is what loads — the raw one is opt-in."""
    with tempfile.TemporaryDirectory() as td:
        processed, raw, _ = _write_region(pathlib.Path(td))

        default = load_noaa_region(processed)
        rawdop = load_noaa_region(processed, raw_dopplergram=True, raw_dir=raw)

        assert np.nanmax(np.abs(default['cube_dop'])) == 0.0, \
            'the default load did not return 02A\'s calibrated cube'
        assert rawdop['cube_dop'][0, 5, 5] == 2000.0, \
            f"raw_dopplergram gave {rawdop['cube_dop'][0, 5, 5]}"
        return 'calibrated by default, raw only when asked'


def test_both_raw_cubes_at_once():
    """The two flags are independent and can be combined in one call."""
    with tempfile.TemporaryDirectory() as td:
        processed, raw, _ = _write_region(pathlib.Path(td))

        data = load_noaa_region(processed, raw_magnetogram=True, raw_dopplergram=True,
                                raw_dir=raw)

        assert data['cube_mag'][0, 5, 5] == 1000.0, f"cube_mag {data['cube_mag'][0, 5, 5]}"
        assert data['cube_dop'][0, 5, 5] == 2000.0, f"cube_dop {data['cube_dop'][0, 5, 5]}"
        return 'raw magnetogram and raw dopplergram loaded together'


def test_spatially_mismatched_raw_dopplergram_is_refused_clearly():
    """A cropped region must be refused for the Doppler cube too, not broadcast-error."""
    with tempfile.TemporaryDirectory() as td:
        processed, raw, _ = _write_region(pathlib.Path(td), dop_shape=(NY + 4, NX + 4))

        try:
            load_noaa_region(processed, raw_dopplergram=True, raw_dir=raw)
        except ValueError as exc:
            msg = str(exc)
            assert 'dopplergram' in msg.lower() and str(NY + 4) in msg, \
                f'error does not say which cube and which shapes disagree: {msg}'
            return f'refused with: {msg[:70]}...'
        raise AssertionError('a 16x16 raw dopplergram against a 12x12 continuum was accepted')


def test_raw_dir_is_required_for_the_dopplergram_too():
    """raw_dopplergram=True with no raw_dir must name that flag, not the magnetogram."""
    with tempfile.TemporaryDirectory() as td:
        processed, _, _ = _write_region(pathlib.Path(td))
        try:
            load_noaa_region(processed, raw_dopplergram=True)
        except ValueError as exc:
            assert 'raw_dopplergram' in str(exc), f'error names the wrong flag: {exc}'
            return f'refused with: {str(exc)[:70]}'
        raise AssertionError('raw_dopplergram=True with no raw_dir was accepted')


if __name__ == '__main__':
    import warnings

    warnings.filterwarnings('ignore')
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    failures = 0
    for test in tests:
        try:
            detail = test()
        except AssertionError as exc:
            print(f'FAIL  {test.__name__}: {exc}')
            failures += 1
        except Exception as exc:
            print(f'ERROR {test.__name__}: {exc!r}')
            failures += 1
        else:
            print(f'PASS  {test.__name__}: {detail}')
    print(f'\n{len(tests) - failures} passed, {failures} failed')
    sys.exit(1 if failures else 0)

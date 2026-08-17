"""Verification for the time-grid helpers in src/utilities.py.

Run directly:  python tests/test_time_grid.py

These are the join that replaced the old timestamp *intersection*: a series missing a frame
mid-window must produce a NaN frame at the right time, not a shortened axis that slides
every later frame out of step. The synthetic cases are exhaustive; the last check reads the
real NOAA 11536 frame directory when it is present.
"""

import glob
import pathlib
import sys
from datetime import datetime, timedelta

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.utilities import (  # noqa: E402
    parse_frame_timestamp,
    regular_time_grid,
    reindex_on_grid,
    reindex_series_on_grid,
)

REAL_FRAMES = 'data/raw/NOAA_11536_2012-07-31/region_01'

T0 = datetime(2012, 7, 31)
CADENCE = 720.0


def _times(*slots):
    """Timestamps at the given slot indices of a 720 s grid starting at T0."""
    return [T0 + timedelta(seconds=k * CADENCE) for k in slots]


def test_grid_spans_the_union():
    """The grid must cover every timestamp in every series, at a uniform spacing."""
    grid, cadence = regular_time_grid([_times(0, 1, 2, 3), _times(0, 2, 3), _times(1, 3)])
    assert cadence == CADENCE, f'inferred cadence {cadence}, expected {CADENCE}'
    assert len(grid) == 4, f'grid has {len(grid)} slots, expected 4'
    diffs = {(grid[i + 1] - grid[i]).total_seconds() for i in range(len(grid) - 1)}
    assert diffs == {CADENCE}, f'grid spacing is not uniform: {diffs}'
    return f'{len(grid)} slots at {cadence:g} s from the union of 3 series'


def test_gap_survives_as_a_slot():
    """A frame missing from the *middle* is the whole point: the slot must stay.

    Under the old intersection join this case lost the slot entirely and shifted every
    later frame, which is what silently mismatched the three series.
    """
    grid, cadence = regular_time_grid([_times(0, 1, 2, 3, 4)])
    cube = np.arange(4 * 2 * 2, dtype=np.float32).reshape(4, 2, 2)
    out, present = reindex_on_grid(cube, _times(0, 1, 3, 4), grid, cadence)

    assert out.shape == (5, 2, 2), f'output has shape {out.shape}, expected (5, 2, 2)'
    assert list(present) == [True, True, False, True, True], f'present is {present}'
    assert np.isnan(out[2]).all(), 'the gap slot is not NaN'
    # Frames after the gap must still be in their own slots, not shifted down one.
    assert np.array_equal(out[3], cube[2]), 'frames after the gap were shifted'
    assert np.array_equal(out[4], cube[3]), 'frames after the gap were shifted'
    return 'slot 2 is NaN, slots 3-4 keep their own frames'


def test_trailing_and_leading_gaps():
    """A series short at either end must NaN-fill there rather than shrink the grid."""
    grid, cadence = regular_time_grid([_times(0, 1, 2, 3), _times(1, 2)])
    cube = np.ones((2, 3, 3), dtype=np.float32)
    out, present = reindex_on_grid(cube, _times(1, 2), grid, cadence)
    assert list(present) == [False, True, True, False]
    assert np.isnan(out[0]).all() and np.isnan(out[3]).all()
    return 'leading and trailing gaps both NaN-filled'


def test_series_reindex_matches_cube_reindex():
    """A 1-D per-frame series (a correction term's mean) must gap the same way."""
    grid, cadence = regular_time_grid([_times(0, 1, 2, 3)])
    values = reindex_series_on_grid([10.0, 20.0, 30.0], _times(0, 2, 3), grid, cadence)
    assert np.isnan(values[1]), 'the gap slot is not NaN'
    assert list(values[[0, 2, 3]]) == [10.0, 20.0, 30.0]
    return f'series gapped at slot 1: {values}'


def test_off_grid_timestamp_raises():
    """A timestamp that doesn't fit its slot means the cadence is wrong — refuse it.

    Snapping it silently is exactly the class of error this join exists to prevent, so the
    failure has to be loud.
    """
    grid, cadence = regular_time_grid([_times(0, 1, 2, 3)])
    bad = [T0 + timedelta(seconds=CADENCE + 300)]  # 300 s off slot 1, tolerance is 360...
    # 300 s is inside the default tolerance of 360 s, so tighten it to make the point.
    try:
        reindex_on_grid(np.zeros((1, 2, 2), np.float32), bad, grid, cadence, tolerance_s=60)
    except ValueError as exc:
        assert 'tolerance' in str(exc)
        return 'a timestamp off its slot by more than the tolerance is refused'
    raise AssertionError('expected a ValueError for an off-grid timestamp')


def test_colliding_timestamps_raise():
    """Two frames in one slot means the assumed cadence is too coarse."""
    grid, cadence = regular_time_grid([_times(0, 1, 2)])
    times = [T0, T0 + timedelta(seconds=60)]  # both nearest slot 0
    try:
        reindex_on_grid(np.zeros((2, 2, 2), np.float32), times, grid, cadence)
    except ValueError as exc:
        assert 'grid slot' in str(exc)
        return 'two timestamps in one slot are refused'
    raise AssertionError('expected a ValueError for colliding timestamps')


def test_jitter_within_tolerance_is_accepted():
    """Half a cadence of jitter is snapped, by design — and that is the whole tolerance.

    `regular_time_grid` builds its grid from the pooled span and the *median* spacing, and
    every timestamp then claims its nearest slot. Rounding to the nearest slot can never be
    off by more than half a cadence, so with the default `tolerance_s = cadence_s / 2`
    nothing except a collision can be refused. That is deliberate — HMI timestamps are exact
    multiples of the cadence, and jitter smaller than half a frame is not a different
    observation — but it means "refuses irregular input" is only true relative to a stated
    tolerance. `test_irregular_input_is_caught_with_a_tight_tolerance` is that case.
    """
    times = [T0, T0 + timedelta(seconds=720), T0 + timedelta(seconds=1500)]
    grid, cadence = regular_time_grid([times])
    assert cadence == 750.0, f'inferred cadence {cadence}, expected the median 750'
    assert len(grid) == 3, f'grid has {len(grid)} slots, expected 3'
    return f'720/780 s spacing snapped onto a {cadence:g} s grid'


def test_irregular_input_is_caught_with_a_tight_tolerance():
    """State how much jitter is real, and a series that exceeds it must be refused.

    This also covers the verification pass at the end of `regular_time_grid`, which exists so
    a wrong cadence surfaces there rather than as a subtly mis-slotted cube much further
    downstream.
    """
    times = [T0, T0 + timedelta(seconds=720), T0 + timedelta(seconds=1500)]
    try:
        regular_time_grid([times], tolerance_s=10)
    except ValueError as exc:
        assert 'tolerance' in str(exc)
        return 'a series jittering past the stated tolerance is refused'
    raise AssertionError('expected a ValueError for an irregular series')


def test_real_region_gaps():
    """The case that motivated all of this, against the actual downloaded frames.

    NOAA 11536's Dopplergram is missing 2012-08-01 09:48 and 2012-08-02 21:48 — both
    *mid-window*, which is exactly the case the old intersection join handled wrongly: it
    threw away two good continuum/magnetogram frames and put two 1440 s jumps into an
    otherwise 720 s axis.

    Nothing here is pinned to a frame *count*. The window grows whenever another day is
    downloaded (it has already gone from 3 days to 4), and a test that asserts the count
    fails on new data rather than on a regression. What must hold regardless is that the
    grid is uniform, covers every frame on disk, and puts the gaps at those two instants.
    """
    series = {}
    for name, pattern in [('cont', 'hmi.ic_*.continuum.fits'),
                          ('mag', 'hmi.m_*.magnetogram.fits'),
                          ('dop', 'hmi.v_*.fits')]:
        found = sorted(filter(None, (parse_frame_timestamp(p)
                                     for p in glob.glob(f'{REAL_FRAMES}/{pattern}'))))
        if not found:
            return None
        series[name] = found

    grid, cadence = regular_time_grid(list(series.values()))
    assert cadence == 720.0, f'inferred cadence {cadence}, expected 720'
    pooled = {t for times in series.values() for t in times}
    assert pooled <= set(grid), 'the grid does not cover every frame on disk'
    spacing = {(grid[i + 1] - grid[i]).total_seconds() for i in range(len(grid) - 1)}
    assert spacing == {720.0}, f'grid spacing is not uniform: {spacing}'

    gaps = {}
    for name, times in series.items():
        _, present = reindex_on_grid(
            np.zeros((len(times), 1, 1), np.float32), times, grid, cadence)
        gaps[name] = [grid[i] for i in np.flatnonzero(~present)]

    assert gaps['cont'] == [] and gaps['mag'] == [], f'unexpected gaps: {gaps}'
    assert gaps['dop'] == [datetime(2012, 8, 1, 9, 48), datetime(2012, 8, 2, 21, 48)], \
        f"Dopplergram gaps are {gaps['dop']}"
    return (f'{len(grid)} slots at 720 s; Dopplergram gaps at '
            + ', '.join(t.strftime('%m-%d %H:%M') for t in gaps['dop']))


if __name__ == '__main__':
    import warnings

    warnings.filterwarnings('ignore')
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    failures, skipped = 0, 0
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
            if detail is None:
                print(f'SKIP  {test.__name__} (no data at {REAL_FRAMES})')
                skipped += 1
            else:
                print(f'PASS  {test.__name__}: {detail}')
    print(f'\n{len(tests) - failures - skipped} passed, {skipped} skipped, {failures} failed')
    sys.exit(1 if failures else 0)

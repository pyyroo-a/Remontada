"""Fill short ball gaps (Plan 3 step B): when the tracker loses the ball for a moment, join the
last seen position and the next one with a straight line, so the clues keep working.

Only gaps with a seen frame directly before and after them inside the same graded run are
filled, and only up to max_frames long. Filled frames are marked, so nothing treats them as a
measured touch (touch.touches(..., ignore=mask)). The rules allow improving the tracking with our
own algorithm.
"""
import numpy as np


def fill_ball_gaps(ball_xy, runs, max_frames):
    """(filled_xy, filled_mask): a copy of ball_xy with gaps of 1 to max_frames frames filled."""
    filled = np.array(ball_xy, dtype=np.float64, copy=True)
    mask = np.zeros(len(filled), dtype=bool)
    if max_frames <= 0:
        return filled, mask
    unseen = np.isnan(filled).any(axis=1)
    for start, stop in runs:
        if stop - start < 3:
            continue
        edges = np.diff(np.concatenate(([0], unseen[start:stop].astype(np.int8), [0])))
        gap_starts = np.flatnonzero(edges == 1) + start  # first unseen frame of each gap
        gap_stops = np.flatnonzero(edges == -1) + start  # first seen frame after it
        for a, b in zip(gap_starts, gap_stops):
            if a == start or b == stop or b - a > max_frames:
                continue  # open at a run edge, or too long
            share = (np.arange(a, b) - (a - 1)) / (b - (a - 1))
            filled[a:b] = filled[a - 1] + share[:, None] * (filled[b] - filled[a - 1])
            mask[a:b] = True
    return filled, mask

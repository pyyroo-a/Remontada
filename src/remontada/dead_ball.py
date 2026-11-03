"""Dead-ball clues: ball out of play, ball sitting still, ball missing for long, players slowing.

On IDSSE the ball freezes exactly when play stops (checked 3 Nov 2026), so "still" is a strong
clue there; on BAR-ATH it is learned like every other clue.
"""
import warnings

import numpy as np

from remontada.motion import BALL_HALF_WINDOW, PLAYER_HALF_WINDOW, runs_at_least, speed

OUT_MARGIN_M = 1.0  # the law says wholly over the line; the margin absorbs tracking error
STILL_MIN_FRAMES = 25
STILL_SPEED = 0.5  # m/s
MISSING_MIN_FRAMES = 75


def players_median_speed(table):
    """Median speed of the visible players per frame; NaN when nobody is visible."""
    s = speed(table.player_xy, table.fps, PLAYER_HALF_WINDOW)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN frames give NaN, as wanted
        return np.nanmedian(s, axis=1)


def ball_out(table, margin=OUT_MARGIN_M):
    length, width = table.pitch
    with np.errstate(invalid="ignore"):
        x, y = np.abs(table.ball_xy).T
        return (x > length / 2 + margin) | (y > width / 2 + margin)


def ball_still(table):
    """Inside a run of at least STILL_MIN_FRAMES seen frames where the ball is not moving.

    The speed looks BALL_HALF_WINDOW frames each way, so it only reads slow once both ends are
    at rest; widening each slow frame by that window gives back the whole resting stretch.
    """
    k = BALL_HALF_WINDOW
    with np.errstate(invalid="ignore"):
        slow = speed(table.ball_xy, table.fps, k) < STILL_SPEED
    near_slow = np.convolve(slow, np.ones(2 * k + 1), mode="same") > 0
    return runs_at_least(near_slow & ~np.isnan(table.ball_xy[:, 0]), STILL_MIN_FRAMES)


def ball_missing_long(table):
    """Inside a gap of unseen ball at least MISSING_MIN_FRAMES long (looks ahead, issue #15)."""
    return runs_at_least(np.isnan(table.ball_xy[:, 0]), MISSING_MIN_FRAMES)

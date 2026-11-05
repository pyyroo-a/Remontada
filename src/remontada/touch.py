"""Who has the ball: touches (the ball changes speed or direction next to a player) and control
(one team's player stays right by the ball). Team codes are 0 home, 1 away, -1 none.

Starting values, set 3 Nov 2026: on J03WMX the nearest player is within 1 m on 40.7% of live
frames and is on the right team 93.3% of the time there.
"""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from remontada.motion import BALL_HALF_WINDOW, runs_at_least

TOUCH_DV = 3.0  # m/s change in ball velocity
TOUCH_RADIUS = 1.5  # m
CONTROL_RADIUS = 1.5  # m
CONTROL_MIN_FRAMES = 10  # 0.4 s
CARRY_FRAMES = 75  # 3 s


def nearest_by_team(table):
    """Distance from the ball to the nearest home and nearest away player, per frame.

    inf when that team has nobody visible, NaN when the ball is unseen.
    """
    gap = np.hypot(*(table.player_xy - table.ball_xy[:, None, :]).transpose(2, 0, 1))
    unseen_ball = np.isnan(table.ball_xy[:, 0])
    out = []
    for team in (0, 1):
        team_gap = np.where(np.isnan(gap[:, table.player_team == team]), np.inf,
                            gap[:, table.player_team == team])
        nearest = team_gap.min(axis=1) if team_gap.shape[1] else np.full(table.n_frames, np.inf)
        nearest[unseen_ball] = np.nan
        out.append(nearest)
    return out[0], out[1]


def _nearest_team(home, away):
    return np.where(home <= away, 0, 1).astype(np.int8)


def touches(table, dv=TOUCH_DV, radius=TOUCH_RADIUS, ignore=None):
    """Team code on frames where the ball's velocity jumps next to a player, -1 elsewhere.

    The jump is |v_after - v_before| over K frames each side. Only jumps next to a player can be
    touches, and of those only the largest within K frames counts (the first one on a tie), so a
    bigger glitch far from everyone cannot hide a real touch (plan 2 review).

    ignore marks frames whose ball position was not measured (filled gaps, gaps.py): a jump worked
    out from any of them, so on frames up to K away, is never a touch.
    """
    k, n = BALL_HALF_WINDOW, table.n_frames
    out = np.full(n, -1, dtype=np.int8)
    if n <= 2 * k:
        return out
    home, away = nearest_by_team(table)
    with np.errstate(invalid="ignore"):
        near = np.fmin(home, away) <= radius

    b = table.ball_xy
    v_before = (b[k:-k] - b[:-2 * k]) * table.fps / k
    v_after = (b[2 * k:] - b[k:-k]) * table.fps / k
    jump = np.full(n, -np.inf)
    jump[k:-k] = np.nan_to_num(np.hypot(*(v_after - v_before).T), nan=-np.inf)
    jump[~near] = -np.inf
    if ignore is not None:
        jump[_within(np.asarray(ignore, dtype=bool), k)] = -np.inf

    padded = np.concatenate((np.full(k, -np.inf), jump, np.full(k, -np.inf)))
    window = sliding_window_view(padded, 2 * k + 1)  # window[t] covers t-k..t+k
    peak = (jump >= window.max(axis=1)) & (jump > window[:, :k].max(axis=1))

    hit = peak & (jump >= dv)
    out[hit] = _nearest_team(home, away)[hit]
    return out


def _within(mask, k):
    """True on frames up to k away from a True frame of mask."""
    padded = np.concatenate((np.zeros(k, dtype=bool), mask, np.zeros(k, dtype=bool)))
    return sliding_window_view(padded, 2 * k + 1).any(axis=1)


def control(table, radius=CONTROL_RADIUS, min_frames=CONTROL_MIN_FRAMES):
    """Team code on frames inside a run of at least min_frames where that team's nearest player
    is within radius and nearer than the other team's; -1 elsewhere."""
    home, away = nearest_by_team(table)
    out = np.full(table.n_frames, -1, dtype=np.int8)
    with np.errstate(invalid="ignore"):
        owners = ((home <= radius) & (home < away), (away <= radius) & (away < home))
    for team, owner in enumerate(owners):
        out[runs_at_least(owner, min_frames)] = team
    return out


def carry_forward(events, max_age):
    """The most recent non -1 value within the last max_age frames (inclusive), else -1."""
    frames = np.arange(len(events))
    last = np.maximum.accumulate(np.where(events >= 0, frames, -1)) if len(events) else frames
    out = np.where(last >= 0, events[np.maximum(last, 0)], -1).astype(np.int8)
    out[frames - last > max_age] = -1
    return out

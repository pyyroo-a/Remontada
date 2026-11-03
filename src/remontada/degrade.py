"""Make clean tracking look messy, at several levels, so testing on IDSSE is honest.

The levels span a wide range on purpose and are not taken from BAR-ATH (decided 30 Oct 2026): a
setting is only kept if it works across the whole sweep.

Degrade v2 (decided 3 Nov 2026): positions wobble by a slow drift, not a fresh jump every frame,
and half of the hidden ball frames start where DATA.md says real trackers lose the ball (ball in
the air, or players crowded around it).
"""
from collections import deque
from dataclasses import dataclass, replace

import numpy as np

SPARE_SLOTS_PER_TEAM = 2
WOBBLE_TIME_S = 1.0  # how long a wobble takes to fade
HARD_SHARE = 0.5  # share of the hidden ball frames placed on hard frames first
AIR_Z_M = 1.0
CROWD_RADIUS_M = 3.0
CROWD_PLAYERS = 3


@dataclass(frozen=True)
class Level:
    ball_hidden_share: float
    ball_gap_mean_s: float
    position_noise_m: float
    track_breaks_per_min: float  # per player


LEVELS = {
    "none": Level(0.0, 0.0, 0.0, 0.0),
    "light": Level(0.10, 0.5, 0.10, 0.5),
    "medium": Level(0.30, 1.0, 0.25, 1.0),
    "heavy": Level(0.60, 2.0, 0.50, 2.0),
}


def hard_frames(table):
    """Frames where a real tracker tends to lose the ball: ball in the air, or a crowd near it."""
    with np.errstate(invalid="ignore"):
        air = table.ball_z > AIR_Z_M
        gap = np.hypot(*(table.player_xy - table.ball_xy[:, None, :]).transpose(2, 0, 1))
        crowd = (gap <= CROWD_RADIUS_M).sum(axis=1) >= CROWD_PLAYERS
    return air | crowd


def _hide_ball(table, level, rng):
    """NaN stretches of exponential length until the target share of frames is hidden.

    The first HARD_SHARE of the target starts on hard frames that are still visible, the rest
    anywhere.
    """
    hidden = np.zeros(table.n_frames, dtype=bool)
    target = int(level.ball_hidden_share * table.n_frames)
    mean_frames = level.ball_gap_mean_s * table.fps
    hard = hard_frames(table)
    count = 0
    while count < target:
        open_hard = np.flatnonzero(hard & ~hidden) if count < HARD_SHARE * target else []
        if len(open_hard):
            start = int(open_hard[rng.integers(len(open_hard))])
        else:  # past the hard share, or every hard frame is already hidden
            start = int(rng.integers(table.n_frames))
        length = max(1, int(rng.exponential(mean_frames)))
        stretch = hidden[start:start + length]
        fresh = np.flatnonzero(~stretch)[: target - count]  # never overshoot the target
        stretch[fresh] = True
        count += len(fresh)
    return hidden


def _wobble(player_xy, noise_m, fps, rng):
    """Slow drift per slot and axis: each frame keeps most of the last frame's error."""
    rho = np.exp(-1.0 / (fps * WOBBLE_TIME_S))
    kicks = rng.normal(0.0, noise_m, player_xy.shape)
    error = np.empty_like(kicks)
    error[0] = kicks[0]
    fresh = np.sqrt(1.0 - rho ** 2)
    for t in range(1, len(kicks)):
        error[t] = rho * error[t - 1] + fresh * kicks[t]
    return player_xy + error


def _break_tracks(player_xy, teams, level, fps, rng):
    """Cut each slot at random frames and move the pieces between slots of the same team.

    Like a real tracker, a new piece reuses whichever slot of its team has been free longest, so
    the slot count stays small (two spares per team) however many breaks there are.
    """
    n_frames, n_slots, _ = player_xy.shape
    expected = level.track_breaks_per_min * n_frames / (fps * 60)
    output_slots, output_teams = 0, []
    pieces_by_output = []
    for team in (0, 1):
        sources = np.flatnonzero(teams == team)
        pieces = []
        for source in sources:
            cuts = np.sort(rng.integers(1, n_frames, size=rng.poisson(expected)))
            edges = np.concatenate(([0], cuts, [n_frames]))
            pieces += [(int(a), int(b), int(source)) for a, b in zip(edges[:-1], edges[1:]) if b > a]
        pieces.sort()

        total = len(sources) + SPARE_SLOTS_PER_TEAM
        free = deque(range(output_slots, output_slots + total))
        active = []  # (stop, slot)
        for start, stop, source in pieces:
            for item in sorted(item for item in active if item[0] <= start):
                active.remove(item)
                free.append(item[1])
            slot = free.popleft()
            active.append((stop, slot))
            pieces_by_output.append((start, stop, source, slot))
        output_slots += total
        output_teams += [team] * total

    out = np.full((n_frames, output_slots, 2), np.nan)
    for start, stop, source, slot in pieces_by_output:
        out[start:stop, slot] = player_xy[start:stop, source]
    return out, np.array(output_teams, dtype=np.int8)


def degrade(table, level, seed):
    """A messier copy of the table. Same level and seed always give the same result."""
    rng = np.random.default_rng(seed)
    ball_xy, ball_z = table.ball_xy, table.ball_z
    if level.ball_hidden_share > 0:
        hidden = _hide_ball(table, level, rng)
        ball_xy = ball_xy.copy()
        ball_z = ball_z.copy()
        ball_xy[hidden] = np.nan
        ball_z[hidden] = np.nan

    player_xy, teams = table.player_xy, table.player_team
    if level.position_noise_m > 0:
        player_xy = _wobble(player_xy, level.position_noise_m, table.fps, rng)
    if level.track_breaks_per_min > 0:
        player_xy, teams = _break_tracks(player_xy, teams, level, table.fps, rng)

    return replace(table, ball_xy=ball_xy, ball_z=ball_z, player_xy=player_xy, player_team=teams)

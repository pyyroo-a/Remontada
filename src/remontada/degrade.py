"""Make clean tracking look messy, at several levels, so testing on IDSSE is honest.

The levels span a wide range on purpose and are not taken from BAR-ATH (decided 30 Oct 2026): a
setting is only kept if it works across the whole sweep.
"""
from collections import deque
from dataclasses import dataclass, replace

import numpy as np

SPARE_SLOTS_PER_TEAM = 2


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


def _hide_ball(table, level, rng):
    """NaN stretches of exponential length until the target share of frames is hidden."""
    hidden = np.zeros(table.n_frames, dtype=bool)
    target = int(level.ball_hidden_share * table.n_frames)
    mean_frames = level.ball_gap_mean_s * table.fps
    count = 0
    while count < target:
        start = int(rng.integers(table.n_frames))
        length = max(1, int(rng.exponential(mean_frames)))
        stretch = hidden[start:start + length]
        fresh = np.flatnonzero(~stretch)[: target - count]  # never overshoot the target
        stretch[fresh] = True
        count += len(fresh)
    return hidden


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
        player_xy = player_xy + rng.normal(0.0, level.position_noise_m, player_xy.shape)
    if level.track_breaks_per_min > 0:
        player_xy, teams = _break_tracks(player_xy, teams, level, table.fps, rng)

    return replace(table, ball_xy=ball_xy, ball_z=ball_z, player_xy=player_xy, player_team=teams)

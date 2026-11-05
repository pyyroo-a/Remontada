"""The learned evidence: how often each clue pattern goes with H, A or a dead ball, counted on IDSSE.

Only the honest parts of IDSSE are learned (decided 3 Nov 2026): who has the ball from live frames,
and players slowing from all frames. Ball-dead clues are fixed-rule hints instead (hints.py).

A team cell joins three clues, all seen from the home side: nearness (6 bins), the last touch
carried forward (3) and control (3), so 54 cells. Every live frame is counted twice, as it is and
mirrored (home and away swapped), so the table cannot favour home or away.
"""
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from remontada.clues import DEFAULT_CLUES
from remontada.dead_ball import players_median_speed
from remontada.touch import carry_forward, control, nearest_by_team, touches
from remontada.vendor.possession_metrics import A, D, H, X

NEAR_EDGES = [-3.0, -1.0, 1.0, 3.0]  # margin = d_away - d_home in m; bin 0 is "ball unseen"
SLOW_EDGES = [0.5, 1.0, 1.5, 2.0, 3.0]  # players' median speed in m/s; last bin is "unknown"
N_TEAM_CELLS = 6 * 3 * 3
N_SLOW_CELLS = len(SLOW_EDGES) + 2
NONE = 2  # "home | away | none" for the touch and control parts


@dataclass(frozen=True)
class Counts:
    team: np.ndarray  # (54, 2): live frames, column 0 H, column 1 A
    slow: np.ndarray  # (7, 2): labelled frames, column 0 live, column 1 dead

    def __add__(self, other):
        return Counts(self.team + other.team, self.slow + other.slow)


def _team_parts(table, clues, ignore=None):
    home, away = nearest_by_team(table)
    with np.errstate(invalid="ignore"):
        margin = away - home  # NaN when the ball is unseen or neither team is visible
    near = np.where(np.isnan(margin), 0, 1 + np.digitize(margin, NEAR_EDGES))
    last_touch = carry_forward(touches(table, clues.touch_dv, clues.touch_radius, ignore), clues.carry_frames)
    owner = control(table, clues.control_radius, clues.control_min_frames)
    return near, np.where(last_touch < 0, NONE, last_touch), np.where(owner < 0, NONE, owner)


def _cells(near, last_touch, owner):
    return (near * 9 + last_touch * 3 + owner).astype(np.int16)


def team_cells(table, clues=DEFAULT_CLUES, ignore=None):
    """ignore: frames whose ball position is filled, not measured (never a touch)."""
    return _cells(*_team_parts(table, clues, ignore))


def team_cells_mirrored(table, clues=DEFAULT_CLUES, ignore=None):
    """The cells the same frames would have with home and away swapped."""
    near, last_touch, owner = _team_parts(table, clues, ignore)
    swap = np.array([1, 0, NONE])
    return _cells(np.where(near == 0, 0, 6 - near), swap[last_touch], swap[owner])


def slow_cells(table):
    speed = players_median_speed(table)
    return np.where(np.isnan(speed), N_SLOW_CELLS - 1, np.digitize(speed, SLOW_EDGES)).astype(np.int16)


def count(table, labels, clues=DEFAULT_CLUES):
    return count_cells(team_cells(table, clues), team_cells_mirrored(table, clues), slow_cells(table),
                       labels)


def count_cells(team_cell, team_cell_mirrored, slow_cell, labels):
    """count() from cells worked out already (the tuning reuses them across settings)."""
    team = np.zeros((N_TEAM_CELLS, 2), dtype=np.int64)
    live = (labels == H) | (labels == A)
    np.add.at(team, (team_cell[live], labels[live]), 1)
    np.add.at(team, (team_cell_mirrored[live], 1 - labels[live]), 1)

    slow = np.zeros((N_SLOW_CELLS, 2), dtype=np.int64)
    labelled = labels != X
    np.add.at(slow, (slow_cell[labelled], (labels[labelled] == D).astype(int)), 1)
    return Counts(team, slow)


def save_counts(counts, path):
    Path(path).write_text(json.dumps({"team": counts.team.tolist(), "slow": counts.slow.tolist()}),
                          encoding="utf-8")


def load_counts(path):
    saved = json.loads(Path(path).read_text(encoding="utf-8"))
    return Counts(np.array(saved["team"], dtype=np.int64), np.array(saved["slow"], dtype=np.int64))


def _probabilities(table_counts):
    """P(cell | column) with 1 added to every cell."""
    smoothed = table_counts + 1.0
    return smoothed / smoothed.sum(axis=0)


def emission_scores(table, counts, temperature, clues=DEFAULT_CLUES):
    """(n, 3) scores in column order H, A, D. Team clues never push towards or against D."""
    return scores_from_cells(team_cells(table, clues), slow_cells(table), counts, temperature)


def scores_from_cells(team_cell, slow_cell, counts, temperature):
    """emission_scores() from cells worked out already."""
    p_team, p_slow = _probabilities(counts.team), _probabilities(counts.slow)
    team = p_team[team_cell]
    slow = p_slow[slow_cell]
    scores = np.empty((len(team_cell), 3))
    scores[:, H] = np.log(team[:, 0]) + np.log(slow[:, 0])
    scores[:, A] = np.log(team[:, 1]) + np.log(slow[:, 0])
    scores[:, D] = np.log(team.mean(axis=1)) + np.log(slow[:, 1])
    return temperature * scores


def hint_weight(counts, temperature):
    """Decision 5: the typical H-vs-A pull of one live frame's team clues.

    The count-weighted median over team cells of temperature * |log P(c | H) - log P(c | A)|.
    """
    p_team = _probabilities(counts.team)
    pull = temperature * np.abs(np.log(p_team[:, 0]) - np.log(p_team[:, 1]))
    weight = counts.team.sum(axis=1)
    order = np.argsort(pull, kind="stable")
    cumulative = np.cumsum(weight[order])
    return float(pull[order][np.searchsorted(cumulative, cumulative[-1] / 2)])

"""A small learned scorer for who has the ball while play is live (Plan 3 step D).

It replaces only the H versus A part of the tally. The dead-ball side stays as it was: players'
speed from the tally, plus the fixed ball hints. Learned only on IDSSE live frames (the honest
part of IDSSE), mirrored home and away, leave one match out for every practice score.

Clues are computed from positions the same way on both datasets, per graded run. Never ball out or
ball still: IDSSE freezes its dead ball, so a model would learn a shortcut real tracking lacks.

The model is a logistic regression on binned clues, mirror-symmetric by construction: every input
is a team side (+1 home, -1 away, 0 neither) times a learned weight for a bin, so swapping home and
away flips the sign of the sum exactly and p becomes 1 - p. A team-neutral clue (ball speed, how
old the last touch is, ...) can only matter by making a team side count for more or less, so each
one is paired with each of the three team sides: the nearer team, the last touch, the control.
Plus the plan 2 tally's cells (nearness x last touch x control) as one block. Settings fixed in
advance: 8 quantile bins per clue plus a NaN bin, L2 1.0, at most 25 Newton steps.
"""
import json
from dataclasses import replace

import numpy as np

from remontada.evidence import NEAR_EDGES, NONE
from remontada.frames import run_slice
from remontada.gaps import fill_ball_gaps
from remontada.motion import BALL_HALF_WINDOW, speed
from remontada.touch import carry_forward, control, nearest_by_team, touches
from remontada.vendor.possession_metrics import A, H

FRAME_CLUES = ("margin", "nearest", "touch_team", "touch_age", "control_team", "control_age",
               "ball_speed", "ball_seen", "since_seen", "ball_filled", "abs_x", "abs_y")
SIGNED = ("margin", "touch_team", "control_team")  # flip sign when home and away swap
SIDES = SIGNED  # the three team sides every bin is multiplied by
SIZES = ("margin", "nearest", "touch_age", "control_age", "ball_speed", "ball_seen", "since_seen",
         "ball_filled", "abs_x", "abs_y")  # binned; "margin" is binned by its size |margin|
MARGIN_CLIP = 20.0  # m
N_BINS = 8  # quantile bins per clue, plus one for NaN
P_CLIP = 1e-4
EVERY = 5  # training uses every 5th frame of each case
_COL = {name: i for i, name in enumerate(FRAME_CLUES)}


# ---------------------------------------------------------------- clues


def _age(events_on, fps):
    """Seconds since the latest frame where events_on is True (0 on it), NaN before the first."""
    frames = np.arange(len(events_on))
    last = np.maximum.accumulate(np.where(events_on, frames, -1)) if len(frames) else frames
    return np.where(last >= 0, (frames - last) / fps, np.nan)


def _run_clues(part, clues):
    n, fps = part.n_frames, part.fps
    out = np.full((n, len(FRAME_CLUES)), np.nan)
    unseen = np.isnan(part.ball_xy).any(axis=1)
    filled = np.zeros(n, dtype=bool)
    if clues.gap_fill_frames > 0:
        ball_xy, filled = fill_ball_gaps(part.ball_xy, part.graded_runs, clues.gap_fill_frames)
        part = replace(part, ball_xy=ball_xy)
    home, away = nearest_by_team(part)
    with np.errstate(invalid="ignore"):
        margin = np.clip(away - home, -MARGIN_CLIP, MARGIN_CLIP)
    nearest = np.fmin(home, away)
    out[:, _COL["margin"]] = margin
    out[:, _COL["nearest"]] = np.where(np.isfinite(nearest), nearest, np.nan)

    touch = touches(part, clues.touch_dv, clues.touch_radius, filled)
    last = carry_forward(touch, clues.carry_frames)
    out[:, _COL["touch_team"]] = np.select([last == 0, last == 1], [1.0, -1.0], 0.0)
    out[:, _COL["touch_age"]] = np.where(last >= 0, _age(touch >= 0, fps), np.nan)

    owner = control(part, clues.control_radius, clues.control_min_frames)
    out[:, _COL["control_team"]] = np.select([owner == 0, owner == 1], [1.0, -1.0], 0.0)
    starts = np.concatenate(([True], owner[1:] != owner[:-1]))
    out[:, _COL["control_age"]] = np.where(owner >= 0, _age(starts, fps), np.nan)

    out[:, _COL["ball_speed"]] = speed(part.ball_xy, fps, BALL_HALF_WINDOW)
    out[:, _COL["ball_seen"]] = (~unseen).astype(float)
    out[:, _COL["since_seen"]] = _age(~unseen, fps)
    out[:, _COL["ball_filled"]] = filled.astype(float)
    out[:, _COL["abs_x"]] = np.abs(part.ball_xy[:, 0])
    out[:, _COL["abs_y"]] = np.abs(part.ball_xy[:, 1])
    return out


def frame_clues(table, clues):
    """(n, len(FRAME_CLUES)) float64, worked out on each graded run on its own. Frames outside
    every run know nothing: NaN, with no team side and no ball."""
    out = np.full((table.n_frames, len(FRAME_CLUES)), np.nan)
    for name in ("touch_team", "control_team", "ball_seen", "ball_filled"):
        out[:, _COL[name]] = 0.0
    for start, stop in table.graded_runs:
        if stop > start:
            out[start:stop] = _run_clues(run_slice(table, start, stop), clues)
    return out


def mirror(X):
    """The same frames with home and away swapped: the team-signed columns change sign."""
    out = np.array(X, dtype=np.float64, copy=True)
    for name in SIGNED:
        out[:, _COL[name]] = -out[:, _COL[name]]
    return out


def training_rows(X, labels, every=EVERY):
    """(rows, y): every `every`-th frame that is live (H or A), y 1 for H and 0 for A."""
    pick = np.zeros(len(labels), dtype=bool)
    pick[::every] = True
    pick &= (labels == H) | (labels == A)
    return X[pick], (labels[pick] == H).astype(np.int8)


# ---------------------------------------------------------------- the model


def _cell_pairs():
    """Each tally cell's column in the cell block and its sign: a cell and its mirror share a
    column with opposite signs; a cell that is its own mirror gets sign 0."""
    swap = np.array([1, 0, NONE])
    cells = np.arange(6 * 3 * 3)
    near, touch, ctrl = cells // 9, (cells // 3) % 3, cells % 3
    mirrored = np.where(near == 0, 0, 6 - near) * 9 + swap[touch] * 3 + swap[ctrl]
    canonical = np.minimum(cells, mirrored)
    reps = np.unique(canonical[cells != mirrored])
    column = np.searchsorted(reps, canonical)
    sign = np.where(cells == mirrored, 0, np.where(cells < mirrored, 1, -1))
    return np.where(sign == 0, 0, column), sign, len(reps)


CELL_COLUMN, CELL_SIGN, N_CELL_PAIRS = _cell_pairs()
N_BLOCKS = len(SIDES) * len(SIZES) + 1
N_WEIGHTS = len(SIDES) * len(SIZES) * (N_BINS + 1) + N_CELL_PAIRS


def _sides(X):
    return {name: np.nan_to_num(np.sign(X[:, _COL[name]])) for name in SIDES}


def _cells(X, sides):
    margin = X[:, _COL["margin"]]
    near = np.where(np.isnan(margin), 0, 1 + np.digitize(np.nan_to_num(margin), NEAR_EDGES))
    code = lambda side: np.select([side > 0, side < 0], [0, 1], NONE)  # noqa: E731
    return near * 9 + code(sides["touch_team"]) * 3 + code(sides["control_team"])


def _bin(values, edges):
    return np.where(np.isnan(values), N_BINS, np.digitize(np.nan_to_num(values), edges))


def _size(X, name):
    values = X[:, _COL[name]]
    return np.abs(values) if name == "margin" else values


def _design(X, edges):
    """(local index, global column, value) per row and block: one non-zero input per block."""
    sides = _sides(X)
    n = len(X)
    local = np.empty((n, N_BLOCKS), dtype=np.int64)
    value = np.empty((n, N_BLOCKS))
    b = 0
    for side in SIDES:
        for name in SIZES:
            local[:, b] = _bin(_size(X, name), edges[name])
            value[:, b] = sides[side]
            b += 1
    cells = _cells(X, sides)
    local[:, b], value[:, b] = CELL_COLUMN[cells], CELL_SIGN[cells]
    offsets = np.concatenate(([0], np.cumsum([N_BINS + 1] * (N_BLOCKS - 1))))
    return local, local + offsets, value, offsets


def _p(z):
    return np.exp(-np.logaddexp(0.0, -z))


class Scorer:
    def __init__(self, edges, weights, hint_weight):
        self.edges = {name: np.asarray(e, dtype=np.float64) for name, e in edges.items()}
        self.weights = np.asarray(weights, dtype=np.float64)
        self.hint_weight = float(hint_weight)

    def _z(self, X):
        _, column, value, _ = _design(np.asarray(X, dtype=np.float64), self.edges)
        return (self.weights[column] * value).sum(axis=1)

    def p_home(self, X):
        """P(home has the ball | clues), for live play."""
        return _p(self._z(X))

    @classmethod
    def fit(cls, X, y, l2=1.0, iterations=25, mirrored=False):
        """Newton steps on the L2-penalised log-likelihood, from all-zero weights.

        mirrored=True counts every row twice, as it is and mirrored with H and A swapped: the
        same as passing the mirrored copies, because the model is mirror-symmetric.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        edges = {}
        for name in SIZES:
            values = _size(X, name)
            values = values[~np.isnan(values)]
            qs = np.arange(1, N_BINS) / N_BINS
            edges[name] = np.quantile(values, qs, method="inverted_cdf") if len(values) else np.zeros(N_BINS - 1)
        _, column, value, _ = _design(X, edges)
        used = (value != 0).any(axis=1)  # rows with no team side at all cannot move any weight
        column, value, yu = column[used], value[used], y[used]
        scale = 2.0 if mirrored else 1.0
        w = np.zeros(N_WEIGHTS)
        for _ in range(iterations):
            p = _p((w[column] * value).sum(axis=1))
            grad = scale * np.bincount(column.ravel(), (value * (yu - p)[:, None]).ravel(), N_WEIGHTS) - l2 * w
            hess = _hessian(column, value, scale * p * (1 - p)) + l2 * np.eye(N_WEIGHTS)
            step = np.linalg.solve(hess, grad)
            w += step
            if np.abs(step).max() < 1e-10:
                break
        model = cls(edges, w, 0.0)
        p = np.clip(model.p_home(X), P_CLIP, 1 - P_CLIP)
        model.hint_weight = float(np.median(np.abs(np.log(p) - np.log(1 - p))))
        return model

    def to_json(self):
        return json.dumps({"frame_clues": list(FRAME_CLUES), "n_bins": N_BINS,
                           "edges": {name: e.tolist() for name, e in self.edges.items()},
                           "weights": self.weights.tolist(), "hint_weight": self.hint_weight}, indent=1)

    @classmethod
    def from_json(cls, text):
        saved = json.loads(text)
        if saved["frame_clues"] != list(FRAME_CLUES) or saved["n_bins"] != N_BINS:
            raise ValueError("scorer file was made for other clues")
        return cls(saved["edges"], saved["weights"], saved["hint_weight"])

    def weight_table(self):
        """[(side, clue, bin, weight)] for the technical report, the cell block left out."""
        rows, b = [], 0
        for side in SIDES:
            for name in SIZES:
                for k in range(N_BINS + 1):
                    rows.append((side, name, k, float(self.weights[b * (N_BINS + 1) + k])))
                b += 1
        return rows


def _hessian(column, value, weight, chunk=100_000):
    """sum over rows of weight * x x^T, built from dense chunks of the (mostly zero) inputs; the
    chunks are added in row order, so a rerun on the same machine gives the same numbers."""
    hess = np.zeros((N_WEIGHTS, N_WEIGHTS))
    for a in range(0, len(weight), chunk):
        dense = np.zeros((min(chunk, len(weight) - a), N_WEIGHTS))
        np.put_along_axis(dense, column[a:a + chunk], value[a:a + chunk], axis=1)
        hess += dense.T @ (dense * weight[a:a + chunk, None])
    return hess

"""The decision model: pick the H, A, D sequence that best fits the clues, where every change of
label costs something (Viterbi). Short blips that are not worth their switching cost go away.
"""
from dataclasses import dataclass

import numpy as np

from remontada.vendor.possession_metrics import A, D, H, X

SCORE_DECIMALS = 6  # scores are rounded before decoding, so float noise between machines cannot flip a choice


@dataclass(frozen=True)
class Costs:
    switch: float  # H to A or A to H
    to_dead: float  # H or A to D
    from_dead: float  # D to H or A


def _cost_matrix(costs):
    """cost[from, to]; staying costs 0."""
    cost = np.zeros((3, 3))
    cost[H, A] = cost[A, H] = costs.switch
    cost[H, D] = cost[A, D] = costs.to_dead
    cost[D, H] = cost[D, A] = costs.from_dead
    return cost


def viterbi(scores, costs, entry_bonus=None):
    """Codes maximising the summed scores minus transition costs. scores is (n, 3) or (b, n, 3).

    entry_bonus (same shape as scores, Plan 3 step C) makes moving into state k at frame t cheaper:
    a change from j to k costs max(cost[j, k] - entry_bonus[t, k], 0), never less than nothing, and
    staying still costs 0. Ties go to the lower code. One loop over frames, vectorised over the
    batch and the states.
    """
    single = scores.ndim == 2
    s = scores[None] if single else scores
    bonus = None if entry_bonus is None else (entry_bonus[None] if single else entry_bonus)
    batch, n, _ = s.shape
    out = np.empty((batch, n), dtype=np.int8)
    if n == 0:
        return out[0] if single else out
    cost = _cost_matrix(costs)
    stay = np.eye(3, dtype=bool)
    back = np.empty((batch, n, 3), dtype=np.int8)
    best = s[:, 0].astype(np.float64)
    for t in range(1, n):
        if bonus is None:
            step_cost = cost
        else:
            step_cost = np.maximum(cost - bonus[:, t, None, :], 0.0)  # (batch, from, to)
            step_cost[:, stay] = 0.0
        options = best[:, :, None] - step_cost  # (batch, from, to)
        back[:, t] = options.argmax(axis=1)  # first maximum, so the lower code on a tie
        best = options.max(axis=1) + s[:, t]
    rows = np.arange(batch)
    out[:, -1] = best.argmax(axis=1)
    for t in range(n - 1, 0, -1):
        out[:, t - 1] = back[rows, t, out[:, t]]
    return out[0] if single else out


def decode_runs(scores, runs, costs, entry_bonus=None):
    """X outside runs; each run decoded on its own, all runs in one batch.

    Scores are rounded to SCORE_DECIMALS first. Runs are padded at the end with zero scores (and
    zero bonus), which cannot change the best path before the padding (staying costs 0 and every
    switch costs at least 0).
    """
    scores = np.round(scores, SCORE_DECIMALS)
    out = np.full(len(scores), X, dtype=np.int8)
    runs = [(a, b) for a, b in runs if b > a]
    if not runs:
        return out
    longest = max(b - a for a, b in runs)
    batch = np.zeros((len(runs), longest, 3))
    bonus = None if entry_bonus is None else np.zeros((len(runs), longest, 3))
    for i, (a, b) in enumerate(runs):
        batch[i, :b - a] = scores[a:b]
        if bonus is not None:
            bonus[i, :b - a] = entry_bonus[a:b]
    paths = viterbi(batch, costs, bonus)
    for i, (a, b) in enumerate(runs):
        out[a:b] = paths[i, :b - a]
    return out

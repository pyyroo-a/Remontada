"""The decision model: pick the H, A, D sequence that best fits the clues, where every change of
label costs something (Viterbi). Short blips that are not worth their switching cost go away.
"""
from dataclasses import dataclass

import numpy as np

from remontada.vendor.possession_metrics import A, D, H, X


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


def viterbi(scores, costs):
    """Codes maximising the summed scores minus transition costs. scores is (n, 3) or (b, n, 3).

    Ties go to the lower code. One loop over frames, vectorised over the batch and the states.
    """
    single = scores.ndim == 2
    s = scores[None] if single else scores
    batch, n, _ = s.shape
    out = np.empty((batch, n), dtype=np.int8)
    if n == 0:
        return out[0] if single else out
    cost = _cost_matrix(costs)
    back = np.empty((batch, n, 3), dtype=np.int8)
    best = s[:, 0].astype(np.float64)
    for t in range(1, n):
        options = best[:, :, None] - cost  # (batch, from, to)
        back[:, t] = options.argmax(axis=1)  # first maximum, so the lower code on a tie
        best = options.max(axis=1) + s[:, t]
    rows = np.arange(batch)
    out[:, -1] = best.argmax(axis=1)
    for t in range(n - 1, 0, -1):
        out[:, t - 1] = back[rows, t, out[:, t]]
    return out[0] if single else out


def decode_runs(scores, runs, costs):
    """X outside runs; each run decoded on its own, all runs in one batch.

    Runs are padded at the end with zero scores, which cannot change the best path before the
    padding (staying costs 0 and every switch costs at least 0).
    """
    out = np.full(len(scores), X, dtype=np.int8)
    runs = [(a, b) for a, b in runs if b > a]
    if not runs:
        return out
    longest = max(b - a for a, b in runs)
    batch = np.zeros((len(runs), longest, 3))
    for i, (a, b) in enumerate(runs):
        batch[i, :b - a] = scores[a:b]
    paths = viterbi(batch, costs)
    for i, (a, b) in enumerate(runs):
        out[a:b] = paths[i, :b - a]
    return out

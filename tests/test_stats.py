import numpy as np

from remontada.stats import possession_summary
from remontada.vendor.possession_metrics import A, D, H, X


def test_summary_counts_runs_inside_halves_only():
    codes = np.array([H, H, A, A, A, D, X, D, H, H], dtype=np.int8)
    s = possession_summary(codes, [(0, 6), (7, 10)], fps=1)
    assert s["n_possessions"] == 3
    assert s["n_dead"] == 2
    assert s["n_under_1s"] == 0
    assert s["mean_s"] == (2 + 3 + 2) / 3
    assert s["alive_share"] == 7 / 9

import numpy as np

from remontada.scoreboard import mask_ungraded, score
from remontada.vendor.possession_metrics import A, D, H, X


def test_mask_ungraded():
    out = mask_ungraded(np.array([H, H, A, A, D], dtype=np.int8), [(1, 3)])
    assert out.tolist() == [X, H, A, X, X]


def test_score_reports_the_headline_numbers():
    gt = np.array([H, H, H, D, A, A, A, X], dtype=np.int8)
    s = score(gt, gt)
    assert s["spell_f1"] == 1.0
    assert s["macro_f1"] == 1.0
    assert s["n_pred_spells"] == s["n_gt_spells"] == 2


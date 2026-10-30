import numpy as np
import pytest

from remontada.baseline import nearest_team, smooth
from remontada.load_bar_ath import load_bar_ath
from remontada.scoreboard import mask_ungraded, score
from remontada.vendor import PUBLIC_GT_PATH
from remontada.vendor import possession_metrics as pm
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


@pytest.mark.realdata
def test_bar_ath_public_window_matches_readme():
    gt = pm.label_codes(pm.read_label_csv(PUBLIC_GT_PATH))
    s = score(gt, smooth(nearest_team(load_bar_ath()), 50))
    assert round(s["spell_f1"], 4) == 0.3256
    assert round(s["macro_f1"], 4) == 0.6207

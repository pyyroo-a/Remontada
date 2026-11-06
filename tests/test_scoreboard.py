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



def test_idsse_features_use_the_given_clues(monkeypatch):
    # the scoreboard must score the decoder we ship, clue settings included
    from dataclasses import replace
    from remontada import lomo
    from remontada.clues import DEFAULT_CLUES
    seen = []
    monkeypatch.setattr(lomo, "MATCH_IDS", ["m1"])
    monkeypatch.setattr(lomo, "load_idsse_cached", lambda match_id: ("table", "labels"))
    monkeypatch.setattr(lomo, "degrade", lambda table, level, seed: table)
    monkeypatch.setattr(lomo, "features", lambda table, clues=DEFAULT_CLUES: seen.append(clues))
    clues = replace(DEFAULT_CLUES, control_radius=2.0, control_min_frames=5)
    lomo.idsse_features(0, clues)
    assert len(seen) == 4 and all(c == clues for c in seen)

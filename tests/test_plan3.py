import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


def _plan3():
    import plan3
    return plan3


def _scores(value_by_level, matches=("m1", "m2")):
    """Fake per-case scores: one spellF1 per level, the same on every match, style and seed."""
    from remontada import practice as p
    return {(m, lv, st, s): (value_by_level[lv], 0.5) for m in matches for lv in p.LEVEL_NAMES
            for st in p.STYLES for s in p.SEEDS}


def _write(tmp_path, a_levels, v2_levels):
    from remontada import practice as p
    lv = dict(zip(p.LEVEL_NAMES, (0.6, 0.5, 0.4, 0.3)))
    p.save_results(tmp_path / "start.json", {"plan2": _scores(lv), "v2": _scores(v2_levels)})
    state = {"steps": {"A-clues": {"verdict": "KEEP", "levels": a_levels, "contexts": {}}}}
    (tmp_path / "state.json").write_text(json.dumps(state), encoding="utf-8")


def _levels(none, light, medium, heavy):
    return {"none": none, "light": light, "medium": medium, "heavy": heavy}


def _args(tmp_path):
    return type("Args", (), {"results_dir": tmp_path, "matches": ["m1", "m2"]})()


def test_compare_v2_keeps_a_rerun_that_clears_the_bar(tmp_path):
    _write(tmp_path, _levels(0.60, 0.55, 0.475, 0.32), _levels(0.62, 0.56, 0.45, 0.32))
    assert _plan3().compare_v2(_args(tmp_path)) == "KEEP"  # medium + heavy up 0.0125
    saved = json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))["steps"]["A-vs-v2"]
    assert saved["verdict"] == "KEEP" and saved["against"] == "v2"


def test_compare_v2_drops_a_rerun_under_the_bar(tmp_path):
    _write(tmp_path, _levels(0.60, 0.55, 0.46, 0.32), _levels(0.62, 0.56, 0.45, 0.32))
    assert _plan3().compare_v2(_args(tmp_path)) == "DROP"  # up only 0.005


def test_compare_v2_drops_a_rerun_that_falls_on_clean_data(tmp_path):
    _write(tmp_path, _levels(0.55, 0.55, 0.50, 0.35), _levels(0.62, 0.56, 0.45, 0.32))
    assert _plan3().compare_v2(_args(tmp_path)) == "DROP"  # none down 0.07, over the 0.04 guard


def test_freeze_refuses_when_the_rerun_was_dropped(tmp_path):
    _write(tmp_path, _levels(0.60, 0.55, 0.46, 0.32), _levels(0.62, 0.56, 0.45, 0.32))
    _plan3().compare_v2(_args(tmp_path))
    with pytest.raises(SystemExit, match="stays"):
        _plan3().freeze(_args(tmp_path))

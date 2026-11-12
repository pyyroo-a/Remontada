"""The results page is printed by scripts/results_tables.py from the committed results/ files."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import results_tables  # noqa: E402
from remontada import practice

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "docs" / "results.md"


def _toy_start(path):
    """Two matches, every level, mess style and seed; each point scores a fixed number per level."""
    value = {"baseline": 0.1, "plan2": 0.3, "v2": 0.4}
    step = {"none": 0.3, "light": 0.2, "medium": 0.1, "heavy": 0.0}
    results = {}
    for point, base in value.items():
        results[point] = {}
        for match in ("M1", "M2"):
            for level in practice.LEVEL_NAMES:
                for style in practice.STYLES:
                    for seed in practice.SEEDS:
                        bonus = 0.05 if style.endswith("smooth") else 0.0
                        results[point][(match, level, style, seed)] = (base + step[level] + bonus, 0.5)
    practice.save_results(path, results)


def test_v4_levels_from_toy_results(tmp_path):
    _toy_start(tmp_path / "start.json")
    rerun = {"none": 0.7, "light": 0.6, "medium": 0.5, "heavy": 0.4}
    (tmp_path / "state.json").write_text(json.dumps({"steps": {"A-clues": {"levels": rerun}}}), encoding="utf-8")
    table = results_tables.v4_levels(tmp_path)
    lines = table.splitlines()
    assert lines[0].startswith("| Messiness |")
    assert len(lines) == 2 + 4 + 1  # header, rule, four levels, medium + heavy
    # Plan 2 at medium: 0.3 + 0.1, plus 0.05 on half the cases (smooth ball) = 0.4250
    assert "| medium | 0.2250 | 0.4250 | 0.5250 | 0.5000 |" in table
    # medium + heavy: Plan 2 (0.4250 + 0.3250) / 2 = 0.3750, rerun (0.5 + 0.4) / 2 = 0.4500
    assert lines[-1] == "| **medium + heavy** | 0.1750 | 0.3750 | **0.4750** | 0.4500 |"


def test_style_splits_from_toy_results(tmp_path):
    _toy_start(tmp_path / "start.json")
    table = results_tables.v4_by_style(tmp_path)
    # smooth ball, Plan 2: medium 0.45 and heavy 0.35, so 0.4000; rough: 0.40 and 0.30, so 0.3500
    assert "| smooth ball | 0.4000 | 0.5000 |" in table
    assert "| rough ball | 0.3500 | 0.4500 |" in table
    assert "| ball hidden at random | 0.3750 | 0.4750 |" in table


@pytest.mark.parametrize("name", ["v3-levels", "v4-levels", "v4-by-style", "steps", "selection-gap"])
def test_results_page_matches_the_saved_results(name):
    table = results_tables.tables(ROOT / "results")[name]
    assert table in PAGE.read_text(encoding="utf-8")


def test_results_page_has_no_dashes():
    text = PAGE.read_text(encoding="utf-8")
    assert "—" not in text and "–" not in text

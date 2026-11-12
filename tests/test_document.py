"""The technical document: the organisers' six headings, the team authors block, numbers that match
results/, working links, and a PDF export that fails loudly without its tools."""
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import export_pdf  # noqa: E402
import results_tables  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "remontada.md"
AUTHORS = ["Vimukthi Ashen", "Mohammed Sherin", "Hussain Abdul Kadhir", "Mishal Ashraf", "Aman Mohammed"]
HEADINGS = ["Remontada in one minute", "Approach", "Assets used and dropped", "Training data",
            "Possession rules", "Limitations", "Reproduction"]


def _doc():
    return DOC.read_text(encoding="utf-8")


def test_six_headings_in_order():
    found = re.findall(r"^## (.+?)\s*$", _doc(), re.M)
    assert found == HEADINGS


def test_authors_block_for_the_team():
    head = _doc().split("\n## ", 1)[0]
    assert "Team Remontada" in head
    authors = [line for line in head.splitlines() if line.lstrip("*_ ").startswith("Authors")]
    assert authors
    positions = [authors[0].find(name) for name in AUTHORS]
    assert -1 not in positions and positions == sorted(positions), positions  # every name, in this order
    assert "Member" not in head


def _medium_heavy(table):
    return re.findall(r"\| \*\*medium \+ heavy\*\* \|(.+)\|", table)[0].replace("*", "").split("|")


def test_document_numbers_match_the_results():
    tables = results_tables.tables(ROOT / "results")
    text = _doc()
    for name in ("v3-levels", "v4-levels"):
        for value in _medium_heavy(tables[name]):
            assert value.strip() in text, (name, value)


def test_document_has_no_dashes():
    text = _doc()
    assert "—" not in text and "–" not in text


def test_document_links_exist():
    for target in re.findall(r"\]\(([^)#]+)(?:#[^)]*)?\)", _doc()):
        if "://" in target:
            continue
        assert (DOC.parent / target).resolve().exists(), target


def test_export_reports_missing_tools(monkeypatch):
    monkeypatch.setattr(export_pdf.shutil, "which", lambda name: None)
    monkeypatch.setattr(export_pdf, "EDGE_PATHS", ())
    with pytest.raises(SystemExit, match="pandoc"):
        export_pdf.main([])

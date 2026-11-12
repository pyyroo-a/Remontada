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
    assert "\u2014" not in text and "\u2013" not in text


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


def test_pdf_has_no_local_links():
    pdf = (ROOT / "docs" / "remontada.pdf").read_bytes()
    assert b"file:" not in pdf and b"AppData" not in pdf


def test_relative_links_become_github_links():
    text = "[a](results.md) and [b](../README.md#setup) and [c](https://x.org/y)"
    out = export_pdf.absolute_links(text, ROOT / "docs")
    assert "(https://github.com/pyyroo-a/Remontada/blob/main/docs/results.md)" in out
    assert "(https://github.com/pyyroo-a/Remontada/blob/main/README.md#setup)" in out
    assert "(https://x.org/y)" in out


def test_export_fails_when_the_browser_writes_nothing(monkeypatch, tmp_path):
    stale = tmp_path / "old.pdf"
    stale.write_bytes(b"%PDF old")
    monkeypatch.setattr(export_pdf.shutil, "which", lambda name: "C:/fake/" + name)
    monkeypatch.setattr(export_pdf.subprocess, "run", lambda *a, **k: None)
    with pytest.raises(SystemExit, match="did not write"):
        export_pdf.main(["--out", str(stale)])


def test_document_claims_only_what_was_measured():
    text = _doc()
    for overclaim in ("Every score is", "far below", "did not fit", "exactly where possession changes",
                      "only input at prediction time"):
        assert overclaim not in text, overclaim
    assert "issue #15" in text  # the look-ahead permission, cited
    training = text.split("## Training data", 1)[1].split("## Possession rules", 1)[0]
    assert "labelled" in training and "none by us" in training

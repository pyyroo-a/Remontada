import pytest

from remontada.make_submission import main
from remontada.submission import sha256
from remontada.vendor import possession_metrics as pm


@pytest.mark.realdata
def test_submission_is_valid_and_deterministic(tmp_path):
    a, b = (main(["--method", "baseline", "--out", str(tmp_path / f"{i}.csv")]) for i in range(2))
    assert pm.validate_submission(pm.read_label_csv(a), 199319) is None
    assert sha256(a) == sha256(b)


@pytest.mark.realdata
def test_hmm_submission_is_valid_and_deterministic(tmp_path):
    a, b = (main(["--method", "hmm", "--out", str(tmp_path / f"{i}.csv")]) for i in range(2))
    assert pm.validate_submission(pm.read_label_csv(a), 199319) is None
    assert sha256(a) == sha256(b)


@pytest.mark.realdata
def test_submission_prints_validity_only(tmp_path, capsys):
    # rules clarification (4 Nov 2026): from our match Claude may only see that the file is valid
    main(["--method", "hmm", "--out", str(tmp_path / "s.csv")])
    out = capsys.readouterr().out
    shows_counts = any(word in out for word in ("labels", "possessions", "mean"))
    shows_validity = "valid" in out and "sha256" in out and "rows 199319" in out
    assert not shows_counts
    assert shows_validity

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

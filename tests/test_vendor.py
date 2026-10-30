import hashlib
from pathlib import Path

from remontada.vendor import PUBLIC_GT_PATH
from remontada.vendor import possession_metrics as pm

# SHA-256 of the upstream files at metrica-sports/more-than-a-hack-2027 commit f8f2629, with \n
# line endings, so the check gives the same answer whatever git does to line endings on Windows.
VENDORED_SHA = "341bad48904b489d6e875397b27687f5d6a4d22a397f3dc8af3a00d48f16874d"
PUBLIC_GT_SHA = "e596a9af9626c623bb94be6fd350e12ee9a75c76a696cfadca3bc01735e92487"


def sha256_of(path):
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def test_vendored_scorer_is_unchanged():
    assert sha256_of(pm.__file__) == VENDORED_SHA
    assert sha256_of(PUBLIC_GT_PATH) == PUBLIC_GT_SHA


def test_public_gt_scores_itself_perfectly():
    gt = pm.label_codes(pm.read_label_csv(PUBLIC_GT_PATH))
    result = pm.evaluate(gt, gt)
    assert len(gt) == 199319
    assert result["spell_f1_mean"] == 1.0
    assert result["n_gt_spells"] == 17
    assert pm.valid_runs(gt) == [(11185, 18685)]

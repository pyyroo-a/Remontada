import hashlib

from remontada.vendor import PUBLIC_GT_PATH
from remontada.vendor import possession_metrics as pm

# SHA-256 of the upstream files at metrica-sports/more-than-a-hack-2027 commit f8f2629.
VENDORED_SHA = "32c878630c74fb52a90c301485d6dff806509d8e933b1177b6035ee8e8f42b80"
PUBLIC_GT_SHA = "24d13807dfd2173b3b567a5f1fa0c597d42653a21d808da6027f98583d446e34"


def sha256_of(path):
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


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

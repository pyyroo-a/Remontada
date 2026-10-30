"""Score label arrays with the official metric code. Summary numbers only."""
import numpy as np

from remontada.vendor import possession_metrics as pm


def score(gt, pred):
    """The headline numbers from pm.evaluate."""
    result = pm.evaluate(gt, pred)
    return {
        "spell_f1": result["spell_f1_mean"],
        "macro_f1": result["macro_f1"],
        "n_pred_spells": result["n_pred_spells"],
        "n_gt_spells": result["n_gt_spells"],
    }


def mask_ungraded(codes, runs):
    """A copy of the labels with X everywhere outside the graded runs."""
    masked = np.full(len(codes), pm.X, dtype=np.int8)
    for start, stop in runs:
        masked[start:stop] = codes[start:stop]
    return masked

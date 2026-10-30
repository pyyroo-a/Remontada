"""Score the baseline (or later, our pipeline) on every IDSSE match at every messiness level.

    python scripts/scoreboard.py

BAR-ATH is scored on the public 5 minutes only, as a pipeline check, never to choose anything.
"""
import numpy as np

from remontada.baseline import nearest_team, smooth
from remontada.degrade import LEVELS, degrade
from remontada.load_bar_ath import load_bar_ath
from remontada.load_idsse import MATCH_IDS, load_idsse
from remontada.scoreboard import score
from remontada.vendor import PUBLIC_GT_PATH
from remontada.vendor import possession_metrics as pm

WINDOW = 50
SEED = 0


def main():
    print("IDSSE baseline (+/-2 s): spellF1 / macroF1 / possessions (truth in brackets)")
    print("match    " + "".join("{:<26}".format(name) for name in LEVELS))
    totals = {name: [] for name in LEVELS}
    for match_id in MATCH_IDS:
        table, labels = load_idsse(match_id)
        cells = []
        for name, level in LEVELS.items():
            s = score(labels, smooth(nearest_team(degrade(table, level, SEED)), WINDOW))
            totals[name].append((s["spell_f1"], s["macro_f1"]))
            cells.append("{:.4f} {:.4f} {:4d} ({:d})".format(
                s["spell_f1"], s["macro_f1"], s["n_pred_spells"], s["n_gt_spells"]).ljust(26))
        print("{}   {}".format(match_id, "".join(cells)))
    means = [np.mean(totals[name], axis=0) for name in LEVELS]
    print("mean     " + "".join("{:.4f} {:.4f}".format(*m).ljust(26) for m in means))

    gt = pm.label_codes(pm.read_label_csv(PUBLIC_GT_PATH))
    s = score(gt, smooth(nearest_team(load_bar_ath()), WINDOW))
    print("\nBAR-ATH public 5 minutes (pipeline check only): spellF1 {:.4f} macroF1 {:.4f}".format(
        s["spell_f1"], s["macro_f1"]))


if __name__ == "__main__":
    main()

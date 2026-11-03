"""Score the baseline or our decoder on every IDSSE match at every messiness level.

    python scripts/scoreboard.py                              # the official baseline
    python scripts/scoreboard.py --method hmm                 # our decoder, settings in hmm_v1.json
    python scripts/scoreboard.py --method hmm --no-ball-hints

For the decoder, the evidence table that scores a match is learned on the other six matches.
BAR-ATH is scored on the public 5 minutes only, as a pipeline check, never to choose anything.
"""
import argparse
from dataclasses import replace

import numpy as np

from remontada import pipeline
from remontada.baseline import nearest_team, smooth
from remontada.degrade import LEVELS, degrade
from remontada.evidence import load_counts
from remontada.load_bar_ath import load_bar_ath
from remontada.load_idsse import MATCH_IDS, load_idsse_cached
from remontada.lomo import counts_by_item, decode_all, idsse_features, leave_one_out_counts
from remontada.scoreboard import score
from remontada.smart_tagging import load_smart_tagging
from remontada.vendor import PUBLIC_GT_PATH
from remontada.vendor import possession_metrics as pm

WINDOW = 50
SEED = 0


def baseline_codes():
    codes, labels = {}, {}
    for match_id in MATCH_IDS:
        table, labels[match_id] = load_idsse_cached(match_id)
        for name, level in LEVELS.items():
            codes[match_id, name] = smooth(nearest_team(degrade(table, level, SEED)), WINDOW)
    return codes, labels


def hmm_codes(settings):
    feats, labels = idsse_features(SEED)
    return decode_all(feats, leave_one_out_counts(counts_by_item(feats, labels)), settings), labels


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--method", default="baseline", choices=["baseline", "hmm"])
    parser.add_argument("--no-ball-hints", action="store_true")
    args = parser.parse_args(argv)

    settings = pipeline.load_settings()
    if args.no_ball_hints:
        settings = replace(settings, ball_hints=False)
    codes, labels = baseline_codes() if args.method == "baseline" else hmm_codes(settings)

    print("IDSSE {} (+/-2 s): spellF1 / macroF1 / possessions (truth in brackets)".format(
        args.method + (" without ball hints" if args.no_ball_hints else "")))
    print("match    " + "".join("{:<26}".format(name) for name in LEVELS))
    totals = {name: [] for name in LEVELS}
    for match_id in MATCH_IDS:
        cells = []
        for name in LEVELS:
            s = score(labels[match_id], codes[match_id, name])
            totals[name].append((s["spell_f1"], s["macro_f1"]))
            cells.append("{:.4f} {:.4f} {:4d} ({:d})".format(
                s["spell_f1"], s["macro_f1"], s["n_pred_spells"], s["n_gt_spells"]).ljust(26))
        print("{}   {}".format(match_id, "".join(cells)))
    means = [np.mean(totals[name], axis=0) for name in LEVELS]
    print("mean     " + "".join("{:.4f} {:.4f}".format(*m).ljust(26) for m in means))

    table = load_bar_ath()
    if args.method == "baseline":
        pred = smooth(nearest_team(table), WINDOW)
    else:
        pred = pipeline.predict(table, load_counts(pipeline.EVIDENCE_PATH), settings,
                                load_smart_tagging())
    gt = pm.label_codes(pm.read_label_csv(PUBLIC_GT_PATH))
    s = score(gt, pred)
    print("\nBAR-ATH public 5 minutes (pipeline check only): spellF1 {:.4f} macroF1 {:.4f}".format(
        s["spell_f1"], s["macro_f1"]))


if __name__ == "__main__":
    main()

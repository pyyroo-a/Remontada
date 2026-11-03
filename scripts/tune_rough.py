"""A rough first choice of the decoder's settings on IDSSE, leaving each scored match out.

    python scripts/tune_rough.py                 # the grid, then the best setting
    python scripts/tune_rough.py --write-counts  # learn the table from all 7 matches at all levels

Rough on purpose: plan 3 replaces it with full leave-one-match-out tuning.
"""
import argparse
import itertools
import time

import numpy as np

from remontada.degrade import LEVELS
from remontada.evidence import save_counts
from remontada.hmm import Costs
from remontada.lomo import counts_by_item, decode_all, idsse_features, leave_one_out_counts
from remontada.pipeline import EVIDENCE_PATH, Settings
from remontada.scoreboard import score

TEMPERATURES = (0.1, 0.2, 0.4)
SWITCH = (4, 8, 16, 32)
DEAD = (4, 8, 16, 32)  # to_dead = from_dead


def level_means(codes, labels):
    return [float(np.mean([score(labels[m], codes[m, name])["spell_f1"] for m in labels]))
            for name in LEVELS]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--write-counts", action="store_true")
    args = parser.parse_args(argv)

    start = time.perf_counter()
    feats, labels = idsse_features()
    by_item = counts_by_item(feats, labels)
    print("features ready in {:.0f} s".format(time.perf_counter() - start), flush=True)
    if args.write_counts:
        total = sum_counts(by_item.values())
        save_counts(total, EVIDENCE_PATH)
        print("wrote {} ({} team frames, {} slow frames)".format(
            EVIDENCE_PATH.name, total.team.sum(), total.slow.sum()))
        return

    loo = leave_one_out_counts(by_item)
    print("mean spellF1 per level: " + " ".join(LEVELS) + "   (setting: temperature switch dead)")
    results = []
    for t, sw, dead in itertools.product(TEMPERATURES, SWITCH, DEAD):
        settings = Settings(t, Costs(sw, dead, dead), ball_hints=True, tag_hints=False)
        means = level_means(decode_all(feats, loo, settings), labels)
        results.append((float(np.mean(means)), (t, sw, dead), means))
        print("{:.1f} {:2d} {:2d}   ".format(t, sw, dead) + " ".join("{:.4f}".format(m) for m in means)
              + "   mean {:.4f}".format(results[-1][0]), flush=True)
    best = max(results, key=lambda r: r[0])
    print("best: temperature {} switch {} dead {} -> mean of level means {:.4f} ({:.0f} s)".format(
        *best[1], best[0], time.perf_counter() - start))


def sum_counts(counts):
    total = None
    for c in counts:
        total = c if total is None else total + c
    return total


if __name__ == "__main__":
    main()

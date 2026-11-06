"""A rough first choice of the decoder's settings on IDSSE, leaving each scored match out.

    python scripts/tune_rough.py                 # the grid, then the best setting

Rough on purpose: plan 3 replaced it with nested tuning (scripts/plan3.py). Kept for the record of
how plan 2 was tuned. It no longer writes evidence_idsse.json: only `plan3.py freeze` does, so the
shipped tally always matches the decoder that was measured.
"""
import argparse
import itertools
import time

import numpy as np

from remontada.degrade import LEVELS
from remontada.hmm import Costs
from remontada.lomo import counts_by_item, decode_all, idsse_features, leave_one_out_counts
from remontada.pipeline import Settings
from remontada.scoreboard import score

TEMPERATURES = (0.1, 0.2, 0.4)
SWITCH = (4, 8, 16, 32)
DEAD = (4, 8, 16, 32)  # to_dead = from_dead


def level_means(codes, labels):
    return [float(np.mean([score(labels[m], codes[m, name])["spell_f1"] for m in labels]))
            for name in LEVELS]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    args = parser.parse_args(argv)

    start = time.perf_counter()
    feats, labels = idsse_features()
    by_item = counts_by_item(feats, labels)
    print("features ready in {:.0f} s".format(time.perf_counter() - start), flush=True)
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


if __name__ == "__main__":
    main()

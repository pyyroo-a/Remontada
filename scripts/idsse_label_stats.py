"""Possession statistics for all 7 IDSSE matches: the convention check from the design.

    python scripts/idsse_label_stats.py
"""
import numpy as np

from remontada.load_idsse import MATCH_IDS, load_idsse
from remontada.stats import possession_summary

COLUMNS = ("n_possessions", "mean_s", "median_s", "n_under_1s", "n_dead", "mean_dead_s", "alive_share")


def main():
    rows = []
    print("match    poss  mean_s  median_s  under_1s  stops  stop_s  alive")
    for match_id in MATCH_IDS:
        table, labels = load_idsse(match_id)
        s = possession_summary(labels, table.graded_runs, table.fps)
        rows.append([s[c] for c in COLUMNS])
        print("{}  {:4d}  {:6.1f}  {:8.1f}  {:8d}  {:5d}  {:6.1f}  {:5.1%}".format(match_id, *rows[-1]))
    mean = np.mean(rows, axis=0)
    print("mean    {:5.0f}  {:6.1f}  {:8.1f}  {:8.0f}  {:5.0f}  {:6.1f}  {:5.1%}".format(*mean))


if __name__ == "__main__":
    main()

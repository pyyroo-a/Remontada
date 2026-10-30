"""Regenerate our submission from the data bundle with one command.

    python -m remontada.make_submission --method baseline --out outputs/remontada.csv

Prints summary numbers only: never rows, positions or frames of the match data.
"""
import argparse
from collections import Counter
from pathlib import Path

from remontada.baseline import nearest_team, smooth
from remontada.load_bar_ath import load_bar_ath
from remontada.scoreboard import mask_ungraded
from remontada.stats import possession_summary
from remontada.submission import sha256, write_csv
from remontada.vendor import possession_metrics as pm

N_ROWS = 199319
BASELINE_WINDOW = 50


def predict(method):
    """Label codes for every frame of BAR-ATH, and the graded runs."""
    table = load_bar_ath()
    if method == "baseline":
        return smooth(nearest_team(table), BASELINE_WINDOW), table.graded_runs
    raise ValueError("unknown method: {}".format(method))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--method", default="baseline", choices=["baseline"])
    parser.add_argument("--out", default="outputs/remontada.csv")
    args = parser.parse_args(argv)

    labels, runs = predict(args.method)
    masked = mask_ungraded(labels, runs)
    path = write_csv(masked, Path(args.out))

    problem = pm.validate_submission(pm.read_label_csv(path), N_ROWS)
    if problem:
        raise SystemExit("invalid submission: {}".format(problem))

    counts = Counter(pm.LABELS[code] for code in masked.tolist())
    summary = possession_summary(masked, runs)
    print("rows {}  labels {}".format(len(masked), dict(sorted(counts.items()))))
    print("possessions {}  mean {:.1f} s".format(summary["n_possessions"], summary["mean_s"]))
    print("sha256 {}".format(sha256(path)))
    return path


if __name__ == "__main__":
    main()

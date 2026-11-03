"""Regenerate our submission from the data bundle with one command.

    python -m remontada.make_submission --method hmm --out outputs/remontada_hmm.csv

Prints only that the file is valid (row count, official checker, fingerprint): never label or
possession counts, rows, positions or frames of the match data (rules clarification, 4 Nov 2026).
"""
import argparse
from pathlib import Path

from remontada import pipeline
from remontada.baseline import nearest_team, smooth
from remontada.evidence import load_counts
from remontada.load_bar_ath import load_bar_ath
from remontada.scoreboard import mask_ungraded
from remontada.smart_tagging import load_smart_tagging
from remontada.submission import sha256, write_csv
from remontada.vendor import possession_metrics as pm

N_ROWS = 199319
BASELINE_WINDOW = 50


def predict(method):
    """Label codes for every frame of BAR-ATH, and the graded runs."""
    table = load_bar_ath()
    if method == "baseline":
        return smooth(nearest_team(table), BASELINE_WINDOW), table.graded_runs
    if method == "hmm":
        counts = load_counts(pipeline.EVIDENCE_PATH)
        codes = pipeline.predict(table, counts, pipeline.load_settings(), load_smart_tagging())
        return codes, table.graded_runs
    raise ValueError("unknown method: {}".format(method))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--method", default="baseline", choices=["baseline", "hmm"])
    parser.add_argument("--out", default="outputs/remontada.csv")
    args = parser.parse_args(argv)

    labels, runs = predict(args.method)
    masked = mask_ungraded(labels, runs)
    path = write_csv(masked, Path(args.out))

    problem = pm.validate_submission(pm.read_label_csv(path), N_ROWS)
    if problem:
        raise SystemExit("invalid submission: {}".format(problem))

    # Validity only (rules clarification, 4 Nov 2026): no label or possession counts from our match.
    print("rows {}  valid".format(len(masked)))
    print("sha256 {}".format(sha256(path)))
    return path


if __name__ == "__main__":
    main()

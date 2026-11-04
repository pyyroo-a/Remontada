"""Plan 3 practice runs on IDSSE (external data only, CC BY 4.0), every number nested.

    python scripts/plan3.py start   # baseline and Plan 2 on degrade v3: the numbers to beat

Results go to results/plan3/*.json (IDSSE scores only). Nothing here touches our match: the rules
(clarified 4 Nov 2026) allow tuning and checking only on external data.
"""
import argparse
import sys
import time
from pathlib import Path

from remontada import practice
from remontada.baseline import nearest_team, smooth
from remontada.degrade import DEGRADE_VERSION, LEVELS, degrade
from remontada.load_idsse import MATCH_IDS, load_idsse_cached
from remontada.pipeline import CONFIG_DIR, features, load_settings
from remontada.scoreboard import score

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results" / "plan3"
BASELINE_WINDOW = 50


def _log(start, text):
    print("[{:5.0f} s] {}".format(time.perf_counter() - start, text), flush=True)


def print_levels(title, levels, macro=None):
    row = " ".join("{:>7.4f}".format(levels[lv]) for lv in practice.LEVEL_NAMES)
    mh = sum(levels[lv] for lv in practice.MEDIUM_HEAVY) / len(practice.MEDIUM_HEAVY)
    print("{:<34} {}   medium+heavy {:.4f}".format(title, row, mh))
    if macro is not None:
        print("{:<34} {}".format("  macroF1", " ".join("{:>7.4f}".format(macro[lv]) for lv in practice.LEVEL_NAMES)))


def print_header():
    print("{:<34} {}".format("spellF1 (nested, IDSSE)", " ".join("{:>7}".format(lv) for lv in practice.LEVEL_NAMES)))


def start(args):
    t0 = time.perf_counter()
    plan2 = load_settings(CONFIG_DIR / "hmm_v1.json")
    assert not plan2.tag_hints
    feats, labels, baseline = {}, {}, {}
    for match_id in MATCH_IDS:
        table, labels[match_id] = load_idsse_cached(match_id)
        for case in practice.all_cases([match_id]):
            _, level, hiding, seed = case
            messy = degrade(table, LEVELS[level], seed, hiding)
            s = score(labels[match_id], smooth(nearest_team(messy), BASELINE_WINDOW))
            baseline[case] = (float(s["spell_f1"]), float(s["macro_f1"]))
            feats[case] = features(messy, plan2.clues)
        _log(t0, "{} ready".format(match_id))
    results = {"baseline": baseline, "plan2": practice.score_point(feats, labels, plan2)}
    practice.save_results(RESULTS_DIR / "start.json", results)
    _log(t0, "scored, wrote results/plan3/start.json")

    print()
    print("Degrade v{}: {} cases ({} matches x {} levels x 2 hiding styles x {} seeds)".format(
        DEGRADE_VERSION, len(baseline), len(MATCH_IDS), len(LEVELS), len(practice.SEEDS)))
    print_header()
    for name in results:
        # one setting each, so the nested estimate is the plain leave-one-match-out mean
        print_levels(name, practice.level_means(results[name]), practice.level_means(results[name], index=1))
        for hiding in ("hard", "random"):
            only = {c: v for c, v in results[name].items() if c[2] == hiding}
            print_levels("  {} hiding only".format(hiding), practice.level_means(only))
    spread = practice.seed_spread(results["plan2"])
    print("Plan 2 seed spread (medium+heavy, seed 0 vs seed 1): {:.4f}".format(spread))
    print("ACCEPT_BAR: {:.4f}".format(practice.accept_bar(spread)))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("start", help="baseline and Plan 2 on degrade v3")
    args = parser.parse_args(argv)
    {"start": start}[args.command](args)
    return 0


if __name__ == "__main__":
    sys.exit(main())

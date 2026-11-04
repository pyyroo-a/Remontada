"""Plan 3 practice runs on IDSSE (external data only, CC BY 4.0), every number nested.

    python scripts/plan3.py start          # baseline and Plan 2 on degrade v3: the numbers to beat
    python scripts/plan3.py search-costs   # step A, part 1: 125 cost points
    python scripts/plan3.py search-clues   # step A, part 2: coordinate search over the clues

Each step is judged by the acceptance rule against the best version so far and recorded as KEEP
or DROP in results/plan3/state.json. Every match keeps its own path through the steps (chosen on
the other six), so the estimate stays nested; the "all" path, chosen on all 7, is what we ship.

Results go to results/plan3/*.json (IDSSE scores only). Nothing here touches our match: the rules
(clarified 4 Nov 2026) allow tuning and checking only on external data.
"""
import argparse
import json
import sys
import time
from pathlib import Path

from remontada import practice
from remontada.baseline import nearest_team, smooth
from remontada.degrade import DEGRADE_VERSION, LEVELS, degrade
from remontada.load_idsse import MATCH_IDS, load_idsse_cached
from remontada.pipeline import CONFIG_DIR, features, load_settings, settings_from_dict, settings_to_dict
from remontada.scoreboard import score

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results" / "plan3"
BASELINE_WINDOW = 50
STEP_ORDER = ("A-costs", "A-clues", "B-gaps", "C-touch", "D-scorer", "E-final")
T0 = time.perf_counter()


def log(text):
    print("[{:6.0f} s] {}".format(time.perf_counter() - T0, text), flush=True)


def print_header():
    print("{:<34} {}".format("spellF1 (nested, IDSSE)", " ".join("{:>7}".format(lv) for lv in practice.LEVEL_NAMES)))


def print_levels(title, levels, macro=None):
    row = " ".join("{:>7.4f}".format(levels[lv]) for lv in practice.LEVEL_NAMES)
    mh = sum(levels[lv] for lv in practice.MEDIUM_HEAVY) / len(practice.MEDIUM_HEAVY)
    print("{:<34} {}   medium+heavy {:.4f}".format(title, row, mh))
    if macro is not None:
        print("{:<34} {}".format("  macroF1", " ".join("{:>7.4f}".format(macro[lv]) for lv in practice.LEVEL_NAMES)))


# ---------------------------------------------------------------- start


def start(args):
    plan2 = load_settings(CONFIG_DIR / "hmm_v1.json")
    assert not plan2.tag_hints
    feats, labels, baseline = {}, {}, {}
    for match_id in args.matches:
        table, labels[match_id] = load_idsse_cached(match_id)
        for case in practice.all_cases([match_id]):
            _, level, hiding, seed = case
            messy = degrade(table, LEVELS[level], seed, hiding)
            s = score(labels[match_id], smooth(nearest_team(messy), BASELINE_WINDOW))
            baseline[case] = (float(s["spell_f1"]), float(s["macro_f1"]))
            feats[case] = features(messy, plan2.clues)
        log("{} ready".format(match_id))
    results = {"baseline": baseline, "plan2": practice.score_point(feats, labels, plan2)}
    practice.save_results(args.results_dir / "start.json", results)
    log("scored, wrote start.json")

    print()
    print("Degrade v{}: {} cases ({} matches x {} levels x 2 hiding styles x {} seeds)".format(
        DEGRADE_VERSION, len(baseline), len(args.matches), len(LEVELS), len(practice.SEEDS)))
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


# ---------------------------------------------------------------- state


def load_state(results_dir):
    path = results_dir / "state.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {"steps": {}}


def current_best(results_dir, before):
    """(label, {context: Settings}, {level: nested mean}, bar) of the best version before a step:
    the last KEEP among the earlier steps, else Plan 2 on degrade v3."""
    start_results = practice.load_results(results_dir / "start.json")["plan2"]
    bar = practice.accept_bar(practice.seed_spread(start_results))
    state = load_state(results_dir)["steps"]
    for step in reversed(STEP_ORDER[:STEP_ORDER.index(before)]):
        if state.get(step, {}).get("verdict") == "KEEP":
            saved = state[step]
            contexts = {c: settings_from_dict(d) for c, d in saved["contexts"].items()}
            return step, contexts, saved["levels"], bar
    plan2 = load_settings(CONFIG_DIR / "hmm_v1.json")
    matches = sorted({case[0] for case in start_results})
    return "plan2", {c: plan2 for c in [*matches, practice.ALL]}, practice.level_means(start_results), bar


def record_step(results_dir, step, levels, contexts, current_label, current_levels, bar):
    verdict = "KEEP" if practice.accepts(levels, current_levels, bar) else "DROP"
    state = load_state(results_dir)
    state["steps"][step] = {
        "verdict": verdict,
        "against": current_label,
        "bar": bar,
        "levels": levels,
        "against_levels": current_levels,
        "contexts": {c: settings_to_dict(s) for c, s in contexts.items()},
        "chosen": {c: practice.point_name(s) for c, s in contexts.items()},
    }
    (results_dir / "state.json").write_text(json.dumps(state, indent=1), encoding="utf-8")

    print()
    print("Step {} against {} (bar {:.4f}, guard {:.2f})".format(step, current_label, bar, practice.GUARD))
    print_header()
    print_levels(current_label, current_levels)
    print_levels(step + " (nested)", levels)
    for c, s in contexts.items():
        print("  {:<7} chose: {}".format(c, practice.point_name(s)))
    print("VERDICT {}: {}".format(step, verdict))
    return verdict


# ---------------------------------------------------------------- scoring


class Scorer:
    """Scores Settings on every case, working out features once per clue setting."""

    def __init__(self, matches):
        self.matches = matches
        self.feats, self.counts, self.labels = {}, {}, None

    def prepare(self, clue_list):
        missing = [c for c in dict.fromkeys(clue_list) if c not in self.feats]
        if missing:
            log("features for {} clue setting(s)".format(len(missing)))
            by_clues, self.labels = practice.case_features_many(missing, self.matches)
            for clues, feats in by_clues.items():
                self.feats[clues] = feats
                self.counts[clues] = practice.left_out_counts(feats, self.labels)

    def score_many(self, settings_list):
        self.prepare([s.clues for s in settings_list])
        out = {}
        for i, s in enumerate(settings_list, 1):
            out[practice.point_name(s)] = practice.score_point(self.feats[s.clues], self.labels, s, self.counts[s.clues])
            log("{}/{} {}  objective {:.4f}".format(i, len(settings_list), practice.point_name(s),
                                                     practice.objective(out[practice.point_name(s)])))
        return out

    def forget_except(self, keep):
        for clues in list(self.feats):
            if clues not in keep:
                del self.feats[clues], self.counts[clues]


# ---------------------------------------------------------------- step A


def search_costs(args):
    label, contexts, levels_now, bar = current_best(args.results_dir, "A-costs")
    clues = contexts[practice.ALL].clues
    assert all(s.clues == clues for s in contexts.values())
    grid = practice.cost_grid(clues)[: args.limit]
    results = Scorer(args.matches).score_many(grid)
    practice.save_results(args.results_dir / "search-costs.json", results)
    names = [practice.point_name(s) for s in grid]
    levels, chosen = practice.nested_step(results, {c: names for c in contexts})
    by_name = dict(zip(names, grid))
    record_step(args.results_dir, "A-costs", levels, {c: by_name[n] for c, n in chosen.items()}, label, levels_now, bar)


def search_clues(args):
    label, contexts, levels_now, bar = current_best(args.results_dir, "A-clues")
    if label == "plan2":  # the cost step was dropped: the clue search starts from the cost winner
        cost_state = load_state(args.results_dir)["steps"]["A-costs"]
        contexts = {c: settings_from_dict(d) for c, d in cost_state["contexts"].items()}
        log("A-costs was dropped; starting the clue search from each path's cost winner")
    scorer = Scorer(args.matches)
    steps = practice.clue_steps()[: args.limit]

    def score_many(settings_list):
        out = scorer.score_many(settings_list)
        scorer.forget_except({s.clues for s in settings_list})
        return out

    final, results = practice.coordinate_search(contexts, steps, score_many)
    practice.save_results(args.results_dir / "search-clues.json", results)
    names = {c: [practice.point_name(s)] for c, s in final.items()}
    levels, _ = practice.nested_step(results, names)
    record_step(args.results_dir, "A-clues", levels, final, label, levels_now, bar)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["start", "search-costs", "search-clues"])
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--matches", default=",".join(MATCH_IDS), help="for quick checks only")
    parser.add_argument("--limit", type=int, default=None, help="first N points or steps, for quick checks only")
    args = parser.parse_args(argv)
    args.matches = args.matches.split(",")
    args.results_dir.mkdir(parents=True, exist_ok=True)
    log("plan3 {} on {} matches".format(args.command, len(args.matches)))
    {"start": start, "search-costs": search_costs, "search-clues": search_clues}[args.command](args)
    return 0


if __name__ == "__main__":
    sys.exit(main())

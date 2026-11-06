"""The honest practice harness for Plan 3: every number nested, on IDSSE only.

A case is (match, level, mess style, seed): 7 matches x 4 levels x 4 mess styles (hard or random
hiding, rough or smooth ball: degrade v4) x 2 seeds = 224 cases. Result files written before v4 hold
only the two rough-ball styles, named "hard" and "random" as before, and load unchanged. For a setting, every case is scored with a tally learned without its match. For a
search, every point is scored on every case; then for each match m the point that does best on
the other six is scored on m (nested_estimate). Ties go to the first point, so reruns agree.

Acceptance rule (spec section 5): a step is kept only if the medium-and-heavy mean rises by at
least the bar and no level falls by more than GUARD.
"""
import json
from dataclasses import fields, replace
from pathlib import Path

import numpy as np

from remontada.clues import DEFAULT_CLUES
from remontada.degrade import BALL_STYLES, HIDING_STYLES, LEVELS, degrade
from remontada.evidence import Counts
from remontada.hmm import Costs
from remontada.lomo import decode_all
from remontada.pipeline import Settings, features
from remontada.scoreboard import score

SEEDS = (0, 1)
STYLES = tuple(h if b == "rough" else h + "-" + b for b in BALL_STYLES for h in HIDING_STYLES)
LEVEL_NAMES = tuple(LEVELS)
MEDIUM_HEAVY = ("medium", "heavy")
BASE_BAR = 0.01  # the smallest rise worth keeping
SPREAD_LIMIT = 0.005  # a seed spread above this raises the bar to twice the spread
GUARD = 0.04  # no level may fall by more than this (the organisers' noise band)


def split_style(style):
    """(hiding, ball) of a mess style: "hard" is hard hiding with a rough ball, "hard-smooth" with
    a smooth one."""
    hiding, _, ball = style.partition("-")
    return hiding, ball or "rough"


def all_cases(matches):
    """Every case for these matches, in a fixed order."""
    return [(m, lv, st, s) for m in matches for lv in LEVEL_NAMES for st in STYLES for s in SEEDS]


def case_features(clues=DEFAULT_CLUES, matches=None):
    """({case: Features}, {match: labels}) for every IDSSE case, degrading one case at a time."""
    by_clues, labels = case_features_many([clues], matches)
    return by_clues[clues], labels


def case_features_many(clue_list, matches=None):
    """({clues: {case: Features}}, {match: labels}): each case is degraded once and its features
    worked out for every clue setting in the list."""
    from remontada.load_idsse import MATCH_IDS, load_idsse_cached
    clue_list = list(dict.fromkeys(clue_list))
    by_clues, labels = {clues: {} for clues in clue_list}, {}
    for match_id in matches or MATCH_IDS:
        table, labels[match_id] = load_idsse_cached(match_id)
        for case in all_cases([match_id]):
            _, level, style, seed = case
            messy = degrade(table, LEVELS[level], seed, *split_style(style))
            for clues in clue_list:
                by_clues[clues][case] = features(messy, clues)
    return by_clues, labels


def _counts_by_match(feats, labels):
    by_match = {}
    for case, f in feats.items():
        c = f.counts(labels[case[0]])
        by_match[case[0]] = c if case[0] not in by_match else by_match[case[0]] + c
    return by_match


def _sum(counts):
    total = None
    for c in counts:
        total = c if total is None else total + c
    return total


def left_out_counts(feats, labels):
    """{match: the tally summed over every case of every other match}."""
    by_match = _counts_by_match(feats, labels)
    total = _sum(by_match.values())
    return {m: Counts(total.team - c.team, total.slow - c.slow) for m, c in by_match.items()}


def all_counts(feats, labels):
    """The tally summed over every case of every match: what ships (the "all" context's tally)."""
    return _sum(_counts_by_match(feats, labels).values())


def score_point(feats, labels, settings, counts=None):
    """{case: (spellF1, macroF1)}, each case decoded with a tally that never saw its match.

    Decoded in batches of one mess style and seed at a time, to keep memory bounded. Pass
    counts (from left_out_counts) to reuse them across settings with the same features.
    """
    counts = counts if counts is not None else left_out_counts(feats, labels)
    out = {}
    for style in STYLES:
        for seed in SEEDS:
            batch = {case: f for case, f in feats.items() if case[2] == style and case[3] == seed}
            if not batch:
                continue
            codes = decode_all(batch, counts, settings)
            for case in batch:
                s = score(labels[case[0]], codes[case])
                out[case] = (float(s["spell_f1"]), float(s["macro_f1"]))
    return {case: out[case] for case in feats}


def _mean(case_scores, keep, index=0):
    values = [v[index] for case, v in case_scores.items() if keep(case)]
    return float(np.mean(values))


def level_means(case_scores, matches=None, index=0):
    """{level: mean spellF1 (index 0) or macroF1 (index 1)} over the given matches, both hiding
    styles and both seeds."""
    chosen = set(matches) if matches is not None else {case[0] for case in case_scores}
    return {lv: _mean(case_scores, lambda c, lv=lv: c[1] == lv and c[0] in chosen, index)
            for lv in LEVEL_NAMES}


def objective(case_scores, matches=None):
    """The number every step is judged on: mean spellF1 over medium and heavy."""
    levels = level_means(case_scores, matches)
    return float(np.mean([levels[lv] for lv in MEDIUM_HEAVY]))


def nested_estimate(results, index=0):
    """({level: held-out mean}, {match: chosen point}) from {point: {case: scores}}.

    For each match m, the point with the best objective on the other matches is scored on m;
    the first point in insertion order wins a tie.
    """
    points = list(results)
    matches = sorted({case[0] for case in results[points[0]]})
    chosen, per_match = {}, []
    for m in matches:
        others = [x for x in matches if x != m]
        best, best_value = points[0], objective(results[points[0]], others)
        for point in points[1:]:
            value = objective(results[point], others)
            if value > best_value:
                best, best_value = point, value
        chosen[m] = best
        per_match.append(level_means(results[best], [m], index))
    return {lv: float(np.mean([pm[lv] for pm in per_match])) for lv in LEVEL_NAMES}, chosen


def accepts(candidate, current, bar):
    """The acceptance rule on two {level: nested mean} tables."""
    rise = np.mean([candidate[lv] for lv in MEDIUM_HEAVY]) - np.mean([current[lv] for lv in MEDIUM_HEAVY])
    guard = all(candidate[lv] >= current[lv] - GUARD for lv in LEVEL_NAMES)
    return bool(rise >= bar - 1e-12 and guard)


def seed_spread(case_scores):
    """|medium-and-heavy mean on seed 0 minus the same on seed 1|."""
    def on_seed(seed):
        return float(np.mean([_mean(case_scores, lambda c, lv=lv: c[1] == lv and c[3] == seed)
                              for lv in MEDIUM_HEAVY]))
    return abs(on_seed(SEEDS[0]) - on_seed(SEEDS[1]))


def accept_bar(spread):
    """BASE_BAR, or twice the seed spread when that spread is over SPREAD_LIMIT (spec section 5)."""
    return 2 * spread if spread > SPREAD_LIMIT else BASE_BAR


def point_name(settings):
    """A readable name for a search point: the costs, plus anything that differs from Plan 2."""
    c = settings.costs
    parts = ["switch={:g} to_dead={:g} from_dead={:g}".format(c.switch, c.to_dead, c.from_dead)]
    if settings.temperature != 1.0:
        parts.append("temperature={:g}".format(settings.temperature))
    if not settings.ball_hints:
        parts.append("no_ball_hints")
    for f in fields(DEFAULT_CLUES):
        value = getattr(settings.clues, f.name)
        if value != f.default:
            parts.append("{}={:g}".format(f.name, value))
    if settings.touch_discount or settings.touch_window:
        parts.append("touch_discount={:g} touch_window={:d}".format(settings.touch_discount, settings.touch_window))
    if settings.scorer != "tally":
        parts.append("scorer=" + settings.scorer)
    return " ".join(parts)


def _case_key(case):
    return "|".join(str(part) for part in case)


def _case_from_key(key):
    match, level, hiding, seed = key.split("|")
    return match, level, hiding, int(seed)


def save_results(path, results):
    """{point: {case: (spellF1, macroF1)}} as JSON, cases written as "match|level|hiding|seed"."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    data = {point: {_case_key(c): list(v) for c, v in cases.items()} for point, cases in results.items()}
    Path(path).write_text(json.dumps(data, indent=1), encoding="utf-8")


def load_results(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {point: {_case_from_key(k): tuple(v) for k, v in cases.items()} for point, cases in data.items()}


# Plan 3 step A: the search grids, fixed before any score was seen (plan Task 5).
SWITCH_GRID = (2.5, 5, 10, 20, 40)
DEAD_GRID = (20, 40, 80, 160, 320)  # to_dead and from_dead searched separately
CLUE_STEPS = (
    ("touch_dv", [2.0, 3.0, 4.5]),
    ("touch_radius", [1.0, 1.5, 2.0]),
    ("control_radius", [1.0, 1.5, 2.0]),
    ("control_min_frames", [5, 10, 20]),
    ("carry_frames", [25, 50, 75, 125]),
    ("out_margin_m", [0.5, 1.0, 1.5]),
)
ALL = "all"  # the context that sees every match: its choice is the one we ship

# Plan 3 step B: longest ball gap filled by a straight line, in frames (0, 0.5, 1, 2, 3 s; plan Task 6).
GAP_GRID = (0, 12, 25, 50, 75)

# Plan 3 step C: the switch discount next to a touch, as a share of the switch cost, and the window
# in frames either side of the touch (plan Task 7).
TOUCH_DISCOUNT_GRID = (0.25, 0.5, 0.75, 1.0)
TOUCH_WINDOW_GRID = (0, 5, 12, 25)

# Plan 3 step D: each cost times half, the same, or double, current costs first (plan Task 8).
SCORER_COST_STEPS = (1.0, 0.5, 2.0)


def cost_grid(clues=DEFAULT_CLUES):
    """125 points at temperature 1 (only cost / temperature ratios matter), grid order fixed."""
    return [Settings(1.0, Costs(float(sw), float(td), float(fd)), True, False, clues=clues)
            for sw in SWITCH_GRID for td in DEAD_GRID for fd in DEAD_GRID]


def clue_steps():
    """The coordinate search over the clue settings, in order: (name, values)."""
    return [(name, list(values)) for name, values in CLUE_STEPS]


def gap_steps():
    """Step B as a one-step coordinate search over gap_fill_frames."""
    return [("gap_fill_frames", list(GAP_GRID))]


def touch_grid(base):
    """No discount first (the current best), then every discount and window: 17 points."""
    return [replace(base, touch_discount=0.0, touch_window=0)] + [
        replace(base, touch_discount=d, touch_window=w) for d in TOUCH_DISCOUNT_GRID for w in TOUCH_WINDOW_GRID]


def scorer_cost_grid(base):
    """27 points with the learned scorer: every cost of base times 1, 0.5 or 2 (base first)."""
    c = base.costs
    return [replace(base, scorer="model", costs=Costs(c.switch * a, c.to_dead * b, c.from_dead * d))
            for a in SCORER_COST_STEPS for b in SCORER_COST_STEPS for d in SCORER_COST_STEPS]


def final_cost_grid(base):
    """Step E: 27 points, every cost of base times 1, 0.5 or 2 (base first), nothing else changed."""
    c = base.costs
    return [replace(base, costs=Costs(c.switch * a, c.to_dead * b, c.from_dead * d))
            for a in SCORER_COST_STEPS for b in SCORER_COST_STEPS for d in SCORER_COST_STEPS]


def selection_gap(results, chosen):
    """(in-sample, nested): the shipped choice scored on the matches it was chosen on, against the
    nested estimate (each match scored by the choice made without it). A big gap means the search
    is fitting the practice matches rather than finding settings that carry over."""
    matches = sorted(c for c in chosen if c != ALL)
    in_sample = objective(results[chosen[ALL]], matches)
    nested = float(np.mean([objective(results[chosen[m]], [m]) for m in matches]))
    return in_sample, nested


def shipped_settings(steps, order, fallback):
    """The "all" settings of the last KEEP in order (tag hints off: Smart Tagging is not ours to
    rely on), else fallback."""
    from remontada.pipeline import settings_from_dict
    for step in reversed(order):
        if steps.get(step, {}).get("verdict") == "KEEP":
            return replace(settings_from_dict(steps[step]["contexts"][ALL]), tag_hints=False)
    return fallback


def train_left_out_scorers(rows_by_match):
    """{match: a scorer trained on the other matches' rows, ALL: one trained on every match}.

    rows_by_match is {match: (rows, y)} from scorer.training_rows; every row also counts mirrored.
    """
    from remontada.scorer import Scorer

    def fit(chosen):
        X = np.concatenate([rows_by_match[m][0] for m in chosen])
        y = np.concatenate([rows_by_match[m][1] for m in chosen])
        return Scorer.fit(X, y, mirrored=True)

    matches = sorted(rows_by_match)
    out = {m: fit([x for x in matches if x != m]) for m in matches}
    out[ALL] = fit(matches)
    return out


def scorer_case_features(clues, matches=None, log=print):
    """({case: Features with the held-out scorer's p_home}, {match: labels}, {context: Scorer}).

    Two passes over the IDSSE cases, one case at a time to keep memory bounded: the first keeps
    the training rows (every 5th live frame), the second works out each case's features with the
    scorer that never saw its match.
    """
    from remontada.load_idsse import MATCH_IDS, load_idsse_cached
    from remontada.scorer import frame_clues, training_rows
    matches = list(matches or MATCH_IDS)
    rows, labels = {}, {}
    for match_id in matches:
        table, labels[match_id] = load_idsse_cached(match_id)
        parts = [training_rows(frame_clues(degrade(table, LEVELS[lv], seed, hiding), clues), labels[match_id])
                 for _, lv, hiding, seed in all_cases([match_id])]
        rows[match_id] = (np.concatenate([x for x, _ in parts]), np.concatenate([y for _, y in parts]))
    log("training rows: {}".format(sum(len(y) for _, y in rows.values())))
    scorers = train_left_out_scorers(rows)
    log("trained {} scorers".format(len(scorers)))
    del rows
    feats = {}
    for match_id in matches:
        table, _ = load_idsse_cached(match_id)
        for case in all_cases([match_id]):
            _, lv, hiding, seed = case
            feats[case] = features(degrade(table, LEVELS[lv], seed, hiding), clues, scorers[match_id])
    return feats, labels, scorers


def context_matches(context, matches):
    """The matches a context may look at: every match for ALL, the other six for a match."""
    return list(matches) if context == ALL else [m for m in matches if m != context]


def choose(results, names, matches):
    """The point among names with the best objective on these matches; the first one on a tie."""
    best, best_value = names[0], objective(results[names[0]], matches)
    for name in names[1:]:
        value = objective(results[name], matches)
        if value > best_value:
            best, best_value = name, value
    return best


def nested_step(results, candidates):
    """({level: nested mean}, {context: chosen point}) when each context has its own candidates.

    candidates is {context: [point names]}, contexts being every match plus ALL. Context m chooses
    on the other matches and is scored on m; ALL's choice is what we would ship.
    """
    matches = sorted(c for c in candidates if c != ALL)
    chosen = {c: choose(results, names, context_matches(c, matches)) for c, names in candidates.items()}
    per_match = [level_means(results[chosen[m]], [m]) for m in matches]
    return {lv: float(np.mean([pm[lv] for pm in per_match])) for lv in LEVEL_NAMES}, chosen


def coordinate_search(contexts, steps, score_many, results=None):
    """Try each clue value in turn, every context on its own path (so the estimate stays nested).

    contexts is {context: starting Settings}; steps is [(clue name, values)]; score_many(list of
    Settings) returns {point name: case scores} and is only asked for points not scored yet.
    Returns ({context: final Settings}, {point name: case scores}).
    """
    results = dict(results or {})
    matches = sorted(c for c in contexts if c != ALL)
    current = dict(contexts)
    for name, values in steps:
        options = {c: [replace(s, clues=replace(s.clues, **{name: v})) for v in values] for c, s in current.items()}
        needed = {}
        for settings_list in options.values():
            for s in settings_list:
                if point_name(s) not in results:
                    needed.setdefault(point_name(s), s)
        if needed:
            results.update(score_many(list(needed.values())))
        for c, settings_list in options.items():
            best = choose(results, [point_name(s) for s in settings_list], context_matches(c, matches))
            current[c] = next(s for s in settings_list if point_name(s) == best)
    return current, results

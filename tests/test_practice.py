import numpy as np

from conftest import synthetic_table
from remontada.evidence import count
from remontada.hmm import Costs
from remontada.pipeline import Settings, features
from remontada.vendor.possession_metrics import A, H


def _practice():
    from remontada import practice
    return practice


def _cases(matches, value_by_match, macro=0.0):
    p = _practice()
    return {(m, lv, h, s): (value_by_match[m], macro) for m in matches for lv in p.LEVEL_NAMES
            for h in ("hard", "random") for s in (0, 1)}


def test_case_names():
    p = _practice()
    assert p.SEEDS == (0, 1) and p.MEDIUM_HEAVY == ("medium", "heavy")
    assert p.LEVEL_NAMES == ("none", "light", "medium", "heavy")
    assert len(p.all_cases(("m1", "m2"))) == 2 * 4 * 2 * 2


def test_level_means_and_objective():
    p = _practice()
    scores = _cases(["m1", "m2"], {"m1": 0.2, "m2": 0.6})
    scores[("m1", "heavy", "random", 1)] = (1.0, 0.0)
    levels = p.level_means(scores)
    assert abs(levels["none"] - 0.4) < 1e-12
    assert abs(levels["heavy"] - (0.2 * 3 + 1.0 + 0.6 * 4) / 8) < 1e-12
    assert abs(p.level_means(scores, ["m2"])["heavy"] - 0.6) < 1e-12
    assert abs(p.objective(scores) - (levels["medium"] + levels["heavy"]) / 2) < 1e-12


def test_nested_estimate_picks_on_the_other_matches():
    p = _practice()
    ms = ["m1", "m2", "m3"]
    results = {"p": _cases(ms, {"m1": 0.9, "m2": 0.1, "m3": 0.1}),
               "q": _cases(ms, {"m1": 0.2, "m2": 0.3, "m3": 0.3})}
    levels, chosen = p.nested_estimate(results)
    # without m1, q wins (0.3 against 0.1); without m2 or m3, p's 0.9 on m1 wins (0.5 against 0.25),
    # so p is then scored on the held-out match, where it gets only 0.1: an honest, lower estimate
    assert chosen == {"m1": "q", "m2": "p", "m3": "p"}
    assert abs(levels["heavy"] - (0.2 + 0.1 + 0.1) / 3) < 1e-12


def test_nested_estimate_can_choose_differently_per_match():
    p = _practice()
    ms = ["m1", "m2", "m3"]
    results = {"p": _cases(ms, {"m1": 0.9, "m2": 0.9, "m3": 0.0}),
               "q": _cases(ms, {"m1": 0.5, "m2": 0.5, "m3": 0.5})}
    levels, chosen = p.nested_estimate(results)
    assert chosen == {"m1": "q", "m2": "q", "m3": "p"}  # others' means: m1 .45/.5, m2 .45/.5, m3 .9/.5
    assert abs(levels["medium"] - (0.5 + 0.5 + 0.0) / 3) < 1e-12


def test_nested_ties_take_the_first_point():
    p = _practice()
    ms = ["m1", "m2"]
    results = {"p": _cases(ms, {"m1": 0.5, "m2": 0.5}), "q": _cases(ms, {"m1": 0.5, "m2": 0.5})}
    assert set(p.nested_estimate(results)[1].values()) == {"p"}


def test_accepts_needs_the_bar_and_the_guard():
    p = _practice()
    cur = {"none": 0.6, "light": 0.5, "medium": 0.4, "heavy": 0.2}
    assert p.accepts({**cur, "medium": 0.42, "heavy": 0.22}, cur, 0.01)
    assert not p.accepts({**cur, "medium": 0.405, "heavy": 0.205}, cur, 0.01)
    assert not p.accepts({**cur, "none": 0.55, "medium": 0.43, "heavy": 0.23}, cur, 0.01)


def test_seed_spread_and_the_bar():
    p = _practice()
    scores = _cases(["m1"], {"m1": 0.3})
    for lv in p.MEDIUM_HEAVY:
        for h in ("hard", "random"):
            scores[("m1", lv, h, 1)] = (0.32, 0.0)
    assert abs(p.seed_spread(scores) - 0.02) < 1e-12
    assert p.accept_bar(0.004) == 0.01 and p.accept_bar(0.005) == 0.01
    assert abs(p.accept_bar(0.02) - 0.04) < 1e-12


def test_results_round_trip(tmp_path):
    p = _practice()
    results = {"start": _cases(["m1"], {"m1": 0.123456789012345}, macro=0.5)}
    p.save_results(tmp_path / "r.json", results)
    assert p.load_results(tmp_path / "r.json") == results


def _toy_match(flip):
    """200 frames: a home player on the ball, then an away one (or the other way round)."""
    ball = [(-40.0 + 0.08 * t, 0.0) for t in range(200)]
    near, far = (0, 1) if not flip else (1, 0)
    players = [[(x + 0.5, 0.0), (x + 10.0, 5.0)] if t < 100 else [(x + 10.0, 5.0), (x + 0.5, 0.0)]
               for t, (x, _) in enumerate(ball)]
    labels = np.array(([H] * 100 + [A] * 100) if not flip else ([A] * 100 + [H] * 100), dtype=np.int8)
    return synthetic_table(ball, players, teams=[near, far]), labels


def test_left_out_counts_never_use_the_scored_match():
    p = _practice()
    tables = {"m1": _toy_match(False), "m2": _toy_match(True)}
    labels = {m: lab for m, (_, lab) in tables.items()}
    feats = {(m, lv, h, s): features(tables[m][0]) for m in tables for lv in p.LEVEL_NAMES
             for h in ("hard", "random") for s in p.SEEDS}
    loo = p.left_out_counts(feats, labels)
    own = count(*tables["m2"])
    np.testing.assert_array_equal(loo["m1"].team, own.team * 16)
    np.testing.assert_array_equal(loo["m1"].slow, own.slow * 16)


def test_score_point_scores_every_case():
    p = _practice()
    tables = {"m1": _toy_match(False), "m2": _toy_match(True)}
    labels = {m: lab for m, (_, lab) in tables.items()}
    feats = {(m, lv, h, s): features(tables[m][0]) for m in tables for lv in p.LEVEL_NAMES
             for h in ("hard", "random") for s in p.SEEDS}
    out = p.score_point(feats, labels, Settings(1.0, Costs(10, 80, 80), True, False))
    assert list(out) == list(feats)
    assert all(0.0 <= spell <= 1.0 and 0.0 <= macro <= 1.0 for spell, macro in out.values())
    assert out[("m1", "none", "hard", 0)][0] > 0.9  # two clean blocks, found


def test_point_names_are_readable():
    p = _practice()
    name = p.point_name(Settings(1.0, Costs(10, 80, 40), True, False))
    assert name == "switch=10 to_dead=80 from_dead=40"


def test_cost_grid_is_centred_on_plan_2():
    p = _practice()
    grid = p.cost_grid()
    assert len(grid) == 125 and len({g.costs for g in grid}) == 125
    assert all(g.temperature == 1.0 and g.ball_hints and not g.tag_hints for g in grid)
    assert Costs(10, 80, 80) in {g.costs for g in grid}
    assert sorted({g.costs.switch for g in grid}) == [2.5, 5, 10, 20, 40]
    assert sorted({g.costs.to_dead for g in grid}) == sorted({g.costs.from_dead for g in grid}) == [20, 40, 80, 160, 320]


def test_cost_grid_keeps_the_base_clues():
    from dataclasses import replace
    from remontada.clues import DEFAULT_CLUES
    p = _practice()
    clues = replace(DEFAULT_CLUES, carry_frames=50)
    assert all(g.clues == clues for g in p.cost_grid(clues))


def test_clue_steps_start_from_defaults():
    from dataclasses import fields
    from remontada.clues import DEFAULT_CLUES
    p = _practice()
    steps = p.clue_steps()
    assert [name for name, _ in steps] == ["touch_dv", "touch_radius", "control_radius",
                                           "control_min_frames", "carry_frames", "out_margin_m"]
    names = {f.name for f in fields(DEFAULT_CLUES)}
    for name, values in steps:
        assert name in names and getattr(DEFAULT_CLUES, name) in values
    assert dict(steps)["carry_frames"] == [25, 50, 75, 125]


def _fake_scores(value_by_match):
    return _cases(list(value_by_match), value_by_match)


def test_choose_takes_the_first_best_point():
    p = _practice()
    results = {"a": _fake_scores({"m1": 0.5, "m2": 0.5}), "b": _fake_scores({"m1": 0.9, "m2": 0.1}),
               "c": _fake_scores({"m1": 0.5, "m2": 0.5})}
    assert p.choose(results, ["a", "b", "c"], ["m1", "m2"]) == "a"
    assert p.choose(results, ["c", "a"], ["m1", "m2"]) == "c"
    assert p.choose(results, ["a", "b"], ["m1"]) == "b"


def test_nested_step_uses_each_contexts_own_candidates():
    p = _practice()
    results = {"a": _fake_scores({"m1": 0.2, "m2": 0.6}), "b": _fake_scores({"m1": 0.4, "m2": 0.4}),
               "c": _fake_scores({"m1": 0.9, "m2": 0.9})}
    candidates = {"m1": ["a", "b"], "m2": ["b", "c"], "all": ["a", "b"]}
    levels, chosen = p.nested_step(results, candidates)
    # without m1, a (0.6 on m2) beats b; without m2, c beats b; on both, a and b tie at 0.4 -> a
    assert chosen == {"m1": "a", "m2": "c", "all": "a"}
    assert abs(levels["heavy"] - (0.2 + 0.9) / 2) < 1e-12


def test_coordinate_search_runs_each_context_on_its_own_matches():
    from dataclasses import replace
    from remontada.clues import DEFAULT_CLUES
    p = _practice()
    base = Settings(1.0, Costs(10, 80, 80), True, False)
    value = {  # spellF1 by (touch_dv, carry_frames) and match
        (3.0, 75): {"m1": 0.5, "m2": 0.5}, (2.0, 75): {"m1": 0.9, "m2": 0.1},
        (2.0, 25): {"m1": 0.1, "m2": 0.1}, (3.0, 25): {"m1": 0.6, "m2": 0.6},
    }
    asked = []

    def score_many(settings_list):
        asked.append(len(settings_list))
        return {p.point_name(s): _fake_scores(value[s.clues.touch_dv, s.clues.carry_frames]) for s in settings_list}

    steps = [("touch_dv", [2.0, 3.0]), ("carry_frames", [25, 75])]
    contexts = {"m1": base, "m2": base, "all": base}
    final, results = p.coordinate_search(contexts, steps, score_many)
    # m1 (scored on m2): touch_dv 2.0 gives 0.1, 3.0 gives 0.5 -> 3.0; then carry 25 (0.6) -> (3.0, 25)
    # m2 (scored on m1): touch_dv 2.0 (0.9) -> 2.0; then carry 25 gives 0.1, 75 gives 0.9 -> (2.0, 75)
    # all: 2.0 and 3.0 tie at 0.5 -> 2.0 (first); then carry 25 gives 0.1 -> (2.0, 75)
    assert (final["m1"].clues.touch_dv, final["m1"].clues.carry_frames) == (3.0, 25)
    assert (final["m2"].clues.touch_dv, final["m2"].clues.carry_frames) == (2.0, 75)
    assert final["all"].clues == replace(DEFAULT_CLUES, touch_dv=2.0)
    assert sum(asked) == len(results) == 4  # every point scored once, however many contexts ask


def test_settings_dict_round_trip():
    from dataclasses import replace
    from remontada.clues import DEFAULT_CLUES
    from remontada.pipeline import settings_from_dict, settings_to_dict
    s = Settings(1.0, Costs(10, 160, 40), True, False, clues=replace(DEFAULT_CLUES, carry_frames=50),
                 touch_discount=0.5, touch_window=12, scorer="model")
    assert settings_from_dict(settings_to_dict(s)) == s


def test_point_names_tell_temperatures_and_hints_apart():
    p = _practice()
    name = p.point_name(Settings(0.4, Costs(4, 32, 32), False, False))
    assert name == "switch=4 to_dead=32 from_dead=32 temperature=0.4 no_ball_hints"

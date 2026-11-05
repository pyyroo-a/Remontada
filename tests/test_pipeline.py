from dataclasses import replace

import numpy as np

from conftest import NAN_XY, synthetic_table
from remontada.evidence import count
from remontada.hmm import Costs
from remontada.lomo import leave_one_out_counts
from remontada.pipeline import CONFIG_DIR, Settings, features, load_settings, predict
from remontada.evidence import N_SLOW_CELLS, NONE, Counts
from remontada.touch import touches
from remontada.vendor.possession_metrics import A, D, H


def _two_halves_of_possession():
    """400 frames, ball rolling at 2 m/s: a home player on it for 200 frames, then an away one."""
    ball = [(-40.0 + 0.08 * t, 0.0) for t in range(400)]
    players = [[(x + 0.5, 0.0), (x + 10.0, 5.0)] if t < 200 else [(x + 10.0, 5.0), (x + 0.5, 0.0)]
               for t, (x, _) in enumerate(ball)]
    labels = np.array([H] * 200 + [A] * 200, dtype=np.int8)
    return synthetic_table(ball, players, teams=[0, 1]), labels


def test_predict_on_a_synthetic_match_gives_blocks():
    table, labels = _two_halves_of_possession()
    codes = predict(table, count(table, labels), Settings(0.2, Costs(8, 8, 8), True, False))
    assert set(codes.tolist()) <= {H, A, D} and (codes[:150] == H).all() and (codes[250:] == A).all()


def test_tags_are_used_only_when_switched_on():
    from remontada.smart_tagging import Tag
    table, labels = _two_halves_of_possession()
    counts = count(table, labels)
    tags = [Tag("SET PIECES", 0, 400, -1, None)] * 50  # a flood of dead-ball hints on frames 0..199
    off = predict(table, counts, Settings(0.2, Costs(8, 8, 8), True, False), tags)
    on = predict(table, counts, Settings(0.2, Costs(8, 8, 8), True, True), tags)
    assert (off[:150] == H).all() and (on[:200] == D).all()


def test_features_count_like_the_table():
    table, labels = _two_halves_of_possession()
    f = features(table)
    assert len(f.team) == len(f.slow) == len(f.ball_clues) == 400
    np.testing.assert_array_equal(f.counts(labels).team, count(table, labels).team)


def test_leave_one_out_never_uses_the_scored_match():
    one = Counts(np.ones((54, 2), dtype=np.int64), np.ones((7, 2), dtype=np.int64))
    by_item = {("m1", "none"): one, ("m1", "heavy"): one, ("m2", "none"): one + one, ("m3", "none"): one}
    out = leave_one_out_counts(by_item)
    assert out["m1"].team[0, 0] == 3 and out["m2"].team[0, 0] == 3 and out["m3"].team[0, 0] == 4


def test_settings_file_loads():
    s = load_settings()
    assert s.temperature > 0 and s.costs.switch > 0


def test_features_repr_hides_the_match():
    # like FrameTable and Tag: features of our match must never print frame by frame
    table, _ = _two_halves_of_possession()
    assert repr(features(table)) == "Features(frames=400)"


def test_shipped_settings_leave_tags_off():
    # Ashen, 3 Nov: tags stacked up and cannot be tested on IDSSE, so plan 2 ships without them
    assert load_settings().tag_hints is False


def _swapped_halves():
    """Players stand still in both halves but swap ends; one NaN row separates the halves."""
    first = [[(-30.0, 0.0), (30.0, 0.0)]] * 100
    second = [[(30.0, 0.0), (-30.0, 0.0)]] * 100
    table = synthetic_table([(0.0, 0.0)] * 201, first + [[NAN_XY, NAN_XY]] + second, teams=[0, 1])
    return replace(table, graded_runs=[(0, 100), (101, 201)])


def test_clues_do_not_cross_the_half_time_gap():
    # Ruling (Task 2): near a run's edge the centred speed has no frames on one side, so it reads
    # "unknown", as at the end of any match; it must never read the 62.5 m/s of the swap.
    f = features(_swapped_halves())
    edges = np.concatenate((f.slow[88:100], f.slow[101:113]))
    assert set(edges.tolist()) <= {0, N_SLOW_CELLS - 1}
    assert (f.slow[12:88] == 0).all() and (f.slow[113:189] == 0).all()


def test_frames_outside_runs_get_nothing_known():
    f = features(_swapped_halves())
    nothing = 0 * 9 + NONE * 3 + NONE  # nearness unseen, no touch, no control
    assert f.team[100] == nothing and f.team_mirrored[100] == nothing
    assert f.slow[100] == N_SLOW_CELLS - 1 and f.ball_clues[100] == 0


def test_predict_with_the_ball_missing_most_of_a_half():
    table, labels = _two_halves_of_possession()
    ball = table.ball_xy.copy()
    ball[20:380] = np.nan
    codes = predict(replace(table, ball_xy=ball), count(table, labels), Settings(0.2, Costs(8, 8, 8), True, False))
    assert len(codes) == 400 and set(codes.tolist()) <= {H, A, D}


def test_a_pass_flying_1m_from_an_opponent_stays_home():
    ball = [(0.0, 0.0)] * 10 + [(0.5 * (t - 10), 0.0) for t in range(10, 51)] + [(20.0, 0.0)] * 20
    players = [[(-0.5, 0.0), (20.5, 0.0), (10.0, 1.0)]] * len(ball)
    table = synthetic_table(ball, players, teams=[0, 0, 1])
    t = touches(table)
    assert not (t == 1).any()
    labels = np.array([H] * len(ball), dtype=np.int8)
    codes = predict(table, count(table, labels), Settings(0.4, Costs(4, 32, 32), True, False))
    assert (codes[10:51] == H).all()


def test_hmm_v1_still_loads_with_defaults():
    from remontada.clues import DEFAULT_CLUES
    s = load_settings(CONFIG_DIR / "hmm_v1.json")
    assert s.clues == DEFAULT_CLUES and s.touch_discount == 0.0 and s.touch_window == 0
    assert s.scorer == "tally"


def test_settings_round_trip_with_clues(tmp_path):
    import json
    path = tmp_path / "hmm.json"
    path.write_text(json.dumps({"temperature": 1, "switch": 10, "to_dead": 80, "from_dead": 40,
                                "ball_hints": True, "tag_hints": False, "touch_discount": 0.5,
                                "touch_window": 12, "scorer": "model",
                                "clues": {"carry_frames": 50, "touch_radius": 2.0}}), encoding="utf-8")
    s = load_settings(path)
    assert s.clues.carry_frames == 50 and s.clues.touch_radius == 2.0 and s.clues.touch_dv == 3.0
    assert (s.touch_discount, s.touch_window, s.scorer) == (0.5, 12, "model")
    assert s.costs == Costs(10.0, 80.0, 40.0)


def test_predict_uses_the_settings_clues():
    from remontada.clues import DEFAULT_CLUES
    from remontada.hmm import decode_runs
    table, labels = _two_halves_of_possession()
    counts = count(table, labels)
    clues = replace(DEFAULT_CLUES, control_radius=0.1)
    settings = Settings(0.2, Costs(8, 8, 8), True, False, clues=clues)
    assert not np.array_equal(features(table, clues).team, features(table).team)
    expected = decode_runs(features(table, clues).scores(counts, settings), table.graded_runs, settings.costs)
    np.testing.assert_array_equal(predict(table, counts, settings), expected)


def test_a_misspelt_clue_setting_is_refused(tmp_path):
    import json
    import pytest
    path = tmp_path / "hmm.json"
    path.write_text(json.dumps({"temperature": 1, "switch": 10, "to_dead": 80, "from_dead": 80,
                                "ball_hints": True, "tag_hints": False,
                                "clues": {"cary_frames": 50}}), encoding="utf-8")
    with pytest.raises(ValueError, match="cary_frames"):
        load_settings(path)


def test_touch_entry_bonus_marks_the_touching_team_near_each_touch():
    from remontada.pipeline import touch_entry_bonus
    codes = np.full(30, -1, dtype=np.int8)
    codes[10], codes[12], codes[25] = 1, 1, 0  # away touches at 10 and 12, home at 25
    bonus = touch_entry_bonus(codes, 2.5, 3)
    assert bonus.shape == (30, 3)
    assert (bonus[7:16, A] == 2.5).all() and (bonus[:7, A] == 0).all() and (bonus[16:, A] == 0).all()
    assert (bonus[22:29, H] == 2.5).all() and bonus[:22, H].sum() == 0 and bonus[29, H] == 0
    assert (bonus[:, D] == 0).all()  # never stacks, never touches D
    assert (touch_entry_bonus(codes, 2.5, 0)[[10, 12], A] == 2.5).all()
    assert touch_entry_bonus(codes, 2.5, 0)[:, A].sum() == 5.0


def test_features_keep_the_touches_per_run():
    table, _ = _two_halves_of_possession()
    f = features(table)
    np.testing.assert_array_equal(f.touches, touches(table))
    halves = features(_swapped_halves())
    assert halves.touches.shape == (201,) and halves.touches[100] == -1


def _bonus_settings(share, window):
    return Settings(0.2, Costs(8, 8, 8), True, False, touch_discount=share, touch_window=window)


def _a_touch_at_200():
    """As _two_halves_of_possession, but the away player knocks the ball back at frame 200."""
    xs = [-40.0 + 0.08 * t if t <= 200 else -40.0 + 0.08 * (400 - t) for t in range(400)]
    players = [[(x + 0.5, 0.0), (x + 10.0, 5.0)] if t < 200 else [(x + 10.0, 5.0), (x + 0.5, 0.0)]
               for t, x in enumerate(xs)]
    labels = np.array([H] * 200 + [A] * 200, dtype=np.int8)
    return synthetic_table([(x, 0.0) for x in xs], players, teams=[0, 1]), labels


def test_predict_takes_the_touch_discount_as_a_share_of_the_switch_cost():
    from remontada.hmm import decode_runs
    from remontada.pipeline import touch_entry_bonus
    table, labels = _a_touch_at_200()
    assert features(table).touches[200] == 1
    counts = count(table, labels)
    settings = _bonus_settings(0.5, 12)
    f = features(table)
    bonus = touch_entry_bonus(f.touches, 0.5 * 8, 12)
    assert bonus.max() == 4.0
    expected = decode_runs(f.scores(counts, settings), table.graded_runs, settings.costs, entry_bonus=bonus)
    np.testing.assert_array_equal(predict(table, counts, settings), expected)


def test_decode_all_uses_the_touch_bonus_like_predict():
    from remontada.lomo import decode_all
    table, labels = _a_touch_at_200()
    counts = count(table, labels)
    feats = {("m1", "none"): features(table)}
    for settings in (_bonus_settings(0.0, 0), _bonus_settings(1.0, 25)):
        got = decode_all(feats, {"m1": counts}, settings)[("m1", "none")]
        np.testing.assert_array_equal(got, predict(table, counts, settings))


def test_decode_all_moves_the_switch_onto_the_touch():
    from remontada.lomo import decode_all
    from remontada.pipeline import Features

    class Fading(Features):
        def scores(self, counts, settings):  # H fades and A grows over frames 10..30
            fade = np.clip((np.arange(40) - 10) / 20, 0, 1)
            return np.stack([1 - fade, fade, np.zeros(40)], axis=1)

    blank = np.zeros(40, dtype=np.int16)
    codes = np.full(40, -1, dtype=np.int8)
    codes[25] = 1  # an away touch
    f = Fading(blank, blank, blank, blank.astype(np.int8), codes, [(0, 40)])
    plain = Settings(1.0, Costs(5, 50, 50), True, False)
    for settings, first_a in ((plain, 21), (replace(plain, touch_discount=1.0), 25)):
        got = decode_all({("m1", "none"): f}, {"m1": None}, settings)[("m1", "none")]
        assert int(np.argmax(got == A)) == first_a

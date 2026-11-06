from dataclasses import replace

import numpy as np
import pytest

from conftest import synthetic_table
from remontada.clues import DEFAULT_CLUES
from remontada.evidence import count
from remontada.hmm import Costs
from remontada.vendor.possession_metrics import A, D, H


def _s():
    from remontada import scorer
    return scorer


def _random_rows(n=4000, seed=0):
    s = _s()
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, len(s.FRAME_CLUES)))
    return X, (X[:, 0] > 0).astype(int)


def test_frame_clue_names():
    s = _s()
    assert s.FRAME_CLUES == ("margin", "nearest", "touch_team", "touch_age", "control_team", "control_age",
                             "ball_speed", "ball_seen", "since_seen", "ball_filled", "abs_x", "abs_y")
    assert not {"ball_out", "ball_still"} & set(s.FRAME_CLUES)


def test_learns_a_clear_rule():
    X, y = _random_rows()
    p = _s().Scorer.fit(X, y).p_home(X)
    assert ((p > 0.5) == y.astype(bool)).mean() > 0.9


def test_mirror_flips_only_the_team_columns():
    s = _s()
    X, _ = _random_rows(50)
    M = s.mirror(X)
    signed = [s.FRAME_CLUES.index(c) for c in ("margin", "touch_team", "control_team")]
    others = [i for i in range(len(s.FRAME_CLUES)) if i not in signed]
    np.testing.assert_array_equal(M[:, signed], -X[:, signed])
    np.testing.assert_array_equal(M[:, others], X[:, others])


def test_mirror_is_exact():
    s = _s()
    X, y = _random_rows()
    model = s.Scorer.fit(np.concatenate([X, s.mirror(X)]), np.concatenate([y, 1 - y]))
    np.testing.assert_allclose(model.p_home(s.mirror(X)), 1 - model.p_home(X), atol=1e-9)


def test_team_neutral_clues_matter_only_with_a_team_side():
    # a fresh touch is strong evidence, an old one weak: the touch age must change p
    s = _s()
    rng = np.random.default_rng(1)
    n = 6000
    X = np.full((n, len(s.FRAME_CLUES)), np.nan)
    team = rng.choice([-1.0, 1.0], size=n)
    age = rng.uniform(0, 3, size=n)
    X[:, s.FRAME_CLUES.index("touch_team")] = team
    X[:, s.FRAME_CLUES.index("touch_age")] = age
    right = rng.uniform(size=n) < np.where(age < 1, 0.95, 0.55)
    y = np.where(right, team > 0, team < 0).astype(int)
    p = s.Scorer.fit(X, y).p_home(X)
    fresh, old = (team > 0) & (age < 0.5), (team > 0) & (age > 2)
    assert p[fresh].mean() > 0.85 and 0.5 < p[old].mean() < 0.65


def test_nan_clues_give_finite_scores():
    s = _s()
    X, y = _random_rows()
    p = s.Scorer.fit(X, y).p_home(np.full((10, len(s.FRAME_CLUES)), np.nan))
    assert np.isfinite(p).all() and np.allclose(p, 0.5)


def test_training_twice_gives_identical_weights():
    s = _s()
    X, y = _random_rows()
    a, b = s.Scorer.fit(X, y), s.Scorer.fit(X, y)
    assert a.to_json() == b.to_json()


def test_json_round_trip(tmp_path):
    s = _s()
    X, y = _random_rows()
    model = s.Scorer.fit(X, y)
    path = tmp_path / "scorer.json"
    path.write_text(model.to_json(), encoding="utf-8")
    again = s.Scorer.from_json(path.read_text(encoding="utf-8"))
    np.testing.assert_array_equal(again.p_home(X), model.p_home(X))
    assert again.hint_weight == model.hint_weight > 0


def test_hint_weight_is_the_median_pull_on_the_training_frames():
    s = _s()
    X, y = _random_rows()
    model = s.Scorer.fit(X, y)
    p = np.clip(model.p_home(X), s.P_CLIP, 1 - s.P_CLIP)
    assert model.hint_weight == pytest.approx(float(np.median(np.abs(np.log(p) - np.log(1 - p)))))


def test_training_rows_keep_every_5th_live_frame():
    s = _s()
    X = np.arange(40, dtype=float)[:, None] * np.ones((1, len(s.FRAME_CLUES)))
    labels = np.array([H] * 10 + [D] * 10 + [A] * 10 + [3] * 10, dtype=np.int8)
    rows, y = s.training_rows(X, labels)
    assert rows[:, 0].tolist() == [0, 5, 20, 25] and y.tolist() == [1, 1, 0, 0]


def _two_halves_of_possession():
    """400 frames, ball rolling at 2 m/s: a home player on it for 200 frames, then an away one."""
    ball = [(-40.0 + 0.08 * t, 0.0) for t in range(400)]
    players = [[(x + 0.5, 0.0), (x + 10.0, 5.0)] if t < 200 else [(x + 10.0, 5.0), (x + 0.5, 0.0)]
               for t, (x, _) in enumerate(ball)]
    labels = np.array([H] * 200 + [A] * 200, dtype=np.int8)
    return synthetic_table(ball, players, teams=[0, 1]), labels


def _trained_on(table, labels):
    s = _s()
    X = s.frame_clues(table, DEFAULT_CLUES)
    rows, y = s.training_rows(X, labels, every=1)
    return s.Scorer.fit(np.concatenate([rows, s.mirror(rows)]), np.concatenate([y, 1 - y]))


def test_frame_clues_on_a_synthetic_match():
    s = _s()
    table, _ = _two_halves_of_possession()
    X = s.frame_clues(table, DEFAULT_CLUES)
    col = {c: i for i, c in enumerate(s.FRAME_CLUES)}
    assert X.shape == (400, len(s.FRAME_CLUES)) and X.dtype == np.float64
    assert (X[50:150, col["margin"]] > 0).all() and (X[250:350, col["margin"]] < 0).all()
    np.testing.assert_allclose(X[50:150, col["nearest"]], 0.5)
    assert (X[50:150, col["control_team"]] == 1).all() and (X[250:350, col["control_team"]] == -1).all()
    np.testing.assert_allclose(X[50:150, col["ball_speed"]], 2.0, atol=1e-9)
    assert (X[:, col["ball_seen"]] == 1).all() and (X[:, col["since_seen"]] == 0).all()
    assert (X[:, col["ball_filled"]] == 0).all()
    np.testing.assert_allclose(X[:, col["abs_x"]], np.abs(table.ball_xy[:, 0]))


def test_frame_clues_when_the_ball_is_unseen():
    s = _s()
    table, _ = _two_halves_of_possession()
    ball = table.ball_xy.copy()
    ball[100:110] = np.nan
    X = s.frame_clues(replace(table, ball_xy=ball), DEFAULT_CLUES)
    col = {c: i for i, c in enumerate(s.FRAME_CLUES)}
    assert np.isnan(X[100:110, col["margin"]]).all() and (X[100:110, col["ball_seen"]] == 0).all()
    np.testing.assert_allclose(X[100:110, col["since_seen"]], np.arange(1, 11) / 25)
    filled = s.frame_clues(replace(table, ball_xy=ball), replace(DEFAULT_CLUES, gap_fill_frames=12))
    assert (filled[100:110, col["ball_filled"]] == 1).all() and (filled[100:110, col["margin"]] > 0).all()


def test_frame_clues_never_cross_the_half_time_gap():
    s = _s()
    table, _ = _two_halves_of_possession()
    halves = replace(table, graded_runs=[(0, 200), (201, 400)])
    X = s.frame_clues(halves, DEFAULT_CLUES)
    col = {c: i for i, c in enumerate(s.FRAME_CLUES)}
    assert np.isnan(X[200, col["margin"]]) and X[200, col["control_team"]] == 0
    second = s.frame_clues(replace(table, ball_xy=table.ball_xy[201:], ball_z=table.ball_z[201:],
                                   player_xy=table.player_xy[201:], graded_runs=[(0, 199)]), DEFAULT_CLUES)
    np.testing.assert_array_equal(X[201:], second)


def test_model_scores_follow_the_written_rule():
    # H = T log(2p), A = T log(2(1 - p)), team part of D = 0, plus the slow part as before
    from remontada.evidence import _probabilities
    from remontada.pipeline import Settings, features
    s = _s()
    table, labels = _two_halves_of_possession()
    model = _trained_on(table, labels)
    counts = count(table, labels)
    f = features(table, DEFAULT_CLUES, scorer=model)
    p = np.clip(model.p_home(s.frame_clues(table, DEFAULT_CLUES)), s.P_CLIP, 1 - s.P_CLIP)
    np.testing.assert_allclose(f.p_home, p)
    settings = Settings(0.5, Costs(8, 8, 8), False, False, scorer="model")
    slow = _probabilities(counts.slow)[f.slow]
    got = f.scores(counts, settings)
    np.testing.assert_allclose(got[:, H], 0.5 * (np.log(2 * p) + np.log(slow[:, 0])), atol=1e-12)
    np.testing.assert_allclose(got[:, A], 0.5 * (np.log(2 * (1 - p)) + np.log(slow[:, 0])), atol=1e-12)
    np.testing.assert_allclose(got[:, D], 0.5 * np.log(slow[:, 1]), atol=1e-12)
    assert f.hint(counts, settings) == pytest.approx(0.5 * model.hint_weight)


def test_the_tally_still_scores_when_a_scorer_is_attached():
    from remontada.pipeline import Settings, features
    table, labels = _two_halves_of_possession()
    counts = count(table, labels)
    tally = Settings(0.5, Costs(8, 8, 8), True, False)
    np.testing.assert_array_equal(features(table, DEFAULT_CLUES, scorer=_trained_on(table, labels)).scores(counts, tally),
                                  features(table).scores(counts, tally))


def test_mirrored_training_equals_training_on_the_mirrored_copies():
    s = _s()
    X, y = _random_rows()
    y = (np.random.default_rng(2).uniform(size=len(y)) < 0.7 * y + 0.15).astype(int)
    one = s.Scorer.fit(X, y, mirrored=True)
    two = s.Scorer.fit(np.concatenate([X, s.mirror(X)]), np.concatenate([y, 1 - y]))
    np.testing.assert_allclose(one.p_home(X), two.p_home(X), atol=1e-9)


def test_scorer_decodes_a_mostly_missing_ball():
    from remontada.pipeline import Settings, predict
    table, labels = _two_halves_of_possession()
    model = _trained_on(table, labels)
    ball = table.ball_xy.copy()
    ball[20:380] = np.nan
    settings = Settings(1.0, Costs(8, 8, 8), True, False, scorer="model")
    codes = predict(replace(table, ball_xy=ball), count(table, labels), settings, scorer=model)
    assert len(codes) == 400 and set(codes.tolist()) <= {H, A, D}


def test_scorer_finds_the_two_possessions():
    from remontada.pipeline import Settings, predict
    table, labels = _two_halves_of_possession()
    settings = Settings(1.0, Costs(8, 8, 8), True, False, scorer="model")
    codes = predict(table, count(table, labels), settings, scorer=_trained_on(table, labels))
    assert (codes[20:180] == H).all() and (codes[220:380] == A).all()


def test_left_out_scorers_never_see_their_match():
    s = _s()
    from remontada.practice import ALL, train_left_out_scorers
    n = len(s.FRAME_CLUES)
    X = np.zeros((200, n)); X[:, 0] = 5.0  # home is nearer on every frame
    rows = {"m1": (X, np.ones(200, dtype=int)), "m2": (X, np.zeros(200, dtype=int)),
            "m3": (X, np.zeros(200, dtype=int))}
    scorers = train_left_out_scorers(rows)
    assert set(scorers) == {"m1", "m2", "m3", ALL}
    assert scorers["m1"].p_home(X[:1])[0] < 0.1  # learned on m2 and m3 only: nearer home means away
    assert scorers["m2"].p_home(X[:1])[0] == pytest.approx(0.5, abs=0.05)  # m1 and m3 disagree

import numpy as np

from conftest import synthetic_table
from remontada.evidence import count
from remontada.hmm import Costs
from remontada.lomo import leave_one_out_counts
from remontada.pipeline import Settings, features, load_settings, predict
from remontada.evidence import Counts
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

import numpy as np

from conftest import NAN_XY, synthetic_table
from remontada.dead_ball import ball_missing_long, ball_out, ball_still, players_median_speed

PLAYER = [(0.0, 0.0)]


def test_ball_out_uses_the_pitch_and_margin():  # pitch (105, 68): touchline at 34, goal line at 52.5
    ball = [(53.6, 0.0), (53.0, 0.0), (0.0, -35.1), (0.0, 34.9), NAN_XY]
    table = synthetic_table(ball, [PLAYER] * len(ball), teams=[0])
    assert ball_out(table).tolist() == [True, False, True, False, False]


def _still_between_moves(still_frames):
    # rolling at 5 m/s for 20 frames, then still, then rolling again
    x = [0.2 * t for t in range(20)] + [4.0] * still_frames + [4.0 + 0.2 * (t + 1) for t in range(20)]
    return synthetic_table([(v, 0.0) for v in x], [PLAYER] * len(x), teams=[0])


def test_ball_still_needs_a_full_second():
    still = ball_still(_still_between_moves(30))
    assert still[20:50].tolist() == [True] * 30
    assert not still[:20].any() and not still[50:].any()
    assert not ball_still(_still_between_moves(20)).any()
    assert not ball_still(synthetic_table([NAN_XY] * 40, [PLAYER] * 40, teams=[0])).any()


def test_missing_long_marks_whole_gap():
    ball = [(0.0, 0.0)] * 10 + [NAN_XY] * 80 + [(0.0, 0.0)] * 10 + [NAN_XY] * 50 + [(0.0, 0.0)] * 10
    missing = ball_missing_long(synthetic_table(ball, [PLAYER] * len(ball), teams=[0]))
    assert missing[10:90].all() and missing.sum() == 80


def test_players_median_speed_ignores_nan_slots():
    # three players at 2 m/s and two slots that are never seen
    players = [[(0.08 * t, 0.0), (0.0, 0.08 * t), (-0.08 * t, 5.0), NAN_XY, NAN_XY] for t in range(60)]
    table = synthetic_table([(0.0, 0.0)] * 60, players, teams=[0, 1, 0, 1, 0])
    np.testing.assert_allclose(players_median_speed(table)[12:-12], 2.0)
    empty = synthetic_table([(0.0, 0.0)] * 60, [[NAN_XY]] * 60, teams=[0])
    assert np.isnan(players_median_speed(empty)).all()

import numpy as np

from conftest import NAN_XY, synthetic_table
from remontada.touch import CARRY_FRAMES, carry_forward, control, nearest_by_team, touches


def test_nearest_by_team_ignores_empty_slots_and_unseen_ball():
    ball = [(0.0, 0.0), NAN_XY, (0.0, 0.0)]
    players = [
        [(1.0, 0.0), NAN_XY, (0.0, 4.0)],
        [(1.0, 0.0), NAN_XY, (0.0, 4.0)],
        [(1.0, 0.0), NAN_XY, NAN_XY],
    ]
    home, away = nearest_by_team(synthetic_table(ball, players, teams=[0, 1, 1]))
    assert (home[0], away[0]) == (1.0, 4.0)
    assert np.isnan(home[1]) and np.isnan(away[1])
    assert home[2] == 1.0 and away[2] == np.inf


def test_bounce_off_a_home_player_is_a_home_touch():
    # ball runs +x at 10 m/s to x=0 (frame 20), then -x at 10 m/s
    ball = [(-0.4 * abs(t - 20), 0.0) for t in range(41)]
    players = [[(0.0, 0.5), (8.0, 0.0)]] * 41
    t = touches(synthetic_table(ball, players, teams=[0, 1]))
    hits = np.flatnonzero(t >= 0)
    assert len(hits) == 1 and hits[0] in (19, 20, 21) and t[hits[0]] == 0


def test_a_far_glitch_does_not_hide_a_real_touch():
    # real bounce off a home player at frame 20; a 3 m spike at frame 22 with no player near makes
    # a bigger jump inside the same window
    ball = [(-0.4 * abs(t - 20), 0.0) for t in range(41)]
    ball[22] = (ball[22][0], 3.0)
    players = [[(0.0, 0.5), (8.0, 0.0)]] * 41
    t = touches(synthetic_table(ball, players, teams=[0, 1]))
    assert t[20] == 0 and t[22] == -1


def test_straight_ball_has_no_touch():
    ball = [(-8.0 + 0.4 * t, 0.0) for t in range(41)]
    players = [[(0.0, 0.5), (8.0, 0.0)]] * 41
    assert (touches(synthetic_table(ball, players, teams=[0, 1])) == -1).all()


def _ball_near_away_player(frames_near):
    ball = [(0.0, 0.0)] * 30
    players = [[(20.0, 0.0), (0.5, 0.0) if 5 <= t < 5 + frames_near else (10.0, 0.0)] for t in range(30)]
    return synthetic_table(ball, players, teams=[0, 1])


def test_control_needs_time_on_the_ball():
    c = control(_ball_near_away_player(15))
    assert (c[5:20] == 1).all() and (c[:5] == -1).all() and (c[20:] == -1).all()
    assert (control(_ball_near_away_player(5)) == -1).all()


def test_pass_past_an_opponent_keeps_the_team():
    # home player kicks at frame 10, the ball flies 20 m in 40 frames passing 2 m from an away
    # player, and a home teammate receives it at frame 50
    ball = [(0.0, 0.0)] * 10 + [(0.5 * (t - 10), 0.0) for t in range(10, 51)] + [(20.0, 0.0)] * 20
    players = [[(-0.5, 0.0), (20.5, 0.0), (10.0, 2.0)]] * len(ball)
    t = touches(synthetic_table(ball, players, teams=[0, 0, 1]))
    assert t[10] == 0
    assert (carry_forward(t, CARRY_FRAMES)[11:50] == 0).all()


def test_carry_forward_ages_out():
    assert carry_forward(np.array([1, -1, -1, -1], dtype=np.int8), 2).tolist() == [1, 1, 1, -1]


def test_mostly_hidden_ball_gives_none_not_errors():
    rng = np.random.default_rng(0)
    ball = rng.uniform(-30, 30, size=(1000, 2))
    ball[rng.random(1000) < 0.95] = np.nan
    players = rng.uniform(-30, 30, size=(1000, 4, 2))
    table = synthetic_table(ball.tolist(), players.tolist(), teams=[0, 0, 1, 1])
    for out in (touches(table), control(table)):
        assert out.dtype == np.int8 and len(out) == 1000 and (out == -1).mean() > 0.9

from dataclasses import replace

import numpy as np

from conftest import synthetic_table
from remontada.clues import DEFAULT_CLUES
from remontada.touch import touches

NAN2 = [np.nan, np.nan]


def _fill(*args):
    from remontada.gaps import fill_ball_gaps
    return fill_ball_gaps(*args)


def test_fills_a_short_gap_in_a_straight_line():
    xy = np.array([[0, 0], NAN2, NAN2, [3, 6]], float)
    filled, mask = _fill(xy, [(0, 4)], 5)
    np.testing.assert_allclose(filled, [[0, 0], [1, 2], [2, 4], [3, 6]])
    assert mask.tolist() == [False, True, True, False]
    assert np.isnan(xy[1, 0])  # the input is left as it was


def test_long_gaps_and_open_ends_stay_unseen():
    xy = np.array([NAN2, [0, 0]] + [NAN2] * 6 + [[6, 0], NAN2], float)
    filled, mask = _fill(xy, [(0, 10)], 5)
    assert not mask.any()
    np.testing.assert_array_equal(np.isnan(filled), np.isnan(xy))


def test_a_gap_of_exactly_max_frames_is_filled():
    xy = np.array([[0, 0]] + [NAN2] * 5 + [[6, 0]], float)
    filled, mask = _fill(xy, [(0, 7)], 5)
    assert mask.sum() == 5
    np.testing.assert_allclose(filled[:, 0], [0, 1, 2, 3, 4, 5, 6])


def test_gaps_never_bridge_two_runs():
    xy = np.array([[0, 0], NAN2, [2, 0]], float)
    assert not _fill(xy, [(0, 2), (2, 3)], 5)[1].any()


def test_frames_outside_every_run_are_never_filled():
    xy = np.array([[0, 0], NAN2, [2, 0], NAN2, [4, 0]], float)
    filled, mask = _fill(xy, [(2, 5)], 5)
    assert mask.tolist() == [False, False, False, True, False]
    assert np.isnan(filled[1, 0])


def test_zero_max_frames_fills_nothing():
    xy = np.array([[0, 0], NAN2, [2, 0]], float)
    assert not _fill(xy, [(0, 3)], 0)[1].any()


def _kink_through_a_gap():
    """The ball rolls along x, vanishes for 10 frames, and comes back 8 m to the side. A home
    player runs right next to it the whole time, so only the filled frames make the kink."""
    seen = [(-20.0 + 0.16 * t, 0.0) for t in range(20)] + [(-16.8, 8.0 + 0.16 * t) for t in range(20)]
    xy = np.array(seen[:20] + [NAN2] * 10 + seen[20:], float)
    filled, mask = _fill(xy, [(0, 50)], 25)
    players = [[(x + 0.5, y), (x + 30.0, y + 20.0)] for x, y in filled]
    table = synthetic_table(filled.tolist(), players, teams=[0, 1])
    return table, mask


def test_the_filled_kink_would_look_like_a_touch():
    table, mask = _kink_through_a_gap()
    assert mask.sum() == 10
    assert (touches(table) >= 0).any()  # so the next test is not passing by accident


def test_filled_frames_never_make_a_touch():
    table, mask = _kink_through_a_gap()
    # nor does any velocity jump worked out from a filled position, at the seen frames next to it
    assert (touches(table, ignore=mask) == -1).all()


def test_ignore_leaves_measured_touches_alone():
    table, mask = _kink_through_a_gap()
    nowhere = np.zeros(table.n_frames, dtype=bool)
    np.testing.assert_array_equal(touches(table, ignore=nowhere), touches(table))


def _gap_match():
    """A home player dribbles with the ball; the ball is unseen for 6 frames in the middle."""
    ball = [(-10.0 + 0.1 * t, 0.0) for t in range(60)]
    players = [[(x + 0.5, 0.0), (x + 20.0, 5.0)] for x, _ in ball]
    for t in range(27, 33):
        ball[t] = (np.nan, np.nan)
    return synthetic_table(ball, players, teams=[0, 1])


def test_features_see_the_ball_through_a_short_gap():
    from remontada.pipeline import features
    table = _gap_match()
    plain = features(table)
    filled = features(table, replace(DEFAULT_CLUES, gap_fill_frames=12))
    gap = slice(27, 33)
    assert (plain.team[gap] // 9 == 0).all()  # nearness bin 0: ball unseen
    assert (filled.team[gap] // 9 > 0).all()  # the ball is seen again, by the straight line
    np.testing.assert_array_equal(filled.team[:27], plain.team[:27])


def test_features_leave_long_gaps_unseen():
    from remontada.pipeline import features
    table = _gap_match()
    filled = features(table, replace(DEFAULT_CLUES, gap_fill_frames=5))
    assert (filled.team[27:33] // 9 == 0).all()

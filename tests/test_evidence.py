from dataclasses import replace

import numpy as np

from conftest import NAN_XY, synthetic_table
from remontada.evidence import (Counts, count, emission_scores, hint_weight, load_counts,
                                save_counts, slow_cells, team_cells, team_cells_mirrored)
from remontada.vendor.possession_metrics import A, D, H, X


def mirror(table):
    return replace(table, player_team=(1 - table.player_team).astype(np.int8))


def _near_home(n=100):
    """Ball still, a home player 0.5 m from it and an away player 5 m from it."""
    return synthetic_table([(0.0, 0.0)] * n, [[(0.5, 0.0), (5.0, 0.0)]] * n, teams=[0, 1])


def test_cells_cover_the_range():
    rng = np.random.default_rng(0)
    ball = rng.uniform(-20, 20, size=(500, 2))
    ball[rng.random(500) < 0.3] = np.nan
    players = rng.uniform(-20, 20, size=(500, 4, 2))
    players[rng.random((500, 4)) < 0.3] = np.nan
    table = synthetic_table(ball.tolist(), players.tolist(), teams=[0, 0, 1, 1])
    for cells, size in ((team_cells(table), 54), (team_cells_mirrored(table), 54), (slow_cells(table), 7)):
        assert cells.dtype == np.int16 and len(cells) == 500
        assert cells.min() >= 0 and cells.max() < size


def test_home_nearness_learned_then_mirrored():
    table = _near_home()
    counts = count(table, np.full(100, H, dtype=np.int8))
    s = emission_scores(table, counts, 1.0)
    assert (s[:, H] > s[:, A]).all()
    s2 = emission_scores(mirror(table), counts, 1.0)
    np.testing.assert_allclose(s2[:, A] - s2[:, H], s[:, H] - s[:, A])


def test_mirrored_cells_match_a_mirrored_match():
    rng = np.random.default_rng(1)
    ball = rng.uniform(-20, 20, size=(300, 2)).tolist()
    players = rng.uniform(-20, 20, size=(300, 4, 2)).tolist()
    table = synthetic_table(ball, players, teams=[0, 0, 1, 1])
    np.testing.assert_array_equal(team_cells_mirrored(table), team_cells(mirror(table)))


def test_team_clues_are_neutral_on_dead():
    rng = np.random.default_rng(2)
    counts = Counts(team=rng.integers(0, 50, size=(54, 2)), slow=np.full((7, 2), 10))
    ball = rng.uniform(-20, 20, size=(300, 2)).tolist()
    players = rng.uniform(-20, 20, size=(300, 4, 2)).tolist()
    s = emission_scores(synthetic_table(ball, players, teams=[0, 0, 1, 1]), counts, 1.0)
    low, high = np.minimum(s[:, H], s[:, A]), np.maximum(s[:, H], s[:, A])
    assert ((low <= s[:, D]) & (s[:, D] <= high)).all()


def test_slow_players_point_to_dead():
    # players run at 2.5 m/s while play is on (H), then stand still while it is dead (D)
    xs = [0.1 * t for t in range(100)] + [9.9] * 100
    players = [[(x, 0.0), (x, 5.0)] for x in xs]
    table = synthetic_table([NAN_XY] * 200, players, teams=[0, 1])
    labels = np.array([H] * 100 + [D] * 100, dtype=np.int8)
    s = emission_scores(table, count(table, labels), 1.0)
    assert (s[125:188, D] > s[125:188, H]).all()
    assert (s[12:88, H] > s[12:88, D]).all()


def test_x_frames_are_ignored():
    labels = np.full(100, X, dtype=np.int8)
    counts = count(_near_home(), labels)
    assert counts.team.sum() == 0 and counts.slow.sum() == 0


def test_counts_round_trip_and_add(tmp_path):
    table = _near_home()
    counts = count(table, np.full(100, H, dtype=np.int8))
    save_counts(counts, tmp_path / "c.json")
    back = load_counts(tmp_path / "c.json")
    np.testing.assert_array_equal(back.team, counts.team)
    np.testing.assert_array_equal(back.slow, counts.slow)
    both = counts + back
    assert both.team.sum() == 2 * counts.team.sum() and both.slow.sum() == 2 * counts.slow.sum()
    assert counts.team.sum() == 200  # each live frame counted once as is and once mirrored


def test_hint_weight_on_a_hand_built_table():
    # one cell (90, 10) seen 100 times, one (150, 150) seen 300 times, the rest empty:
    # the count-weighted median is the second cell's smoothed ratio, close to 0
    team = np.zeros((54, 2), dtype=np.int64)
    team[0], team[1] = (90, 10), (150, 150)
    counts = Counts(team=team, slow=np.zeros((7, 2), dtype=np.int64))
    expected = abs(np.log(151 / (240 + 54)) - np.log(151 / (160 + 54)))
    assert np.isclose(hint_weight(counts, 1.0), expected)
    assert np.isclose(hint_weight(counts, 0.2), 0.2 * expected)

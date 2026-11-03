from dataclasses import replace

import numpy as np
import pytest

from conftest import synthetic_table
from remontada.degrade import LEVELS, Level, degrade, hard_frames

N_FRAMES = 25000


@pytest.fixture(scope="module")
def clean_table():
    """A long clean match: ball always visible, four players always visible, two per team."""
    rng = np.random.default_rng(1)
    ball = rng.uniform(-30, 30, size=(N_FRAMES, 2)).tolist()
    players = rng.uniform(-40, 40, size=(N_FRAMES, 4, 2)).tolist()
    return synthetic_table(ball, players, teams=[0, 0, 1, 1])


def test_levels_cover_the_sweep():
    assert list(LEVELS) == ["none", "light", "medium", "heavy"]
    assert LEVELS["medium"] == Level(0.30, 1.0, 0.25, 1.0)


def test_none_level_is_identity(clean_table):
    out = degrade(clean_table, LEVELS["none"], seed=0)
    np.testing.assert_array_equal(out.ball_xy, clean_table.ball_xy)
    np.testing.assert_array_equal(out.player_xy, clean_table.player_xy)


def test_ball_hidden_share_close_to_target(clean_table):
    out = degrade(clean_table, LEVELS["medium"], seed=0)
    assert abs(np.isnan(out.ball_xy[:, 0]).mean() - 0.30) < 0.03


def test_heavy_never_hides_every_frame(clean_table):
    assert not np.isnan(degrade(clean_table, LEVELS["heavy"], seed=0).ball_xy[:, 0]).all()


def test_same_seed_same_output(clean_table):
    a, b = (degrade(clean_table, LEVELS["heavy"], seed=7) for _ in range(2))
    np.testing.assert_array_equal(a.player_xy, b.player_xy)
    np.testing.assert_array_equal(a.ball_xy, b.ball_xy)


def test_position_noise_is_small_but_real(clean_table):
    out = degrade(clean_table, Level(0.0, 0.0, 0.5, 0.0), seed=0)
    spread = np.nanstd(out.player_xy - clean_table.player_xy)
    assert 0.4 < spread < 0.6


def test_track_breaks_keep_team_and_player_count(clean_table):
    out = degrade(clean_table, LEVELS["heavy"], seed=0)
    assert out.player_xy.shape[1] > clean_table.player_xy.shape[1]

    def visible(table, team):
        return (~np.isnan(table.player_xy[:, table.player_team == team, 0])).sum(axis=1)

    for team in (0, 1):
        np.testing.assert_array_equal(visible(out, team), visible(clean_table, team))


@pytest.mark.parametrize("n_frames", [25, 100, 250])
def test_short_tables_never_lose_every_ball_frame(n_frames):
    table = synthetic_table([(0.0, 0.0)] * n_frames, [[(1.0, 1.0)]] * n_frames, teams=[0])
    for seed in range(300):
        hidden = np.isnan(degrade(table, LEVELS["heavy"], seed).ball_xy[:, 0]).sum()
        assert hidden == int(0.60 * n_frames)


def test_wobble_drifts_slowly(clean_table):
    out = degrade(clean_table, Level(0.0, 0.0, 0.5, 0.0), seed=0)
    noise = out.player_xy - clean_table.player_xy
    step = np.nanstd(np.diff(noise, axis=0))
    assert step < 0.2  # per-frame jumps of the old noise were about 0.71 m


def test_hard_frames_are_air_or_crowd():
    ball = [(0.0, 0.0)] * 3
    players = [[(10, 0), (20, 0), (30, 0)], [(1, 0), (0, 1), (-1, 0)], [(10, 0), (20, 0), (30, 0)]]
    table = replace(synthetic_table(ball, players, teams=[0, 1, 0]), ball_z=np.array([0.2, 0.2, 1.5]))
    assert hard_frames(table).tolist() == [False, True, True]


def test_hiding_prefers_hard_frames(clean_table):
    # ball in the air for 10 s in every 50 s; heavy hiding must hit those frames more often
    hard = (np.arange(N_FRAMES) // 250) % 5 == 0
    table = replace(clean_table, ball_z=np.where(hard, 2.0, 0.1))
    hidden = np.isnan(degrade(table, LEVELS["heavy"], seed=0).ball_xy[:, 0])
    assert hidden.sum() == int(0.60 * len(hidden))
    assert hidden[hard].mean() > hidden[~hard].mean() + 0.1

import numpy as np
import pytest

from conftest import NAN_XY, synthetic_table
from remontada.baseline import nearest_team, smooth
from remontada.load_bar_ath import load_bar_ath
from remontada.submission import sha256, write_csv
from remontada.vendor.possession_metrics import A, D, H


def test_ball_without_players_gives_no_answer():
    table = synthetic_table(ball=[(0.0, 0.0)], players=[[NAN_XY, NAN_XY]])
    assert nearest_team(table).tolist() == [-1]


def test_missing_ball_gives_no_answer():
    table = synthetic_table(ball=[NAN_XY], players=[[(1.0, 0.0), (2.0, 0.0)]])
    assert nearest_team(table).tolist() == [-1]


def test_nearest_player_team():
    table = synthetic_table(ball=[(0.0, 0.0)], players=[[(3.0, 0.0), (1.0, 0.0)]], teams=[0, 1])
    assert nearest_team(table).tolist() == [1]


def test_first_slot_wins_a_tie():
    table = synthetic_table(ball=[(0.0, 0.0)], players=[[(1.0, 0.0), (-1.0, 0.0)]], teams=[1, 0])
    assert nearest_team(table).tolist() == [1]


def test_smooth_majority_and_no_votes():
    codes = np.array([0, 0, 1, 0, 0, -1, -1, -1, -1, -1, -1], dtype=np.int8)
    assert smooth(codes, 0).tolist() == [H, H, A, H, H, D, D, D, D, D, D]
    assert smooth(codes, 1).tolist() == [H, H, H, H, H, H, D, D, D, D, D]


def test_write_csv_uses_plain_newlines(tmp_path):
    path = write_csv(np.array([H, A, D], dtype=np.int8), tmp_path / "x.csv")
    assert path.read_bytes() == b"frame,label\n0,H\n1,A\n2,D\n"


@pytest.mark.realdata
@pytest.mark.parametrize("window,expected_sha", [
    (0, "9471afeb95722454a96189481561adfb73b3963365a83969f04a986930f12511"),
    (25, "c1fd857c2c0bd5299260b6c55dcd419bde41a4c5af46f9bbcb41b4b3b577353f"),
    (50, "6d4e04b80b9016cc16ada99697bf219d3a74b58750dd6d5ec5a4cd5fc32ab4b9"),
    (125, "41704ddccc1edcddf6a0781e3fdfa06ae07b19fb9ead3bf67f663bfc7b1900ba"),
])
def test_reproduces_official_baselines(tmp_path, window, expected_sha):
    # Hashes of upstream evaluation/results/*.csv with \n line endings.
    codes = smooth(nearest_team(load_bar_ath()), window)
    assert sha256(write_csv(codes, tmp_path / "b.csv")) == expected_sha

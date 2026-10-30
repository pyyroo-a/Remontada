import numpy as np
import pytest

from conftest import write_synthetic_bar_ath
from remontada import paths
from remontada.load_bar_ath import load_bar_ath
from remontada.paths import DataNotFound


def test_parses_synthetic_atd(tmp_path):
    table = load_bar_ath(write_synthetic_bar_ath(tmp_path))
    assert table.n_frames == 3
    assert table.player_team.tolist() == [0, 1]
    assert np.isnan(table.ball_xy[1]).all()
    assert table.ball_xy[0].tolist() == pytest.approx([(0.6 - 0.5) * 105.3, (0.25 - 0.5) * 68.0])
    assert np.isnan(table.player_xy[1, 1]).all()
    assert table.graded_runs == [(11185, 90652), (110808, 190198)]


def test_missing_data_dir_names_the_variable(monkeypatch):
    monkeypatch.delenv("MTAH_DATA_DIR", raising=False)
    with pytest.raises(DataNotFound, match="MTAH_DATA_DIR"):
        paths.data_dir()


def test_missing_file_names_the_path(tmp_path):
    with pytest.raises(DataNotFound, match="BAR-ATH_tactical_atd.txt"):
        load_bar_ath(tmp_path / "more than a hack")


@pytest.mark.realdata
def test_real_bar_ath_shape():
    table = load_bar_ath()
    assert table.n_frames == 199319
    assert table.player_xy.shape[1] == 46
    assert int((table.player_team == 0).sum()) == 24
    assert int((~np.isnan(table.ball_xy[:, 0])).sum()) == 113488

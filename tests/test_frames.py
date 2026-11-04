import tomllib
from pathlib import Path

import numpy as np

from conftest import synthetic_table
from remontada.frames import run_slice


def test_repr_never_shows_positions():
    table = synthetic_table(ball=[(12.3456, 65.4321)], players=[[(23.4567, 76.5432)]], teams=[0])
    text = repr(table)
    for value in ("12.3456", "65.4321", "23.4567", "76.5432"):
        assert value not in text
    assert "SYNTH" in text and "frames=1" in text and "slots=1" in text


def test_pytest_tracebacks_stay_short_by_default():
    # Failure output must never print match data, even when someone runs plain pytest.
    config = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    assert "--tb=line" in config["tool"]["pytest"]["ini_options"]["addopts"]


def test_run_slice_is_one_graded_run():
    table = synthetic_table([(float(t), 0.0) for t in range(10)], [[(0.0, float(t))] for t in range(10)], teams=[0])
    part = run_slice(table, 3, 7)
    assert part.n_frames == 4 and part.graded_runs == [(0, 4)]
    np.testing.assert_array_equal(part.ball_xy[:, 0], [3, 4, 5, 6])
    np.testing.assert_array_equal(part.player_xy[:, 0, 1], [3, 4, 5, 6])
    assert part.fps == table.fps and part.pitch == table.pitch

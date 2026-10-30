import tomllib
from pathlib import Path

from conftest import synthetic_table


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

"""Shared test helpers: synthetic matches in our formats, and the realdata skip."""
import os
from pathlib import Path

import numpy as np
import pytest

from remontada.frames import FrameTable

NAN_XY = (float("nan"), float("nan"))


def synthetic_table(ball, players, teams=None):
    """A FrameTable in metres from plain lists. One graded run covers every frame."""
    ball_xy = np.array(ball, dtype=np.float64).reshape(len(ball), 2)
    player_xy = np.array(players, dtype=np.float64).reshape(len(players), -1, 2)
    if teams is None:
        teams = [slot % 2 for slot in range(player_xy.shape[1])]
    return FrameTable(
        match_id="SYNTH",
        fps=25,
        pitch=(105.0, 68.0),
        ball_xy=ball_xy,
        ball_z=np.full(len(ball), np.nan),
        player_xy=player_xy,
        player_team=np.array(teams, dtype=np.int8),
        graded_runs=[(0, len(ball))],
    )


def write_synthetic_bar_ath(root):
    """A 3-frame, 2-slot BAR-ATH bundle in the official file formats. Ball missing on frame 2."""
    folder = Path(root) / "tactical"
    folder.mkdir(parents=True)
    (folder / "BAR-ATH_tactical_atd.xml").write_text(
        '<main><Metadata><Teams>'
        '<Team id="HOME"><Name>Home FC</Name></Team>'
        '<Team id="AWAY"><Name>Away FC</Name></Team>'
        '</Teams><Players>'
        '<Player id="P1" teamId="HOME"><Name>Track 1</Name></Player>'
        '<Player id="P2" teamId="AWAY"><Name>Track 2</Name></Player>'
        '</Players></Metadata></main>',
        encoding="utf-8",
    )
    (folder / "BAR-ATH_tactical_atd.txt").write_text(
        "1:0.50000,0.50000,1.0;0.70000,0.25000,2.0:0.60000,0.25000,0.0\n"
        "2:0.51000,0.50000,1.0;NaN,NaN,NaN:NaN,NaN,NaN\n"
        "3:NaN,NaN,NaN;0.71000,0.26000,2.0:0.55000,0.50000,0.0\n",
        encoding="utf-8",
    )
    return Path(root)


def _bar_ath_available():
    root = os.environ.get("MTAH_DATA_DIR")
    return bool(root) and (Path(root) / "tactical" / "BAR-ATH_tactical_atd.txt").is_file()


def pytest_collection_modifyitems(config, items):
    skip = pytest.mark.skip(reason="needs MTAH_DATA_DIR with the match data")
    for item in items:
        if "realdata" in item.keywords and "bar_ath" in item.nodeid and not _bar_ath_available():
            item.add_marker(skip)

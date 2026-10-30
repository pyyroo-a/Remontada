"""Read the BAR-ATH tactical tracking (Metrica FIFA/EPTS export) into a FrameTable.

Formats follow DATA.md and the official starter code: a metadata XML whose first Team is home and
whose Player elements, in file order, are the track slots; and a raw text file with one line per
frame, "<frame_id>:<slot>;<slot>;...:<ball>", each group "x,y,..." normalised 0 to 1, NaN when
missing. Video frame = frame_id - 1.
"""
import xml.etree.ElementTree as ET

import numpy as np

from remontada.frames import FrameTable
from remontada.paths import bar_ath_files

FPS = 25
PITCH = (105.3, 68.0)
GRADED_RUNS = [(11185, 90652), (110808, 190198)]


def _slot_teams(meta_path):
    root = ET.parse(meta_path).getroot()
    team_ids = [team.get("id") for team in root.iter("Team")]
    if len(team_ids) != 2:
        raise ValueError("{}: expected 2 teams, found {}".format(meta_path.name, len(team_ids)))
    slots = [player.get("teamId") for player in root.iter("Player")]
    return np.array([0 if team == team_ids[0] else 1 for team in slots], dtype=np.int8)


def _xy(group):
    first, second = group.split(",")[:2]
    return float(first), float(second)


def load_bar_ath(data_dir=None):
    """The whole tactical recording, one row per video frame (frame_id - 1)."""
    meta_path, raw_path = bar_ath_files(["atd_meta", "atd_raw"], data_dir)
    teams = _slot_teams(meta_path)

    rows = []
    with open(raw_path, encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split(":")
            if len(parts) != 3:
                continue
            rows.append((int(parts[0]) - 1, parts[1].split(";"), parts[2]))

    n_frames = max(frame for frame, _, _ in rows) + 1
    ball_xy = np.full((n_frames, 2), np.nan)
    player_xy = np.full((n_frames, len(teams), 2), np.nan)
    length, width = PITCH
    for frame, slots, ball in rows:
        ball_xy[frame] = _xy(ball)
        for slot, group in enumerate(slots[: len(teams)]):
            player_xy[frame, slot] = _xy(group)

    scale = np.array([length, width])
    return FrameTable(
        match_id="BAR-ATH",
        fps=FPS,
        pitch=PITCH,
        ball_xy=(ball_xy - 0.5) * scale,
        ball_z=np.full(n_frames, np.nan),
        player_xy=(player_xy - 0.5) * scale,
        player_team=teams,
        graded_runs=list(GRADED_RUNS),
    )

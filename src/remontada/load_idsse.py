"""Read one IDSSE match (DFL / Sportec open data, CC BY 4.0) into a FrameTable plus labels.

Rows are the first half, one separator row (NaN, label X), then the second half, so the official
scorer's valid_runs sees two halves. Each ball frame carries BallStatus (1 alive, 0 dead) and
BallPossession (1 home, 2 away; checked against pass events on 29 Oct 2026): our label is D when
dead, else H or A. Coordinates are already metres from the centre spot.

The positions file is about 420 MB, so it is streamed line by line and read with a regex.
"""
import re

import numpy as np

from remontada.frames import FrameTable
from remontada.paths import DataNotFound, idsse_dir
from remontada.vendor import possession_metrics as pm

MATCH_IDS = ("J03WMX", "J03WN1", "J03WOH", "J03WOY", "J03WPY", "J03WQQ", "J03WR9")
COMPETITION = {"J03WMX": "DFL-COM-000001", "J03WN1": "DFL-COM-000001"}  # the rest are 000002
FPS = 25
SECTIONS = ("firstHalf", "secondHalf")
_ATTR = re.compile(r'(\w+)="([^"]*)"')


def file_names(match_id):
    """The three IDSSE files for one match, as named on Hugging Face."""
    tail = "{}_DFL-MAT-{}.xml".format(COMPETITION.get(match_id, "DFL-COM-000002"), match_id)
    return {
        "info": "DFL_02_01_matchinformation_" + tail,
        "events": "DFL_03_02_events_raw_" + tail,
        "positions": "DFL_04_03_positions_raw_observed_" + tail,
    }


def _find(folder, prefix, match_id):
    hits = sorted(folder.glob("{}_*_DFL-MAT-{}.xml".format(prefix, match_id)))
    if not hits:
        raise DataNotFound("IDSSE file not found: {}/{}_*_DFL-MAT-{}.xml".format(folder, prefix, match_id))
    return hits[0]


def _read_positions(path):
    """{section: {"ball": {N: (x, y, z, poss, status)}, "players": {pid: (team, {N: (x, y)})}}}."""
    halves = {section: {"ball": {}, "players": {}} for section in SECTIONS}
    order = []
    current = None
    complete = False
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("<Frame "):
                if current is not None:
                    a = dict(_ATTR.findall(line))
                    current(a)
            elif line.startswith("<FrameSet"):
                a = dict(_ATTR.findall(line))
                half = halves.get(a.get("GameSection"))
                team, person = a.get("TeamId"), a.get("PersonId")
                if half is None or team == "referee":
                    current = None
                elif team == "BALL":
                    ball = half["ball"]
                    current = lambda f, b=ball: b.__setitem__(int(f["N"]), (
                        float(f["X"]), float(f["Y"]), float(f.get("Z", "nan")),
                        f.get("BallPossession"), f.get("BallStatus")))
                else:
                    if person not in order:
                        order.append(person)
                    frames = half["players"].setdefault(person, (team, {}))[1]
                    current = lambda f, p=frames: p.__setitem__(int(f["N"]), (float(f["X"]), float(f["Y"])))
            elif line.startswith("</FrameSet"):
                current = None
            elif line.startswith("</PutDataRequest"):
                complete = True
    if not complete or not all(halves[s]["ball"] for s in SECTIONS):
        raise ValueError("IDSSE positions file looks truncated: {}".format(path.name))
    return halves, order


def load_idsse(match_id, folder=None):
    """(FrameTable, int8 label codes) for one IDSSE match."""
    folder = folder or idsse_dir()
    info = _find(folder, "DFL_02_01_matchinformation", match_id).read_text(encoding="utf-8")
    home = re.search(r'HomeTeamId="([^"]+)"', info).group(1)
    halves, order = _read_positions(_find(folder, "DFL_04_03_positions_raw_observed", match_id))

    teams = {}
    for section in SECTIONS:
        for person, (team, _) in halves[section]["players"].items():
            teams[person] = 0 if team == home else 1
    slot = {person: index for index, person in enumerate(order)}

    lengths = [len(halves[s]["ball"]) for s in SECTIONS]
    n = lengths[0] + 1 + lengths[1]
    ball_xy = np.full((n, 2), np.nan)
    ball_z = np.full(n, np.nan)
    player_xy = np.full((n, len(order), 2), np.nan)
    labels = np.full(n, pm.X, dtype=np.int8)
    offsets = (0, lengths[0] + 1)

    for section, offset in zip(SECTIONS, offsets):
        ball = halves[section]["ball"]
        first = min(ball)
        for number, (x, y, z, possession, status) in ball.items():
            row = offset + number - first
            ball_xy[row] = (x, y)
            ball_z[row] = z
            labels[row] = pm.D if status == "0" else (pm.H if possession == "1" else pm.A)
        for person, (_, frames) in halves[section]["players"].items():
            for number, xy in frames.items():
                row = number - first
                if 0 <= row < len(ball):
                    player_xy[offset + row, slot[person]] = xy

    return (
        FrameTable(
            match_id=match_id,
            fps=FPS,
            pitch=(105.0, 68.0),
            ball_xy=ball_xy,
            ball_z=ball_z,
            player_xy=player_xy,
            player_team=np.array([teams[p] for p in order], dtype=np.int8),
            graded_runs=[(0, lengths[0]), (offsets[1], n)],
        ),
        labels,
    )

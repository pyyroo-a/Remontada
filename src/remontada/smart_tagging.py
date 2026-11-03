"""Smart Tagging: the broadcaster's tagged intervals (BUILD UP, SET PIECES and so on) as frames.

Parsing follows the organisers' starter/load_smart_tagging.py. Times are seconds from the start of
the tracking file, so frame = round(seconds * fps); intervals are half-open like graded_runs.
"""
import xml.etree.ElementTree as ET
from dataclasses import dataclass

from remontada.paths import bar_ath_file

TEAMS = {"barcelona": 0, "atletic club": 1}  # names as written in DATA.md; anything else is -1


@dataclass(frozen=True, repr=False)
class Tag:
    code: str
    start_frame: int
    end_frame: int
    team: int
    type: str | None

    def __repr__(self):
        # Never the frames: a printed tag or a failed test must not show match data.
        return "Tag(...)"


def parse_smart_tagging(path, fps=25):
    """Every instance in file order."""
    tags = []
    for instance in ET.parse(path).getroot().iter("instance"):
        labels = {}
        for label in instance.findall("label"):
            labels.setdefault(label.findtext("group", "").strip(), label.findtext("text", "").strip())
        tags.append(Tag(
            code=instance.findtext("code", "").strip().upper(),
            start_frame=round(float(instance.findtext("start")) * fps),
            end_frame=round(float(instance.findtext("end")) * fps),
            team=TEAMS.get(labels.get("Team", "").lower(), -1),
            type=labels.get("Type"),
        ))
    return tags


def load_smart_tagging(data_dir=None):
    return parse_smart_tagging(bar_ath_file("smart_tagging", data_dir))

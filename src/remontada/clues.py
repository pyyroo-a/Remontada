"""The clue settings in one place, so the Plan 3 search can try them (decided 4 Nov 2026).

The defaults are the Plan 2 starting values, kept as module constants where each clue lives.
"""
from dataclasses import dataclass

from remontada.dead_ball import OUT_MARGIN_M
from remontada.touch import CARRY_FRAMES, CONTROL_MIN_FRAMES, CONTROL_RADIUS, TOUCH_DV, TOUCH_RADIUS


@dataclass(frozen=True)
class ClueSettings:
    touch_dv: float = TOUCH_DV  # m/s change in ball velocity that can be a touch
    touch_radius: float = TOUCH_RADIUS  # m from the nearest player
    control_radius: float = CONTROL_RADIUS  # m
    control_min_frames: int = CONTROL_MIN_FRAMES
    carry_frames: int = CARRY_FRAMES  # how long the last touch is carried forward
    out_margin_m: float = OUT_MARGIN_M  # ball past the line by this much is "out"
    gap_fill_frames: int = 0  # longest ball gap filled by a straight line (0: none)


DEFAULT_CLUES = ClueSettings()

"""One shape for every match: per frame, the ball and every tracked object, in metres."""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FrameTable:
    """Positions in metres with the origin at the centre spot; NaN where something is not seen.

    player_xy is (frames, slots, 2). A slot is a track, not a person: on BAR-ATH slots appear,
    vanish and swap, so nothing may depend on who a slot is. player_team is 0 home, 1 away.
    graded_runs are half-open frame spans that are scored (the two halves).
    """

    match_id: str
    fps: int
    pitch: tuple
    ball_xy: np.ndarray
    ball_z: np.ndarray
    player_xy: np.ndarray
    player_team: np.ndarray
    graded_runs: list

    @property
    def n_frames(self):
        return len(self.ball_xy)

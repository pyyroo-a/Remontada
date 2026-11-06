"""One shape for every match: per frame, the ball and every tracked object, in metres."""
from dataclasses import dataclass, replace

import numpy as np


@dataclass(frozen=True, repr=False, eq=False)
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

    def __repr__(self):
        # Never the arrays: a printed table or a failed test must not show match data.
        return "FrameTable({}, frames={}, slots={})".format(
            self.match_id, self.n_frames, self.player_xy.shape[1])


def run_slice(table, start, stop):
    """Rows start:stop as a table of their own, graded as one run (used to keep clues per half)."""
    return replace(
        table,
        ball_xy=table.ball_xy[start:stop],
        ball_z=table.ball_z[start:stop],
        player_xy=table.player_xy[start:stop],
        graded_runs=[(0, stop - start)],
    )

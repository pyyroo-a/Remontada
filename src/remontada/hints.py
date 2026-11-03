"""Fixed-rule hints: ball-dead clues and Smart Tagging tags nudge a state's score by one weight.

The weight comes from the written rule (evidence.hint_weight, decided 3 Nov 2026) and is never
tuned. Tag meanings come from the code names only, never from looking at BAR-ATH frames.
"""
import numpy as np

from remontada.dead_ball import ball_missing_long, ball_out, ball_still
from remontada.vendor.possession_metrics import D

ATTACKING = {"BUILD UP", "PROGRESSION", "ATTACKING TRANSITION", "CREATING CHANCES",
             "BALL IN FINAL THIRD", "BALL IN THE BOX", "LONG BALL"}  # hint the tagged team
DEFENDING = {"DEFENDING IN DEFENSIVE THIRD", "DEFENDING IN MIDDLE THIRD",
             "DEFENDING IN ATTACKING THIRD", "DEFENSIVE TRANSITION"}  # hint the other team
SET_PIECES = "SET PIECES"  # hint D on the first half: the ball is dead before the restart


def ball_hints(table, weight):
    """weight added to D for each ball-dead clue that is true on a frame."""
    out = np.zeros((table.n_frames, 3))
    out[:, D] = weight * (ball_out(table).astype(int) + ball_still(table) + ball_missing_long(table))
    return out


def tag_hints(tags, n_frames, weight):
    """(n_frames, 3) hints from Smart Tagging; intervals are clipped to the match, overlaps add."""
    out = np.zeros((n_frames, 3))
    for tag in tags:
        start, end = tag.start_frame, tag.end_frame
        if tag.code == SET_PIECES:
            state, end = D, start + (end - start) // 2
        elif tag.team not in (0, 1):
            continue
        elif tag.code in ATTACKING:
            state = tag.team
        elif tag.code in DEFENDING:
            state = 1 - tag.team
        else:
            continue
        out[max(start, 0):max(min(end, n_frames), 0), state] += weight
    return out

import numpy as np

from conftest import synthetic_table
from remontada.hints import ball_hints, tag_hints
from remontada.smart_tagging import Tag
from remontada.vendor.possession_metrics import A, D, H


def test_ball_hints_add_one_weight_per_clue():
    # ball resting 7.5 m past the goal line: out and still on every frame, never missing
    table = synthetic_table([(60.0, 0.0)] * 40, [[(0.0, 0.0)]] * 40, teams=[0])
    h = ball_hints(table, 0.5)
    assert h.shape == (40, 3)
    assert (h[:, D] == 1.0).all() and (h[:, [H, A]] == 0).all()


def test_set_piece_hints_dead_before_the_restart():
    h = tag_hints([Tag("SET PIECES", 100, 150, -1, "Throw-in")], 200, 0.7)
    assert (h[100:125, D] == 0.7).all() and (h[125:, D] == 0).all() and (h[:100, D] == 0).all()
    assert (h[:, :2] == 0).all()


def test_attacking_and_defending_codes():
    h = tag_hints([Tag("BUILD UP", 0, 10, 0, None), Tag("DEFENDING IN MIDDLE THIRD", 10, 20, 0, None)], 20, 1.0)
    assert (h[:10, H] == 1).all() and (h[10:, A] == 1).all() and h[:, D].sum() == 0
    assert h[:10, A].sum() == 0 and h[10:, H].sum() == 0


def test_unknown_code_or_team_gives_nothing():
    tags = [Tag("PLAYERS IN THE BOX", 0, 10, 0, None), Tag("GOALS", 0, 10, 1, None),
            Tag("BUILD UP", 0, 10, -1, None), Tag("NEW CODE", 0, 10, 0, None)]
    assert (tag_hints(tags, 10, 1.0) == 0).all()


def test_overlapping_intervals_add():
    h = tag_hints([Tag("BUILD UP", 0, 10, 1, None), Tag("LONG BALL", 5, 10, 1, None)], 10, 1.0)
    assert h[:, A].tolist() == [1] * 5 + [2] * 5


def test_intervals_outside_the_match_are_clipped():
    h = tag_hints([Tag("BUILD UP", -50, 5, 1, None), Tag("BUILD UP", 195, 400, 1, None)], 200, 1.0)
    assert h.shape == (200, 3) and h[:5, A].sum() == 5 and h[195:, A].sum() == 5
    assert (tag_hints([Tag("SET PIECES", 300, 400, -1, None), Tag("BUILD UP", 50, 40, 0, None)], 200, 1.0) == 0).all()

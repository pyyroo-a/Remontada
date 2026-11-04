from dataclasses import replace

import numpy as np
import pytest

from conftest import synthetic_table
from remontada import dead_ball, touch
from remontada.pipeline import features


def _clues():
    from remontada.clues import DEFAULT_CLUES, ClueSettings
    return DEFAULT_CLUES, ClueSettings


def _bounce():
    """Ball runs +x at 10 m/s to x=0 (frame 20), bounces off a home player 0.5 m away, runs back."""
    ball = [(-0.4 * abs(t - 20), 0.0) for t in range(41)]
    players = [[(0.0, 0.5), (8.0, 0.0)]] * 41
    return synthetic_table(ball, players, teams=[0, 1])


def _two_halves():
    """400 frames, ball rolling at 2 m/s: a home player on it for 200 frames, then an away one."""
    ball = [(-40.0 + 0.08 * t, 0.0) for t in range(400)]
    players = [[(x + 0.5, 0.0), (x + 10.0, 5.0)] if t < 200 else [(x + 10.0, 5.0), (x + 0.5, 0.0)]
               for t, (x, _) in enumerate(ball)]
    return synthetic_table(ball, players, teams=[0, 1])


def _past_the_line():
    """Ball rolls over the goal line, from 52 m to 54 m out of 52.5 m, players far away."""
    ball = [(52.0 + 0.04 * t, 0.0) for t in range(50)]
    return synthetic_table(ball, [[(0.0, 0.0), (10.0, 0.0)]] * 50, teams=[0, 1])


def test_default_clues_are_the_module_constants():
    default, clue_settings = _clues()
    assert default == clue_settings()
    assert (default.touch_dv, default.touch_radius) == (touch.TOUCH_DV, touch.TOUCH_RADIUS)
    assert (default.control_radius, default.control_min_frames) == (touch.CONTROL_RADIUS, touch.CONTROL_MIN_FRAMES)
    assert default.carry_frames == touch.CARRY_FRAMES and default.out_margin_m == dead_ball.OUT_MARGIN_M
    assert default.gap_fill_frames == 0


def test_default_clues_change_nothing():
    default, _ = _clues()
    for table in (_bounce(), _two_halves(), _past_the_line()):
        a, b = features(table), features(table, default)
        for name in ("team", "team_mirrored", "slow", "ball_clues"):
            np.testing.assert_array_equal(getattr(a, name), getattr(b, name))


@pytest.mark.parametrize("make, field, value", [
    (_bounce, "touch_dv", 30.0),
    (_bounce, "touch_radius", 0.2),
    (_bounce, "carry_frames", 5),
    (_two_halves, "control_radius", 0.1),
    (_two_halves, "control_min_frames", 300),
])
def test_clue_settings_reach_the_team_cells(make, field, value):
    default, _ = _clues()
    table = make()
    changed = features(table, replace(default, **{field: value}))
    assert not np.array_equal(changed.team, features(table).team)
    assert not np.array_equal(changed.team_mirrored, features(table).team_mirrored)


def test_out_margin_reaches_the_ball_clues():
    default, _ = _clues()
    table = _past_the_line()
    assert features(table, replace(default, out_margin_m=0.5)).ball_clues.sum() > features(table).ball_clues.sum()


def test_touch_and_control_take_their_settings():
    table = _bounce()
    assert (touch.touches(table) >= 0).any()
    assert (touch.touches(table, dv=30.0) == -1).all() and (touch.touches(table, radius=0.2) == -1).all()
    rolling = _two_halves()
    assert (touch.control(rolling) >= 0).any() and (touch.control(rolling, radius=0.1) == -1).all()
    assert (touch.control(rolling, min_frames=300) == -1).all()

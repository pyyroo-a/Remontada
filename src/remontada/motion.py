"""Speeds worked out from positions, the same way for every match (decided 3 Nov 2026).

Trackers report their own speeds, but each in its own way; computing them here keeps IDSSE and
BAR-ATH on one footing. A centred difference over +/- half_window frames smooths tracker wobble.
"""
import numpy as np

BALL_HALF_WINDOW = 2  # 0.16 s: short, so a kick shows up as a sharp change
PLAYER_HALF_WINDOW = 12  # about 1 s: players change speed slowly


def velocity(xy, fps, half_window):
    """(xy[t+k] - xy[t-k]) / (2k / fps); NaN at the ends, where either end is NaN, and where
    xy[t] itself is NaN (an unseen frame never gets a speed)."""
    k = half_window
    out = np.full(xy.shape, np.nan)
    if len(xy) > 2 * k:
        out[k:-k] = (xy[2 * k:] - xy[:-2 * k]) * (fps / (2 * k))
    out[np.isnan(xy)] = np.nan
    return out


def speed(xy, fps, half_window):
    """Size of the velocity, in m/s. Shape of xy without its last axis."""
    v = velocity(xy, fps, half_window)
    return np.hypot(v[..., 0], v[..., 1])


def runs_at_least(mask, min_len):
    """True on every stretch of consecutive True frames that is at least min_len long."""
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    starts, stops = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    keep = np.zeros(len(mask) + 1, dtype=np.int8)
    long_enough = stops - starts >= min_len
    np.add.at(keep, starts[long_enough], 1)
    np.add.at(keep, stops[long_enough], -1)
    return np.cumsum(keep)[:-1] > 0

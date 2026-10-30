"""The official reference baseline on any FrameTable: nearest player, then a majority vote.

Behaviour matches upstream starter/baseline.py exactly; tests reproduce its four files byte for
byte. Team codes here are 0 home, 1 away, -1 no answer; smooth() returns scorer label codes.
"""
import numpy as np

from remontada.vendor.possession_metrics import A, D, H

NO_ANSWER = -1
CHUNK = 20000


def nearest_team(table):
    """Team of the nearest tracked player per frame. -1 where the ball or every player is missing.

    The first slot wins a tie, as in upstream's strict comparison.
    """
    teams = np.full(table.n_frames, NO_ANSWER, dtype=np.int8)
    seen = np.flatnonzero(~np.isnan(table.ball_xy[:, 0]))
    for start in range(0, len(seen), CHUNK):
        frames = seen[start:start + CHUNK]
        offset = table.player_xy[frames] - table.ball_xy[frames, np.newaxis, :]
        distance = np.nan_to_num((offset ** 2).sum(axis=2), nan=np.inf)
        best = distance.argmin(axis=1)
        anyone = np.isfinite(distance.min(axis=1))
        teams[frames[anyone]] = table.player_team[best[anyone]]
    return teams


def smooth(teams, window):
    """Scorer label codes from a centred majority vote over +/- window frames. Ties go to home."""
    home_votes = np.concatenate(([0], np.cumsum(teams == 0)))
    away_votes = np.concatenate(([0], np.cumsum(teams == 1)))
    index = np.arange(len(teams))
    low = np.maximum(0, index - window)
    high = np.minimum(len(teams), index + window + 1)
    home = home_votes[high] - home_votes[low]
    away = away_votes[high] - away_votes[low]

    labels = np.full(len(teams), D, dtype=np.int8)
    seen = (home + away) > 0
    labels[seen] = np.where(home[seen] >= away[seen], H, A)
    return labels

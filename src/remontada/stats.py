"""Plain possession statistics for a label array. Nothing here is a score."""
import numpy as np

from remontada.vendor.possession_metrics import D, H, A


def _run_lengths(codes):
    """(label, length) for each maximal run of one label."""
    edges = np.flatnonzero(np.diff(codes)) + 1
    starts = np.concatenate(([0], edges))
    stops = np.concatenate((edges, [len(codes)]))
    return [(int(codes[start]), int(stop - start)) for start, stop in zip(starts, stops)]


def possession_summary(codes, runs, fps=25):
    """Counts and lengths of possessions and stoppages. Nothing crosses a run boundary."""
    possessions, stoppages = [], []
    graded = 0
    for start, stop in runs:
        graded += stop - start
        for label, length in _run_lengths(codes[start:stop]):
            if label in (H, A):
                possessions.append(length / fps)
            elif label == D:
                stoppages.append(length / fps)
    return {
        "n_possessions": len(possessions),
        "mean_s": float(np.mean(possessions)) if possessions else 0.0,
        "median_s": float(np.median(possessions)) if possessions else 0.0,
        "n_under_1s": sum(1 for length in possessions if length < 1.0),
        "n_dead": len(stoppages),
        "mean_dead_s": float(np.mean(stoppages)) if stoppages else 0.0,
        "alive_share": sum(possessions) * fps / graded if graded else 0.0,
    }

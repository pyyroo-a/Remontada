"""The whole decoder: learned evidence plus fixed-rule hints, decoded per half with Viterbi.

    codes = predict(table, load_counts(EVIDENCE_PATH), load_settings(), tags)
"""
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from remontada.evidence import (count_cells, hint_weight, scores_from_cells, slow_cells,
                                team_cells, team_cells_mirrored)
from remontada.hints import ball_hints, tag_hints
from remontada.hmm import Costs, decode_runs
from remontada.vendor.possession_metrics import D

CONFIG_DIR = Path(__file__).parent / "config"
SETTINGS_PATH = CONFIG_DIR / "hmm_v1.json"
EVIDENCE_PATH = CONFIG_DIR / "evidence_idsse.json"


@dataclass(frozen=True)
class Settings:
    temperature: float
    costs: Costs
    ball_hints: bool
    tag_hints: bool


def load_settings(path=None):
    saved = json.loads(Path(path or SETTINGS_PATH).read_text(encoding="utf-8"))
    return Settings(
        temperature=float(saved["temperature"]),
        costs=Costs(float(saved["switch"]), float(saved["to_dead"]), float(saved["from_dead"])),
        ball_hints=bool(saved["ball_hints"]),
        tag_hints=bool(saved["tag_hints"]),
    )


@dataclass(frozen=True)
class Features:
    """Everything the decoder needs from one table, worked out once and reused across settings."""
    team: np.ndarray
    team_mirrored: np.ndarray
    slow: np.ndarray
    ball_clues: np.ndarray  # how many ball-dead clues are true on each frame (0 to 3)
    graded_runs: list

    def counts(self, labels):
        return count_cells(self.team, self.team_mirrored, self.slow, labels)

    def scores(self, counts, settings):
        """Learned evidence plus ball hints (when switched on); no Smart Tagging."""
        scores = scores_from_cells(self.team, self.slow, counts, settings.temperature)
        if settings.ball_hints:
            scores[:, D] += hint_weight(counts, settings.temperature) * self.ball_clues
        return scores


def features(table):
    return Features(
        team=team_cells(table),
        team_mirrored=team_cells_mirrored(table),
        slow=slow_cells(table),
        ball_clues=ball_hints(table, 1.0)[:, D].astype(np.int8),
        graded_runs=list(table.graded_runs),
    )


def predict(table, counts, settings, tags=None):
    """Label codes for every frame: H, A or D inside the graded runs, X outside."""
    scores = features(table).scores(counts, settings)
    if settings.tag_hints and tags is not None:
        scores += tag_hints(tags, table.n_frames, hint_weight(counts, settings.temperature))
    return decode_runs(scores, table.graded_runs, settings.costs)

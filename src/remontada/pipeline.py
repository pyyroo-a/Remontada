"""The whole decoder: learned evidence plus fixed-rule hints, decoded per half with Viterbi.

    codes = predict(table, load_counts(EVIDENCE_PATH), load_settings(), tags)
"""
import json
from dataclasses import dataclass, fields, replace
from pathlib import Path

import numpy as np

from remontada.clues import DEFAULT_CLUES, ClueSettings
from remontada.evidence import (N_SLOW_CELLS, NONE, count_cells, hint_weight, scores_from_cells,
                                slow_cells, slow_logs, team_cells, team_cells_mirrored)
from remontada.frames import run_slice
from remontada.gaps import fill_ball_gaps
from remontada.hints import ball_hints, tag_hints
from remontada.hmm import Costs, decode_runs
from remontada.scorer import P_CLIP, Scorer, frame_clues
from remontada.touch import touches, within
from remontada.vendor.possession_metrics import A, D, H

CONFIG_DIR = Path(__file__).parent / "config"
SETTINGS_PATH = CONFIG_DIR / "hmm_v1.json"
EVIDENCE_PATH = CONFIG_DIR / "evidence_idsse.json"
SCORER_PATH = CONFIG_DIR / "scorer_v2.json"


@dataclass(frozen=True)
class Settings:
    temperature: float
    costs: Costs
    ball_hints: bool
    tag_hints: bool
    clues: ClueSettings = DEFAULT_CLUES
    touch_discount: float = 0.0  # Plan 3 step C: share of the switch cost taken off next to a touch
    touch_window: int = 0  # frames either side of a touch where the discount applies
    scorer: str = "tally"  # Plan 3 step D: "tally" (the 54-cell table) or "model"


def _clue_settings(saved):
    """ClueSettings from a saved {"name": value} object; missing names keep their defaults."""
    kinds = {f.name: type(f.default) for f in fields(ClueSettings)}
    unknown = sorted(set(saved) - set(kinds))
    if unknown:
        raise ValueError(f"unknown clue settings: {unknown}")
    return ClueSettings(**{name: kinds[name](value) for name, value in saved.items()})


def settings_to_dict(settings):
    """The saved form of Settings (the layout of hmm_v1.json, plus the Plan 3 keys)."""
    return {
        "temperature": settings.temperature,
        "switch": settings.costs.switch,
        "to_dead": settings.costs.to_dead,
        "from_dead": settings.costs.from_dead,
        "ball_hints": settings.ball_hints,
        "tag_hints": settings.tag_hints,
        "clues": {f.name: getattr(settings.clues, f.name) for f in fields(ClueSettings)},
        "touch_discount": settings.touch_discount,
        "touch_window": settings.touch_window,
        "scorer": settings.scorer,
    }


def load_settings(path=None):
    """Settings from a JSON file; the Plan 3 keys are optional, so hmm_v1.json loads unchanged."""
    return settings_from_dict(json.loads(Path(path or SETTINGS_PATH).read_text(encoding="utf-8")))


def load_scorer(path=None):
    """The learned scorer for Settings.scorer == "model" (Plan 3 step D)."""
    return Scorer.from_json(Path(path or SCORER_PATH).read_text(encoding="utf-8"))


def settings_from_dict(saved):
    return Settings(
        temperature=float(saved["temperature"]),
        costs=Costs(float(saved["switch"]), float(saved["to_dead"]), float(saved["from_dead"])),
        ball_hints=bool(saved["ball_hints"]),
        tag_hints=bool(saved["tag_hints"]),
        clues=_clue_settings(saved.get("clues", {})),
        touch_discount=float(saved.get("touch_discount", 0.0)),
        touch_window=int(saved.get("touch_window", 0)),
        scorer=str(saved.get("scorer", "tally")),
    )


@dataclass(frozen=True, repr=False)
class Features:
    """Everything the decoder needs from one table, worked out once and reused across settings."""
    team: np.ndarray
    team_mirrored: np.ndarray
    slow: np.ndarray
    ball_clues: np.ndarray  # how many ball-dead clues are true on each frame (0 to 3)
    touches: np.ndarray  # team code of each touch (0 home, 1 away), -1 elsewhere
    graded_runs: list
    p_home: np.ndarray = None  # the learned scorer's P(home has the ball), when one was given
    model_hint: float = 0.0  # that scorer's typical pull (its hint weight before temperature)

    def __repr__(self):
        # Never the arrays: on BAR-ATH they are frame-by-frame extracts of the match.
        return "Features(frames={})".format(len(self.team))

    def counts(self, labels):
        return count_cells(self.team, self.team_mirrored, self.slow, labels)

    def scores(self, counts, settings):
        """Learned evidence plus ball hints (when switched on); no Smart Tagging.

        With settings.scorer == "model" the team part comes from the learned scorer:
        H = log(2p), A = log(2(1 - p)) and 0 for D (the average of the two likelihoods is 1);
        players' speed still comes from the tally, all times the temperature.
        """
        if settings.scorer == "model":
            if self.p_home is None:
                raise ValueError("these features were worked out without a scorer")
            slow = slow_logs(self.slow, counts)
            scores = np.empty((len(self.slow), 3))
            scores[:, H] = np.log(2 * self.p_home) + slow[:, 0]
            scores[:, A] = np.log(2 * (1 - self.p_home)) + slow[:, 0]
            scores[:, D] = slow[:, 1]
            scores *= settings.temperature
        else:
            scores = scores_from_cells(self.team, self.slow, counts, settings.temperature)
        if settings.ball_hints:
            scores[:, D] += self.hint(counts, settings) * self.ball_clues
        return scores

    def hint(self, counts, settings):
        """The hint weight for these settings: the written rule for the tally or the scorer."""
        if settings.scorer == "model":
            return settings.temperature * self.model_hint
        return hint_weight(counts, settings.temperature)

    def entry_bonus(self, settings):
        """The touch bonus for the decoder, or None when step C is off."""
        if settings.touch_discount <= 0:
            return None
        return touch_entry_bonus(self.touches, settings.touch_discount * settings.costs.switch,
                                 settings.touch_window)


def touch_entry_bonus(touch_codes, discount, window):
    """(n, 3): discount on the H column within window frames of a home touch, on the A column
    near an away touch (once, however many touches are near), and never on D (Plan 3 step C)."""
    bonus = np.zeros((len(touch_codes), 3))
    for team, column in ((0, H), (1, A)):
        bonus[within(touch_codes == team, window), column] = discount
    return bonus


NOTHING_KNOWN_TEAM_CELL = NONE * 3 + NONE  # nearness unseen (bin 0), no touch, no control


def features(table, clues=DEFAULT_CLUES, scorer=None):
    """Clues worked out on each graded run on its own, so nothing crosses the half-time gap.

    Frames outside every run get the "nothing known" cells: unseen nearness, no touch, no
    control, unknown speed and no ball clues. With clues.gap_fill_frames > 0, short ball gaps in
    each run are filled by a straight line first (gaps.py); filled frames are never touches.
    """
    n = table.n_frames
    team = np.full(n, NOTHING_KNOWN_TEAM_CELL, dtype=np.int16)
    team_mirrored = team.copy()
    slow = np.full(n, N_SLOW_CELLS - 1, dtype=np.int16)
    ball_clues = np.zeros(n, dtype=np.int8)
    touch_codes = np.full(n, -1, dtype=np.int8)
    for start, stop in table.graded_runs:
        if stop <= start:
            continue
        part = run_slice(table, start, stop)
        filled = None
        if clues.gap_fill_frames > 0:
            ball_xy, filled = fill_ball_gaps(part.ball_xy, part.graded_runs, clues.gap_fill_frames)
            part = replace(part, ball_xy=ball_xy)
        team[start:stop] = team_cells(part, clues, filled)
        team_mirrored[start:stop] = team_cells_mirrored(part, clues, filled)
        slow[start:stop] = slow_cells(part)
        touch_codes[start:stop] = touches(part, clues.touch_dv, clues.touch_radius, filled)
        ball_clues[start:stop] = ball_hints(part, 1.0, clues.out_margin_m)[:, D]
    if scorer is None:
        return Features(team, team_mirrored, slow, ball_clues, touch_codes, list(table.graded_runs))
    p_home = np.clip(scorer.p_home(frame_clues(table, clues)), P_CLIP, 1 - P_CLIP)
    return Features(team, team_mirrored, slow, ball_clues, touch_codes, list(table.graded_runs),
                    p_home, scorer.hint_weight)


def predict(table, counts, settings, tags=None, scorer=None):
    """Label codes for every frame: H, A or D inside the graded runs, X outside.

    With settings.scorer == "model", scorer (default: the file at SCORER_PATH) gives the H versus
    A part."""
    if settings.scorer == "model":
        scorer = scorer if scorer is not None else load_scorer()
    else:
        scorer = None
    f = features(table, settings.clues, scorer)
    scores = f.scores(counts, settings)
    if settings.tag_hints and tags is not None:
        scores += tag_hints(tags, table.n_frames, f.hint(counts, settings))
    return decode_runs(scores, table.graded_runs, settings.costs, f.entry_bonus(settings))

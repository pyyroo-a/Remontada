"""Leave one match out on IDSSE: the evidence table that scores a match never saw that match.

Features for every match and level are worked out once; then each setting costs one batched
Viterbi call over every half of every match at every level.
"""
import numpy as np

from remontada.degrade import LEVELS, degrade
from remontada.evidence import Counts
from remontada.hmm import decode_runs
from remontada.load_idsse import MATCH_IDS, load_idsse_cached
from remontada.pipeline import features

SEED = 0


def idsse_features(seed=SEED):
    """{(match, level): Features} and {match: labels} for all 7 matches at all 4 levels."""
    feats, labels = {}, {}
    for match_id in MATCH_IDS:
        table, labels[match_id] = load_idsse_cached(match_id)
        for name, level in LEVELS.items():
            feats[match_id, name] = features(degrade(table, level, seed))
    return feats, labels


def counts_by_item(feats, labels):
    return {(m, name): f.counts(labels[m]) for (m, name), f in feats.items()}


def leave_one_out_counts(by_item):
    """{match: the summed counts of every other match at every level}."""
    total = None
    for c in by_item.values():
        total = c if total is None else total + c
    out = {}
    for match_id in {m for m, _ in by_item}:
        own = [c for (m, _), c in by_item.items() if m == match_id]
        mine = own[0]
        for c in own[1:]:
            mine = mine + c
        out[match_id] = Counts(total.team - mine.team, total.slow - mine.slow)
    return out


def decode_all(feats, loo_counts, settings):
    """{(match, level): codes}, every item decoded with its own match's left-out counts."""
    keys = list(feats)
    scores, bonus, runs, offsets, offset = [], [], [], {}, 0
    for key in keys:
        f = feats[key]
        scores.append(f.scores(loo_counts[key[0]], settings))
        bonus.append(f.entry_bonus(settings))
        runs += [(offset + a, offset + b) for a, b in f.graded_runs]
        offsets[key] = (offset, offset + len(f.team))
        offset += len(f.team)
    entry_bonus = None if bonus[0] is None else np.concatenate(bonus)
    codes = decode_runs(np.concatenate(scores), runs, settings.costs, entry_bonus)
    return {key: codes[a:b] for key, (a, b) in offsets.items()}

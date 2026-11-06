import time

import numpy as np

from remontada.hmm import Costs, decode_runs, viterbi
from remontada.vendor.possession_metrics import A, D, H, X


def test_clear_evidence_gives_two_blocks():
    s = np.zeros((100, 3)); s[:50, H] = 1; s[50:, A] = 1
    assert viterbi(s, Costs(5, 5, 5)).tolist() == [H] * 50 + [A] * 50


def test_switch_cost_removes_a_blip():
    s = np.zeros((60, 3)); s[:, H] = 1; s[30, A] = 3  # one frame favours A by 2
    assert (viterbi(s, Costs(10, 10, 10)) == H).all()
    assert viterbi(s, Costs(0.5, 10, 10))[30] == A


def test_dead_stretch():
    s = np.zeros((90, 3)); s[:30, H] = 1; s[30:60, D] = 1; s[60:, A] = 1
    assert viterbi(s, Costs(5, 5, 5)).tolist() == [H] * 30 + [D] * 30 + [A] * 30


def test_ties_go_to_the_lower_code():
    assert (viterbi(np.zeros((10, 3)), Costs(1, 1, 1)) == H).all()


def test_batch_matches_single():
    rng = np.random.default_rng(0)
    s = rng.normal(size=(2, 500, 3))
    costs = Costs(2, 3, 1.5)
    batched = viterbi(s, costs)
    assert batched.shape == (2, 500) and batched.dtype == np.int8
    for i in range(2):
        np.testing.assert_array_equal(batched[i], viterbi(s[i], costs))


def test_decode_runs_marks_outside_x_and_never_chains():
    s = np.zeros((11, 3)); s[:5, H] = 1; s[6:, A] = 1
    out = decode_runs(s, [(0, 5), (6, 11)], Costs(100, 100, 100))
    assert out.tolist() == [H] * 5 + [X] + [A] * 5


def test_decode_runs_with_no_runs_is_all_x():
    assert (decode_runs(np.zeros((4, 3)), [], Costs(1, 1, 1)) == X).all()


def test_speed():
    s = np.random.default_rng(1).normal(size=(2, 150_000, 3))
    start = time.perf_counter()
    viterbi(s, Costs(8, 8, 8))
    assert time.perf_counter() - start < 10


def _path_value(s, path, costs, bonus):
    from remontada.hmm import _cost_matrix
    cost = _cost_matrix(costs)
    value = s[0, path[0]]
    for t in range(1, len(path)):
        j, k = path[t - 1], path[t]
        value += s[t, k] - (0.0 if j == k else max(cost[j, k] - bonus[t, k], 0.0))
    return value


def test_entry_bonus_matches_brute_force():
    import itertools
    rng = np.random.default_rng(7)
    mismatches = 0
    for _ in range(300):
        n = int(rng.integers(1, 7))
        s = rng.normal(size=(n, 3))
        bonus = rng.uniform(0, 4, size=(n, 3)) * (rng.uniform(size=(n, 3)) < 0.5)
        costs = Costs(*rng.uniform(0.1, 3, size=3))
        best = max(_path_value(s, p, costs, bonus) for p in itertools.product(range(3), repeat=n))
        got = _path_value(s, viterbi(s, costs, entry_bonus=bonus), costs, bonus)
        mismatches += abs(best - got) > 1e-9
    assert mismatches == 0


def test_no_bonus_is_the_old_decoder():
    s = np.random.default_rng(3).normal(size=(2, 400, 3))
    costs = Costs(2, 3, 1.5)
    np.testing.assert_array_equal(viterbi(s, costs, entry_bonus=np.zeros_like(s)), viterbi(s, costs))


def test_a_bonus_bigger_than_the_cost_never_makes_switching_free_to_flicker():
    s = np.zeros((40, 3)); s[:, H] = 1
    bonus = np.zeros((40, 3)); bonus[:, A] = 100.0
    assert (viterbi(s, Costs(5, 5, 5), entry_bonus=bonus) == H).all()


def test_switch_lands_on_the_touch():
    # H evidence fades over frames 10..30 and A grows (tied at 20, A ahead from 21); a touch by A at 25
    s = np.zeros((40, 3))
    fade = np.clip((np.arange(40) - 10) / 20, 0, 1)
    s[:, H], s[:, A] = 1 - fade, fade
    costs = Costs(5, 50, 50)
    assert int(np.argmax(viterbi(s, costs) == A)) == 21
    bonus = np.zeros((40, 3)); bonus[25, A] = 5.0
    assert int(np.argmax(viterbi(s, costs, entry_bonus=bonus) == A)) == 25


def test_decode_runs_passes_the_bonus_per_run():
    s = np.zeros((81, 3))
    fade = np.clip((np.arange(40) - 10) / 20, 0, 1)
    for a in (0, 41):
        s[a:a + 40, H], s[a:a + 40, A] = 1 - fade, fade
    bonus = np.zeros((81, 3)); bonus[25, A] = bonus[41 + 22, A] = 5.0
    out = decode_runs(s, [(0, 40), (41, 81)], Costs(5, 50, 50), entry_bonus=bonus)
    assert int(np.argmax(out[:40] == A)) == 25 and int(np.argmax(out[41:] == A)) == 22
    assert out[40] == X


def test_decode_runs_rounds_scores_to_6_decimals():
    # A is ahead by far less than a millionth: after rounding it is a tie, which goes to H
    s = np.array([[0.5, 0.5 + 1e-9, 0.0]])
    assert viterbi(s, Costs(1, 1, 1))[0] == A
    assert decode_runs(s, [(0, 1)], Costs(1, 1, 1))[0] == H

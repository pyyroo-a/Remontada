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

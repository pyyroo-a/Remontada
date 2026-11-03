import numpy as np

from remontada.motion import runs_at_least, speed


def test_speed_of_steady_run():
    xy = np.column_stack([np.arange(50) * 0.2, np.zeros(50)])  # 0.2 m per frame is 5 m/s
    s = speed(xy, 25, 2)
    assert np.isnan(s[:2]).all() and np.isnan(s[-2:]).all()
    np.testing.assert_allclose(s[2:-2], 5.0)


def test_speed_is_nan_next_to_a_gap():
    xy = np.column_stack([np.arange(20.0), np.zeros(20)])
    xy[10] = np.nan
    s = speed(xy, 25, 2)
    assert np.isnan(s[[8, 10, 12]]).all()  # both ends seen is not enough: frame 10 itself is unseen
    assert not np.isnan(s[[5, 9, 11, 15]]).any()


def test_speed_works_per_slot():
    xy = np.zeros((30, 3, 2))
    xy[:, 1, 0] = np.arange(30) * 0.1
    s = speed(xy, 25, 12)
    assert s.shape == (30, 3)
    np.testing.assert_allclose(s[12:-12, 1], 2.5)
    np.testing.assert_allclose(s[12:-12, 0], 0.0)


def test_runs_at_least():
    assert runs_at_least(np.array([1, 1, 0, 1, 1, 1], bool), 3).tolist() == [False, False, False, True, True, True]
    assert runs_at_least(np.array([1, 1, 1], bool), 3).tolist() == [True, True, True]
    assert runs_at_least(np.zeros(0, bool), 3).tolist() == []

import numpy as np
import pytest
from scipy.optimize import linear_sum_assignment

from or_algorithms import hungarian, reduce_matrix


def test_course_example():
    cost = [[150, 200, 125], [280, 150, 180], [300, 220, 340]]
    rows, cols, total = hungarian(cost)
    r, c = linear_sum_assignment(cost)
    assert total == pytest.approx(np.asarray(cost)[r, c].sum())
    assert sorted(cols.tolist()) == [0, 1, 2]


def test_maximize():
    profit = [[9, 2, 7], [6, 4, 3], [5, 8, 1]]
    _, _, total = hungarian(profit, maximize=True)
    r, c = linear_sum_assignment(profit, maximize=True)
    assert total == pytest.approx(np.asarray(profit)[r, c].sum())


@pytest.mark.parametrize('seed', range(200))
def test_random_matrices_match_scipy(seed):
    rng = np.random.default_rng(seed)
    n, m = rng.integers(1, 9, size=2)
    cost = rng.integers(-50, 100, (n, m)) if seed % 2 else rng.uniform(0, 1000, (n, m))
    rows, cols, total = hungarian(cost)
    r, c = linear_sum_assignment(cost)
    assert total == pytest.approx(cost[r, c].sum())
    assert len(rows) == min(n, m)
    assert len(set(rows.tolist())) == len(rows) and len(set(cols.tolist())) == len(cols)


def test_reduce_matrix_has_a_zero_in_every_row_and_column():
    _, reduced = reduce_matrix([[150, 200, 125], [280, 150, 180], [300, 220, 340]])
    assert np.all(reduced.min(axis=1) == 0) and np.all(reduced.min(axis=0) == 0)


def test_rejects_bad_input():
    with pytest.raises(ValueError):
        hungarian([[1, np.inf]])

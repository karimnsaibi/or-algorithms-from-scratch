import numpy as np
import pytest
from scipy.optimize import linprog

from or_algorithms import solve_lp


def scipy_solve(c, A, b, senses, maximize):
    """Same problem through SciPy/HiGHS; returns (status, objective, shadow prices)."""
    c = np.asarray(c, float)
    A_ub, b_ub, A_eq, b_eq, kind = [], [], [], [], []
    for row, rhs, s in zip(A, b, senses):
        if s == '<=':
            A_ub.append(row); b_ub.append(rhs); kind.append(('ub', len(A_ub) - 1, 1))
        elif s == '>=':
            A_ub.append([-a for a in row]); b_ub.append(-rhs); kind.append(('ub', len(A_ub) - 1, -1))
        else:
            A_eq.append(row); b_eq.append(rhs); kind.append(('eq', len(A_eq) - 1, 1))
    res = linprog(-c if maximize else c,
                  A_ub=A_ub or None, b_ub=b_ub or None, A_eq=A_eq or None, b_eq=b_eq or None,
                  bounds=[(0, None)] * len(c), method='highs')
    status = {0: 'optimal', 2: 'infeasible', 3: 'unbounded'}.get(res.status, 'other')
    if status != 'optimal':
        return status, None, None
    sign = -1 if maximize else 1
    shadow = []
    for k, idx, flip in kind:
        marg = res.ineqlin.marginals[idx] if k == 'ub' else res.eqlin.marginals[idx]
        shadow.append(sign * flip * marg)
    return status, sign * res.fun, np.array(shadow)


def test_zenith_production_mix():
    r = solve_lp([70, 50], [[4, 3], [2, 1], [1, 1]], [240, 100, 60])
    assert r.status == 'optimal'
    np.testing.assert_allclose(r.x, [40, 20])
    assert r.objective == pytest.approx(3800)
    np.testing.assert_allclose(r.shadow_prices, [0, 20, 30], atol=1e-9)
    np.testing.assert_allclose(r.slacks, [20, 0, 0], atol=1e-9)


def test_minimization_with_ge_constraint():
    r = solve_lp([2, 3], [[1, 1], [1, 0]], [4, 1], ['>=', '<='], maximize=False)
    assert r.objective == pytest.approx(11)
    np.testing.assert_allclose(r.x, [1, 3])
    np.testing.assert_allclose(r.shadow_prices, [3, -1], atol=1e-9)


def test_equality_and_negative_rhs():
    # x1 - x2 = -2  (negative right-hand side), x1 + x2 <= 6, maximize x1 + 2 x2
    r = solve_lp([1, 2], [[1, -1], [1, 1]], [-2, 6], ['=', '<='])
    assert r.status == 'optimal'
    np.testing.assert_allclose(r.x, [2, 4])
    assert r.objective == pytest.approx(10)


def test_unbounded():
    assert solve_lp([1, 1], [[1, -1]], [1]).status == 'unbounded'


def test_infeasible():
    assert solve_lp([1], [[1], [1]], [1, 2], ['<=', '>=']).status == 'infeasible'


def test_redundant_equality_rows():
    # The second equality is twice the first: Phase I must drop it, not fail.
    r = solve_lp([1, 1], [[1, 1], [2, 2], [1, 0]], [4, 8, 3], ['=', '=', '<='], maximize=False)
    assert r.status == 'optimal'
    assert r.objective == pytest.approx(4)


def test_degenerate_problem_does_not_cycle():
    # Beale's classic cycling example (cycles under the pure Dantzig rule without anti-cycling).
    c = [0.75, -150, 0.02, -6]
    A = [[0.25, -60, -0.04, 9], [0.5, -90, -0.02, 3], [0, 0, 1, 0]]
    r = solve_lp(c, A, [0, 0, 1])
    assert r.status == 'optimal'
    assert r.objective == pytest.approx(0.05)


@pytest.mark.parametrize('seed', range(150))
def test_random_problems_match_scipy(seed):
    rng = np.random.default_rng(seed)
    m, n = rng.integers(2, 7), rng.integers(2, 7)
    A = rng.uniform(-3, 6, (m, n)).round(2)
    x0 = rng.uniform(0, 5, n)
    senses = list(rng.choice(['<=', '>=', '='], size=m, p=[0.6, 0.25, 0.15]))
    activity = A @ x0
    b = np.where(np.array(senses) == '<=', activity + rng.uniform(0, 5, m),
                 np.where(np.array(senses) == '>=', activity - rng.uniform(0, 5, m), activity)).round(3)
    if seed % 3 == 0:                      # some deliberately infeasible-or-unbounded cases
        b = rng.uniform(-10, 10, m).round(2)
    c = rng.uniform(-5, 10, n).round(2)
    maximize = bool(seed % 2)
    A, b = A.tolist(), b.tolist()

    mine = solve_lp(c, A, b, senses, maximize)
    status, obj, shadow = scipy_solve(c, A, b, senses, maximize)
    assert mine.status == status
    if status == 'optimal':
        assert mine.objective == pytest.approx(obj, rel=1e-6, abs=1e-6)
        assert np.all(mine.x >= -1e-9)
        for row, rhs, s, val in zip(A, b, senses, np.asarray(A) @ mine.x):
            assert {'<=': val <= rhs + 1e-6, '>=': val >= rhs - 1e-6, '=': abs(val - rhs) < 1e-6}[s]
        check_duals(mine, c, A, b, senses, maximize)


def check_duals(res, c, A, b, senses, maximize):
    """Shadow prices must form an optimal dual solution (they need not be unique).

    Strong duality: b @ y equals the optimal objective. Dual feasibility: with y_i = dZ/db_i,
    a max problem needs A^T y >= c with y >= 0 on '<=' rows and y <= 0 on '>=' rows;
    a min problem needs A^T y <= c with the opposite signs.
    """
    y, A, c = res.shadow_prices, np.asarray(A), np.asarray(c)
    tol = 1e-6 * (1 + np.abs(A).sum())
    assert np.asarray(b) @ y == pytest.approx(res.objective, rel=1e-6, abs=1e-6)
    s = 1 if maximize else -1
    assert np.all(s * (A.T @ y - c) >= -tol)
    for yi, sense in zip(y, senses):
        if sense == '<=':
            assert s * yi >= -tol
        elif sense == '>=':
            assert s * yi <= tol


def test_shadow_prices_match_scipy_on_nondegenerate_case():
    c, A, b, senses = [3, 5, 4], [[2, 3, 1], [1, 1, 2], [3, 2, 4]], [12, 8, 20], ['<='] * 3
    mine = solve_lp(c, A, b, senses)
    _, obj, shadow = scipy_solve(c, A, b, senses, True)
    assert mine.objective == pytest.approx(obj)
    np.testing.assert_allclose(mine.shadow_prices, shadow, atol=1e-8)

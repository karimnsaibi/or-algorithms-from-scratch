"""Two-phase simplex method (tableau form), written from scratch with NumPy.

Solves
    max / min   c @ x
    subject to  A[i] @ x  (<= | >= | =)  b[i]     for every row i
                x >= 0

Phase I finds a feasible basis by minimizing the sum of artificial variables;
Phase II optimizes the real objective from that basis. Entering variables are
chosen with Dantzig's rule (most negative reduced cost), switching to Bland's
rule after a degenerate pivot so the method cannot cycle.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

EPS = 1e-9


@dataclass
class LPResult:
    status: str                      # 'optimal' | 'infeasible' | 'unbounded'
    x: np.ndarray | None = None      # values of the decision variables
    objective: float | None = None
    shadow_prices: np.ndarray | None = None   # d(objective) / d(b_i), one per constraint
    slacks: np.ndarray | None = None          # b_i - A_i @ x (0 = binding constraint)
    reduced_costs: np.ndarray | None = None   # how much c_j must improve before x_j enters
    iterations: int = 0
    history: list = field(default_factory=list)   # tableau snapshots when trace=True

    def summary(self, var_names=None, con_names=None) -> str:
        if self.status != 'optimal':
            return f'Status: {self.status}'
        var_names = var_names or [f'x{j + 1}' for j in range(len(self.x))]
        con_names = con_names or [f'c{i + 1}' for i in range(len(self.shadow_prices))]
        lines = [f'Status: optimal after {self.iterations} pivots',
                 f'Objective: {self.objective:,.4f}', 'Decision variables:']
        lines += [f'  {n:<20} {v:>12,.4f}' for n, v in zip(var_names, self.x)]
        lines.append('Constraints (slack, shadow price):')
        lines += [f'  {n:<20} {s:>12,.4f} {p:>12,.4f}'
                  for n, s, p in zip(con_names, self.slacks, self.shadow_prices)]
        return '\n'.join(lines)


class SimplexSolver:
    """Tableau simplex.

    Example (Zenith Furniture production mix)::

        solver = SimplexSolver(c=[70, 50],
                               A=[[4, 3], [2, 1], [1, 1]],
                               b=[240, 100, 60],
                               senses=['<=', '<=', '<='],
                               maximize=True)
        result = solver.solve()
    """

    def __init__(self, c, A, b, senses=None, maximize=True, trace=False):
        self.c = np.asarray(c, dtype=float)
        self.A = np.atleast_2d(np.asarray(A, dtype=float))
        self.b = np.asarray(b, dtype=float)
        m, n = self.A.shape
        if self.c.shape != (n,) or self.b.shape != (m,):
            raise ValueError(f'Shapes do not match: c {self.c.shape}, A {self.A.shape}, b {self.b.shape}')
        self.senses = list(senses) if senses is not None else ['<='] * m
        if len(self.senses) != m or any(s not in ('<=', '>=', '=') for s in self.senses):
            raise ValueError("senses must contain one of '<=', '>=', '=' per constraint")
        self.maximize = maximize
        self.trace = trace
        self.m, self.n = m, n

    # ------------------------------------------------------------------ setup
    def _standard_form(self):
        """Build [A | slack/surplus | artificial] with b >= 0 and an initial basis."""
        A, b, senses = self.A.copy(), self.b.copy(), list(self.senses)
        self.row_sign = np.ones(self.m)
        for i in range(self.m):
            if b[i] < 0:                       # make the right-hand side non-negative
                A[i], b[i] = -A[i], -b[i]
                self.row_sign[i] = -1
                senses[i] = {'<=': '>=', '>=': '<=', '=': '='}[senses[i]]

        n_slack = sum(s != '=' for s in senses)
        n_art = sum(s != '<=' for s in senses)
        N = self.n + n_slack + n_art
        self.n_real = self.n + n_slack          # columns that survive into Phase II
        S = np.zeros((self.m, N))
        S[:, :self.n] = A
        basis = []
        k_slack, k_art = self.n, self.n_real
        self.slack_col = {}
        for i, s in enumerate(senses):
            if s == '<=':
                S[i, k_slack] = 1.0
                self.slack_col[i] = k_slack
                basis.append(k_slack)
                k_slack += 1
            elif s == '>=':
                S[i, k_slack] = -1.0
                self.slack_col[i] = k_slack
                k_slack += 1
                S[i, k_art] = 1.0
                basis.append(k_art)
                k_art += 1
            else:
                S[i, k_art] = 1.0
                basis.append(k_art)
                k_art += 1
        return S, b, basis

    # ----------------------------------------------------------------- pivots
    def _optimize(self, T, basis, cost, allowed):
        """Maximize cost @ x over the tableau T = [B^-1 A | B^-1 b]. Returns status, pivots."""
        pivots, bland = 0, False
        while True:
            cb = cost[basis]
            reduced = cb @ T[:, :-1] - cost           # z_j - c_j
            if self.trace:
                self._snapshot(T, basis, reduced, cb @ T[:, -1])
            candidates = [j for j in allowed if reduced[j] < -EPS]
            if not candidates:
                return 'optimal', pivots
            col = min(candidates) if bland else min(candidates, key=lambda j: reduced[j])

            ratios = [(T[i, -1] / T[i, col], basis[i], i)
                      for i in range(T.shape[0]) if T[i, col] > EPS]
            if not ratios:
                return 'unbounded', pivots
            step, _, row = min(ratios)                # ties broken by smallest basic index
            if step < EPS:
                bland = True                           # degenerate pivot: avoid cycling

            T[row] /= T[row, col]
            for i in range(T.shape[0]):
                if i != row and abs(T[i, col]) > EPS:
                    T[i] -= T[i, col] * T[row]
            basis[row] = col
            pivots += 1

    def _snapshot(self, T, basis, reduced, value):
        self.history.append({'basis': list(basis), 'tableau': T.copy(),
                             'reduced_costs': reduced.copy(), 'value': value})

    # ------------------------------------------------------------------ solve
    def solve(self) -> LPResult:
        self.history = []
        S, b, basis = self._standard_form()
        T = np.hstack([S, b[:, None]])
        N = S.shape[1]
        rows = list(range(self.m))                  # original row index of each tableau row
        pivots = 0

        # Phase I: maximize -(sum of artificials)
        if N > self.n_real:
            cost1 = np.zeros(N)
            cost1[self.n_real:] = -1.0
            _, p = self._optimize(T, basis, cost1, allowed=range(N))
            pivots += p
            if cost1[basis] @ T[:, -1] < -1e-7:
                return LPResult('infeasible', iterations=pivots, history=self.history)
            # Drive artificial variables still in the basis (at zero) out of it.
            for r in range(len(basis) - 1, -1, -1):
                if basis[r] >= self.n_real:
                    col = next((j for j in range(self.n_real) if abs(T[r, j]) > 1e-7), None)
                    if col is None:                 # redundant constraint: drop the row
                        T = np.delete(T, r, axis=0)
                        del basis[r], rows[r]
                        continue
                    T[r] /= T[r, col]
                    for i in range(T.shape[0]):
                        if i != r:
                            T[i] -= T[i, col] * T[r]
                    basis[r] = col
            T = np.hstack([T[:, :self.n_real], T[:, -1:]])

        # Phase II: the real objective (always maximized internally)
        sign = 1.0 if self.maximize else -1.0
        cost2 = np.zeros(self.n_real)
        cost2[:self.n] = sign * self.c
        status, p = self._optimize(T, basis, cost2, allowed=range(self.n_real))
        pivots += p
        if status == 'unbounded':
            return LPResult('unbounded', iterations=pivots, history=self.history)

        values = np.zeros(self.n_real)
        values[basis] = T[:, -1]
        x = values[:self.n]
        x[np.abs(x) < EPS] = 0.0

        # Dual values y = c_B B^-1 from the original standard-form columns
        B = S[np.ix_(rows, basis)]
        y_internal = np.linalg.solve(B.T, cost2[basis])
        shadow = np.zeros(self.m)
        for k, i in enumerate(rows):
            shadow[i] = sign * self.row_sign[i] * y_internal[k]
        reduced = cost2[basis] @ T[:, :-1] - cost2
        return LPResult(
            status='optimal',
            x=x,
            objective=float(self.c @ x),
            shadow_prices=shadow,
            slacks=self.b - self.A @ x,
            reduced_costs=reduced[:self.n],
            iterations=pivots,
            history=self.history,
        )


def solve_lp(c, A, b, senses=None, maximize=True, trace=False) -> LPResult:
    """Shortcut for SimplexSolver(...).solve()."""
    return SimplexSolver(c, A, b, senses, maximize, trace).solve()


def format_tableau(snapshot, var_names) -> str:
    """Pretty-print one tableau snapshot from LPResult.history."""
    header = 'Basis    ' + ''.join(f'{v:>10}' for v in var_names) + f'{"RHS":>12}'
    lines = [header, '-' * len(header)]
    T = snapshot['tableau']
    for i, bv in enumerate(snapshot['basis']):
        lines.append(f'{var_names[bv]:<9}' + ''.join(f'{v:>10.3f}' for v in T[i, :len(var_names)])
                     + f'{T[i, -1]:>12.3f}')
    lines.append(f'{"z_j-c_j":<9}' + ''.join(f'{v:>10.3f}' for v in snapshot['reduced_costs'][:len(var_names)])
                 + f'{snapshot["value"]:>12.3f}')
    return '\n'.join(lines)

"""Hungarian method for the assignment problem, written from scratch.

Assigns each row (e.g. worker) to a distinct column (e.g. task) at minimum total
cost. Works for square and rectangular matrices: with more columns than rows,
some columns stay unassigned (and vice versa).

Two entry points:
- `hungarian(cost)`: the O(n^2 m) potentials version of the method, used for real problems.
- `reduce_matrix(cost)`: the classroom steps (row then column reduction) for illustration.
"""
from __future__ import annotations

import numpy as np


def hungarian(cost, maximize: bool = False):
    """Return (rows, cols, total) of an optimal assignment.

    rows[k] is assigned to cols[k]; total is the sum of cost[rows, cols].
    """
    C = np.asarray(cost, dtype=float)
    if C.ndim != 2 or C.size == 0:
        raise ValueError('cost must be a non-empty 2-D matrix')
    if not np.all(np.isfinite(C)):
        raise ValueError('cost must contain only finite values')
    work = -C if maximize else C
    transposed = work.shape[0] > work.shape[1]
    if transposed:                      # the algorithm needs rows <= columns
        work = work.T
    n, m = work.shape

    # Potentials u (rows) and v (columns); p[j] = row assigned to column j (1-based, 0 = none).
    INF = float('inf')
    u = np.zeros(n + 1)
    v = np.zeros(m + 1)
    p = np.zeros(m + 1, dtype=int)
    way = np.zeros(m + 1, dtype=int)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(m + 1, INF)
        used = np.zeros(m + 1, dtype=bool)
        while True:                      # grow an alternating tree until a free column is reached
            used[j0] = True
            i0 = p[j0]
            delta, j1 = INF, 0
            for j in range(1, m + 1):
                if not used[j]:
                    cur = work[i0 - 1, j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j], way[j] = cur, j0
                    if minv[j] < delta:
                        delta, j1 = minv[j], j
            for j in range(m + 1):       # update potentials (the "subtract the minimum" step)
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:                      # flip the augmenting path
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    pairs = sorted((p[j] - 1, j - 1) for j in range(1, m + 1) if p[j] != 0)
    rows = np.array([r for r, _ in pairs], dtype=int)
    cols = np.array([c for _, c in pairs], dtype=int)
    if transposed:
        rows, cols = cols, rows
        order = np.argsort(rows)
        rows, cols = rows[order], cols[order]
    return rows, cols, float(C[rows, cols].sum())


def reduce_matrix(cost):
    """Classroom steps 1-2: subtract each row's minimum, then each column's minimum."""
    M = np.asarray(cost, dtype=float)
    after_rows = M - M.min(axis=1, keepdims=True)
    after_cols = after_rows - after_rows.min(axis=0, keepdims=True)
    return after_rows, after_cols

"""A tiny simplex, written out so the solver stops being magic.

The method walks from corner to corner of the feasible polygon, improving the
objective at every step, until no neighbouring corner is better -- that corner
is the optimum (notebook 01's geometry, made mechanical). This implementation is
deliberately small: standard form only,

    maximize  c . x   subject to   A x <= b,  x >= 0,   with every b_i >= 0,

which is exactly the shape of the product-mix problem and enough to see the
engine work. It is validated against the libraries in ``tests/test_simplex.py``;
it is not meant to compete with CBC or HiGHS.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

OPTIMAL = "optimal"
UNBOUNDED = "unbounded"
ITERATION_LIMIT = "iteration_limit"


@dataclass
class SimplexResult:
    """Outcome of a from-scratch simplex solve."""

    status: str
    objective: float
    x: list[float]
    iterations: int
    vertices: list[list[float]] = field(default_factory=list)


def simplex_max(
    c: np.ndarray | list[float],
    a: np.ndarray | list[list[float]],
    b: np.ndarray | list[float],
    max_iter: int = 100,
    tol: float = 1e-9,
) -> SimplexResult:
    """Maximize ``c . x`` s.t. ``A x <= b``, ``x >= 0`` (requires ``b >= 0``)."""
    c = np.asarray(c, dtype=float)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    m, n = a.shape
    if b.shape[0] != m or c.shape[0] != n:
        raise ValueError("shape mismatch between c, A and b")
    if np.any(b < -tol):
        raise ValueError("this teaching simplex needs every b_i >= 0 (all-slack start)")

    # Tableau: m constraint rows + 1 objective row;
    # columns = n structural, m slack, 1 right-hand side.
    tab = np.zeros((m + 1, n + m + 1))
    tab[:m, :n] = a
    tab[:m, n : n + m] = np.eye(m)
    tab[:m, -1] = b
    tab[m, :n] = -c  # objective row holds reduced costs; start at -c

    basis = list(range(n, n + m))  # slacks are the initial basic variables
    vertices = [_structural(tab, basis, n, m)]

    for iteration in range(1, max_iter + 1):
        pivot_col = int(np.argmin(tab[m, :-1]))
        if tab[m, pivot_col] >= -tol:  # no improving direction -> optimal corner
            x = _structural(tab, basis, n, m)
            return SimplexResult(OPTIMAL, float(tab[m, -1]), x, iteration - 1, vertices)

        column = tab[:m, pivot_col]
        positive = column > tol
        if not np.any(positive):
            return SimplexResult(UNBOUNDED, float("inf"), [], iteration, vertices)

        ratios = np.where(positive, tab[:m, -1] / np.where(positive, column, 1.0), np.inf)
        pivot_row = int(np.argmin(ratios))

        tab[pivot_row, :] /= tab[pivot_row, pivot_col]
        for r in range(m + 1):
            if r != pivot_row:
                tab[r, :] -= tab[r, pivot_col] * tab[pivot_row, :]
        basis[pivot_row] = pivot_col
        vertices.append(_structural(tab, basis, n, m))

    x = _structural(tab, basis, n, m)
    return SimplexResult(ITERATION_LIMIT, float(tab[m, -1]), x, max_iter, vertices)


def _structural(tab: np.ndarray, basis: list[int], n: int, m: int) -> list[float]:
    """Read the structural variables' values off the current tableau."""
    x = [0.0] * n
    for row, var in enumerate(basis):
        if var < n:
            x[var] = float(tab[row, -1])
    return x

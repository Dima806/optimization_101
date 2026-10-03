"""Shadow prices the hard way: re-solve and watch the objective move.

A shadow price is a derivative -- how much the optimum changes per unit of a
constraint's right-hand side. This module computes that derivative by brute
force: nudge one rhs by epsilon, re-solve, and divide. It is how the tests prove
the duals the solver reports are real and checkable, not asserted (notebook 03).
"""

from __future__ import annotations

from src.solvers.backends import solve
from src.solvers.problem import LinearProgram


def finite_difference_shadow_prices(
    lp: LinearProgram, backend: str = "scipy", eps: float = 1e-4
) -> dict[str, float]:
    """Estimate every constraint's shadow price by a one-sided re-solve.

    ``shadow[c] ~= (objective(rhs_c + eps) - objective(rhs_c)) / eps``.
    """
    base = solve(lp, backend)
    prices: dict[str, float] = {}
    for con in lp.constraints:
        bumped = solve(lp.with_rhs(con.name, con.rhs + eps), backend)
        prices[con.name] = (bumped.objective - base.objective) / eps
    return prices


def objective_vs_rhs(
    lp: LinearProgram, constraint: str, values: list[float], backend: str = "scipy"
) -> list[float]:
    """Optimal objective as one constraint's right-hand side sweeps a range."""
    return [solve(lp.with_rhs(constraint, rhs), backend).objective for rhs in values]

"""What a solve returns: the optimum, the plan, and the shadow prices.

A predictive model returns a number. A solver returns the number *and* a map of
what every constraint is worth at the margin -- the shadow prices -- which is the
signature deliverable of this whole project (notebook 03).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

OPTIMAL = "optimal"
INFEASIBLE = "infeasible"
UNBOUNDED = "unbounded"
NOT_SOLVED = "not_solved"


@dataclass
class Solution:
    """The result of solving a :class:`~src.solvers.problem.LinearProgram`."""

    status: str
    objective: float
    values: dict[str, float]
    shadow_prices: dict[str, float] = field(default_factory=dict)
    slack: dict[str, float] = field(default_factory=dict)
    reduced_costs: dict[str, float] = field(default_factory=dict)
    solve_time: float = 0.0
    backend: str = ""

    @property
    def is_optimal(self) -> bool:
        """True if the solver proved an optimal solution."""
        return self.status == OPTIMAL

    def value(self, name: str) -> float:
        """The value of one decision variable at the optimum."""
        return self.values[name]

    def binding_constraints(self, tol: float = 1e-6) -> list[str]:
        """Constraints with ~zero slack -- the ones that are actually limiting you."""
        return [name for name, s in self.slack.items() if abs(s) <= tol]

    def solution_frame(self) -> pd.DataFrame:
        """Decision variables and their optimal values, as a tidy table."""
        return pd.DataFrame({"variable": list(self.values), "value": list(self.values.values())})

    def shadow_price_frame(self, tol: float = 1e-6) -> pd.DataFrame:
        """Per-constraint shadow prices and slack -- the bottleneck map."""
        rows = [
            {
                "constraint": name,
                "shadow_price": self.shadow_prices.get(name, 0.0),
                "slack": self.slack.get(name, 0.0),
                "binding": abs(self.slack.get(name, 0.0)) <= tol,
            }
            for name in self.shadow_prices or self.slack
        ]
        return pd.DataFrame(rows)

    def summary(self) -> str:
        """A short human-readable report."""
        lines = [
            f"status      : {self.status}",
            f"backend     : {self.backend}",
            f"objective   : {self.objective:.6g}",
            f"solve_time  : {self.solve_time * 1e3:.2f} ms",
            "solution    : " + ", ".join(f"{k}={v:.4g}" for k, v in self.values.items()),
        ]
        if self.shadow_prices:
            lines.append(
                "shadow      : " + ", ".join(f"{k}={v:.4g}" for k, v in self.shadow_prices.items())
            )
        return "\n".join(lines)

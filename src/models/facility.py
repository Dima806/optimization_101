"""A small fixed-charge facility-location MILP (notebook 04).

Open a depot and you pay a fixed cost; then serve each customer from some open
depot. The open/closed choice is binary -- half a warehouse is meaningless -- so
this is a MILP with the characteristic linking constraint ``x[c,d] <= y[d]``:
you cannot serve a customer from a depot you did not open.
"""

from __future__ import annotations

from src.datasets.scenarios import FacilityScenario
from src.solvers.problem import Constraint, Direction, LinearProgram, Sense, binary_var


def build(scenario: FacilityScenario) -> LinearProgram:
    """Build the fixed-charge facility-location MILP from a scenario."""
    depots, customers = scenario.depots, scenario.customers

    def assign(c: str, d: str) -> str:
        return f"serve_{c}_{d}"

    def is_open(d: str) -> str:
        return f"open_{d}"

    variables = [binary_var(is_open(d)) for d in depots]
    variables += [binary_var(assign(c, d)) for c in customers for d in depots]

    objective = {is_open(d): scenario.open_cost[d] for d in depots}
    objective.update({assign(c, d): scenario.serve_cost[c, d] for c in customers for d in depots})

    constraints: list[Constraint] = []
    for c in customers:  # every customer served from exactly one depot
        constraints.append(
            Constraint(f"serve_{c}", {assign(c, d): 1.0 for d in depots}, Sense.EQ, 1.0)
        )
    for c in customers:  # cannot serve from a depot that is not open
        for d in depots:
            constraints.append(
                Constraint(
                    f"link_{c}_{d}",
                    {assign(c, d): 1.0, is_open(d): -1.0},
                    Sense.LE,
                    0.0,
                )
            )
    return LinearProgram(scenario.name, Direction.MIN, objective, variables, constraints)

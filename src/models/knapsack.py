"""The 0/1 knapsack -- the canonical MILP (notebook 04).

Take or leave each item (a binary decision) to maximise value under a weight
budget. Relax the variables to fractions and you get nonsense -- "take 0.6 of an
item" -- which is exactly why integrality matters and why the solver must search
(branch-and-bound) instead of sliding to a corner.
"""

from __future__ import annotations

from src.datasets.scenarios import KnapsackScenario
from src.solvers.problem import Constraint, Direction, LinearProgram, Sense, binary_var


def build(scenario: KnapsackScenario) -> LinearProgram:
    """Build the 0/1 knapsack MILP from a scenario."""
    variables = [binary_var(item) for item in scenario.items]
    objective = {item: scenario.value[item] for item in scenario.items}
    weight = Constraint(
        name="capacity",
        lhs={item: scenario.weight[item] for item in scenario.items},
        sense=Sense.LE,
        rhs=scenario.capacity,
    )
    return LinearProgram(scenario.name, Direction.MAX, objective, variables, [weight])

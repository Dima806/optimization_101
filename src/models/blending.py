"""The diet / blending LP: least-cost mix that meets a spec (notebook 05).

The historically first LP solved industrially. Decision variables are how much of
each food (or raw material) to use, the objective is total cost, and the
constraints say every nutrient (or chemical spec) clears its minimum.
"""

from __future__ import annotations

from src.datasets.scenarios import BlendingScenario
from src.solvers.problem import Constraint, Direction, LinearProgram, Sense, continuous_var


def build(scenario: BlendingScenario) -> LinearProgram:
    """Build the least-cost blending LP from a scenario."""
    variables = [continuous_var(food) for food in scenario.foods]
    objective = {food: scenario.cost[food] for food in scenario.foods}
    constraints = [
        Constraint(
            name=nutrient,
            lhs={
                food: scenario.content[nutrient][food]
                for food in scenario.foods
                if scenario.content[nutrient].get(food, 0.0) != 0.0
            },
            sense=Sense.GE,
            rhs=scenario.minimum[nutrient],
        )
        for nutrient in scenario.nutrients
    ]
    return LinearProgram(scenario.name, Direction.MIN, objective, variables, constraints)

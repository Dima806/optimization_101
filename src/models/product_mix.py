"""The product-mix linear program -- the showpiece (notebook 02).

Several products, each with a profit and a resource footprint; limited resources;
maximise profit. Three ingredients: the quantities are the *decision variables*,
total profit is the *objective*, the resource limits are the *constraints*. That
is the whole model, and a free solver returns the provably best mix.
"""

from __future__ import annotations

import numpy as np

from src.datasets.scenarios import ProductMixScenario
from src.solvers.problem import Constraint, Direction, LinearProgram, Sense, continuous_var


def build(scenario: ProductMixScenario) -> LinearProgram:
    """Build the product-mix LP from a scenario."""
    variables = [continuous_var(p) for p in scenario.products]
    objective = {p: scenario.profit[p] for p in scenario.products}
    constraints = [
        Constraint(
            name=resource,
            lhs={
                p: scenario.usage[resource][p]
                for p in scenario.products
                if scenario.usage[resource].get(p, 0.0) != 0.0
            },
            sense=Sense.LE,
            rhs=scenario.capacity[resource],
        )
        for resource in scenario.resources
    ]
    return LinearProgram(scenario.name, Direction.MAX, objective, variables, constraints)


def as_matrices(
    scenario: ProductMixScenario,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The same LP in raw matrix form ``(c, A, b)`` for ``maximize c.x, A x <= b``.

    This is the SciPy view of the identical model built by :func:`build`: readers
    see that algebraic modelling and matrices are two descriptions of one problem.
    """
    products = scenario.products
    c = np.array([scenario.profit[p] for p in products], dtype=float)
    a = np.array(
        [[scenario.usage[r].get(p, 0.0) for p in products] for r in scenario.resources],
        dtype=float,
    )
    b = np.array([scenario.capacity[r] for r in scenario.resources], dtype=float)
    return c, a, b

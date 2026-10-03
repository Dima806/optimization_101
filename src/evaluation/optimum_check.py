"""Brute-force oracles: proof that the solver's optimum is the optimum.

The project's core promise is exactness, so on instances small enough we compute
the answer a second, independent way -- by enumeration -- and check the solver
matches. If brute force and the solver disagree, something is wrong, and the
tests say so.
"""

from __future__ import annotations

import itertools
import math

from src.datasets.scenarios import (
    AssignmentScenario,
    FacilityScenario,
    KnapsackScenario,
)


def brute_force_knapsack(
    values: list[float], weights: list[float], capacity: float
) -> tuple[float, tuple[int, ...]]:
    """Best value over every subset of items (2^n enumeration)."""
    n = len(values)
    best_value = 0.0
    best_set: tuple[int, ...] = ()
    for mask in range(1 << n):
        chosen = tuple(i for i in range(n) if mask & (1 << i))
        if sum(weights[i] for i in chosen) <= capacity:
            total = sum(values[i] for i in chosen)
            if total > best_value:
                best_value, best_set = total, chosen
    return best_value, best_set


def brute_force_knapsack_scenario(scenario: KnapsackScenario) -> float:
    """Convenience wrapper returning just the optimal value for a scenario."""
    values = [scenario.value[i] for i in scenario.items]
    weights = [scenario.weight[i] for i in scenario.items]
    best, _ = brute_force_knapsack(values, weights, scenario.capacity)
    return best


def brute_force_assignment(cost: list[list[float]]) -> tuple[float, tuple[int, ...]]:
    """Cheapest one-to-one assignment over every permutation (n! enumeration)."""
    n = len(cost)
    best_cost = math.inf
    best_perm: tuple[int, ...] = ()
    for perm in itertools.permutations(range(n)):
        total = sum(cost[w][perm[w]] for w in range(n))
        if total < best_cost:
            best_cost, best_perm = total, perm
    return best_cost, best_perm


def brute_force_assignment_scenario(scenario: AssignmentScenario) -> float:
    """Convenience wrapper returning just the optimal cost for a scenario."""
    cost = [list(row) for row in scenario.cost]
    best, _ = brute_force_assignment(cost)
    return best


def brute_force_facility(scenario: FacilityScenario) -> tuple[float, tuple[str, ...]]:
    """Best total cost over every non-empty subset of depots to open."""
    depots = scenario.depots
    best_cost = math.inf
    best_open: tuple[str, ...] = ()
    for r in range(1, len(depots) + 1):
        for subset in itertools.combinations(depots, r):
            fixed = sum(scenario.open_cost[d] for d in subset)
            serving = sum(
                min(scenario.serve_cost[c, d] for d in subset) for c in scenario.customers
            )
            total = fixed + serving
            if total < best_cost:
                best_cost, best_open = total, subset
    return best_cost, best_open


def is_close(a: float, b: float, tol: float = 1e-6) -> bool:
    """Absolute/relative closeness check for comparing optima."""
    return math.isclose(a, b, rel_tol=tol, abs_tol=tol)

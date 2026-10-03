"""Each model returns its known optimum -- verified analytically or by brute force."""

from __future__ import annotations

import pytest

from src.datasets import scenarios
from src.evaluation.optimum_check import (
    brute_force_assignment_scenario,
    brute_force_facility,
    brute_force_knapsack_scenario,
    is_close,
)
from src.models import assignment, blending, facility, knapsack, product_mix
from src.solvers.backends import solve


def test_product_mix_optimum() -> None:
    s = scenarios.product_mix_scenario()
    sol = solve(product_mix.build(s), backend="scipy")
    assert sol.is_optimal
    assert is_close(sol.objective, s.optimum)
    for p, qty in s.optimal_mix.items():
        assert is_close(sol.value(p), qty, tol=1e-5)


def test_blending_optimum() -> None:
    s = scenarios.blending_scenario()
    sol = solve(blending.build(s), backend="scipy")
    assert sol.is_optimal
    assert is_close(sol.objective, s.optimum)
    for food, qty in s.optimal_amounts.items():
        assert is_close(sol.value(food), qty, tol=1e-5)


def test_assignment_matches_brute_force() -> None:
    s = scenarios.assignment_scenario()
    sol = solve(assignment.build_assignment(s), backend="scipy")
    assert sol.is_optimal
    assert is_close(sol.objective, brute_force_assignment_scenario(s))
    assert is_close(sol.objective, s.optimum)
    # totally unimodular -> the LP returns a whole-number assignment for free
    for value in sol.values.values():
        assert is_close(value, round(value), tol=1e-6)


def test_transportation_optimum() -> None:
    s = scenarios.transportation_scenario()
    lp = assignment.build_transportation(s)
    sol = solve(lp, backend="scipy")
    assert sol.is_optimal
    assert is_close(sol.objective, s.optimum)
    for d in s.warehouses:  # every warehouse's demand is met
        shipped = sum(sol.value(f"ship_{p}_{d}") for p in s.plants)
        assert shipped >= s.demand[d] - 1e-6


def test_knapsack_matches_brute_force() -> None:
    s = scenarios.knapsack_scenario()
    sol = solve(knapsack.build(s), backend="pulp")
    assert sol.is_optimal
    assert is_close(sol.objective, brute_force_knapsack_scenario(s))
    assert is_close(sol.objective, s.optimum)
    picked = {item for item in s.items if sol.value(item) > 0.5}
    assert picked == set(s.chosen)


def test_facility_matches_brute_force() -> None:
    s = scenarios.facility_scenario()
    sol = solve(facility.build(s), backend="pulp")
    best_cost, open_depots = brute_force_facility(s)
    assert sol.is_optimal
    assert is_close(sol.objective, best_cost)
    assert is_close(sol.objective, s.optimum)
    opened = {d for d in s.depots if sol.value(f"open_{d}") > 0.5}
    assert opened == set(open_depots) == set(s.open_depots)


def test_fractional_relaxation_is_nonsensical() -> None:
    # Notebook 04's point: relax integrality and the knapsack answer goes fractional,
    # and its (looser) optimum overshoots the true integer optimum.
    s = scenarios.knapsack_scenario()
    milp = knapsack.build(s)
    relaxed = solve(milp.relaxation(), backend="scipy")
    assert relaxed.objective > s.optimum + 1e-6
    assert any(abs(v - round(v)) > 1e-6 for v in relaxed.values.values())


@pytest.mark.parametrize("backend", ["scipy", "pulp"])
def test_in_process_backends_agree_on_product_mix(backend: str) -> None:
    s = scenarios.product_mix_scenario()
    sol = solve(product_mix.build(s), backend=backend)
    assert is_close(sol.objective, s.optimum)

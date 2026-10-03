"""Shadow prices are real: they match both the known duals and a re-solve."""

from __future__ import annotations

from src.datasets import scenarios
from src.evaluation.optimum_check import is_close
from src.evaluation.sensitivity import finite_difference_shadow_prices
from src.models import product_mix
from src.solvers.backends import solve


def test_shadow_prices_match_known_values() -> None:
    s = scenarios.product_mix_scenario()
    sol = solve(product_mix.build(s), backend="scipy")
    assert sol.is_optimal
    for constraint, expected in s.shadow_prices.items():
        assert is_close(sol.shadow_prices[constraint], expected, tol=1e-5)


def test_shadow_prices_match_finite_difference() -> None:
    s = scenarios.product_mix_scenario()
    lp = product_mix.build(s)
    sol = solve(lp, backend="scipy")
    fd = finite_difference_shadow_prices(lp, backend="scipy", eps=1e-4)
    for con in lp.constraints:
        assert abs(sol.shadow_prices[con.name] - fd[con.name]) < 1e-3


def test_binding_constraints_identified() -> None:
    s = scenarios.product_mix_scenario()
    sol = solve(product_mix.build(s), backend="scipy")
    # plant1 has slack (shadow price 0); plant2 and plant3 bind.
    binding = set(sol.binding_constraints(tol=1e-6))
    assert binding == {"plant2", "plant3"}

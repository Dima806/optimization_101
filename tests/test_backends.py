"""All four open solvers agree on the same model.

Each backend is solved in its own subprocess (``compare_backends``), because
``highspy`` and ``ortools`` each statically bundle HiGHS and cannot share a
process. The parent test process therefore only ever imports scipy and pulp.
"""

from __future__ import annotations

from src.datasets import scenarios
from src.evaluation.optimum_check import is_close
from src.models import knapsack, product_mix
from src.solvers.backends import available_backends, compare_backends


def test_available_backends_present() -> None:
    found = available_backends()
    assert set(found) == {"scipy", "pulp", "highs", "ortools"}


def test_backends_agree_on_lp() -> None:
    s = scenarios.product_mix_scenario()
    results = compare_backends(product_mix.build(s))
    assert set(results) == set(available_backends())
    for backend, sol in results.items():
        assert sol.is_optimal, f"{backend} did not solve"
        assert is_close(sol.objective, s.optimum, tol=1e-5), backend
        assert is_close(sol.value("doors"), 2.0, tol=1e-5), backend
        assert is_close(sol.value("windows"), 6.0, tol=1e-5), backend


def test_backends_agree_on_lp_shadow_prices() -> None:
    s = scenarios.product_mix_scenario()
    results = compare_backends(product_mix.build(s))
    for backend, sol in results.items():
        for constraint, expected in s.shadow_prices.items():
            assert is_close(sol.shadow_prices[constraint], expected, tol=1e-5), (
                f"{backend}:{constraint}"
            )


def test_backends_agree_on_milp() -> None:
    s = scenarios.knapsack_scenario()
    results = compare_backends(knapsack.build(s))
    for backend, sol in results.items():
        assert sol.is_optimal, f"{backend} did not solve"
        assert is_close(sol.objective, s.optimum, tol=1e-5), backend

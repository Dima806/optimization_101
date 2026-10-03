"""Smoke tests: the plots build without a display."""

from __future__ import annotations

import matplotlib

from src.datasets import scenarios
from src.models import product_mix
from src.solvers.backends import solve
from src.visualisation import plot_feasible_region, plot_shadow_prices


def test_feasible_region_builds() -> None:
    s = scenarios.product_mix_scenario()
    lp = product_mix.build(s)
    sol = solve(lp, backend="scipy")
    fig = plot_feasible_region(lp, sol, x_max=12, y_max=12)
    assert isinstance(fig, matplotlib.figure.Figure)


def test_shadow_price_chart_builds() -> None:
    s = scenarios.product_mix_scenario()
    sol = solve(product_mix.build(s), backend="scipy")
    fig = plot_shadow_prices(sol)
    assert isinstance(fig, matplotlib.figure.Figure)

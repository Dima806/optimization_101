"""The from-scratch simplex matches the library on small LPs."""

from __future__ import annotations

import pytest

from src.datasets import scenarios
from src.evaluation.optimum_check import is_close
from src.models import product_mix
from src.simplex.from_scratch import OPTIMAL, simplex_max
from src.solvers.backends import solve


def test_simplex_matches_library_on_product_mix() -> None:
    s = scenarios.product_mix_scenario()
    c, a, b = product_mix.as_matrices(s)
    result = simplex_max(c, a, b)
    assert result.status == OPTIMAL
    assert is_close(result.objective, s.optimum, tol=1e-6)
    for value, name in zip(result.x, s.products, strict=True):
        assert is_close(value, s.optimal_mix[name], tol=1e-6)


def test_simplex_matches_scipy_on_a_second_lp() -> None:
    # max 2x + 3y  s.t.  x + y <= 4 ; x <= 3 ; y <= 3
    c = [2.0, 3.0]
    a = [[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]]
    b = [4.0, 3.0, 3.0]
    result = simplex_max(c, a, b)
    assert result.status == OPTIMAL
    # optimum at (1, 3): 2*1 + 3*3 = 11
    assert is_close(result.objective, 11.0, tol=1e-6)


def test_simplex_walks_corners() -> None:
    s = scenarios.product_mix_scenario()
    c, a, b = product_mix.as_matrices(s)
    result = simplex_max(c, a, b)
    # starts at the origin corner and ends at the optimal corner
    assert result.vertices[0] == [0.0, 0.0]
    assert is_close(result.vertices[-1][0], s.optimal_mix["doors"], tol=1e-6)


def test_simplex_rejects_negative_rhs() -> None:
    with pytest.raises(ValueError, match="b_i >= 0"):
        simplex_max([1.0], [[1.0]], [-1.0])


def test_matches_a_full_solver_backend() -> None:
    s = scenarios.product_mix_scenario()
    c, a, b = product_mix.as_matrices(s)
    scratch = simplex_max(c, a, b)
    library = solve(product_mix.build(s), backend="scipy")
    assert is_close(scratch.objective, library.objective, tol=1e-6)

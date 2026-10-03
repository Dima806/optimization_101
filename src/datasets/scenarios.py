"""Named, self-contained business problems with known or checkable optima.

Every instance is small on purpose: a 2-CPU solve is instant, and the optimum is
either analytic (product mix, blending) or confirmable by brute force (assignment,
knapsack, facility) in the tests. No external data, no downloads -- the project's
promise is exactness, so every claimed optimum has to be checkable.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProductMixScenario:
    """Make several products from limited resources to maximise profit."""

    name: str
    products: tuple[str, ...]
    profit: dict[str, float]
    resources: tuple[str, ...]
    usage: dict[str, dict[str, float]]  # resource -> {product: units consumed}
    capacity: dict[str, float]
    optimum: float
    optimal_mix: dict[str, float]
    shadow_prices: dict[str, float]


@dataclass(frozen=True)
class BlendingScenario:
    """Least-cost blend of foods meeting nutrient minimums (the first LP)."""

    name: str
    foods: tuple[str, ...]
    cost: dict[str, float]
    nutrients: tuple[str, ...]
    content: dict[str, dict[str, float]]  # nutrient -> {food: amount per unit}
    minimum: dict[str, float]
    optimum: float
    optimal_amounts: dict[str, float]


@dataclass(frozen=True)
class AssignmentScenario:
    """Match workers to jobs one-to-one at minimum total cost."""

    name: str
    workers: tuple[str, ...]
    jobs: tuple[str, ...]
    cost: tuple[tuple[float, ...], ...]  # cost[worker][job]
    optimum: float


@dataclass(frozen=True)
class TransportationScenario:
    """Ship from plants to warehouses at minimum cost (supply meets demand)."""

    name: str
    plants: tuple[str, ...]
    warehouses: tuple[str, ...]
    supply: dict[str, float]
    demand: dict[str, float]
    cost: dict[tuple[str, str], float]
    optimum: float


@dataclass(frozen=True)
class KnapsackScenario:
    """Pick items under a weight budget to maximise value (the canonical MILP)."""

    name: str
    items: tuple[str, ...]
    value: dict[str, float]
    weight: dict[str, float]
    capacity: float
    optimum: float
    chosen: tuple[str, ...]


@dataclass(frozen=True)
class FacilityScenario:
    """Open depots (fixed cost) and serve every customer at least total cost."""

    name: str
    depots: tuple[str, ...]
    customers: tuple[str, ...]
    open_cost: dict[str, float]
    serve_cost: dict[tuple[str, str], float]  # (customer, depot) -> cost
    optimum: float
    open_depots: tuple[str, ...]


def product_mix_scenario() -> ProductMixScenario:
    """The Wyndor Glass product mix (Hillier & Lieberman): optimum 36 at (2, 6)."""
    return ProductMixScenario(
        name="wyndor_glass",
        products=("doors", "windows"),
        profit={"doors": 3.0, "windows": 5.0},
        resources=("plant1", "plant2", "plant3"),
        usage={
            "plant1": {"doors": 1.0},
            "plant2": {"windows": 2.0},
            "plant3": {"doors": 3.0, "windows": 2.0},
        },
        capacity={"plant1": 4.0, "plant2": 12.0, "plant3": 18.0},
        optimum=36.0,
        optimal_mix={"doors": 2.0, "windows": 6.0},
        shadow_prices={"plant1": 0.0, "plant2": 1.5, "plant3": 1.0},
    )


def blending_scenario() -> BlendingScenario:
    """A tiny diet problem: optimum cost 18 at grain=2, meat=6."""
    return BlendingScenario(
        name="diet",
        foods=("grain", "meat"),
        cost={"grain": 3.0, "meat": 2.0},
        nutrients=("vitamin", "protein"),
        content={
            "vitamin": {"grain": 1.0, "meat": 1.0},
            "protein": {"grain": 2.0, "meat": 1.0},
        },
        minimum={"vitamin": 8.0, "protein": 10.0},
        optimum=18.0,
        optimal_amounts={"grain": 2.0, "meat": 6.0},
    )


def assignment_scenario() -> AssignmentScenario:
    """A 3x3 assignment problem; optimum 9 (W1->J2, W2->J1, W3->J3)."""
    return AssignmentScenario(
        name="assign3",
        workers=("W1", "W2", "W3"),
        jobs=("J1", "J2", "J3"),
        cost=(
            (9.0, 2.0, 7.0),
            (6.0, 4.0, 3.0),
            (5.0, 8.0, 1.0),
        ),
        optimum=9.0,
    )


def transportation_scenario() -> TransportationScenario:
    """Two plants, three warehouses, balanced supply/demand; optimum 465."""
    cost = {
        ("P1", "D1"): 8.0,
        ("P1", "D2"): 6.0,
        ("P1", "D3"): 10.0,
        ("P2", "D1"): 9.0,
        ("P2", "D2"): 12.0,
        ("P2", "D3"): 13.0,
    }
    return TransportationScenario(
        name="transport",
        plants=("P1", "P2"),
        warehouses=("D1", "D2", "D3"),
        supply={"P1": 20.0, "P2": 30.0},
        demand={"D1": 10.0, "D2": 25.0, "D3": 15.0},
        cost=cost,
        optimum=465.0,
    )


def knapsack_scenario() -> KnapsackScenario:
    """The textbook 0/1 knapsack; optimum value 220 (items b and c)."""
    return KnapsackScenario(
        name="knapsack3",
        items=("a", "b", "c"),
        value={"a": 60.0, "b": 100.0, "c": 120.0},
        weight={"a": 10.0, "b": 20.0, "c": 30.0},
        capacity=50.0,
        optimum=220.0,
        chosen=("b", "c"),
    )


def facility_scenario() -> FacilityScenario:
    """Two depots, three customers; optimum 155 (open only depot d0)."""
    serve = {
        ("c0", "d0"): 10.0,
        ("c0", "d1"): 20.0,
        ("c1", "d0"): 25.0,
        ("c1", "d1"): 15.0,
        ("c2", "d0"): 20.0,
        ("c2", "d1"): 22.0,
    }
    return FacilityScenario(
        name="facility",
        depots=("d0", "d1"),
        customers=("c0", "c1", "c2"),
        open_cost={"d0": 100.0, "d1": 100.0},
        serve_cost=serve,
        optimum=155.0,
        open_depots=("d0",),
    )

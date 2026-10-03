"""Assignment and transportation problems (notebook 05).

Both are network LPs with a surprising property: although the "match one-to-one"
and "ship integer units" decisions sound integer, the constraint matrices are
totally unimodular, so the LP relaxation already returns a whole-number optimum.
That is why continuous variables suffice and the answer still comes out integral.
"""

from __future__ import annotations

from src.datasets.scenarios import AssignmentScenario, TransportationScenario
from src.solvers.problem import Constraint, Direction, LinearProgram, Sense, continuous_var


def build_assignment(scenario: AssignmentScenario) -> LinearProgram:
    """Minimum-cost one-to-one assignment of workers to jobs."""
    workers, jobs = scenario.workers, scenario.jobs

    def x(w: str, j: str) -> str:
        return f"x_{w}_{j}"

    variables = [continuous_var(x(w, j), 0.0, 1.0) for w in workers for j in jobs]
    objective = {
        x(w, j): scenario.cost[wi][ji] for wi, w in enumerate(workers) for ji, j in enumerate(jobs)
    }
    constraints: list[Constraint] = []
    for w in workers:  # each worker takes exactly one job
        constraints.append(Constraint(f"worker_{w}", {x(w, j): 1.0 for j in jobs}, Sense.EQ, 1.0))
    for j in jobs:  # each job gets exactly one worker
        constraints.append(Constraint(f"job_{j}", {x(w, j): 1.0 for w in workers}, Sense.EQ, 1.0))
    return LinearProgram(scenario.name, Direction.MIN, objective, variables, constraints)


def build_transportation(scenario: TransportationScenario) -> LinearProgram:
    """Minimum-cost shipping from plants (supply) to warehouses (demand)."""
    plants, warehouses = scenario.plants, scenario.warehouses

    def x(p: str, d: str) -> str:
        return f"ship_{p}_{d}"

    variables = [continuous_var(x(p, d)) for p in plants for d in warehouses]
    objective = {x(p, d): scenario.cost[p, d] for p in plants for d in warehouses}
    constraints: list[Constraint] = []
    for p in plants:  # do not ship more than a plant can supply
        constraints.append(
            Constraint(
                f"supply_{p}", {x(p, d): 1.0 for d in warehouses}, Sense.LE, scenario.supply[p]
            )
        )
    for d in warehouses:  # meet every warehouse's demand
        constraints.append(
            Constraint(f"demand_{d}", {x(p, d): 1.0 for p in plants}, Sense.GE, scenario.demand[d])
        )
    return LinearProgram(scenario.name, Direction.MIN, objective, variables, constraints)

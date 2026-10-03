"""One uniform interface over four open-source solvers.

The same :class:`~src.solvers.problem.LinearProgram` can be handed to any of:

* ``"scipy"``   -- ``scipy.optimize.linprog`` / ``milp`` (HiGHS under the hood),
* ``"pulp"``    -- PuLP modeling with the bundled open CBC solver,
* ``"highs"``   -- HiGHS directly via ``highspy`` (MIT, the high-performance option),
* ``"ortools"`` -- Google OR-Tools (GLOP for LPs, CBC for MILPs).

Every one returns a :class:`~src.solvers.report.Solution`. Shadow prices and
reduced costs are reported only for pure LPs, where duals are defined; for a
MILP they are left empty. All duals are normalised to one convention:

    shadow_price[c] = d(optimal objective) / d(rhs of c)

so they are directly comparable across backends and against a finite-difference
re-solve (see ``src/evaluation/sensitivity.py``).
"""

from __future__ import annotations

import importlib.util
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

import numpy as np

from src.solvers.problem import INF, Constraint, Direction, LinearProgram, Sense, VarType
from src.solvers.report import INFEASIBLE, NOT_SOLVED, OPTIMAL, UNBOUNDED, Solution

BACKENDS = ("scipy", "pulp", "highs", "ortools")

# highspy and ortools each statically bundle HiGHS and export its C++ symbols
# with global visibility; whichever imports first interposes on the other and
# the second import fails (an upstream packaging clash, independent of version).
# scipy's HiGHS is vendored privately and conflicts with neither. So we keep the
# parent process on scipy+pulp and run the all-backends comparison through
# per-solve subprocesses (see ``solve_isolated`` / ``compare_backends``).
_PACKAGES = {
    "scipy": "scipy",
    "pulp": "pulp",
    "highs": "highspy",
    "ortools": "ortools",
}
_CONFLICTING = {"highs", "ortools"}


def solve(lp: LinearProgram, backend: str = "pulp", time_limit: float | None = None) -> Solution:
    """Solve ``lp`` with the named backend (in-process) and return a uniform report.

    Note: ``"highs"`` and ``"ortools"`` cannot both be used in one process. Mixing
    them raises a clear error pointing you at :func:`compare_backends`, which runs
    each backend in its own subprocess.
    """
    if backend not in BACKENDS:
        raise ValueError(f"unknown backend {backend!r}; choose from {BACKENDS}")
    solver = {
        "scipy": _solve_scipy,
        "pulp": _solve_pulp,
        "highs": _solve_highs,
        "ortools": _solve_ortools,
    }[backend]
    try:
        sol = solver(lp, time_limit)
    except ImportError as exc:  # symbol clash from the other HiGHS-bundling backend
        if backend in _CONFLICTING and "symbol" in str(exc):
            raise RuntimeError(
                f"{backend!r} cannot share a process with the other HiGHS backend; "
                "use compare_backends() / solve_isolated() (one subprocess per solve)."
            ) from exc
        raise
    _attach_slack(lp, sol)
    return sol


def available_backends() -> tuple[str, ...]:
    """Backends whose package is installed (checked without importing the binaries)."""
    return tuple(
        name for name, mod in _PACKAGES.items() if importlib.util.find_spec(mod) is not None
    )


def _isolated_solve(lp: LinearProgram, backend: str, time_limit: float | None) -> Solution:
    """Module-level target so it is picklable for ``spawn`` subprocesses."""
    return solve(lp, backend, time_limit)


def solve_isolated(lp: LinearProgram, backend: str, time_limit: float | None = None) -> Solution:
    """Solve in a fresh subprocess, so ``highs`` and ``ortools`` never coexist."""
    ctx = get_context("spawn")
    with ProcessPoolExecutor(max_workers=1, mp_context=ctx) as pool:
        return pool.submit(_isolated_solve, lp, backend, time_limit).result()


def compare_backends(
    lp: LinearProgram,
    backends: tuple[str, ...] | None = None,
    time_limit: float | None = None,
) -> dict[str, Solution]:
    """Solve the same model through every backend (each isolated) and collect results.

    This is the honest "do they all agree?" demo (notebook 06): same problem, four
    open solvers, one optimum.
    """
    chosen = backends or available_backends()
    return {b: solve_isolated(lp, b, time_limit) for b in chosen}


def _attach_slack(lp: LinearProgram, sol: Solution) -> None:
    """Fill slack from the primal solution (same definition for every backend)."""
    if not sol.is_optimal:
        return
    for con in lp.constraints:
        sol.slack[con.name] = con.rhs - lp.activity(con.name, sol.values)


def _sign_for_direction(lp: LinearProgram) -> float:
    """Multiplier that maps a minimiser's duals onto d(obj)/d(rhs) for our sense."""
    return -1.0 if lp.direction is Direction.MAX else 1.0


# --------------------------------------------------------------------------- #
# SciPy
# --------------------------------------------------------------------------- #
def _solve_scipy(lp: LinearProgram, time_limit: float | None) -> Solution:
    from scipy.optimize import Bounds, LinearConstraint, linprog, milp

    names = lp.var_names
    index = {n: i for i, n in enumerate(names)}
    n = len(names)
    cost = np.array([lp.objective.get(name, 0.0) for name in names], dtype=float)
    sign = -1.0 if lp.direction is Direction.MAX else 1.0  # linprog/milp minimise

    lowers = np.array([-np.inf if v.lb == -INF else v.lb for v in lp.variables])
    uppers = np.array([np.inf if v.ub == INF else v.ub for v in lp.variables])

    if lp.is_integer:
        return _solve_scipy_milp(
            lp, index, n, cost, sign, lowers, uppers, Bounds, LinearConstraint, milp, time_limit
        )

    a_ub, b_ub, a_eq, b_eq, ub_cons, eq_cons = [], [], [], [], [], []
    for con in lp.constraints:
        row = np.zeros(n)
        for var, coeff in con.lhs.items():
            row[index[var]] = coeff
        if con.sense is Sense.LE:
            a_ub.append(row)
            b_ub.append(con.rhs)
            ub_cons.append(con)
        elif con.sense is Sense.GE:  # -row . x <= -rhs
            a_ub.append(-row)
            b_ub.append(-con.rhs)
            ub_cons.append(con)
        else:
            a_eq.append(row)
            b_eq.append(con.rhs)
            eq_cons.append(con)

    bounds = list(zip(lowers, uppers, strict=True))
    t0 = time.perf_counter()
    res = linprog(
        sign * cost,
        A_ub=np.array(a_ub) if a_ub else None,
        b_ub=np.array(b_ub) if b_ub else None,
        A_eq=np.array(a_eq) if a_eq else None,
        b_eq=np.array(b_eq) if b_eq else None,
        bounds=bounds,
        method="highs",
    )
    dt = time.perf_counter() - t0

    if not res.success:
        return Solution(_scipy_status(res.status), 0.0, {}, backend="scipy", solve_time=dt)

    values = {name: float(res.x[index[name]]) for name in names}
    objective = sign * res.fun + lp.objective_constant

    # linprog marginals are d(min)/d(scipy_b); map back to d(our obj)/d(our rhs).
    shadow: dict[str, float] = {}
    dir_sign = _sign_for_direction(lp)
    if a_ub:
        for con, m in zip(ub_cons, res.ineqlin.marginals, strict=True):
            b_sign = 1.0 if con.sense is Sense.LE else -1.0  # GE rows were negated
            shadow[con.name] = dir_sign * float(m) * b_sign
    if a_eq:
        for con, m in zip(eq_cons, res.eqlin.marginals, strict=True):
            shadow[con.name] = dir_sign * float(m)

    reduced = {name: dir_sign * float(res.lower.marginals[index[name]]) for name in names}
    return Solution(OPTIMAL, objective, values, shadow, {}, reduced, dt, "scipy")


def _solve_scipy_milp(
    lp, index, n, cost, sign, lowers, uppers, bounds_cls, lincon_cls, milp_fn, time_limit
) -> Solution:
    rows, con_lb, con_ub = [], [], []
    for con in lp.constraints:
        row = np.zeros(n)
        for var, coeff in con.lhs.items():
            row[index[var]] = coeff
        rows.append(row)
        if con.sense is Sense.LE:
            con_lb.append(-np.inf)
            con_ub.append(con.rhs)
        elif con.sense is Sense.GE:
            con_lb.append(con.rhs)
            con_ub.append(np.inf)
        else:
            con_lb.append(con.rhs)
            con_ub.append(con.rhs)

    integrality = np.array([0 if v.kind is VarType.CONTINUOUS else 1 for v in lp.variables])
    constraints = lincon_cls(np.array(rows), con_lb, con_ub) if rows else None
    options = {"time_limit": time_limit} if time_limit else None
    t0 = time.perf_counter()
    res = milp_fn(
        sign * cost,
        constraints=constraints,
        integrality=integrality,
        bounds=bounds_cls(lowers, uppers),
        options=options,
    )
    dt = time.perf_counter() - t0
    if not res.success:
        return Solution(INFEASIBLE, 0.0, {}, backend="scipy", solve_time=dt)
    values = {name: float(res.x[index[name]]) for name in lp.var_names}
    objective = sign * res.fun + lp.objective_constant
    return Solution(OPTIMAL, objective, values, backend="scipy", solve_time=dt)


def _scipy_status(code: int) -> str:
    return {2: INFEASIBLE, 3: UNBOUNDED}.get(code, NOT_SOLVED)


# --------------------------------------------------------------------------- #
# PuLP + CBC
# --------------------------------------------------------------------------- #
def _solve_pulp(lp: LinearProgram, time_limit: float | None) -> Solution:
    import pulp

    sense = pulp.LpMaximize if lp.direction is Direction.MAX else pulp.LpMinimize
    prob = pulp.LpProblem(lp.name, sense)
    cat = {
        VarType.CONTINUOUS: pulp.LpContinuous,
        VarType.INTEGER: pulp.LpInteger,
        VarType.BINARY: pulp.LpBinary,
    }
    pvars = {
        v.name: pulp.LpVariable(
            v.name,
            lowBound=None if v.lb == -INF else v.lb,
            upBound=None if v.ub == INF else v.ub,
            cat=cat[v.kind],
        )
        for v in lp.variables
    }
    prob += (
        pulp.lpSum(coeff * pvars[name] for name, coeff in lp.objective.items())
        + lp.objective_constant
    )
    for con in lp.constraints:
        expr = pulp.lpSum(coeff * pvars[var] for var, coeff in con.lhs.items())
        if con.sense is Sense.LE:
            prob += (expr <= con.rhs, con.name)
        elif con.sense is Sense.GE:
            prob += (expr >= con.rhs, con.name)
        else:
            prob += (expr == con.rhs, con.name)

    cmd = pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit)
    t0 = time.perf_counter()
    prob.solve(cmd)
    dt = time.perf_counter() - t0

    status = {
        pulp.LpStatusOptimal: OPTIMAL,
        pulp.LpStatusInfeasible: INFEASIBLE,
        pulp.LpStatusUnbounded: UNBOUNDED,
    }.get(prob.status, NOT_SOLVED)
    if status != OPTIMAL:
        return Solution(status, 0.0, {}, backend="pulp", solve_time=dt)

    values = {name: float(var.value() or 0.0) for name, var in pvars.items()}
    objective = float(pulp.value(prob.objective))
    shadow: dict[str, float] = {}
    reduced: dict[str, float] = {}
    if not lp.is_integer:
        for con in lp.constraints:
            pi = prob.constraints[con.name].pi
            shadow[con.name] = float(pi) if pi is not None else 0.0
        for name, var in pvars.items():
            reduced[name] = float(var.dj) if var.dj is not None else 0.0
    return Solution(OPTIMAL, objective, values, shadow, {}, reduced, dt, "pulp")


# --------------------------------------------------------------------------- #
# HiGHS via highspy
# --------------------------------------------------------------------------- #
def _solve_highs(lp: LinearProgram, time_limit: float | None) -> Solution:
    import highspy

    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    if time_limit:
        h.setOptionValue("time_limit", float(time_limit))

    names = lp.var_names
    index = {n: i for i, n in enumerate(names)}
    inf = highspy.kHighsInf
    for v in lp.variables:
        lb = -inf if v.lb == -INF else v.lb
        ub = inf if v.ub == INF else v.ub
        cost = lp.objective.get(v.name, 0.0)
        h.addCol(cost, lb, ub, 0, np.empty(0, dtype=np.int32), np.empty(0))
    h.changeObjectiveOffset(lp.objective_constant)
    h.changeObjectiveSense(
        highspy.ObjSense.kMaximize if lp.direction is Direction.MAX else highspy.ObjSense.kMinimize
    )
    for i, v in enumerate(lp.variables):
        if v.kind is not VarType.CONTINUOUS:
            h.changeColIntegrality(i, highspy.HighsVarType.kInteger)

    for con in lp.constraints:
        idx = np.array([index[var] for var in con.lhs], dtype=np.int32)
        val = np.array(list(con.lhs.values()), dtype=float)
        if con.sense is Sense.LE:
            lower, upper = -inf, con.rhs
        elif con.sense is Sense.GE:
            lower, upper = con.rhs, inf
        else:
            lower, upper = con.rhs, con.rhs
        h.addRow(lower, upper, len(idx), idx, val)

    t0 = time.perf_counter()
    h.run()
    dt = time.perf_counter() - t0

    model_status = h.getModelStatus()
    if model_status != highspy.HighsModelStatus.kOptimal:
        name = h.modelStatusToString(model_status).lower()
        status = (
            INFEASIBLE
            if "infeasible" in name
            else (UNBOUNDED if "unbounded" in name else NOT_SOLVED)
        )
        return Solution(status, 0.0, {}, backend="highs", solve_time=dt)

    sol = h.getSolution()
    values = {name: float(sol.col_value[index[name]]) for name in names}
    objective = float(h.getObjectiveValue())
    shadow: dict[str, float] = {}
    reduced: dict[str, float] = {}
    if not lp.is_integer:
        # HiGHS row duals are d(obj)/d(rhs) for a minimise; flip for maximise.
        dir_sign = _sign_for_direction(lp)
        for i, con in enumerate(lp.constraints):
            shadow[con.name] = -dir_sign * float(sol.row_dual[i])
        for name in names:
            reduced[name] = -dir_sign * float(sol.col_dual[index[name]])
    return Solution(OPTIMAL, objective, values, shadow, {}, reduced, dt, "highs")


# --------------------------------------------------------------------------- #
# OR-Tools
# --------------------------------------------------------------------------- #
def _solve_ortools(lp: LinearProgram, time_limit: float | None) -> Solution:
    from ortools.linear_solver import pywraplp

    solver = pywraplp.Solver.CreateSolver("GLOP" if not lp.is_integer else "CBC")
    if solver is None:
        raise RuntimeError("OR-Tools solver unavailable")
    if time_limit:
        solver.set_time_limit(int(time_limit * 1000))
    inf = solver.infinity()
    ovars = {}
    for v in lp.variables:
        lb = -inf if v.lb == -INF else v.lb
        ub = inf if v.ub == INF else v.ub
        if v.kind is VarType.CONTINUOUS:
            ovars[v.name] = solver.NumVar(lb, ub, v.name)
        else:
            ovars[v.name] = solver.IntVar(lb, ub, v.name)

    objective = solver.Objective()
    for name, coeff in lp.objective.items():
        objective.SetCoefficient(ovars[name], coeff)
    objective.SetOffset(lp.objective_constant)
    if lp.direction is Direction.MAX:
        objective.SetMaximization()
    else:
        objective.SetMinimization()

    ocons: dict[str, Constraint] = {}
    ort_cons = {}
    for con in lp.constraints:
        if con.sense is Sense.LE:
            lo, hi = -inf, con.rhs
        elif con.sense is Sense.GE:
            lo, hi = con.rhs, inf
        else:
            lo, hi = con.rhs, con.rhs
        ct = solver.Constraint(lo, hi, con.name)
        for var, coeff in con.lhs.items():
            ct.SetCoefficient(ovars[var], coeff)
        ort_cons[con.name] = ct
        ocons[con.name] = con

    t0 = time.perf_counter()
    result = solver.Solve()
    dt = time.perf_counter() - t0

    if result == pywraplp.Solver.INFEASIBLE:
        return Solution(INFEASIBLE, 0.0, {}, backend="ortools", solve_time=dt)
    if result == pywraplp.Solver.UNBOUNDED:
        return Solution(UNBOUNDED, 0.0, {}, backend="ortools", solve_time=dt)
    if result not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        return Solution(NOT_SOLVED, 0.0, {}, backend="ortools", solve_time=dt)

    values = {name: float(var.solution_value()) for name, var in ovars.items()}
    shadow: dict[str, float] = {}
    reduced: dict[str, float] = {}
    if not lp.is_integer:
        for name, ct in ort_cons.items():
            shadow[name] = float(ct.dual_value())
        for name, var in ovars.items():
            reduced[name] = float(var.reduced_cost())
    return Solution(OPTIMAL, float(objective.Value()), values, shadow, {}, reduced, dt, "ortools")

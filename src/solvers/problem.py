"""Backend-agnostic representation of a linear / mixed-integer program.

The three ingredients, turned into data:

* **decision variables** -- the choices (name, bounds, kind),
* an **objective** -- linear coefficients plus a direction to push them,
* **constraints** -- a linear combination, a sense, and a right-hand side.

A model in ``src/models`` builds one :class:`LinearProgram`; a backend in
``src/solvers/backends`` solves it. Keeping the problem separate from the solver
is exactly what lets the *same* model run through PuLP, SciPy, HiGHS and
OR-Tools and agree (notebook 06).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import StrEnum

INF = float("inf")


class Direction(StrEnum):
    """Whether we push the objective up or down."""

    MIN = "minimize"
    MAX = "maximize"


class Sense(StrEnum):
    """The relation between a constraint's left- and right-hand sides."""

    LE = "<="
    GE = ">="
    EQ = "=="


class VarType(StrEnum):
    """A decision variable's domain."""

    CONTINUOUS = "continuous"
    INTEGER = "integer"
    BINARY = "binary"


@dataclass(frozen=True)
class Variable:
    """A single decision variable with its bounds and domain."""

    name: str
    lb: float = 0.0
    ub: float = INF
    kind: VarType = VarType.CONTINUOUS


def continuous_var(name: str, lb: float = 0.0, ub: float = INF) -> Variable:
    """A continuous decision variable."""
    return Variable(name, lb, ub, VarType.CONTINUOUS)


def integer_var(name: str, lb: float = 0.0, ub: float = INF) -> Variable:
    """An integer decision variable."""
    return Variable(name, lb, ub, VarType.INTEGER)


def binary_var(name: str) -> Variable:
    """A 0/1 decision variable (build it or do not)."""
    return Variable(name, 0.0, 1.0, VarType.BINARY)


@dataclass(frozen=True)
class Constraint:
    """A linear constraint: ``sum(lhs[v] * v) <sense> rhs``."""

    name: str
    lhs: dict[str, float]
    sense: Sense
    rhs: float


@dataclass
class LinearProgram:
    """A complete linear or mixed-integer program."""

    name: str
    direction: Direction
    objective: dict[str, float]
    variables: list[Variable]
    constraints: list[Constraint] = field(default_factory=list)
    objective_constant: float = 0.0

    @property
    def var_names(self) -> list[str]:
        """Variable names, in declaration order (the canonical column order)."""
        return [v.name for v in self.variables]

    def variable(self, name: str) -> Variable:
        """Look up a variable by name."""
        for v in self.variables:
            if v.name == name:
                return v
        raise KeyError(name)

    @property
    def is_integer(self) -> bool:
        """True if any variable is integer or binary (i.e. this is a MILP)."""
        return any(v.kind is not VarType.CONTINUOUS for v in self.variables)

    def activity(self, name: str, values: dict[str, float]) -> float:
        """Left-hand-side value of a constraint at the given solution."""
        con = self.constraint(name)
        return sum(coeff * values.get(var, 0.0) for var, coeff in con.lhs.items())

    def constraint(self, name: str) -> Constraint:
        """Look up a constraint by name."""
        for con in self.constraints:
            if con.name == name:
                return con
        raise KeyError(name)

    def with_rhs(self, name: str, rhs: float) -> LinearProgram:
        """Copy with one constraint's right-hand side changed (for sensitivity)."""
        new_cons = [replace(c, rhs=rhs) if c.name == name else c for c in self.constraints]
        return replace(self, constraints=new_cons)

    def relaxation(self) -> LinearProgram:
        """Continuous relaxation: drop integrality, keep bounds (notebook 04)."""
        relaxed = [replace(v, kind=VarType.CONTINUOUS) for v in self.variables]
        return replace(self, name=f"{self.name}_relaxed", variables=relaxed)

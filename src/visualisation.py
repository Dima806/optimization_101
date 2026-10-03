"""Plots for the notebooks and the app: the feasible region and the bottleneck map.

The two-variable feasible region is the picture the whole field rests on -- a
polygon with the optimum sitting at a corner -- and the shadow-price bar chart is
the strategic payoff, showing which constraint is binding and what it is worth.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless: works in CI and notebooks without a display

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.solvers.problem import LinearProgram, Sense  # noqa: E402
from src.solvers.report import Solution  # noqa: E402


def plot_feasible_region(
    lp: LinearProgram,
    solution: Solution | None = None,
    x_max: float = 10.0,
    y_max: float = 10.0,
) -> plt.Figure:
    """Shade a two-variable LP's feasible region and mark the optimal corner."""
    if len(lp.variables) != 2:
        raise ValueError("feasible-region plot needs exactly two variables")
    xname, yname = lp.var_names
    grid = 400
    xs = np.linspace(0, x_max, grid)
    ys = np.linspace(0, y_max, grid)
    xx, yy = np.meshgrid(xs, ys)

    feasible = (xx >= 0) & (yy >= 0)
    for con in lp.constraints:
        ax_c = con.lhs.get(xname, 0.0) * xx + con.lhs.get(yname, 0.0) * yy
        if con.sense is Sense.LE:
            feasible &= ax_c <= con.rhs + 1e-9
        elif con.sense is Sense.GE:
            feasible &= ax_c >= con.rhs - 1e-9
        else:
            feasible &= np.isclose(ax_c, con.rhs, atol=1e-6)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.contourf(xx, yy, feasible.astype(float), levels=[0.5, 1.5], colors=["#4c78a8"], alpha=0.3)
    for con in lp.constraints:
        a = con.lhs.get(xname, 0.0)
        b = con.lhs.get(yname, 0.0)
        if b != 0.0:
            ax.plot(xs, (con.rhs - a * xs) / b, label=con.name, linewidth=1.5)
        elif a != 0.0:
            ax.axvline(con.rhs / a, label=con.name, linewidth=1.5)

    if solution is not None and solution.is_optimal:
        ax.plot(
            solution.value(xname),
            solution.value(yname),
            "o",
            color="#d62728",
            markersize=11,
            label="optimum",
            zorder=5,
        )

    ax.set_xlim(0, x_max)
    ax.set_ylim(0, y_max)
    ax.set_xlabel(xname)
    ax.set_ylabel(yname)
    ax.set_title("Feasible region")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    return fig


def plot_shadow_prices(solution: Solution) -> plt.Figure:
    """Bar chart of per-constraint shadow prices -- the bottleneck map."""
    frame = solution.shadow_price_frame()
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ["#d62728" if binding else "#9ecae1" for binding in frame["binding"]]
    ax.bar(frame["constraint"], frame["shadow_price"], color=colors)
    ax.set_ylabel("shadow price (value per unit relaxed)")
    ax.set_title("Bottleneck map: what each binding constraint is worth")
    ax.axhline(0, color="black", linewidth=0.8)
    fig.tight_layout()
    return fig

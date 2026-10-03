"""Streamlit app: see optimization answer business questions, live.

Four tabs mirror the notebooks -- a feasible-region explorer, the product-mix
solver, the bottleneck (shadow-price) map, and an LP-vs-MILP toggle. Run with
``make run``.
"""

from __future__ import annotations

import streamlit as st

from src.datasets.scenarios import ProductMixScenario, knapsack_scenario, product_mix_scenario
from src.models import knapsack, product_mix
from src.solvers.backends import solve
from src.solvers.problem import VarType
from src.visualisation import plot_feasible_region, plot_shadow_prices

st.set_page_config(page_title="Optimization 101", layout="wide")
st.title("Optimization 101")
st.caption("Your business problem is a set of equations, and a solver already knows the answer.")

tab_region, tab_mix, tab_bottleneck, tab_milp = st.tabs(
    ["Feasible region", "Product mix", "Bottleneck map", "LP vs MILP"]
)


def _edited_scenario() -> ProductMixScenario:
    """Let the user edit the product-mix numbers in the sidebar."""
    base = product_mix_scenario()
    st.sidebar.header("Product mix inputs")
    profit = {
        p: st.sidebar.number_input(f"profit: {p}", value=float(base.profit[p]), step=0.5)
        for p in base.products
    }
    capacity = {
        r: st.sidebar.number_input(f"capacity: {r}", value=float(base.capacity[r]), step=1.0)
        for r in base.resources
    }
    return ProductMixScenario(
        name=base.name,
        products=base.products,
        profit=profit,
        resources=base.resources,
        usage=base.usage,
        capacity=capacity,
        optimum=base.optimum,
        optimal_mix=base.optimal_mix,
        shadow_prices=base.shadow_prices,
    )


scenario = _edited_scenario()
lp = product_mix.build(scenario)
solution = solve(lp, backend="scipy")

with tab_region:
    st.subheader("Drag the constraints, watch the optimal corner move")
    st.pyplot(plot_feasible_region(lp, solution, x_max=12, y_max=12))

with tab_mix:
    st.subheader("The optimal product mix")
    if solution.is_optimal:
        st.metric("maximum profit", f"{solution.objective:.2f}")
        st.dataframe(solution.solution_frame(), hide_index=True)
    else:
        st.error(f"solver status: {solution.status}")

with tab_bottleneck:
    st.subheader("What is each binding constraint worth?")
    if solution.is_optimal:
        st.pyplot(plot_shadow_prices(solution))
        st.dataframe(solution.shadow_price_frame(), hide_index=True)

with tab_milp:
    st.subheader("Toggle integrality: watch the fractional answer snap to the integer optimum")
    ks = knapsack_scenario()
    milp = knapsack.build(ks)
    integer = st.checkbox("require 0/1 (integer) decisions", value=True)
    model = milp if integer else milp.relaxation()
    result = solve(model, backend="scipy")
    st.metric("objective (value packed)", f"{result.objective:.2f}")
    st.metric("solve time", f"{result.solve_time * 1e3:.2f} ms")
    fractional = [
        name for name, value in result.values.items() if abs(value - round(value)) > 1e-6
    ]
    if not integer and fractional:
        st.warning(f"fractional (nonsensical) picks: {', '.join(fractional)}")
    st.caption(
        "Binary variables: "
        + ", ".join(v.name for v in milp.variables if v.kind is VarType.BINARY)
    )
    st.dataframe(result.solution_frame(), hide_index=True)

# Project: optimization_101

> Guidance for Claude Code working in this repo. Keep it accurate as the code changes.
> Status: implemented — `src/` library, 6 executed notebooks, Streamlit app, and tests all
> land; `make ci` (sync + lint + test) and `make notebooks` are green. PRD at
> `.llm/PRD_optimization_101.md`.

## Identity & thesis

A hands-on **101 introduction to mathematical optimization** (a.k.a. mathematical
programming / operations research) for a data-science audience. Educational portfolio
project: 6 Jupyter notebooks + 1 Streamlit app + a small `src/` library.

**The claim the whole repo defends:** a huge class of business decisions (what to make,
what to ship, whom to assign, what to buy) are **not prediction problems** — they are
**constrained optimization problems with a provably optimal answer** a solver returns in
milliseconds. Writing one down is just naming three things: the decisions, the goal, the
rules. And the solver hands back a bonus no ML model ever gives you: the **shadow price**,
which tells you what your binding constraint is worth.

Teaching arc (notebooks build strictly in this order, each resting on the last):
`three ingredients → product mix → shadow prices → integer decisions → classic problems → solver landscape`

This is the **foundation for a planned optimization track**: `supply_chain_arena`, `routing_arena`,
`portfolio_optimization_101`, `constraint_programming_101` (not yet created — do **not** link them
as live repos). Existing sibling projects that this one connects to, under `github.com/Dima806`:
`gradient_descent_101` (iterative/continuous optimization), `threshold_lab` (score → decision),
`metrics_101` (the metric is the objective). The README links these three.

## Non-negotiables (the project's promises — do not break these)

- **All solvers open-source. No commercial solver, no license, no activation step, ever.**
  CBC (via PuLP), HiGHS (via `highspy`, MIT), SciPy (`linprog`/`milp`, HiGHS-backed), OR-Tools.
  Gurobi/CPLEX may be *mentioned* as faster on very large industrial models — never imported,
  never required.
- **Everything runs on a 2-CPU / 8 GB Codespace, no GPU.** Optimization solvers are CPU tools.
  All instances are tiny (tens to low-hundreds of variables); a solve is sub-second.
- **Exactness is the core promise. Every claimed optimum must be checkable.** Verify each
  optimum against brute force or an analytic value; verify each shadow price against a
  finite-difference re-solve. If you can't verify it, don't assert it.
- **`uv` only. No `pip`, no `conda`.**
- **Each notebook runs end-to-end in < 3 minutes** on 2 CPUs. CI executes all of them.
- **Teach by translation, not theory.** Name the decisions, the goal, the rules → call a
  solver. Geometry and code over simplex/duality proofs. The from-scratch simplex is a small
  teaching aid (2–3 vars), not a performance contender.

## Stack

- Python **3.11+**.
- Solvers: `pulp` (bundles open CBC), `highspy` (HiGHS direct, the high-performance option),
  `scipy` (`linprog`/`milp`), `ortools`.
  - **`pulp` is pinned `<4`**: PuLP 4.0 is a Rust rewrite that broke the classic
    `LpVariable(name, lowBound=...)` / `PULP_CBC_CMD` API the notebooks teach. Stay on 3.x.
    Its 4.0 `DeprecationWarning`s are muted in `pyproject.toml`'s `filterwarnings`.
  - **`highspy` and `ortools` cannot share one process.** Both statically bundle HiGHS and
    export its C++ symbols globally; whichever imports first makes the other's import fail
    (any version, either order). `scipy`'s HiGHS is private and clashes with neither. So the
    parent process stays on `scipy`+`pulp`, and the all-backends comparison runs each solve
    in its own subprocess via `backends.compare_backends` / `solve_isolated` (spawn). Never
    `import highspy` and `import ortools` in the same kernel/test process.
- Core: `numpy`, `pandas`, `matplotlib`, `plotly`, `streamlit`, `pydantic`,
  `pydantic-settings`, `pyyaml`.
- Dev: `pytest`, `ruff`, `ty`, `jupyter`, `ipykernel`, `nbconvert`, `nbclient`.
- Config lives in `config/settings.yaml` (solver choice, instance sizes, seeds), loaded via
  `pydantic-settings` through `src/config.py`.

## Repository structure

```
notebooks/   01_three_ingredients  02_the_product_mix  03_shadow_prices
             04_integer_decisions  05_classic_problems  06_solver_landscape
src/
  config.py                      # pydantic-settings over config/settings.yaml
  models/      product_mix.py     # canonical LP: build() + as_matrices() (PuLP & SciPy views)
               blending.py        # diet/blending LP (least-cost mix hitting a spec)
               assignment.py      # assignment + transportation problems
               knapsack.py        # 0/1 knapsack — the canonical MILP
               facility.py        # fixed-charge facility choice (MILP w/ binaries)
  solvers/     problem.py         # LinearProgram IR: Variable/Constraint/Direction/Sense/VarType
               backends.py        # ONE interface over PuLP/CBC, SciPy, HiGHS, OR-Tools (+ compare_backends)
               report.py          # Solution: value, solution, shadow prices, slack, solve time
  simplex/     from_scratch.py    # tiny teaching simplex (<=3 vars), records the corners it walks
  datasets/    scenarios.py       # named business scenarios with KNOWN optima
  evaluation/  optimum_check.py   # solver optimum vs brute force on small instances
               sensitivity.py     # shadow prices + how the optimum moves with the data
  exporters.py                    # save_fig() -> outputs/figures, save_json() -> outputs/data
  visualisation.py                # feasible region + bottleneck-map plots (Agg backend)
app/           streamlit_app.py
tests/         test_models.py  test_shadow_prices.py  test_simplex.py  test_backends.py  test_visualisation.py
config/        settings.yaml
outputs/       figures/ (PNG, generated)   data/ (JSON inputs+results, generated)
```

(Note British spelling `visualisation.py`, matching the PRD. `src/` is installed editable via
the hatchling `[build-system]`, so `import src...` resolves in tests, notebooks, and the app.)

## Key concepts (shared vocabulary — reuse these terms exactly)

- **The three ingredients.** *Decision variables* (the choices you control), *objective*
  (the one quantity to max/min), *constraints* (rules the choices must obey). That's the
  whole field. Every model and notebook frames itself this way.
- **LP vs MILP — the line that matters most.** A **linear program** (continuous variables,
  everything a weighted sum) has a convex feasible region and solves fast to the *global*
  optimum. The moment a variable must be a whole number or binary (build the factory or
  don't), it's a **mixed-integer program**: combinatorially harder, the solver must *search*
  (branch-and-bound). Don't avoid integer decisions you need (→ nonsense like half a factory);
  don't scatter them freely (→ the solver crawls).
- **LP optima sit at a corner (vertex) of the feasible polytope.** The simplex method walks
  corner to corner, improving each step. This is the geometric intuition notebook 01 builds
  and everything else rests on. The solver is not magic.
- **Shadow price (dual value) — the signature deliverable.** For each constraint, how much
  the objective improves if that constraint is relaxed by one unit. It's the *bottleneck map*:
  which resource is binding and what one more unit of it is worth. Valid only within an
  allowable RHS range. **No predictive model produces anything like it.** Always verify it by
  re-solving with the constraint nudged and confirming the objective moved by exactly that much.
- **Binding vs slack.** A binding constraint holds with equality at the optimum (shadow price
  usually > 0); a slack constraint has room left (shadow price 0 — relaxing it buys nothing).
- **The reframe.** "What will happen?" = prediction (ML). "What should we do?" = optimization.
  Recognizing the shape (a goal, decisions, rules) is the single most valuable skill here —
  it redirects the practitioner from an approximate tool to an exact one. Draw this contrast
  explicitly, especially in notebook 02.

## Solver guidance (which backend to reach for)

- **SciPy** `linprog` / `milp` — small LPs/MILPs with zero extra dependency. Matrix form.
- **PuLP + CBC** — the everyday MILP workhorse; readable algebraic modeling.
- **highspy (HiGHS)** — high-performance LP/MILP engine when CBC runs out of road; MIT.
- **OR-Tools** — larger or structured problems.
- `product_mix.py` is built **twice** (PuLP *and* SciPy) on purpose: readers see the same
  problem as readable algebra (`build`) and as raw matrices (`as_matrices`) and learn they're
  identical. `backends.py` lets the *same* `LinearProgram` run through every backend; notebook
  06 shows they agree via `compare_backends` (subprocess-isolated, see the highspy/ortools note).
- Architecture: models build a backend-agnostic `LinearProgram` (`src/solvers/problem.py`);
  backends return a uniform `Solution` (`src/solvers/report.py`) with duals normalised to
  `shadow_price = d(objective)/d(rhs)` so every backend's shadow prices are directly comparable.

## Verification & testing discipline

Tests encode the exactness promise — treat them as specs, not afterthoughts:

- `test_models.py` — each model returns its known optimum.
- `test_shadow_prices.py` — shadow prices match a finite-difference re-solve to tight tolerance.
- `test_simplex.py` — from-scratch simplex matches the library on 2–3 variable LPs.
- `test_backends.py` — all backends agree on a shared model.
- Assignment/knapsack/facility instances are sized so **brute force can confirm** the optimum.
- Scenarios in `datasets/scenarios.py` are seeded and carry their known/brute-forceable optima.
- Compare floats with tolerances, not `==`. Prefer `pytest.approx` / `np.isclose`.

## Workflow (use the Makefile; it wraps `uv run`)

```
make setup      # first-time: install uv if missing, uv sync --all-extras, register kernel
make sync       # uv sync --all-extras
make lint       # format + check + typecheck
make format     # ruff format   src/ tests/ app/
make check      # ruff check --fix   src/ tests/ app/
make typecheck  # ty check src/
make test       # pytest
make test-cov   # pytest with coverage
make notebooks  # execute all notebooks IN PLACE (nbconvert --inplace, 180s each; regenerates outputs/)
make run        # streamlit run app/streamlit_app.py (port 8501)
make lab        # jupyter lab
make ci         # sync + lint + test      (what CI runs)
make dev        # lint + test             (fast local loop)
```

Run `make lint && make test` before declaring anything done. **Target: zero lint errors,
all tests pass.** Don't invoke `python`/`pytest`/`ruff` directly — go through `uv run` or the
Makefile so the locked environment is used.

## Conventions

- **ruff**: line-length 99, `target-version = py311`, lint select
  `E F W I UP N B A SIM PTH`, `E501` ignored. Let ruff format and fix; don't hand-fight it.
- **Types**: `ty check src/` must pass. Type public functions in `src/`; annotate what models
  and the report helper return.
- **Config** flows through `src/config.py` (pydantic-settings) — don't hardcode seeds, sizes,
  or solver choice; read them from `config/settings.yaml`.
- **Models** return a consistent report object (optimal value, solution, shadow prices, slack,
  solve time) via `solvers/report.py` — keep that shape uniform across models and backends.
- **Artifacts:** notebooks save figures with `exporters.save_fig(fig, "NN_name")` → `outputs/figures/*.png`
  and inputs + numerical results with `exporters.save_json(data, "NN_name")` → `outputs/data/*.json`.
  Paths are repo-root-anchored, so they work whether run from `notebooks/` or the root. Plots come
  from `visualisation.py` (matplotlib Agg); notebooks display the saved PNG via `IPython.display.Image`.
- License headers / project license: **Apache-2.0**.

## Notebook rules

- Build in the fixed order above; each notebook assumes the prior one's intuition.
- Notebook 01 uses **no solver** — pure geometry (feasible polygon, sliding objective line,
  optimum at a corner). Notebook 02 is the showpiece (product mix, PuLP + SciPy, verified).
  Notebook 03 is the shadow-price payoff. Notebook 04 draws the LP/MILP line (show the
  fractional relaxation giving nonsense, then branch-and-bound finding the integer optimum).
  Notebook 05 is the pattern library (blending, assignment, transportation) with forward links.
  Notebook 06 is the solver tour: from-scratch simplex, then `compare_backends` (subprocess-isolated,
  so highspy+ortools never share the kernel) showing all four agree.
- Keep each < 3 min; verify every optimum shown (assert against the known/brute-force value in-cell);
  seed everything; no external data, no downloads. Save every figure as PNG and every input/result as
  JSON via `exporters` (see Conventions). Notebooks are committed **with outputs**; `make notebooks`
  re-executes them in place.
- Notebook 06 (or any cell) must **not** `import highspy` and `import ortools` into the same kernel —
  use `compare_backends` for multi-backend work; in-kernel single solves should use scipy or pulp.

## Streamlit app (4 tabs)

1. Feasible-region explorer — drag a 2-var LP's constraints, watch polygon + optimal corner move.
2. Product-mix solver — edit profits/limits, see optimal mix, total profit, every shadow price live.
3. Bottleneck map — bar chart of shadow prices flagging the binding constraint and its value.
4. LP-vs-MILP demo — toggle the integer constraint, watch the fractional answer snap to the
   integer optimum, with solve time shown.

## Working style / token rules

- **Caveman mode: ON.** Code-first, terse prose. (See `.claude/skills/caveman.md` if present.)
- `/compact` at notebook boundaries.
- Prefer showing working code over explaining it; the PRD already carries the narrative.

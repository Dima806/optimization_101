"""Save notebook artifacts: figures as PNG, inputs and results as JSON.

Every number and every figure it shows is written to
``outputs/`` so the repo share one source of truth: figures land
in ``outputs/figures/`` and data in ``outputs/data/``. Paths are anchored to the
repo root, so a notebook writes to the same place whether it is run from the
``notebooks/`` directory (``make notebooks``) or the project root.
"""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from matplotlib.figure import Figure

_ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = _ROOT / "outputs" / "figures"
DATA_DIR = _ROOT / "outputs" / "data"


def _ensure_dirs() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def save_fig(fig: Figure, name: str, dpi: int = 150) -> Path:
    """Write a matplotlib figure to ``outputs/figures/<name>.png`` and return its path."""
    _ensure_dirs()
    path = FIGURES_DIR / f"{name}.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path


def _json_default(obj: Any) -> Any:
    """Make dataclasses, sets/tuples and numpy scalars JSON-serialisable."""
    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj)
    if isinstance(obj, set | frozenset | tuple):
        return list(obj)
    item = getattr(obj, "item", None)  # numpy scalars expose .item()
    if callable(item):
        return item()
    raise TypeError(f"not JSON-serialisable: {type(obj)!r}")


def save_json(data: Any, name: str, indent: int = 2) -> Path:
    """Write a dict/structure to ``outputs/data/<name>.json`` and return its path."""
    _ensure_dirs()
    path = DATA_DIR / f"{name}.json"
    path.write_text(json.dumps(data, indent=indent, default=_json_default))
    return path

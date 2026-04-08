# algorithms/ga/__init__.py
"""
GA package exports & compatibility shims.

Goals
-----
- Keep stable entry points regardless of internal file names.
- Avoid importing heavy submodules until actually used (lazy exports).
- Preserve back-compat aliases for notebooks / older code.
"""

from __future__ import annotations

import importlib
import warnings
from typing import TYPE_CHECKING

# Toggle to emit a gentle warning when a legacy alias is accessed.
_ALIAS_WARNINGS = False

# Canonical exports (public API)
_EXPORTS = {
    "run_ga_flat": ("flat", "run_ga_flat"),
    "run_ga_equivclass_exhaustive": ("equivclass_exhaustive", "run_ga_equivclass_exhaustive"),
    "run_ga_equivclass_joint": ("equivclass_joint", "run_ga_equivclass_joint"),
}

# Back-compat function aliases
_ALIASES = {
    "maximize_by_class_combinations": "run_ga_equivclass_exhaustive",
    "run_joint_ga": "run_ga_equivclass_joint",
}

__all__ = list(_EXPORTS.keys()) + list(_ALIASES.keys())


def __getattr__(name: str):
    """
    Lazy attribute loader (PEP 562). Imports the submodule only when an exported
    symbol is first accessed. Also resolves legacy aliases without eager imports.
    """
    # Resolve aliases to their canonical names
    canonical = _ALIASES.get(name, name)

    if canonical not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    if name in _ALIASES and _ALIAS_WARNINGS:
        warnings.warn(
            f"'{name}' is a compatibility alias for '{canonical}'. "
            f"Prefer '{canonical}' going forward.",
            DeprecationWarning,
            stacklevel=2,
        )

    mod_name, attr_name = _EXPORTS[canonical]
    try:
        # Import the submodule relative to this package (e.g., algorithms.ga.flat)
        module = importlib.import_module(f"{__name__}.{mod_name}")
    except ImportError as e:
        # Provide a clearer message when optional deps (e.g., DEAP) are missing
        raise ImportError(
            f"Failed to import '{canonical}' from '{mod_name}'. "
            f"This algorithm may require optional dependencies (e.g., 'deap'). "
            f"Install missing packages and retry. Original error: {e}"
        ) from e

    obj = getattr(module, attr_name)

    # Cache on the module to avoid repeated imports / lookups
    globals()[canonical] = obj
    if name != canonical:  # also cache the alias binding
        globals()[name] = obj

    return obj


def __dir__():
    return sorted(__all__)


# Static typing support: make symbols visible to type checkers without
# importing submodules at runtime.
if TYPE_CHECKING:  # pragma: no cover
    from .flat import run_ga_flat as run_ga_flat
    from .equivclass_exhaustive import (
        run_ga_equivclass_exhaustive as run_ga_equivclass_exhaustive,
    )
    from .equivclass_joint import run_ga_equivclass_joint as run_ga_equivclass_joint

    # Back-compat aliases
    maximize_by_class_combinations = run_ga_equivclass_exhaustive
    run_joint_ga = run_ga_equivclass_joint

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Hyperparameter Candidate Generator (LHS + Discrete Mesh)

This script builds per-algorithm hyperparameter candidate files from a single
experiment plan JSON (e.g., ft2_plan.json). It supports two complementary
sampling modes per parameter:

1) Continuous ranges (Latin Hypercube Sampling, LHS)
   - A parameter specified as a two-length list `[low, high]` is interpreted as
     a closed interval. The generator draws `n_samples_per_algo` points in the
     unit hypercube and maps them to the provided ranges.

2) Discrete grids (Cartesian product / mesh)
   - A parameter specified as an object `{"grid": [v1, v2, ...]}` is treated as
     a finite set. If *all* parameters for an algorithm are given as grids, the
     generator builds the full Cartesian product.
   - If only a subset is declared as grid and the rest as ranges, this script
     defaults to LHS on the ranged subset and *ignores* the grid subset for the
     mixed case to avoid combinatorial blow-up (see rationale below). Users are
     encouraged to use either all-grid or all-range per algorithm for clarity.

Output files are written under a `configs/` sibling directory:

    - configs/candidates_<algo>.json

Each output is a JSON object with the shape:

    {
      "algo": "<algo-name>",
      "candidates": [
        {"id": 1, "hparams": {...}},
        {"id": 2, "hparams": {...}},
        ...
      ]
    }

Special handling for PSO:
------------------------
When all PSO parameters are ranges, the script samples in a reparameterized
space `(w, phi, balance)` where `phi := c1 + c2` is constrained to [2, 4]
(soft), and `balance ∈ [0.25, 0.75]`. Then `c1 := balance * phi`,
`c2 := (1 - balance) * phi`, and both are clipped to user-provided hard bounds
`[c1_low, c1_high]`, `[c2_low, c2_high]`. This improves coverage under the
soft sum constraint 2.0 ≤ c1 + c2 ≤ 4.0 without ad-hoc rejection.

Deterministic capping:
----------------------
If a pure grid (Cartesian product) yields more than `n_samples_per_algo`
combinations, the script performs a deterministic sub-sampling without
replacement using a seed from the plan (or 42 by default).

Design decisions:
-----------------
- The generator is intentionally *conservative* with mixed grid+range setups.
  Supporting hybrid “LHS × partial-grid expansion” is possible, but it tends to
  explode combinatorially and complicates accounting. The current design keeps
  behavior predictable: either all-grid (mesh) or all-range (LHS).
- Parameters are emitted in a stable, lexicographic order of names to ensure
  reproducible mapping from vectors to dictionaries across runs.

Usage:
------
    python candidates_generator.py --plan /path/to/ft2_plan.json

The plan must define:
- seeds: list of ints (only used for reproducible sub-sampling)
- n_samples_per_algo: int
- bounds: mapping algo_name -> mapping param -> spec
  where spec is either [low, high] or {"grid": [...]}
- Optional: algorithms: list of algo_names to generate
- Optional: seed: int (top-level, used for deterministic choices)

Example parameter specs:
  "generic": {
    "cxpb": [0.01, 0.99],
    "mutpb": [0.01, 0.50],
    "mutation_rate": [0.01, 0.20]
  }

  "macro_micro": {
    "tourn_size": {"grid": [2,3,5]},
    "lambda_in":  {"grid": [0.10,0.25,0.40]},
    "lambda_out": {"grid": [0.30,0.50,0.70]},
    "sel_mutation":  {"grid": [0.005,0.01,0.015]},
    "tail_mutation": {"grid": [0.005,0.01,0.015]}
  }
"""

from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

# Optional SciPy LHS; falls back to uniform if not available.
try:
    from scipy.stats import qmc
    _HAS_SCIPY = True
except Exception:  # pragma: no cover
    _HAS_SCIPY = False


# =============================================================================
# Core utilities
# =============================================================================

def _lhs(n: int, d: int, seed: int) -> np.ndarray:
    """
    Draws LHS samples in [0, 1]^d. Falls back to uniform sampling when SciPy
    is unavailable.

    Parameters
    ----------
    n : int
        Number of samples to draw.
    d : int
        Dimension of the design space.
    seed : int
        Seed for reproducibility.

    Returns
    -------
    np.ndarray
        Array of shape (n, d) with values in [0, 1).
    """
    rng = np.random.default_rng(seed)
    if _HAS_SCIPY:
        sampler = qmc.LatinHypercube(d=d, seed=rng)
        return sampler.random(n=n)
    return rng.random((n, d))


def _map_to_bounds(unit: np.ndarray, low: List[float], high: List[float]) -> np.ndarray:
    """
    Affine maps unit-hypercube samples to the hyper-rectangle defined by `low`
    and `high`.

    Parameters
    ----------
    unit : np.ndarray
        Unit samples of shape (n, d).
    low : List[float]
        Lower bounds per dimension.
    high : List[float]
        Upper bounds per dimension.

    Returns
    -------
    np.ndarray
        Mapped samples of shape (n, d).
    """
    l = np.asarray(low, dtype=float)
    h = np.asarray(high, dtype=float)
    return l + unit * (h - l)


def _save_json(path: Path, payload: Dict[str, Any]) -> None:
    """
    Persists a JSON payload with UTF-8 encoding, creating parent directories
    as needed.

    Parameters
    ----------
    path : Path
        Destination file path.
    payload : Dict[str, Any]
        JSON-serializable object.

    Returns
    -------
    None
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _is_grid_spec(spec: Any) -> bool:
    """
    Checks whether a parameter specification is a discrete grid.

    Parameters
    ----------
    spec : Any
        Parameter specification.

    Returns
    -------
    bool
        True iff `spec` is a dict with a list under key "grid".
    """
    return isinstance(spec, dict) and "grid" in spec and isinstance(spec["grid"], list)


def _is_range_spec(spec: Any) -> bool:
    """
    Checks whether a parameter specification is a numeric range.

    Parameters
    ----------
    spec : Any
        Parameter specification.

    Returns
    -------
    bool
        True iff `spec` is a length-2 list/tuple of numbers.
    """
    if isinstance(spec, (list, tuple)) and len(spec) == 2:
        try:
            float(spec[0]); float(spec[1])
            return True
        except Exception:
            return False
    return False


def _validate_algo_bounds(algo: str, bounds: Dict[str, Any]) -> Tuple[List[str], bool, bool]:
    """
    Validates the per-algorithm bounds and determines whether the configuration
    is all-grid, all-range, or mixed.

    Parameters
    ----------
    algo : str
        Algorithm name (used for error messages).
    bounds : Dict[str, Any]
        Mapping parameter name -> spec.

    Returns
    -------
    Tuple[List[str], bool, bool]
        (ordered parameter names, is_all_grid, is_all_range)

    Raises
    ------
    ValueError
        If a parameter specification is neither a valid range nor a grid.
    """
    names = sorted(bounds.keys())
    has_grid = False
    has_range = False
    for k in names:
        spec = bounds[k]
        if _is_grid_spec(spec):
            has_grid = True
        elif _is_range_spec(spec):
            has_range = True
        else:
            raise ValueError(
                f"[{algo}] Unsupported spec for parameter '{k}': {spec!r}. "
                "Expected either a 2-length [low, high] range or {'grid': [...]}."
            )
    return names, has_grid and not has_range, has_range and not has_grid


def _deterministic_subsample(
    items: List[Dict[str, Any]],
    k: int,
    seed: int
) -> List[Dict[str, Any]]:
    """
    Deterministically subsamples without replacement when the mesh cardinality
    exceeds the requested number of samples.

    Parameters
    ----------
    items : List[Dict[str, Any]]
        Full list of candidate dicts (with 'hparams' inside).
    k : int
        Target number of items.
    seed : int
        Seed to ensure reproducibility.

    Returns
    -------
    List[Dict[str, Any]]
        Subsampled list of items of size k (or the original list if len ≤ k).
    """
    if len(items) <= k:
        return items
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(items), size=k, replace=False)
    idx.sort()
    return [items[i] for i in idx]


# =============================================================================
# Algorithm-specific generators
# =============================================================================

def _gen_mesh(bounds: Dict[str, Any], algo: str, seed: int, cap: int | None) -> Dict[str, Any]:
    """
    Generates candidates from a pure grid specification via Cartesian product.

    Parameters
    ----------
    bounds : Dict[str, Any]
        Mapping of parameter -> {'grid': [...]}.
    algo : str
        Algorithm name (for the output envelope).
    seed : int
        Seed used only if downsampling is required.
    cap : Optional[int]
        Maximum number of candidates to emit (None means no cap).

    Returns
    -------
    Dict[str, Any]
        {"algo": <algo>, "candidates": [{"id": 1, "hparams": {...}}, ...]}
    """
    names = sorted(bounds.keys())
    grids: List[List[float]] = []
    for k in names:
        grid = bounds[k]["grid"]
        # Allow integers/floats; cast to float for consistency.
        grids.append([float(v) for v in grid])

    # Cartesian product in a fixed parameter order.
    combos = product(*grids)
    candidates: List[Dict[str, Any]] = []
    cid = 1
    for tpl in combos:
        hp = {k: v for k, v in zip(names, tpl)}
        candidates.append({"id": cid, "hparams": hp})
        cid += 1

    # Deterministic downsampling if needed.
    if cap is not None and len(candidates) > cap:
        candidates = _deterministic_subsample(candidates, cap, seed)

    return {"algo": algo, "candidates": candidates}


def _gen_lhs_generic(bounds: Dict[str, Any], n: int, seed: int, algo: str) -> Dict[str, Any]:
    """
    Generates LHS candidates for algorithms with only range parameters.

    Parameters
    ----------
    bounds : Dict[str, Any]
        Mapping parameter -> [low, high].
    n : int
        Number of samples.
    seed : int
        RNG seed.
    algo : str
        Algorithm name.

    Returns
    -------
    Dict[str, Any]
        Envelope with "algo" and "candidates".
    """
    names = sorted(bounds.keys())
    low = [float(bounds[k][0]) for k in names]
    high = [float(bounds[k][1]) for k in names]
    unit = _lhs(n, d=len(names), seed=seed)
    vals = _map_to_bounds(unit, low, high)
    cands = [{"id": i + 1,
              "hparams": {k: float(v) for k, v in zip(names, row)}}
             for i, row in enumerate(vals)]
    return {"algo": algo, "candidates": cands}


def _gen_lhs_pso(bounds: Dict[str, Any], n: int, seed: int) -> Dict[str, Any]:
    """
    Generates PSO candidates from ranges using a reparameterization that keeps
    c1 + c2 in a soft interval while honoring hard clipping.

    Parameters
    ----------
    bounds : Dict[str, Any]
        Must contain at least ranges for 'w', 'c1', 'c2'.
        Optional 'c_sum_soft' range constrains phi := c1 + c2 (default [2, 4]).
    n : int
        Number of samples.
    seed : int
        RNG seed.

    Returns
    -------
    Dict[str, Any]
        Envelope with "algo": "pso" and "candidates".
    """
    # Extract ranges and soft constraint.
    w_low, w_high = map(float, bounds["w"])
    c1_low, c1_high = map(float, bounds["c1"])
    c2_low, c2_high = map(float, bounds["c2"])
    phi_low, phi_high = map(float, bounds.get("c_sum_soft", [2.0, 4.0]))

    # Sample in (w, phi, balance); keep balance away from extremes for stability.
    unit = _lhs(n, d=3, seed=seed + 33)
    bal_low, bal_high = 0.25, 0.75

    mapped = np.column_stack([
        w_low + unit[:, 0] * (w_high - w_low),          # w
        phi_low + unit[:, 1] * (phi_high - phi_low),    # phi = c1 + c2
        bal_low + unit[:, 2] * (bal_high - bal_low)     # balance
    ])

    candidates: List[Dict[str, Any]] = []
    for i, (w, phi, bal) in enumerate(mapped, start=1):
        c1 = float(np.clip(bal * phi, c1_low, c1_high))
        c2 = float(np.clip((1.0 - bal) * phi, c2_low, c2_high))
        candidates.append({"id": i, "hparams": {"w": float(w), "c1": c1, "c2": c2}})

    return {"algo": "pso", "candidates": candidates}


def _gen_for_algo(
    algo: str,
    bounds: Dict[str, Any],
    n_samples: int,
    seed: int
) -> Dict[str, Any]:
    """
    Dispatches to the appropriate generator depending on the parameter specs.

    Parameters
    ----------
    algo : str
        Algorithm name (as appears in the plan).
    bounds : Dict[str, Any]
        Mapping parameter -> spec (either range or grid).
    n_samples : int
        Number of LHS samples or mesh cap for grids.
    seed : int
        Seed for reproducible downsampling or LHS.

    Returns
    -------
    Dict[str, Any]
        Envelope {"algo": algo, "candidates": [...]}
    """
    _, is_all_grid, is_all_range = _validate_algo_bounds(algo, bounds)

    # Pure grid: Cartesian product with deterministic capping if needed.
    if is_all_grid:
        return _gen_mesh(bounds=bounds, algo=algo, seed=seed, cap=n_samples)

    # Pure ranges: LHS. Use a specialized path for "pso" to honor c1+c2 soft constraints.
    if is_all_range:
        if algo.lower() == "pso":
            return _gen_lhs_pso(bounds=bounds, n=n_samples, seed=seed)
        return _gen_lhs_generic(bounds=bounds, n=n_samples, seed=seed, algo=algo)

    # Mixed specifications are intentionally not expanded to avoid combinatorial growth.
    # Users should prefer either all-range (LHS) or all-grid (mesh) per algorithm.
    raise ValueError(
        f"[{algo}] Mixed parameter specification detected (both grids and ranges). "
        "Please convert parameters so they are either all ranges or all grids for this algorithm."
    )


# =============================================================================
# CLI
# =============================================================================

def main() -> None:
    """
    CLI entry point. Reads a plan JSON and emits per-algorithm candidate files
    under a sibling 'configs/' directory.

    Required plan fields:
      - n_samples_per_algo : int
      - bounds : dict

    Optional plan fields:
      - algorithms : list[str] (defaults to sorted(bounds.keys()))
      - seed : int (defaults to 42)

    Returns
    -------
    None
    """
    ap = argparse.ArgumentParser(
        description="Generate per-algorithm hyperparameter candidates (LHS or mesh) from a single plan JSON."
    )
    ap.add_argument("--plan", type=Path, required=True, help="Path to ft2_plan.json.")
    args = ap.parse_args()

    plan = json.loads(args.plan.read_text(encoding="utf-8"))

    # Top-level controls
    n_samples = int(plan["n_samples_per_algo"])
    seed = int(plan.get("seed", 42))

    # Determine which algorithms to emit.
    bounds_all: Dict[str, Any] = plan["bounds"]
    algos: List[str] = [str(a) for a in plan.get("algorithms", sorted(bounds_all.keys()))]

    # Output directory colocated with this script: ../configs
    out_dir = Path(__file__).resolve().parents[1] / "configs"

    emitted: List[str] = []
    for algo in algos:
        if algo not in bounds_all:
            raise KeyError(f"Algorithm '{algo}' listed in 'algorithms' but missing under 'bounds'.")
        bundle = _gen_for_algo(
            algo=algo,
            bounds=bounds_all[algo],
            n_samples=n_samples,
            seed=seed
        )
        _save_json(out_dir / f"candidates_{algo}.json", bundle)
        emitted.append(algo)

    # Console summary
    if emitted:
        print("Candidate files created in configs/:")
        for a in emitted:
            print(f"- candidates_{a}.json")
    else:
        print("No candidates were generated (empty algorithm list).")


if __name__ == "__main__":
    main()

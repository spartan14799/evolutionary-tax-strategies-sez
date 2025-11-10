#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generates Latin Hypercube Samples (LHS) of hyperparameters per algorithm
from a single experiment plan JSON (ft2_plan.json).

Outputs:
- configs/candidates_generic.json
- configs/candidates_pso.json
- configs/candidates_joint.json

PSO sampling enforces the soft sum constraint (2.0 ≤ c1+c2 ≤ 4.0) by
sampling in (phi, balance): phi ∈ [2,4], balance ∈ [0.25,0.75], then
c1 = balance*phi, c2 = (1-balance)*phi. Both are finally clipped into
[0.80, 2.20] as requested.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

# Try SciPy LHS; fallback to uniform if not available
try:
    from scipy.stats import qmc
    _HAS_SCIPY = True
except Exception:  # pragma: no cover
    _HAS_SCIPY = False


def _lhs(n: int, d: int, seed: int) -> np.ndarray:
    """
    Draws LHS samples in [0,1]^d. Falls back to uniform if SciPy is missing.

    Args:
        n (int): Number of samples.
        d (int): Dimension.
        seed (int): RNG seed.

    Returns:
        np.ndarray: Array (n, d) with values in [0,1).
    """
    rng = np.random.default_rng(seed)
    if _HAS_SCIPY:
        sampler = qmc.LatinHypercube(d=d, seed=rng)
        return sampler.random(n=n)
    return rng.random((n, d))


def _map_to_bounds(unit: np.ndarray, low: List[float], high: List[float]) -> np.ndarray:
    """
    Maps unit-hypercube samples to [low[i], high[i]] intervals.

    Args:
        unit (np.ndarray): Unit samples (n, d).
        low (List[float]): Lower bounds per dimension.
        high (List[float]): Upper bounds per dimension.

    Returns:
        np.ndarray: Mapped samples (n, d).
    """
    low = np.asarray(low, dtype=float)
    high = np.asarray(high, dtype=float)
    return low + unit * (high - low)


def _save_json(path: Path, payload: Dict) -> None:
    """
    Writes a JSON payload to disk with UTF-8 encoding.

    Args:
        path (Path): Output path.
        payload (Dict): JSON-compatible dictionary.

    Returns:
        None
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _gen_generic(bounds: Dict[str, List[float]], n: int, seed: int) -> Dict:
    """
    Generates generic GA candidates via LHS.

    Args:
        bounds (Dict[str, List[float]]): Bounds for cxpb, mutpb, mutation_rate.
        n (int): Number of candidates.
        seed (int): RNG seed.

    Returns:
        Dict: {"algo": "generic", "candidates": [{"id": 1, "hparams": {...}}, ...]}
    """
    names = ["cxpb", "mutpb", "mutation_rate"]
    low = [bounds[k][0] for k in names]
    high = [bounds[k][1] for k in names]
    unit = _lhs(n, d=len(names), seed=seed)
    vals = _map_to_bounds(unit, low, high)
    cands = [{"id": i + 1, "hparams": {k: float(v) for k, v in zip(names, row)}} for i, row in enumerate(vals)]
    return {"algo": "generic", "candidates": cands}


def _gen_joint(bounds: Dict[str, List[float]], n: int, seed: int) -> Dict:
    """
    Generates joint GA candidates via LHS.

    Args:
        bounds (Dict[str, List[float]]): Bounds for parents_rate, sel_mutation, tail_mutation.
        n (int): Number of candidates.
        seed (int): RNG seed.

    Returns:
        Dict: {"algo": "joint", "candidates": [{"id": 1, "hparams": {...}}, ...]}
    """
    names = ["parents_rate", "sel_mutation", "tail_mutation"]
    low = [bounds[k][0] for k in names]
    high = [bounds[k][1] for k in names]
    unit = _lhs(n, d=len(names), seed=seed + 17)  # offset seed per algo
    vals = _map_to_bounds(unit, low, high)
    cands = [{"id": i + 1, "hparams": {k: float(v) for k, v in zip(names, row)}} for i, row in enumerate(vals)]
    return {"algo": "joint", "candidates": cands}


def _gen_pso(bounds: Dict[str, List[float]], n: int, seed: int) -> Dict:
    """
    Generates PSO candidates using a reparameterization that respects the soft sum constraint.

    Args:
        bounds (Dict[str, List[float]]): Bounds for w, c1, c2 and c_sum_soft.
        n (int): Number of candidates.
        seed (int): RNG seed.

    Returns:
        Dict: {"algo": "pso", "candidates": [{"id": 1, "hparams": {...}}, ...]}
    """
    rng = np.random.default_rng(seed + 33)
    # 3D LHS: (w, phi, balance)
    unit = _lhs(n, d=3, seed=seed + 33)
    w_low, w_high = bounds["w"]
    phi_low, phi_high = bounds.get("c_sum_soft", [2.0, 4.0])
    # Restrict balance to avoid extreme splits (keeps c1,c2 within [0.8,2.2] more often)
    bal_low, bal_high = 0.25, 0.75

    mapped = np.column_stack([
        w_low + unit[:, 0] * (w_high - w_low),          # w
        phi_low + unit[:, 1] * (phi_high - phi_low),    # phi = c1 + c2
        bal_low + unit[:, 2] * (bal_high - bal_low)     # balance
    ])

    cands = []
    for i, (w, phi, bal) in enumerate(mapped, start=1):
        c1 = bal * phi
        c2 = (1.0 - bal) * phi
        # Clip to hard bounds (requested)
        c1 = float(np.clip(c1, bounds["c1"][0], bounds["c1"][1]))
        c2 = float(np.clip(c2, bounds["c2"][0], bounds["c2"][1]))
        # Soft constraint: keep phi in [2,4] but do not over-repair; accept minor deviations
        cand = {"id": i, "hparams": {"w": float(w), "c1": c1, "c2": c2}}
        cands.append(cand)

    return {"algo": "pso", "candidates": cands}


def main() -> None:
    """
    CLI entry point that reads ft2_plan.json and writes per-algorithm candidate files.

    Args:
        None

    Returns:
        None
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", type=Path, required=True, help="Path to ft2_plan.json.")
    args = ap.parse_args()

    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    n = int(plan["n_samples_per_algo"])
    seed = int(plan.get("seed", 42))  # optional top-level seed

    out_dir = Path(__file__).resolve().parents[1] / "configs"
    bounds = plan["bounds"]

    generic = _gen_generic(bounds["generic"], n=n, seed=seed)
    joint = _gen_joint(bounds["joint"], n=n, seed=seed)
    pso = _gen_pso(bounds["pso"], n=n, seed=seed)

    _save_json(out_dir / "candidates_generic.json", generic)
    _save_json(out_dir / "candidates_joint.json", joint)
    _save_json(out_dir / "candidates_pso.json", pso)

    print("Candidate files created in configs/:")
    print("- candidates_generic.json")
    print("- candidates_joint.json")
    print("- candidates_pso.json")


if __name__ == "__main__":
    main()

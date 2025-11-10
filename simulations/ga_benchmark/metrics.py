# simulations/ga_benchmark/metrics.py
# =============================================================================
# Metrics helpers for FTZ_EvoBench
# -----------------------------------------------------------------------------
# This module computes:
#   - HR(B): hit-rate under evaluation budget B
#   - Time-to-hit (in evaluations) and its ECDF
#   - Anytime curves Pr[hit | B] on a budget grid
#   - Regret at close for misses
# It also derives graph/price complexity tags per environment.
# =============================================================================

from __future__ import annotations

from typing import Dict, List, Tuple, Any
import math
import numpy as np
import networkx as nx
from collections import defaultdict, Counter
from math import log

# -----------------------------------------------------------------------------
# Utilities to accumulate evaluation counts per generation
# -----------------------------------------------------------------------------
def cumulative_evals_from_meta(meta: Dict[str, Any]) -> List[int]:
    """
    Builds a cumulative evaluations series aligned with curves['best'].
    Contract:
      meta['evals_per_gen'] must be a list of ints of length == len(curves.best).
      gen 0 usually equals popsize; subsequent gens are #invalid re-evaluated.
    """
    per_gen = list(map(int, meta.get("evals_per_gen", [])))
    if not per_gen:
        # Fallback: assume popsize at gen0, then popsize per generation
        L = int(meta.get("generations", 0)) + 1
        pop = int(meta.get("popsize", 0)) or 1
        per_gen = [pop] + [pop] * (L - 1)

    cum = []
    s = 0
    for e in per_gen:
        s += int(e)
        cum.append(s)
    return cum

# -----------------------------------------------------------------------------
# Golden best selection
# -----------------------------------------------------------------------------
def compute_golden_best(
    runs: List[Dict[str, Any]],
    prefer_algo: str | None = "exhaustive"
) -> Dict[str, float]:
    """
    For each environment_id, returns U*:
      - If `prefer_algo` is present for that environment, use its max best_utility
      - Otherwise, use the max best_utility across all algos/seeds
    Assumes each run has: env_id, algorithm, result['best_utility'].
    """
    golden: Dict[str, float] = {}
    by_env: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in runs:
        by_env[str(r["env_id"])].append(r)

    for env_id, group in by_env.items():
        u_candidates = []
        if prefer_algo:
            u_candidates = [g["result"]["best_utility"] for g in group
                            if g["algorithm"].lower() == prefer_algo.lower()]
        if not u_candidates:
            u_candidates = [g["result"]["best_utility"] for g in group]
        golden[env_id] = float(max(u_candidates)) if u_candidates else float("-inf")
    return golden

# -----------------------------------------------------------------------------
# Hit / time-to-hit / regret
# -----------------------------------------------------------------------------
def summarize_run_against_golden(
    run: Dict[str, Any],
    u_star: float,
    epsilon: float
) -> Dict[str, Any]:
    """
    Computes hit, hit_eval, regret for a single run.
    hit_eval is the cumulative eval count at the first time curves.best >= (1-eps)*U*.
    """
    best_curve = run["result"].get("curves", {}).get("best", [])
    meta = run["result"].get("meta", {})
    evals_cum = cumulative_evals_from_meta(meta)

    target = u_star * (1.0 - float(epsilon))
    hit_idx = -1
    for i, val in enumerate(best_curve):
        if float(val) >= target:
            hit_idx = i
            break

    hit = int(hit_idx >= 0)
    hit_eval = int(evals_cum[hit_idx]) if hit else None
    final_best = float(best_curve[-1]) if best_curve else float("nan")
    regret = max(0.0, float(u_star) - float(final_best)) if hit == 0 else 0.0

    return {
        "hit": hit,
        "hit_eval": hit_eval,
        "regret": regret,
        "final_best": final_best,
        "evals_total": int(evals_cum[-1]) if evals_cum else 0,
    }

# -----------------------------------------------------------------------------
# Anytime success probability on a budget grid
# -----------------------------------------------------------------------------
def anytime_success(
    per_algo_hits: Dict[str, List[int]],
    per_algo_hit_evals: Dict[str, List[int | None]],
    budgets: List[int],
) -> List[Dict[str, Any]]:
    """
    For each algorithm and each budget B, computes Pr[hit | B]:
      numerator = #runs with hit_eval <= B
      denominator = #runs
    """
    rows: List[Dict[str, Any]] = []
    for algo, hits in per_algo_hits.items():
        evals = per_algo_hit_evals[algo]
        n = len(hits)
        for B in budgets:
            ok = sum(1 for h, he in zip(hits, evals) if h == 1 and he is not None and he <= B)
            rows.append({
                "algorithm": algo,
                "budget_evals": int(B),
                "success_rate": ok / n if n > 0 else float("nan"),
                "runs": n,
            })
    return rows

# -----------------------------------------------------------------------------
# ECDF of time-to-hit
# -----------------------------------------------------------------------------
def ecdf_time_to_hit(
    hit_evals: List[int | None]
) -> List[Tuple[int, float]]:
    """
    Returns a discrete ECDF of time-to-hit. Non-hits are ignored (as standard).
    """
    xs = sorted([int(x) for x in hit_evals if x is not None])
    if not xs:
        return []
    n = len(xs)
    pairs = []
    for i, v in enumerate(xs, start=1):
        pairs.append((v, i / n))
    return pairs

# -----------------------------------------------------------------------------
# Graph / price complexity tags
# -----------------------------------------------------------------------------
def env_complexity_tags(
    edges: List[Tuple[str, str]],
    pmatrix: np.ndarray,
    K: int | None = None,
    alphabet_size: int | None = None
) -> Dict[str, float | int]:
    """
    Computes basic complexity descriptors:
      - N: #nodes
      - E: #edges
      - density (DAG treated as simple directed)
      - depth (longest path length)
      - avg_outdeg, max_outdeg
      - branching_factor ≈ avg_outdeg over non-sinks
      - price stats: var, cv, entropy (discrete over normalized values), cond_ratio, skewness
      - K (optional) and |S| (optional) if alphabet_size is given
    """
    G = nx.DiGraph()
    G.add_edges_from(edges)

    nodes = list(G.nodes())
    N = len(nodes)
    E = G.number_of_edges()

    # Longest path length (depth)
    try:
        depth = nx.dag_longest_path_length(G)
    except Exception:
        depth = 0

    # Degrees
    outd = [G.out_degree(n) for n in nodes]
    avg_outdeg = float(np.mean(outd)) if outd else 0.0
    max_outdeg = int(np.max(outd)) if outd else 0
    non_sinks = [d for d in outd if d > 0]
    branching = float(np.mean(non_sinks)) if non_sinks else 0.0

    # Directed density (use N*(N-1) as denominator if N>1)
    dens = (2 * float(E)) / (N * (N - 1)) if N > 1 else 0.0

    # Price tensor stats
    flat = pmatrix.astype(float).ravel()
    m = float(np.mean(flat)) if flat.size else 0.0
    var = float(np.var(flat)) if flat.size else 0.0
    cv = float(np.std(flat) / m) if flat.size and m != 0 else 0.0
    # condition ratio as max/min (guard against zeros)
    vmax = float(np.max(flat)) if flat.size else 0.0
    vmin = float(np.min(flat)) if flat.size else 0.0
    cond = float(vmax / vmin) if flat.size and vmin != 0 else float("inf")

    # entropy over normalized positive values
    eps = 1e-12
    pos = flat - np.min(flat) + eps
    p = pos / np.sum(pos)
    ent = -float(np.sum(p * np.log(p + eps)))

    # skewness
    skew = float(((flat - m) ** 3).mean() / (np.std(flat) ** 3 + eps)) if flat.size else 0.0

    row = {
        "N": int(N),
        "E": int(E),
        "density": dens,
        "depth": int(depth),
        "avg_outdeg": avg_outdeg,
        "max_outdeg": max_outdeg,
        "branching_factor": branching,
        "price_var": var,
        "price_cv": cv,
        "price_entropy": ent,
        "price_cond_ratio": cond,
        "price_skewness": skew,
    }

    # Optional combinatorial size if K and alphabet are given
    if K is not None and alphabet_size is not None and K >= 0 and alphabet_size >= 1:
        # Very rough |S| proxy: number of compositions with alphabet A over K
        # This mirrors ∏_g C(A_g + k_g - 1, k_g) when A_g ≈ alphabet_size
        # Here we approximate with a single A to avoid exposing GA internals.
        from math import comb
        row["K"] = int(K)
        try:
            row["S_approx"] = float(comb(alphabet_size + K - 1, K))
        except Exception:
            row["S_approx"] = float("inf")

    return row

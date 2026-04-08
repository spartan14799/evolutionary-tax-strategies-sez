#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Mixed-Generic GA Wrapper (Normalized Output Interface)

This module provides a stable execution interface for the mixed-type
equivalence-class GA. It harmonizes inputs and outputs to match the
fine-tuning and experiment orchestration framework.

Public API
----------
- run_mixed_generic_w1(graph_links, price_matrix, agent_info_dict, hyperparams)
- run_mixed_generic_w2(env_dict, base_hyperparams, candidate_hyperparams, env_id=None)

Both return schemas of the form:
{
  "name": "mixed_generic",
  "curves": {"best": [...], "mean": [...], "median": [...]},
  "best_genome": [...],
  "all_best_genomes": [[...], ...],
  "meta": {...},
  "env_id": <env_id or None>,         # only in w2
  "elapsed_sec": <float>              # only in w2
}
"""

from __future__ import annotations
import time
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
from src.algorithms.ga.eq_class_generic import run_eq_class_generic_ga

ALGO_NAME = "mixed_generic"


# ============================================================================
# Normalization Utilities
# ============================================================================

def _normalize_result(res: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize the raw result into a portable schema.
    """
    curves = res.get("curves", {}) or {}
    curves_clean: Dict[str, List[float]] = {}
    for key in ("best", "mean", "median"):
        if isinstance(curves.get(key), list):
            curves_clean[key] = curves[key]

    meta = res.get("meta", {}) or {}

    return {
        "curves": curves_clean,
        "best_genome": res.get("best_genome"),
        "all_best_genomes": res.get("all_best_genomes"),
        "meta": meta,
    }


def _coerce_hparams(h: Dict[str, Any]) -> Dict[str, Any]:
    """
    Coerce hyperparameter types and ensure numeric stability.
    """
    out = dict(h)

    # Integers
    for k in ("generations", "popsize", "elitism", "seed", "verbosity", "log_every"):
        if k in out and out[k] is not None:
            out[k] = int(out[k])

    # Floats
    for k in ("cxpb", "mutpb", "mutation_rate", "selector_mutation_rate"):
        if k in out and out[k] is not None:
            out[k] = float(out[k])

    # Booleans
    if "fix_last_gene" in out and out["fix_last_gene"] is not None:
        out["fix_last_gene"] = bool(out["fix_last_gene"])

    return out


def _merge_base_candidate(base: Dict[str, Any], cand: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Merge base and candidate hyperparameters (flat or nested).
    """
    if cand is None:
        return dict(base)

    if "hparams" in cand and isinstance(cand["hparams"], dict):
        c = {k: v for k, v in cand["hparams"].items()}
    else:
        c = {k: v for k, v in cand.items() if k != "id"}

    return {**base, **c}


def _extract_candidate_id(cand: Optional[Dict[str, Any]]) -> Optional[Any]:
    if isinstance(cand, dict) and "id" in cand:
        return cand["id"]
    return None


# ============================================================================
# Single-Environment Execution
# ============================================================================

def run_mixed_generic_w1(
    graph_links: List[Tuple[str, str]],
    price_matrix: np.ndarray,
    agent_info_dict: Dict[str, Any],
    hyperparams: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Execute the mixed-type GA on a single environment.
    """
    hp = _coerce_hparams(hyperparams)

    res_raw = run_eq_class_generic_ga(
        production_graph=graph_links,
        pmatrix=price_matrix,
        agents_information=agent_info_dict,
        generations=hp.get("generations", 100),
        popsize=hp.get("popsize", 60),
        cxpb=hp.get("cxpb", 0.7),
        mutpb=hp.get("mutpb", 0.2),
        mutation_rate=hp.get("mutation_rate", 0.05),
        selector_mutation_rate=hp.get("selector_mutation_rate", 0.25),
        elitism=hp.get("elitism", 1),
        fix_last_gene=hp.get("fix_last_gene", True),
        seed=hp.get("seed", 44),
        verbosity=hp.get("verbosity", 1),
        log_every=hp.get("log_every", 1),
        evals_cap=hp.get("evals_cap", None),
        time_limit_sec=hp.get("time_limit_sec", None),
    )

    res_raw.setdefault("meta", {})["algo"] = ALGO_NAME

    norm = _normalize_result(res_raw)
    return {"name": ALGO_NAME, **norm}


# ============================================================================
# Batch Execution (Fine-Tuning Mode)
# ============================================================================

def run_mixed_generic_w2(
    env_dict: Dict[str, Any],
    base_hyperparams: Dict[str, Any],
    candidate_hyperparams: Dict[str, Any],
    env_id: str | int | None = None,
) -> Dict[str, Any]:
    """
    Execute mixed_generic in hierarchical hyperparameter mode.
    """
    hp = _merge_base_candidate(base_hyperparams, candidate_hyperparams)
    hp = _coerce_hparams(hp)
    candidate_id = _extract_candidate_id(candidate_hyperparams)

    graph = env_dict["production_graph"]
    prices = env_dict["price_matrix"]
    agents = env_dict["agents_info"]

    name = f"env_{env_id}" if env_id is not None else "env_unknown"

    if hp.get("verbosity", 1) > 0:
        print(
            f"[mixed_generic] Starting {name} | "
            f"gens={hp.get('generations')} | pop={hp.get('popsize')} | "
            f"cxpb={hp.get('cxpb')} | mutpb={hp.get('mutpb')}"
        )

    t0 = time.time()
    res = run_mixed_generic_w1(graph, prices, agents, hp)
    elapsed = time.time() - t0

    res["meta"] = res.get("meta", {})
    res["meta"]["algo"] = ALGO_NAME
    if candidate_id is not None:
        res["meta"]["candidate_id"] = candidate_id

    if hp.get("verbosity", 1) > 0:
        print(f"[mixed_generic] Completed {name} | Elapsed: {elapsed:.2f}s")

    return {"env_id": env_id, "elapsed_sec": float(elapsed), **res}

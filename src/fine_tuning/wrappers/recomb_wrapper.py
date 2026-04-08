#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Recombination-Only GA Wrapper (Normalized Output Interface)

This module exposes a standardized interface for executing the
recombination-only Genetic Algorithm variant across heterogeneous
environments. It ensures outputs conform to the normalized schema used
by the fine-tuning pipeline.

Responsibilities
---------------
- Execute the Recomb GA using base-level and candidate-specific
  hyperparameters.
- Control runtime verbosity in a unified manner.
- Return portable, JSON-serializable payloads.
- Aggregate timing and minimal metadata for downstream consumers.

Public API
----------
- run_recomb_w1(graph_links, price_matrix, agent_info_dict, hyperparams) -> Dict
- run_recomb_w2(env_dict, base_hyperparams, candidate_hyperparams, env_id=None) -> Dict

Both functions return:
{
  "name": "recomb",
  "curves": {"best": [...], "mean": [...], "median": [...]},
  "best_genome": [...],
  "all_best_genomes": [[...], ...],
  "meta": {...},
  # In run_recomb_w2 only:
  "env_id": <env_id or None>,
  "elapsed_sec": <float>
}
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Tuple

import numpy as np
from src.algorithms.ga.recomb_only import run_ga_recomb_only


# =============================================================================
# Internal Output Normalization
# =============================================================================

def _normalize_result(res: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize the raw Recomb GA report to a stable schema.

    Parameters
    ----------
    res : Dict[str, Any]
        Direct mapping returned by `run_ga_recomb_only`.

    Returns
    -------
    Dict[str, Any]
        Normalized mapping with the following keys:
        - curves : Dict[str, List[float]] (subset of available series)
        - best_genome : Optional[List[int]]
        - all_best_genomes : Optional[List[List[int]]]
        - meta : Dict[str, Any]
    """
    curves = res.get("curves", {}) or {}
    curves_clean: Dict[str, List[float]] = {}

    for key in ("best", "mean", "median"):
        val = curves.get(key)
        if isinstance(val, list):
            curves_clean[key] = val

    return {
        "curves": curves_clean,
        "best_genome": res.get("best_genome"),
        "all_best_genomes": res.get("all_best_genomes"),
        "meta": res.get("meta", {}),
    }


# =============================================================================
# Single-Environment Execution
# =============================================================================

def run_recomb_w1(
    graph_links: List[Tuple[str, str]],
    price_matrix: np.ndarray,
    agent_info_dict: Dict[str, Any],
    hyperparams: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Execute the recombination-only GA on a single environment.

    Parameters
    ----------
    graph_links : List[Tuple[str, str]]
        Production DAG edges (predecessor -> successor).
    price_matrix : np.ndarray
        Price tensor/matrix consumed by the Economy implementation.
    agent_info_dict : Dict[str, Any]
        Agent-specific configuration and mappings required by the Economy.
    hyperparams : Dict[str, Any]
        Evolutionary hyperparameters. Accepted keys include (non-exhaustive):
        - generations, popsize
        - parents_rate (preferred) OR parents (absolute); parents_rate ∈ (0,1]
        - elite_fraction, tourn_size, parent_selection, mating_selection
        - p_recomb, sel_mutation, tail_mutation, p_min, tau_percent, fix_last_gene
        - per_good_cap, max_index_probe, mode, seed, verbosity, log_every
        - evals_cap, time_limit_sec

    Returns
    -------
    Dict[str, Any]
        Normalized result with name="recomb".
    """
    popsize = int(hyperparams.get("popsize", 100))

    # Resolve parent pool size from parents_rate when provided.
    if "parents_rate" in hyperparams and hyperparams["parents_rate"] is not None:
        pr = float(hyperparams["parents_rate"])
        pr = max(0.0, min(1.0, pr))
        parents = max(2, int(round(pr * popsize)))
    else:
        parents = int(hyperparams.get("parents", max(2, int(round(0.5 * popsize)))))

    res_raw = run_ga_recomb_only(
        # Domain inputs
        production_graph=graph_links,
        pmatrix=price_matrix,
        agents_information=agent_info_dict,
        # Detection / pools
        mode=hyperparams.get("mode", "graph"),
        per_good_cap=hyperparams.get("per_good_cap", None),
        max_index_probe=int(hyperparams.get("max_index_probe", 3)),
        # Budgets
        generations=int(hyperparams.get("generations", 100)),
        popsize=popsize,
        parents=parents,
        elite_fraction=float(hyperparams.get("elite_fraction", 0.01)),
        # Selection policies
        tourn_size=int(hyperparams.get("tourn_size", 3)),
        parent_selection=str(hyperparams.get("parent_selection", "tournament")),
        mating_selection=str(hyperparams.get("mating_selection", "pairwise_tournament")),
        # Variation
        p_recomb=float(hyperparams.get("p_recomb", 0.50)),
        sel_mutation=hyperparams.get("sel_mutation", None),
        tail_mutation=hyperparams.get("tail_mutation", None),
        p_min=float(hyperparams.get("p_min", 0.30)),
        tau_percent=hyperparams.get("tau_percent", 0.05),
        fix_last_gene=bool(hyperparams.get("fix_last_gene", True)),
        # Reproducibility / logging / early stop
        seed=int(hyperparams.get("seed", 44)),
        verbosity=int(hyperparams.get("verbosity", 1)),
        log_every=int(hyperparams.get("log_every", 1)),
        evals_cap=hyperparams.get("evals_cap", None),
        time_limit_sec=hyperparams.get("time_limit_sec", None),
    )

    norm = _normalize_result(res_raw)
    return {"name": "recomb", **norm}


# =============================================================================
# Candidate Evaluation Mode (Batch Execution)
# =============================================================================

def run_recomb_w2(
    env_dict: Dict[str, Any],
    base_hyperparams: Dict[str, Any],
    candidate_hyperparams: Dict[str, Any],
    env_id: str | int | None = None,
) -> Dict[str, Any]:
    """
    Execute the recombination-only GA using hierarchical hyperparameters:
    base-level defaults overridden by candidate-specific values.

    Parameters
    ----------
    env_dict : Dict[str, Any]
        Environment mapping with keys:
        - "production_graph": edge list or compatible graph structure
        - "price_matrix"   : np.ndarray
        - "agents_info"    : Dict[str, Any]
    base_hyperparams : Dict[str, Any]
        Shared hyperparameters for the experiment suite.
    candidate_hyperparams : Dict[str, Any]
        Hyperparameters overriding the base for this specific run.
    env_id : str | int | None
        Optional identifier for batch tuning or sharded execution.

    Returns
    -------
    Dict[str, Any]
        Mapping including environment identifier, wall-clock time, and
        a normalized payload under the unified schema.
    """
    hp = {**base_hyperparams, **candidate_hyperparams}

    graph = env_dict["production_graph"]
    prices = env_dict["price_matrix"]
    agents = env_dict["agents_info"]

    name = f"env_{env_id}" if env_id is not None else "unknown_env"

    if int(hp.get("verbosity", 1)) > 0:
        print(
            f"Starting Recomb GA | {name} | "
            f"gens={hp.get('generations')} | pop={hp.get('popsize')} | "
            f"parents_rate={hp.get('parents_rate', None)} | "
            f"p_recomb={hp.get('p_recomb', 0.50)}"
        )

    t0 = time.time()
    res = run_recomb_w1(graph, prices, agents, hp)
    elapsed = time.time() - t0

    if int(hp.get("verbosity", 1)) > 0:
        print(f"Completed {name} | Elapsed: {elapsed:.2f}s")

    return {"env_id": env_id, "elapsed_sec": float(elapsed), **res}

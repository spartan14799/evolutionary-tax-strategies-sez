#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Joint GA Wrapper (Normalized Output Interface)

This module provides a standardized interface for executing the Joint
Equivalence-Class Genetic Algorithm across heterogeneous environments.
It ensures that all output returned by the algorithm matches the normalized
schema required by the fine_tuning_2 pipeline.

Functional responsibilities:
- Executes the Joint GA using both base-level and candidate-specific
  hyperparameters.
- Controls runtime console output via a unified verbosity setting.
- Enforces portable and JSON-serializable outputs (ASCII-safe).
- Aggregates metadata and timing information for downstream pipeline usage.

This wrapper is specifically intended for large-scale distributed tuning
and benchmarking of algorithmic performance.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Tuple

import numpy as np
from src.algorithms.ga.equivclass_joint import run_ga_equivclass_joint


# ======================================================================================
# Internal Output Normalization
# ======================================================================================

def _normalize_result(res: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize the raw Joint GA result to a stable schema.

    Parameters
    ----------
    res : Dict[str, Any]
        Direct mapping returned by the Joint GA implementation.

    Returns
    -------
    Dict[str, Any]
        Normalized mapping containing:
        - curves : Dict[str, List[float]]
        - best_genome : Optional[List[int]]
        - all_best_genomes : Optional[List[List[int]]]
        - meta : Dict[str, Any]
    """
    curves = res.get("curves", {}) or {}
    curves_clean: Dict[str, List[float]] = {}

    for key in ("best", "mean", "median", "std", "variance"):
        val = curves.get(key)
        if isinstance(val, list):
            curves_clean[key] = val

    return {
        "curves": curves_clean,
        "best_genome": res.get("best_genome"),
        "all_best_genomes": res.get("all_best_genomes"),
        "meta": res.get("meta", {}),
    }


# ======================================================================================
# Single-Environment Execution
# ======================================================================================

def run_joint_w1(
    graph_links: List[Tuple[str, str]],
    price_matrix: np.ndarray,
    agent_info_dict: Dict[str, Any],
    hyperparams: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Run the Joint GA for one environment, deriving inter-agent crossover structure
    and index-equivalence representation from graph topology.

    Parameters
    ----------
    graph_links : List[Tuple[str, str]]
        Production graph where each edge denotes predecessor → successor relation.
    price_matrix : np.ndarray
        Tensor of dimension (goods × agents × agents).
    agent_info_dict : Dict[str, Any]
        Agent-specific configurations and price mappings.
    hyperparams : Dict[str, Any]
        Evolutionary hyperparameters, including:
        parents_rate, sel_mutation, tail_mutation, generations, popsize, etc.

    Returns
    -------
    Dict[str, Any]
        Normalized result object including:
        - name : "joint"
        - curves : tracked performance statistics
        - best_genome, all_best_genomes : solution records
        - meta : runtime and termination information
    """
    popsize = int(hyperparams.get("popsize", 100))
    parents_rate = float(hyperparams.get("parents_rate", 0.5))
    parents = max(2, int(parents_rate * popsize))

    res_raw = run_ga_equivclass_joint(
        production_graph=graph_links,
        pmatrix=price_matrix,
        agents_information=agent_info_dict,
        mode=hyperparams.get("mode", "graph"),
        generations=int(hyperparams.get("generations", 100)),
        popsize=popsize,
        parents=parents,
        sel_mutation=float(hyperparams.get("sel_mutation", 0.20)),
        tail_mutation=float(hyperparams.get("tail_mutation", 0.05)),
        per_good_cap=hyperparams.get("per_good_cap", None),
        max_index_probe=int(hyperparams.get("max_index_probe", 8)),
        fix_last_gene=bool(hyperparams.get("fix_last_gene", True)),
        seed=int(hyperparams.get("seed", 42)),
        verbosity=int(hyperparams.get("verbosity", 1)),
        log_every=int(hyperparams.get("log_every", 1)),
    )

    norm = _normalize_result(res_raw)
    return {"name": "joint", **norm}


# ======================================================================================
# Candidate Evaluation Mode (Batch Execution)
# ======================================================================================

def run_joint_w2(
    env_dict: Dict[str, Any],
    base_hyperparams: Dict[str, Any],
    candidate_hyperparams: Dict[str, Any],
    env_id: str | int | None = None,
) -> Dict[str, Any]:
    """
    Execute the Joint GA for a single environment using hierarchical hyperparameters.

    Parameters
    ----------
    env_dict : Dict[str, Any]
        Environment mapping including:
        - production_graph
        - price_matrix
        - agents_info
    base_hyperparams : Dict[str, Any]
        Default hyperparameters shared by all GA runs in this experiment suite.
    candidate_hyperparams : Dict[str, Any]
        Hyperparameters that override the base for this specific run.
    env_id : str, int or None
        Optional execution identifier to support batch tuning/sharding.

    Returns
    -------
    Dict[str, Any]
        Mapping including environment identifier, execution time, and normalized payload.
    """
    hp = {**base_hyperparams, **candidate_hyperparams}

    graph = env_dict["production_graph"]
    prices = env_dict["price_matrix"]
    agents = env_dict["agents_info"]

    name = f"env_{env_id}" if env_id is not None else "unknown_env"

    if hp.get("verbosity", 1) > 0:
        print(
            f"Starting Joint GA | {name} | "
            f"gens={hp.get('generations')} | pop={hp.get('popsize')} | "
            f"parents_rate={hp.get('parents_rate')} | "
            f"sel_mut={hp.get('sel_mutation')} | tail_mut={hp.get('tail_mutation')}"
        )

    start = time.time()
    res = run_joint_w1(graph, prices, agents, hp)
    elapsed = time.time() - start

    if hp.get("verbosity", 1) > 0:
        print(f"Completed {name} | Elapsed: {elapsed:.2f}s")

    return {"env_id": env_id, "elapsed_sec": elapsed, **res}

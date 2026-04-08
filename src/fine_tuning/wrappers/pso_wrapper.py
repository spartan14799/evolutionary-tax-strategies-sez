#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Particle Swarm Optimization (PSO) – Normalized Wrapper Interface

This module provides a standardized interface for executing a PSO procedure
across multiple environments. It harmonizes the raw algorithm output into a
portable schema used by the fine_tuning_2 computational pipeline.

Key responsibilities
--------------------
- Automatically calibrates the required genome length using the global
  production graph.
- Evaluates candidate hyperparameter configurations from distributed tuning.
- Controls console printing behavior to avoid issues in non-UTF terminals.
- Ensures reproducible process execution through consolidated hyperparameters.
- Returns a normalized structure containing curves, solution records,
  and execution metadata.

This layer abstracts implementation differences so that downstream analysis
and orchestration logic remain consistent across different algorithms.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Tuple

import numpy as np

from src.algorithms.particle_swarm import run_pso
from src.algorithms.common import make_transactions_builder, calibrate_min_len_via_builder


# ======================================================================================
# Internal Normalization Utilities
# ======================================================================================

def _normalize_result(res: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize raw PSO results, ensuring consistent keys
    across evolutionary algorithm implementations.

    Parameters
    ----------
    res : Dict[str, Any]
        Direct output from the PSO executor.

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
    cleaned_curves: Dict[str, List[float]] = {}

    for key in ("best", "mean", "median", "std", "variance"):
        v = curves.get(key)
        if isinstance(v, list):
            cleaned_curves[key] = v

    return {
        "curves": cleaned_curves,
        "best_genome": res.get("best_genome"),
        "all_best_genomes": res.get("all_best_genomes"),
        "meta": res.get("meta", {}),
    }


# ======================================================================================
# Single-Environment Execution
# ======================================================================================

def run_pso_w1(
    graph_links: List[Tuple[str, str]],
    price_matrix: np.ndarray,
    agent_info_dict: Dict[str, Any],
    hyperparams: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Execute PSO within a single environment and normalize resulting output.

    The required genome length is inferred from the production graph structure.
    Each particle encodes a feasible sequence of trading or production decisions.

    Parameters
    ----------
    graph_links : List[Tuple[str, str]]
        Production graph with directed edges (u -> v).
    price_matrix : np.ndarray
        Price tensor specifying valuations for each good and agent pair.
    agent_info_dict : Dict[str, Any]
        Agent-related behavioral and accounting configuration.
    hyperparams : Dict[str, Any]
        Execution parameters including swarm behavior:
        * c1, c2, w ∈ ℝ → acceleration and inertia
        * generations, popsize → swarm evolution settings

    Returns
    -------
    Dict[str, Any]
        Normalized solution mapping with execution metadata.
    """
    # Determine minimum genome shape based on dependency structure
    txb = make_transactions_builder(graph_links)
    primary_nodes = [u for u, _ in graph_links if not any(v == u for _, v in graph_links)]
    base_L = max(2, len(primary_nodes) + 1)
    L_min = calibrate_min_len_via_builder(txb, base_L=base_L)

    # Execute PSO algorithm
    raw_result = run_pso(
        production_graph=graph_links,
        pmatrix=price_matrix,
        agents_information=agent_info_dict,
        genome_shape=L_min,
        generations=int(hyperparams.get("generations", 100)),
        popsize=int(hyperparams.get("popsize", 100)),
        c1=float(hyperparams.get("c1", 0.3)),
        c2=float(hyperparams.get("c2", 0.3)),
        w=float(hyperparams.get("w", 0.9)),
        fix_last_gene=bool(hyperparams.get("fix_last_gene", True)),
        seed=int(hyperparams.get("seed", 42)),
        verbosity=int(hyperparams.get("verbosity", 1)),
    )

    normalized = _normalize_result(raw_result)
    return {"name": "pso", **normalized}


# ======================================================================================
# Batch-Candidate Execution
# ======================================================================================

def run_pso_w2(
    env_dict: Dict[str, Any],
    base_hyperparams: Dict[str, Any],
    candidate_hyperparams: Dict[str, Any],
    env_id: str | int | None = None,
) -> Dict[str, Any]:
    """
    High-level execution mode for PSO under fine_tuning_2 orchestration.

    Parameters
    ----------
    env_dict : Dict[str, Any]
        Contains:
        - production_graph
        - price_matrix
        - agents_info
    base_hyperparams : Dict[str, Any]
        Hyperparameter defaults shared across evaluated candidates.
    candidate_hyperparams : Dict[str, Any]
        Hyperparameters overriding `base_hyperparams` for this specific candidate.
    env_id : str | int | None
        Optional identifier for distributed sharding experiments.

    Returns
    -------
    Dict[str, Any]
        Consolidated mapping including:
        * env_id
        * elapsed_sec
        * normalized PSO payload
    """
    hp = {**base_hyperparams, **candidate_hyperparams}
    name = f"env_{env_id}" if env_id is not None else "unknown_env"

    # Optional controlled console output
    if hp.get("verbosity", 1) > 0:
        print(
            f"Starting PSO | {name} | "
            f"gens={hp.get('generations')} | "
            f"pop={hp.get('popsize')} | "
            f"c1={hp.get('c1')} | c2={hp.get('c2')} | w={hp.get('w')}"
        )

    start = time.time()
    result = run_pso_w1(
        env_dict["production_graph"],
        env_dict["price_matrix"],
        env_dict["agents_info"],
        hp,
    )
    elapsed = time.time() - start

    if hp.get("verbosity", 1) > 0:
        print(f"Completed PSO | {name} | Elapsed: {elapsed:.2f}s")

    return {"env_id": env_id, "elapsed_sec": elapsed, **result}

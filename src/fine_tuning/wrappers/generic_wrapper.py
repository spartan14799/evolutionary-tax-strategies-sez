#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generic GA Wrapper (Normalized Output Interface)

This module provides a thin, standardized wrapper around the underlying
Generic Genetic Algorithm implementation. It enforces a consistent output
schema compatible with the fine_tuning_2 pipeline:

- Genome length is automatically calibrated from the production graph.
- Execution metadata is aggregated (runtime, evaluations, budget).
- Result objects include stable keys for curves, genomes, and diagnostics.
- Printing/logging is strictly controlled through the `verbosity` parameter.
- JSON-serializable, ASCII-safe output is ensured for distributed execution.

This wrapper is designed for automated batch evaluation of multiple GA
hyperparameter configurations across environments.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Tuple

import numpy as np

from src.algorithms.generic_ga import run_generic_ga
from src.algorithms.common import make_transactions_builder, calibrate_min_len_via_builder


# ======================================================================================
# Internal result normalization utilities
# ======================================================================================

def _normalize_result(res: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize the structure returned by the underlying GA.

    Ensures that:
    - Curve arrays are included only if valid lists.
    - Genome-related fields are always present (default: None).
    - Metadata is returned as a dictionary.

    Parameters
    ----------
    res : Dict[str, Any]
        Raw mapping returned by the GA.

    Returns
    -------
    Dict[str, Any]
        Normalized mapping including:
        - curves : Dict[str, List[float]]
        - best_genome : List[int] or None
        - all_best_genomes : List[List[int]] or None
        - meta : Dict[str, Any]
    """
    curves = res.get("curves", {}) or {}
    curves_clean: Dict[str, List[float]] = {}

    for key in ("best", "mean", "median", "std", "variance"):
        val = curves.get(key)
        if isinstance(val, list):  # Accept only well-formed lists
            curves_clean[key] = val

    return {
        "curves": curves_clean,
        "best_genome": res.get("best_genome"),
        "all_best_genomes": res.get("all_best_genomes"),
        "meta": res.get("meta", {}),
    }


# ======================================================================================
# External API: Single-Environment Execution
# ======================================================================================

def run_generic_w1(
    graph_links: List[Tuple[str, str]],
    price_matrix: np.ndarray,
    agent_info_dict: Dict[str, Any],
    hyperparams: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Execute a Generic GA in a single environment with calibrated genome length.

    Parameters
    ----------
    graph_links : List[Tuple[str, str]]
        Production graph edges representing predecessor → successor relations.
    price_matrix : np.ndarray
        Goods × agents × agents price tensor.
    agent_info_dict : Dict[str, Any]
        Configuration of agents, inventory rules, and price mapping.
    hyperparams : Dict[str, Any]
        Evolutionary hyperparameters (generations, popsize, mutation rates, etc.)

    Returns
    -------
    Dict[str, Any]
        Aggregated result including:
        - name : str, identifying the GA family ("generic")
        - curves : fitness statistics per generation
        - best_genome : best solution found
        - all_best_genomes : equivalent top-performing solutions (if tracked)
        - meta : runtime/diagnostics details
    """
    if not graph_links:
        raise ValueError("Production graph is empty; cannot calibrate genome length.")

    # Genome calibration from directed production topology
    builder = make_transactions_builder(graph_links)
    primary_nodes = [node for node, _ in graph_links
                     if not any(child == node for _, child in graph_links)]
    base_length = max(2, len(primary_nodes) + 1)
    genome_length = calibrate_min_len_via_builder(builder, base_L=base_length)

    # Delegate computation to the underlying GA implementation
    res_raw = run_generic_ga(
        production_graph=graph_links,
        pmatrix=price_matrix,
        agents_information=agent_info_dict,
        genome_shape=genome_length,
        generations=hyperparams.get("generations", 100),
        popsize=hyperparams.get("popsize", 100),
        cxpb=hyperparams.get("cxpb", 0.7),
        mutpb=hyperparams.get("mutpb", 0.2),
        mutation_rate=hyperparams.get("mutation_rate", 0.05),
        elitism=hyperparams.get("elitism", 1),
        fix_last_gene=hyperparams.get("fix_last_gene", True),
        seed=hyperparams.get("seed", 42),
        verbosity=hyperparams.get("verbosity", 1),
        log_every=hyperparams.get("log_every", 1),
    )

    norm = _normalize_result(res_raw)
    return {"name": "generic", **norm}


# ======================================================================================
# External API: Candidate Evaluation Mode (used in batch tuning)
# ======================================================================================

def run_generic_w2(
    env_dict: Dict[str, Any],
    base_hyperparams: Dict[str, Any],
    candidate_hyperparams: Dict[str, Any],
    env_id: str | int | None = None,
) -> Dict[str, Any]:
    """
    Execute the Generic GA using environment data and hierarchical hyperparameters.

    This mode is intended for automated tuning:
    - Base hyperparameters define global defaults for all runs.
    - Candidate hyperparameters override specific elements per execution.
    - Logging is controlled by `verbosity` and is disabled when 0.

    Parameters
    ----------
    env_dict : Dict[str, Any]
        Mapping with keys:
        - production_graph : List[Tuple[str, str]]
        - price_matrix : np.ndarray
        - agents_info : Dict[str, Any]
    base_hyperparams : Dict[str, Any]
        Default hyperparameter set for this algorithm.
    candidate_hyperparams : Dict[str, Any]
        Hyperparameters uniquely applied to this candidate run.
    env_id : str, int, or None, optional
        Identifier used for tracking results across multiple environments.

    Returns
    -------
    Dict[str, Any]
        Result mapping containing:
        - env_id : identifier of the evaluated environment
        - elapsed_sec : wall-clock execution time
        - curves/best_genome/all_best_genomes/meta : from run_generic_w1
    """
    # Merge hierarchical hyperparameters
    hp = {**base_hyperparams, **candidate_hyperparams}

    graph = env_dict["production_graph"]
    prices = env_dict["price_matrix"]
    agents = env_dict["agents_info"]

    name = f"env_{env_id}" if env_id is not None else "unknown_env"

    # Optional logging (ASCII-safe only)
    if hp.get("verbosity", 1) > 0:
        print(f"Starting Generic GA | {name} | gens={hp.get('generations')} "
              f"| pop={hp.get('popsize')} | cxpb={hp.get('cxpb')} "
              f"| mutpb={hp.get('mutpb')} | mut_rate={hp.get('mutation_rate')} "
              f"| elitism={hp.get('elitism')}")

    start = time.time()
    res = run_generic_w1(graph, prices, agents, hp)
    elapsed = time.time() - start

    if hp.get("verbosity", 1) > 0:
        print(f"Completed {name} | Elapsed: {elapsed:.2f}s")

    return {"env_id": env_id, "elapsed_sec": elapsed, **res}

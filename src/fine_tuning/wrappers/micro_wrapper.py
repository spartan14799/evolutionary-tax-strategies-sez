#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Micro-Only GA Wrapper (Normalized Output Interface)

This module exposes a standardized interface for executing the Micro-only
variant of the Macro→Micro Genetic Algorithm across heterogeneous environments.
It force-sets crossover to micro (p_micro=1.0, p_macro=0.0), normalizes outputs
to the fine-tuning schema, and provides single-environment and batch-style
entry points.

Responsibilities
---------------
- Execute the Macro→Micro GA with micro crossover only.
- Ignore or override any candidate/base p_macro/p_micro values.
- Control runtime verbosity in a unified manner.
- Return portable, JSON-serializable payloads with minimal metadata.

Public API
----------
- run_micro_w1(graph_links, price_matrix, agent_info_dict, hyperparams) -> Dict
- run_micro_w2(env_dict, base_hyperparams, candidate_hyperparams, env_id=None) -> Dict

Both functions return:
{
  "name": "baseline",
  "curves": {"best": [...], "mean": [...], "median": [...]},
  "best_genome": [...],
  "all_best_genomes": [[...], ...],
  "meta": {...},
  # In run_micro_w2 only:
  "env_id": <env_id or None>,
  "elapsed_sec": <float>
}
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
from src.search_heuristics.macro_micro import run_ga_macro_micro

# Fixed algorithm name for downstream consumers
ALGO_NAME = "micro"

# Fixed crossover mode for this wrapper
FIXED_P_MACRO = 0.0
FIXED_P_MICRO = 1.0


# =============================================================================
# Internal Utilities
# =============================================================================

def _normalize_result(res: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize the raw Macro→Micro GA report to a stable schema.

    Parameters
    ----------
    res : Dict[str, Any]
        Mapping returned by `run_ga_macro_micro`.

    Returns
    -------
    Dict[str, Any]
        Normalized mapping with keys:
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

    meta = res.get("meta", {}) or {}
    return {
        "curves": curves_clean,
        "best_genome": res.get("best_genome"),
        "all_best_genomes": res.get("all_best_genomes"),
        "meta": meta,
    }


def _coerce_hparams(hp: Dict[str, Any]) -> Dict[str, Any]:
    """
    Coerce hyperparameter types to robust primitives and enforce micro-only mode.
    """
    out = dict(hp)

    # Integers
    for k in ("generations", "popsize", "parents", "tourn_size",
              "max_index_probe", "seed", "verbosity", "log_every"):
        if k in out and out[k] is not None:
            out[k] = int(out[k])

    # Floats
    for k in ("elite_fraction", "lambda_in", "lambda_out",
              "sel_mutation", "tail_mutation", "p_min", "tau_percent"):
        if k in out and out[k] is not None:
            out[k] = float(out[k])

    # Booleans
    for k in ("fix_last_gene",):
        if k in out and out[k] is not None:
            out[k] = bool(out[k])

    # parents_rate bounded to [0,1]
    if "parents_rate" in out and out["parents_rate"] is not None:
        pr = float(out["parents_rate"])
        out["parents_rate"] = max(0.0, min(1.0, pr))

    # Enforce micro-only crossover, regardless of provided values
    out["p_macro"] = FIXED_P_MACRO
    out["p_micro"] = FIXED_P_MICRO

    return out


def _merge_base_candidate(base: Dict[str, Any],
                          candidate: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge base-level defaults with candidate-specific overrides.

    Accepts candidates in two shapes:
      - {"id": <int|str>, "hparams": {...}}
      - {"id": <int|str>, ...flat hyperparams...}  (legacy/flat)
      - Or simply {...flat hyperparams...} (no id)

    Returns merged hyperparameters (no 'id' key).
    """
    if candidate is None:
        merged = dict(base)
    else:
        cand = dict(candidate)
        if "hparams" in cand and isinstance(cand["hparams"], dict):
            cand_hp = cand["hparams"]
        else:
            cand_hp = {k: v for k, v in cand.items() if k != "id"}
        merged = {**base, **cand_hp}

    # Remove any external attempt to set p_macro/p_micro;
    # they will be force-set in _coerce_hparams().
    merged.pop("p_macro", None)
    merged.pop("p_micro", None)
    return merged


def _extract_candidate_id(candidate: Optional[Dict[str, Any]]) -> Optional[Any]:
    """
    Extract candidate id if present.
    """
    if isinstance(candidate, dict) and "id" in candidate:
        return candidate["id"]
    return None


# =============================================================================
# Single-Environment Execution
# =============================================================================

def run_micro_w1(
    graph_links: List[Tuple[str, str]],
    price_matrix: np.ndarray,
    agent_info_dict: Dict[str, Any],
    hyperparams: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Execute the Macro→Micro GA with micro-only crossover on a single environment.

    Parameters
    ----------
    graph_links : List[Tuple[str, str]]
        Production DAG edges (predecessor -> successor).
    price_matrix : np.ndarray
        Price tensor/matrix consumed by the evaluation model.
    agent_info_dict : Dict[str, Any]
        Agent-specific configuration required by the evaluation model.
    hyperparams : Dict[str, Any]
        Evolutionary hyperparameters. Accepted keys include (non-exhaustive):
        - generations, popsize
        - parents_rate (preferred) OR parents (absolute); parents_rate ∈ (0,1]
        - elite_fraction, tourn_size, parent_selection, mating_selection
        - lambda_in, lambda_out
        - sel_mutation, tail_mutation, p_min, tau_percent, fix_last_gene
        - per_good_cap, max_index_probe, mode, seed, verbosity, log_every
        - evals_cap, time_limit_sec

        Note: p_macro/p_micro are ignored and force-set to 0.0 and 1.0 respectively.

    Returns
    -------
    Dict[str, Any]
        Normalized result with name="baseline".
    """
    hp = _coerce_hparams(hyperparams)

    popsize = int(hp.get("popsize", 100))


    # Execute the underlying runner with coerced/derived parameters.
    res_raw = run_ga_macro_micro(
        # Domain inputs
        production_graph=graph_links,
        pmatrix=price_matrix,
        agents_information=agent_info_dict,
        # Detection / pools
        mode=hp.get("mode", "graph"),
        per_good_cap=hp.get("per_good_cap", None),
        max_index_probe=int(hp.get("max_index_probe", 3)),
        # Budgets
        generations=int(hp.get("generations", 100)),
        popsize=popsize,
        elite_fraction=float(hp.get("elite_fraction", 0.01)),
        # Selection policies
        tourn_size=int(hp.get("tourn_size", 3)),
        parent_selection=str(hp.get("parent_selection", "tournament")),
        mating_selection=str(hp.get("mating_selection", "pairwise_tournament")),
        # Variation (Micro-only enforced)
        lambda_in=float(hp.get("lambda_in", 0.25)),
        lambda_out=float(hp.get("lambda_out", 0.50)),
        p_macro=FIXED_P_MACRO,
        p_micro=FIXED_P_MICRO,
        # Mutation
        sel_mutation=hp.get("sel_mutation", None),
        tail_mutation=hp.get("tail_mutation", None),
        p_min=float(hp.get("p_min", 0.30)),
        tau_percent=hp.get("tau_percent", 0.05),
        fix_last_gene=bool(hp.get("fix_last_gene", True)),
        # Reproducibility / logging / early stop
        seed=int(hp.get("seed", 44)),
        verbosity=int(hp.get("verbosity", 1)),
        log_every=int(hp.get("log_every", 1)),
        evals_cap=hp.get("evals_cap", None),
        time_limit_sec=hp.get("time_limit_sec", None),
    )

    # Annotate wrapper notes and algo name in meta.
    meta = res_raw.setdefault("meta", {})
    notes = meta.setdefault("notes", {})
    notes["wrapper"] = "micro_only"
    notes["p_macro"] = FIXED_P_MACRO
    notes["p_micro"] = FIXED_P_MICRO
    meta["algo"] = ALGO_NAME

    norm = _normalize_result(res_raw)
    return {"name": ALGO_NAME, **norm}


# =============================================================================
# Candidate Evaluation Mode (Batch Execution)
# =============================================================================

def run_micro_w2(
    env_dict: Dict[str, Any],
    base_hyperparams: Dict[str, Any],
    candidate_hyperparams: Dict[str, Any],
    env_id: str | int | None = None,
) -> Dict[str, Any]:
    """
    Execute the Micro-only GA using hierarchical hyperparameters:
    base-level defaults overridden by candidate-specific values. Any attempt
    to set p_macro/p_micro is ignored and replaced by 0.0/1.0 respectively.

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
        It may include an "id" field and/or a nested "hparams" mapping.
    env_id : str | int | None
        Optional identifier for batch tuning or sharded execution.

    Returns
    -------
    Dict[str, Any]
        Mapping including environment identifier, wall-clock time, and
        a normalized payload under the unified schema.
    """
    # Merge and coerce; enforce micro-only mode
    hp_merged = _merge_base_candidate(base_hyperparams, candidate_hyperparams)
    hp_merged = _coerce_hparams(hp_merged)
    candidate_id = _extract_candidate_id(candidate_hyperparams)

    graph = env_dict["production_graph"]
    prices = env_dict["price_matrix"]
    agents = env_dict["agents_info"]

    name = f"env_{env_id}" if env_id is not None else "unknown_env"

    if int(hp_merged.get("verbosity", 1)) > 0:
        print(
            f"Starting Micro-Only GA | {name} | "
            f"gens={hp_merged.get('generations')} | pop={hp_merged.get('popsize')} | "
            f"parents_rate={hp_merged.get('parents_rate', None)} | "
            f"p_macro={FIXED_P_MACRO} | p_micro={FIXED_P_MICRO}"
        )

    t0 = time.time()
    res = run_micro_w1(graph, prices, agents, hp_merged)
    elapsed = time.time() - t0

    # Ensure meta contains candidate and algo annotations
    res["meta"] = res.get("meta", {})
    if candidate_id is not None:
        res["meta"]["candidate_id"] = candidate_id
    res["meta"]["algo"] = ALGO_NAME

    if int(hp_merged.get("verbosity", 1)) > 0:
        print(f"Completed {name} | Elapsed: {elapsed:.2f}s")

    return {"env_id": env_id, "elapsed_sec": float(elapsed), **res}

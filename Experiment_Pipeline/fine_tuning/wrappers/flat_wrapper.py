import time
import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Any

from algorithms.ga.flat import run_ga_flat
from algorithms.ga.common import make_transactions_builder, calibrate_min_len_via_builder


# =============================================================================
#  FLAT WRAPPER LEVEL 1
# =============================================================================

def run_flat_w1(
    graph_links: List[Tuple[str, str]],
    P: np.ndarray,
    agent_info_dict: Dict[str, Any],
    hp: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Execute the Flat GA for a single environment using the provided graph,
    prices, and hyperparameters.

    Parameters
    ----------
    graph_links : list of tuple(str, str)
        Directed edges of the production graph.
    P : np.ndarray
        Price matrix or tensor for the environment.
    agent_info_dict : dict
        Information dictionary for agents in the environment.
    hp : dict
        Hyperparameters for the GA. Expected keys:
        'generations', 'popsize', 'parents_rate', 'mutation_rate',
        'fix_last_gene', 'seed'.

    Returns
    -------
    dict
        Dictionary with the GA results including the name ("flat") and metrics.

         "env_id": env_id,
        "parents": parents,
        "parents_rate": parents_rate,
        "mutation_rate": mutation_rate,
        "elapsed_sec": elapsed,
        **res,
        "best_genome": best_genome_list,
        "best_utility": best_val,
        "all_best_genomes": all_best_genomes if all_best_genomes else [list(best_ind)],
        "curves": {"best": best_hist, "mean": mean_hist, "median": median_hist},
        "meta": {
            "genome_shape": L,
            "generations": int(gen if not budget_triggered else gen),
            "popsize": N,
            "parents": P,
            "mutation_rate": float(mutation_rate),
            "fix_last_gene": bool(fix_last_gene),
            "seed": seed,
            "tie_tolerance": tie_tolerance,
            "evals_per_gen": evals_per_gen,
            "evals_cum": evals_cum,
            "runtime_sec": float(runtime),
            "budget": {
                "evals_cap": evals_cap,
                "time_limit_sec": time_limit_sec,
                "triggered": bool(budget_triggered),
                "reason": budget_reason,
                "evals_total": int(sum(evals_per_gen)),
                "time_total_sec": float(runtime),
    """
    # --- Determine minimal feasible genome length ---
    txb = make_transactions_builder(graph_links)
    primary_nodes = [u for u, _ in graph_links if not any(v == u for _, v in graph_links)]
    base_L = max(2, len(primary_nodes) + 1)
    L_min = calibrate_min_len_via_builder(txb, base_L=base_L)

    parents_rate = hp.get("parents_rate", 0.3)
    popsize = hp.get("popsize", 30)
    parents = max(2, int(parents_rate * popsize))

    res = run_ga_flat(
        production_graph=graph_links,
        pmatrix=P,
        agents_information=agent_info_dict,
        genome_shape=L_min,
        generations=hp.get("generations", 12),
        popsize=popsize,
        parents=parents,
        mutation_rate=hp.get("mutation_rate", 0.05),
        fix_last_gene=hp.get("fix_last_gene", True),
        seed=hp.get("seed", 42),
    )
    return {"name": "flat", **res}


# =============================================================================
#  FLAT WRAPPER LEVEL 2 (USED IN FINE-TUNING)
# =============================================================================

def run_flat_w2(
    env_dict: Dict[str, Any],
    base_hyperparams: Dict[str, Any],
    candidate_hyperparams: Dict[str, Any],
    env_id: str | int | None = None,
) -> Dict[str, Any]:
    """
    Wrapper to test a candidate (mutation_rate, parents_rate) combination
    for a given environment, inheriting fixed base parameters.

    Parameters
    ----------
    env_dict : dict
        Environment dictionary with keys:
        'production_graph', 'price_matrix', 'agents_info'.
    base_hyperparams : dict
        Baseline hyperparameters (e.g. generations, popsize, seed).
    candidate_hyperparams : dict
        Candidate-specific hyperparameters (e.g. mutation_rate, parents_rate).
    env_id : str or int, optional
        Identifier for the environment, used in progress logs.

    Returns
    -------
    dict
        GA run results including performance metrics and setup details.

          "best_genome": best_genome_list,
        "best_utility": best_val,
        "all_best_genomes": all_best_genomes if all_best_genomes else [list(best_ind)],
        "curves": {"best": best_hist, "mean": mean_hist, "median": median_hist},
        "meta": {
            "genome_shape": L,
            "generations": int(gen if not budget_triggered else gen),
            "popsize": N,
            "parents": P,
            "mutation_rate": float(mutation_rate),
            "fix_last_gene": bool(fix_last_gene),
            "seed": seed,
            "tie_tolerance": tie_tolerance,
            "evals_per_gen": evals_per_gen,
            "evals_cum": evals_cum,
            "runtime_sec": float(runtime),
            "budget": {
                "evals_cap": evals_cap,
                "time_limit_sec": time_limit_sec,
                "triggered": bool(budget_triggered),
                "reason": budget_reason,
                "evals_total": int(sum(evals_per_gen)),
                "time_total_sec": float(runtime),
    """
    graph_links = env_dict["production_graph"]
    price_tensor = env_dict["price_matrix"]
    agent_info_dict = env_dict["agents_info"]

    hp = {**base_hyperparams, **candidate_hyperparams}

    # --- Extract hyperparameters ---
    parents_rate = hp.get("parents_rate")
    popsize = hp.get("popsize")
    parents = max(2, int(parents_rate * popsize))
    mutation_rate = hp.get("mutation_rate")
    generations = hp.get("generations")

    sel_mut = hp.get("sel_mutation")
    tail_mut = hp.get("tail_mutation")

    env_name = f"env_{env_id}" if env_id is not None else "unknown_env"
    print(f"\n[▶] Starting GA | {env_name} | parents_rate={parents_rate:.3f}, parents={parents}, "
          f"mutation_rate={mutation_rate:.3f}, gens={generations}, pop={popsize}")

    start_time = time.time()
    res = run_flat_w1(graph_links, price_tensor, agent_info_dict, hp)
    elapsed = time.time() - start_time

    print(f"[✔] Completed {env_name} | parents_rate={parents_rate:.3f}, parents={parents}, "
          f"mutation_rate={mutation_rate:.3f} | Elapsed: {elapsed:.2f}s\n")
    return {
        **res,  # unpack first (the GA output)
        "env_id": env_id,
        "parents_rate": parents_rate,
        "parents": int(parents_rate * hp.get("popsize", 28)),
        "sel_mutation": sel_mut,
        "tail_mutation": tail_mut,
        "elapsed_sec": elapsed,
    }
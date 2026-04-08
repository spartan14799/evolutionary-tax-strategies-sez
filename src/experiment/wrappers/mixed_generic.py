from typing import List, Tuple, Dict, Any
import numpy as np

from src.algorithms.eq_class_generic import run_eq_class_generic_ga
from src.algorithms.common import make_transactions_builder, calibrate_min_len_via_builder


def run_mixed_generic_w1(
    graph_links: List[Tuple[str, str]],
    P: np.ndarray,
    agent_info_dict: Dict[str, Any],
    hp: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Run a single-environment **Equivalence-Class Generic GA** (mixed genotype).

    This wrapper calibrates the minimal genome length from the production graph,
    then executes `run_eq_class_generic_ga` using a configuration dictionary of
    hyperparameters (hp) that can be defined per experiment.

    Parameters
    ----------
    graph_links : list[tuple[str, str]]
        Production graph edges (u, v).
    P : np.ndarray
        Price tensor (n_goods × n_agents × n_agents).
    agent_info_dict : dict
        Agent metadata for the Economy simulation.
    hp : dict
        Dictionary of GA hyperparameters.

    Returns
    -------
    dict
        Results from run_eq_class_generic_ga with environment metadata.
    """

    # --- Validate input ---
    if not graph_links:
        raise ValueError("Production graph is empty — cannot calibrate genome length.")

    # --- Determine minimal feasible genome length (for internal consistency) ---
    txb = make_transactions_builder(graph_links)
    primary_nodes = [u for u, _ in graph_links if not any(v == u for _, v in graph_links)]
    base_L = max(2, len(primary_nodes) + 1)
    L_min = calibrate_min_len_via_builder(txb, base_L=base_L)

    # --- Run equivalence-class GA ---
    res = run_eq_class_generic_ga(
        production_graph=graph_links,
        pmatrix=P,
        agents_information=agent_info_dict,
        final_good=hp.get("final_good", None),
        generations=hp.get("generations", 50),
        popsize=hp.get("popsize", 50),
        cxpb=hp.get("cxpb", 0.7),
        mutpb=hp.get("mutpb", 0.2),
        mutation_rate=hp.get("mutation_rate", 0.05),
        selector_mutation_rate=hp.get("selector_mutation_rate", 0.25),
        elitism=hp.get("elitism", 1),
        fix_last_gene=hp.get("fix_last_gene", True),
        seed=hp.get("seed", 42),
        verbosity=hp.get("verbosity", 1),
        log_every=hp.get("log_every", 1),
        evals_cap=hp.get("evals_cap", None),
        time_limit_sec=hp.get("time_limit_sec", None),
    )

    # --- Return with metadata ---
    return {"name": "mixed_generic", "L_min": L_min, **res}

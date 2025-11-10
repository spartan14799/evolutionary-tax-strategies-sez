
from algorithms.ga.blind_random_search import run_random_search

from typing import Any, List, Tuple, Dict 

import numpy as np 

import random 

from algorithms.ga.common import make_transactions_builder, calibrate_min_len_via_builder

def run_blindrandom_w1(
    graph_links: List[Tuple[str, str]],
    P: np.ndarray,
    agent_info_dict: Dict[str, Any],
    hp: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Execute the Blind Random Search for a single environment using the provided
    graph, prices, and hyperparameters.

    Parameters
    ----------
    graph_links : list of tuple(str, str)
        Directed edges of the production graph.
    P : np.ndarray
        Price matrix or tensor for the environment.
    agent_info_dict : dict
        Information dictionary for agents in the environment.
    hp : dict
        Hyperparameters for the random search. Expected keys:
        'generations', 'popsize', 'fix_last_gene', 'seed', 'verbosity',
        'log_every', 'evals_cap', 'time_limit_sec'.

    Returns
    -------
    dict
        Dictionary with the Random Search results including the name ("blindrandom")
        and associated metrics.
    """
    # --- Determine minimal feasible genome length ---
    txb = make_transactions_builder(graph_links)
    primary_nodes = [u for u, _ in graph_links if not any(v == u for _, v in graph_links)]
    base_L = max(2, len(primary_nodes) + 1)
    L_min = calibrate_min_len_via_builder(txb, base_L=base_L)

    # --- Run the random search ---
    res = run_random_search(
        production_graph=graph_links,
        pmatrix=P,
        agents_information=agent_info_dict,
        genome_shape=L_min,
        popsize=hp.get("popsize", 30),
        generations=hp.get("generations", 15),
        fix_last_gene=hp.get("fix_last_gene", True),
        seed=hp.get("seed", 42),
        verbosity=hp.get("verbosity", 1),
        log_every=hp.get("log_every", 1),
        evals_cap=hp.get("evals_cap", None),
        time_limit_sec=hp.get("time_limit_sec", None),
    )

    return {"name": "blindrandom", **res}

from algorithms.ga.common import make_transactions_builder, calibrate_min_len_via_builder
from algorithms.ga.macro_micro import run_ga_macro_micro  
from typing import List, Tuple, Dict, Any
import numpy as np


def run_macro_w1(
    graph_links: List[Tuple[str, str]],
    P: np.ndarray,
    agent_info_dict: Dict[str, Any],
    hp: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Run a single-environment **Macro-only GA** (p_micro = 0) with genome length
    calibrated from a production graph.

    Parameters
    ----------
    graph_links : list[tuple[str, str]]
        Production graph edges (u, v).
    P : np.ndarray
        Price tensor (n_goods × n_agents × n_agents).
    agent_info_dict : dict
        Agent metadata for Economy simulation.
    hp : dict
        Dictionary of GA hyperparameters.

    Returns
    -------
    dict
        Results from run_ga_macro_micro with environment metadata.
    """



    # --- Run Macro→Micro GA (Macro-only mode) ---
    res = run_ga_macro_micro(
        production_graph=graph_links,
        pmatrix=P,
        agents_information=agent_info_dict,
        # --- Prefix / alphabet detection ---
        mode=hp.get("mode", "graph"),
        per_good_cap=hp.get("per_good_cap", None),
        max_index_probe=hp.get("max_index_probe", 3),
        # --- Evolutionary budget / loop ---
        generations=hp.get("generations", 50),
        popsize=hp.get("popsize", 50),
        parents=hp.get("parents", 20),
        elite_fraction=hp.get("elite_fraction", 0.01),
        # --- Selection policies ---
        tourn_size=hp.get("tourn_size", 5),
        parent_selection=hp.get("parent_selection", "tournament"),
        mating_selection=hp.get("mating_selection", "pairwise_tournament"),
        # --- Crossover & mutation ---
        lambda_in=hp.get("lambda_in", 0.25),
        lambda_out=hp.get("lambda_out", 0.50),
        p_macro=hp.get("p_macro", 1.00),
        p_micro=0.0,  # disable micro stage
        sel_mutation=hp.get("sel_mutation", 0.05),
        tail_mutation=hp.get("tail_mutation", 0.05),
        p_min=hp.get("p_min", 0.30),
        tau_percent=hp.get("tau_percent", 0.05),
        fix_last_gene=hp.get("fix_last_gene", True),
        # --- Logging, reproducibility, and budgets ---
        seed=hp.get("seed", 42),
        verbosity=hp.get("verbosity", 1),
        log_every=hp.get("log_every", 1),
        evals_cap=hp.get("evals_cap", None),
        time_limit_sec=hp.get("time_limit_sec", None),
    )

    # Fix: identify correctly as “macro”, not “macro_micro”
    return {"name": "macro", **res}

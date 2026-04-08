import time
import numpy as np
from typing import Dict, List, Tuple, Any

from src.search_heuristics.generic_ga import run_generic_ga
from src.search_heuristics.common import make_transactions_builder, calibrate_min_len_via_builder


def run_generic_ga_w1(
    graph_links: List[Tuple[str, str]],
    P: np.ndarray,
    agent_info_dict: Dict[str, Any],
    hp: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Run a single-environment GA with genome length calibrated from a production graph.

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
        Results from run_generic_ga with environment metadata.
    """

    # --- Validate input ---
    if not graph_links:
        raise ValueError("Production graph is empty — cannot calibrate genome length.")

    # --- Determine minimal feasible genome length ---
    txb = make_transactions_builder(graph_links)
    primary_nodes = [u for u, _ in graph_links if not any(v == u for _, v in graph_links)]
    base_L = max(2, len(primary_nodes) + 1)
    L_min = calibrate_min_len_via_builder(txb, base_L=base_L)

    # --- Run GA ---
    res = run_generic_ga(
        production_graph=graph_links,
        pmatrix=P,
        agents_information=agent_info_dict,
        genome_shape=L_min,
        generations=hp.get("generations", 20),
        popsize=hp.get("popsize", 50),
        cxpb=hp.get("cxpb", 0.7),
        mutpb=hp.get("mutpb", 0.2),
        mutation_rate=hp.get("mutation_rate", 0.05),
        elitism=hp.get("elitism", 1),
        fix_last_gene=hp.get("fix_last_gene", True),
        seed=hp.get("seed", 42),
        verbosity=hp.get("verbosity", 1),
        log_every=hp.get("log_every", 1),
    )

    return {"name": "generic", **res}


def run_generic_ga_w2(
    env_dict: Dict[str, Any],
    base_hyperparams: Dict[str, Any],
    candidate_hyperparams: Dict[str, Any],
    env_id: str | int | None = None
) -> Dict[str, Any]:
    """
    Wrapper to evaluate a given hyperparameter candidate configuration for one environment.

    Parameters
    ----------
    env_dict : dict
        Environment dictionary containing keys:
            'production_graph', 'price_matrix', 'agents_info'.
    base_hyperparams : dict
        Default hyperparameters shared across environments.
    candidate_hyperparams : dict
        Candidate-specific hyperparameters that override the base ones.
    env_id : str or int, optional
        Environment identifier for logging.

    Returns
    -------
    dict
        Aggregated results including GA output and runtime metadata.
    """

    # --- Extract environment data ---
    graph_links = env_dict["production_graph"]
    price_tensor = env_dict["price_matrix"]
    agent_info_dict = env_dict["agents_info"]

    # --- Merge base and candidate hyperparameters ---
    hp = {**base_hyperparams, **candidate_hyperparams}
    env_name = f"env_{env_id}" if env_id is not None else "unknown_env"

    # --- Log run configuration ---
    print(
        f"\n[▶] Starting Generic GA | {env_name} | "
        f"gens={hp.get('generations')} | pop={hp.get('popsize')} | "
        f"cxpb={hp.get('cxpb', 0.7)} | mutpb={hp.get('mutpb', 0.2)} | "
        f"mut_rate={hp.get('mutation_rate', 0.05)} | elitism={hp.get('elitism', 1)}"
    )

    # --- Run GA and time it ---
    start_time = time.time()
    res = run_generic_ga_w1(graph_links, price_tensor, agent_info_dict, hp)
    elapsed = time.time() - start_time

    print(f"[✔] Completed {env_name} | Elapsed: {elapsed:.2f}s\n")

    # --- Return results with metadata ---
    return {
        "env_id": env_id,
        "elapsed_sec": elapsed,
        "cxpb": hp.get("cxpb", 0.7),
        "mutpb": hp.get("mutpb", 0.2),
        "mutation_rate": hp.get("mutation_rate", 0.05),
        "elitism": hp.get("elitism", 1),
        "generations": hp.get("generations", 20),
        "popsize": hp.get("popsize", 50),
        **res,
    }

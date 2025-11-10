import time
import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Any

from algorithms.ga.equivclass_joint import run_ga_equivclass_joint

def print_result_summary(result: Dict[str, Any], max_list_items: int = 3, max_dict_items: int = 3) -> None:
    """
    Pretty print summary of GA result dictionary with nested structures.
    Shows scalars directly, lists with length and preview, and dicts with key summaries.
    """
    print("\n[RESULT SUMMARY]")
    print("-" * 60)
    for k, v in result.items():
        if isinstance(v, (int, float, str)):
            print(f"{k:<22}: {v}")
        elif isinstance(v, list):
            n = len(v)
            preview = ", ".join(map(str, v[:max_list_items]))
            if n > max_list_items:
                preview += ", ..."
            print(f"{k:<22}: list[{n}] -> [{preview}]")
        elif isinstance(v, dict):
            keys = list(v.keys())[:max_dict_items]
            preview = ", ".join(keys)
            if len(v) > max_dict_items:
                preview += ", ..."
            print(f"{k:<22}: dict({len(v)}) keys=[{preview}]")
        else:
            print(f"{k:<22}: {type(v)}")


def run_joint_w1(
    graph_links: List[Tuple[str, str]],
    P: np.ndarray,
    agent_info_dict: Dict[str, Any],
    hp: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Execute the Joint GA for a single environment using the provided graph, prices, and hyperparameters.

    Parameters
    ----------
    graph_links : list of tuple(str, str)
        Directed edges of the production graph.
    P : np.ndarray
        Price matrix or tensor for the environment.
    agent_info_dict : dict
        Information dictionary for agents in the environment.
    hp : dict
        Hyperparameters for the GA. Expected keys include:
        'mode', 'generations', 'popsize', 'parents',
        'sel_mutation', 'tail_mutation', 'per_good_cap',
        'max_index_probe', 'fix_last_gene', 'seed',
        'verbosity', 'log_every'.

    Returns
    -------
    dict
        Dictionary with the GA results including the name ("joint") and metrics.
    """
    parents_rate = hp.get("parents_rate", 0.5)
    popsize = hp.get("popsize", 28)
    parents = int(parents_rate * popsize)
    res = run_ga_equivclass_joint(
        production_graph=graph_links,
        pmatrix=P,
        agents_information=agent_info_dict,
        mode=hp.get("mode", "graph"),
        generations=hp.get("generations", 12),
        popsize=popsize,
        parents=parents,
        sel_mutation=hp.get("sel_mutation", 0.20),
        tail_mutation=hp.get("tail_mutation", 0.05),
        per_good_cap=hp.get("per_good_cap", None),
        max_index_probe=hp.get("max_index_probe", 8),
        fix_last_gene=hp.get("fix_last_gene", True),
        seed=hp.get("seed", 42),
        verbosity=hp.get("verbosity", 1),
        log_every=hp.get("log_every", 2),
    )
    return {"name": "joint", **res}


def run_joint_w2(
    env_dict: Dict[str, Any],
    base_hyperparams: Dict[str, Any],
    candidate_hyperparams: Dict[str, Any],
    env_id: str | int | None = None,
) -> Dict[str, Any]:
    """
    Wrapper to test a candidate (sel_mutation, tail_mutation, parents)
    combination for a given environment, inheriting fixed base parameters.

    Parameters
    ----------
    env_dict : dict
        Environment dictionary with keys:
        'production_graph', 'price_matrix', 'agents_info'.
    base_hyperparams : dict
        Baseline hyperparameters (e.g., generations, popsize, seed).
    candidate_hyperparams : dict
        Candidate-specific hyperparameters (e.g., sel_mutation, tail_mutation, parents).
    env_id : str or int, optional
        Identifier for the environment, used in progress logs.

    Returns
    -------
    dict
        GA run results including performance metrics and setup details.
    """
    graph_links = env_dict["production_graph"]
    price_tensor = env_dict["price_matrix"]
    agent_info_dict = env_dict["agents_info"]

    # Merge hyperparameters: candidate overrides base
    hp = {**base_hyperparams, **candidate_hyperparams}

    # --- Logging: start progress message ---
    env_name = f"env_{env_id}" if env_id is not None else "unknown_env"

    parents_rate = hp.get("parents_rate")
    popsize = hp.get("popsize")


    parents = int(parents_rate * popsize)

    sel_mut = hp.get("sel_mutation")
    tail_mut = hp.get("tail_mutation")
    generations = hp.get("generations")
    

    print(f"\n[▶] Starting Joint GA | {env_name} |popsize={popsize} |, "
          f"sel_mut={sel_mut},parents_rate={parents_rate} ,tail_mut={tail_mut}, gens={generations}, pop={popsize}")
    start_time = time.time()

    res = run_joint_w1(graph_links, price_tensor, agent_info_dict, hp)

    elapsed = time.time() - start_time
    print(f"[✔] Completed {env_name} | parents={parents}, sel_mut={sel_mut}, "
          f"tail_mut={tail_mut} | Elapsed: {elapsed:.2f}s\n")
    

    # Attach metadata for traceability
    return {
        "env_id": env_id,
        "parents_rate": parents_rate,
        "sel_mutation": sel_mut,
        "tail_mutation": tail_mut,
        "elapsed_sec": elapsed,
        **res,
    }
import time
import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Any


from src.algorithms.ga.particle_swarm import run_pso
from src.algorithms.ga.common import make_transactions_builder, calibrate_min_len_via_builder

import time
import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Any

from src.algorithms.ga.particle_swarm import run_pso
from src.algorithms.ga.common import make_transactions_builder, calibrate_min_len_via_builder


def run_pso_w1(
    graph_links: List[Tuple[str, str]],
    P: np.ndarray,
    agent_info_dict: Dict[str, Any],
    hp: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Execute the PSO for a single environment using the provided graph, prices, and hyperparameters.
    """
    txb = make_transactions_builder(graph_links)
    primary_nodes = [u for u, _ in graph_links if not any(v == u for _, v in graph_links)]

    base_L = max(2, len(primary_nodes) + 1)
    L_min = calibrate_min_len_via_builder(txb, base_L=base_L)
    
    genome_shape = L_min

    generations = hp.get("generations", 15)

    fix_last_gene = hp.get("fix_last_gene", True)

    c1 = hp.get("c1", 0.3)
    c2 = hp.get("c2", 0.3)
    w = hp.get("w", 0.9)
    popsize = hp.get("popsize", 30)

    seed = hp.get("seed", 42)
    verbosity = hp.get("verbosity", 1)

    res = run_pso(
        production_graph=graph_links,
        pmatrix=P,
        agents_information=agent_info_dict,
        genome_shape=genome_shape,
        generations=generations,
        popsize=popsize,
        c1=c1,
        c2=c2,
        w=w,
        fix_last_gene=fix_last_gene,
        seed=seed,
        verbosity=verbosity,
    )

    return {"algorithm": "pso", **res}


def run_pso_w2(
    env_dict: Dict[str, Any],
    base_hyperparams: Dict[str, Any],
    candidate_hyperparams: Dict[str, Any],
    env_id: str | int | None = None,
) -> Dict[str, Any]:
    """Execute PSO on a full environment setup."""
    graph_links = env_dict["production_graph"]
    price_tensor = env_dict["price_matrix"]
    agent_info_dict = env_dict["agents_info"]

    hp = {**base_hyperparams, **candidate_hyperparams}
    c1 = hp["c1"]
    c2 = hp["c2"]
    w = hp["w"]
    generations = hp["generations"]
    popsize = hp["popsize"]

    env_name = f"env_{env_id}" if env_id is not None else "unknown_env"
    print(f"\n[▶] Starting PSO | {env_name} | C1={c1:.3f}, C2={c2:.3f}, W={w:.3f}, "
          f"gens={generations}, pop={popsize}")

    start_time = time.time()
    res = run_pso_w1(graph_links, price_tensor, agent_info_dict, hp)
    elapsed = time.time() - start_time

    print(f"[✔] Completed {env_name} | Elapsed: {elapsed:.2f}s\n")

    return {
        **res,
        "env_id": env_id,
        "elapsed_sec": elapsed,
    }

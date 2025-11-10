
from __future__ import annotations

import pandas as pd 

import numpy as np 

from typing import Dict, List, Tuple, Union 

import networkx as nx 



import json
import csv
import yaml
import numpy as np
from pathlib import Path
from collections import defaultdict, deque 
import random
import os 
import datetime 

###----------------------------------------
#
#
# Self Contained that generates a test suite environment to conduct fine tuning using given parameters
#
#------------------------------------------



#-------------------------------------- 1. Graph Generation Functions------------------------------

def generate_max_edges(max_number_nodes: int) -> list[int]:
    """
    Generate the possible range of edge counts for a DAG with `max_number_nodes` nodes.
    The minimum number of edges is (n - 1) — corresponding to the directed path from input to output -like structure.
    The maximum number of edges is n*(n - 1)/2 — a fully connected DAG.
    """
    n = max_number_nodes
    min_edges = n - 1
    max_edges = n * (n - 1) // 2  # use integer division
    edges = list(range(min_edges, max_edges + 1))
    return edges


def get_percentile(list_of_values: list[int], percentile: int) -> int:
    return int(np.percentile(list_of_values, percentile))

def generate_number_edges(max_graph_nodes:int,percentile:int):

    """
    Takes the maximum number of nodes and a percentile and returns the number of edges for the graph

    that correspond to the qth percenile of the number of edges given the number of nodes 
    """

    edge_list = generate_max_edges(max_graph_nodes)

    return get_percentile(edge_list,percentile)




def generate_dag(max_number_nodes: int, num_edges: int, max_depth: int, rng: np.random.Generator):
    """
    Generates a layered DAG with N nodes (X0..X{N-1}) satisfying:
      • Exactly one sink (final good) located at the deepest layer.
      • At least one source exists.
      • Every node has a directed path to the final good.
      • Edges always go from lower to higher layers (acyclic).
    The number of edges is constrained to `num_edges`.
    """
    N = max_number_nodes 
    # --- 1) Assign nodes to layers -------------------------------------------
    all_nodes = [f"X{i}" for i in range(N)]
    layers: list[list[str]] = [[] for _ in range(max_depth)]
    for node in all_nodes:
        layer = int(rng.integers(0, max_depth))
        layers[layer].append(node)

    # Ensure deepest layer non-empty
    if not layers[-1]:
        donor_layers = [i for i in range(max_depth - 1) if layers[i]]
        src_li = int(rng.choice(donor_layers)) if donor_layers else 0
        node = layers[src_li].pop(0)
        layers[-1].append(node)

    # Choose a unique sink (final good)
    final_good = str(rng.choice(layers[-1]))
    others = [n for n in layers[-1] if n != final_good]
    if others:
        target_layer = max(0, max_depth - 2)
        layers[target_layer].extend(others)
        layers[-1] = [final_good]

    # --- 2) Build list of all possible forward edges -------------------------
    possible_edges = []
    for li in range(max_depth - 1):
        for lj in range(li + 1, max_depth):
            for u in layers[li]:
                for v in layers[lj]:
                    if u != v and v != u:  # avoid self-loops
                        possible_edges.append((u, v))

    # Remove any outgoing edges from the sink
    possible_edges = [(u, v) for (u, v) in possible_edges if u != final_good]

    # --- 3) Sample the specified number of edges -----------------------------
    # Limit to the total number of possible edges
    num_edges = min(num_edges, len(possible_edges))
    sampled_edges = rng.choice(len(possible_edges), size=num_edges, replace=False)
    edges = [possible_edges[i] for i in sampled_edges]

    # --- 4) Ensure reachability to the final good ----------------------------
    def build_rev_adj(edgelist):
        rev = defaultdict(list)
        for u, v in edgelist:
            rev[v].append(u)
        return rev

    def reachable_to_sink(edgelist):
        rev = build_rev_adj(edgelist)
        seen = set()
        q = deque([final_good])
        while q:
            x = q.popleft()
            if x in seen:
                continue
            seen.add(x)
            for pred in rev[x]:
                if pred not in seen:
                    q.append(pred)
        return seen

    layer_of = {n: li for li, nodes in enumerate(layers) for n in nodes}

    reachable = reachable_to_sink(edges)
    for u in all_nodes:
        if u in reachable:
            continue
        candidates = [v for v in reachable if layer_of[v] > layer_of[u]]
        if not candidates:
            candidates = [final_good]
        v = str(rng.choice(candidates))
        edges.append((u, v))
        reachable = reachable_to_sink(edges)

    # --- 5) Ensure at least one source ---------------------------------------
    targets = {v for _, v in edges}
    sources = [n for n in all_nodes if n not in targets]
    if not sources:
        non_sink_edges = [(u, v) for (u, v) in edges if v != final_good]
        cut_from = non_sink_edges if non_sink_edges else edges
        if cut_from:
            u, v = cut_from[int(rng.integers(0, len(cut_from)))]
            edges.remove((u, v))

    return edges


def generate_multiple_dags(
    max_number_nodes: int,
    num_edges: int,
    max_depth: int,
    k: int,
    seed: int | None = None
) -> dict[str, list[tuple[str, str]]]:
    """
    Generate k random DAGs with the same parameters.

    Parameters
    ----------
    N : int
        Number of nodes in each DAG.
    num_edges : int
        Desired number of edges per DAG.
    max_depth : int
        Number of possible layers (controls acyclicity).
    k : int
        Number of DAGs to generate.
    seed : int | None
        Optional random seed for reproducibility.

    Returns
    -------
    dict[str, list[tuple[str, str]]]
        Dictionary of graphs: {"G0": [...edges...], "G1": [...], ...}
    """
    N = max_number_nodes 
    rng = np.random.default_rng(seed)
    return {f"G{i}": generate_dag(N, num_edges, max_depth, rng) for i in range(k)}



#---------------------------------- Price Matrix generation Function 

# -----------------------------------------------------------------------------
# Price matrices: monotone-on-average 3x3 blocks with multiplicative noise
# -----------------------------------------------------------------------------

def generate_price_tensor(
    max_number_nodes: int,
    start: float,
    step: float,
    n_agents: int,
    noise_cfg: dict,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generates a tensor of shape (max_number_nodes, n_agents, n_agents)
    where each [i,:,:] is a price matrix centered at (start + i*step)
    with optional multiplicative perturbation.
    Ensures all resulting prices are >= 1.
    """
    N = max_number_nodes
    size = n_agents
    tensor = np.zeros((N, size, size), dtype=float)

    std = float(noise_cfg.get("std", 0.0))
    dist = str(noise_cfg.get("distribution", "normal"))
    apply_noise = std > 0

    for i in range(N):
        base_value = start + i * step
        mat = np.full((size, size), base_value, dtype=float)

        if apply_noise:
            if dist == "normal":
                noise = rng.normal(1.0, std, size=(size, size))
            elif dist == "uniform":
                noise = rng.uniform(1.0 - std, 1.0 + std, size=(size, size))
            else:
                raise ValueError(f"Unknown noise distribution: {dist}")

            mat *= noise
            # Recenter to preserve the mean
            mat *= base_value / np.mean(mat)

        # --- Ensure prices >= 1 ---
        mat = np.clip(mat, a_min=1.0, a_max=None)

        tensor[i] = mat

    return tensor


def generate_suite_dictionary(
    max_number_nodes: int,
    edges_percentile: float,
    max_depth: int,
    k: int,
    start: float,
    step: float,
    n_agents: int,
    noise_cfg: dict,
    rng: np.random.Generator,
) -> dict[str, dict]:
    """
    Generate a test suite of k random environments, each containing:
      • A random DAG (production network)
      • A full 3D price tensor of shape (nodes, agents, agents)
    """
    num_edges = generate_number_edges(max_number_nodes, edges_percentile)
    dags = generate_multiple_dags(max_number_nodes, num_edges, max_depth, k, rng)

    suite = {}
    for i in range(k):
        env_name = f"env{i+1}"
        price_tensor = generate_price_tensor(
            max_number_nodes, start, step, n_agents, noise_cfg, rng
        )
        suite[env_name] = {
            "graph": dags[f"G{i}"],
            "prices": price_tensor.tolist(),  # JSON-safe
        }

    return suite


def export_test_suite_to_json(
    output_dir: str,
    max_number_nodes: int,
    percentile: float,
    max_depth: int,
    k: int,
    start: float,
    step: float,
    n_agents: int,
    noise_cfg: dict,
    seed: int | None = None,
) -> str:
    """Generate suite and export to JSON."""
    rng = np.random.default_rng(seed)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")

    num_edges = generate_number_edges(max_number_nodes, percentile)

    suite = generate_suite_dictionary(
        max_number_nodes=max_number_nodes,
        edges_percentile=percentile,
        max_depth=max_depth,
        k=k,
        start=start,
        step=step,
        n_agents=n_agents,
        noise_cfg=noise_cfg,
        rng=rng,
    )

    suite_with_meta = {
        "used_params": {
            "max_number_nodes": max_number_nodes,
            "percentile": percentile,
            "num_edges": num_edges,
            "max_depth": max_depth,
            "k": k,
            "start": start,
            "step": step,
            "n_agents": n_agents,
            "noise_cfg": noise_cfg,
            "seed": seed,
            "generated_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
    }
    suite_with_meta.update(suite)

    os.makedirs(output_dir, exist_ok=True)
    filename = f"envsuite_{k}envs_{timestamp}.json"
    output_path = os.path.join(output_dir, filename)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(suite_with_meta, f, indent=4, ensure_ascii=False)

    print(f"✅ Test suite exported successfully to: {output_path}")
    return output_path
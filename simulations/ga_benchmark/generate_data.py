#!/usr/bin/env python3
# =============================================================================
# Data generator for GA benchmark (graphs + price matrices + index)
# -----------------------------------------------------------------------------
# This script is intentionally self-contained under simulations/ga_benchmark/.
# It reads the YAML config in ./configs/global_config.yaml and materializes
# synthetic inputs into ./data/{graphs,prices}/ plus an env_index.csv.
# =============================================================================

from __future__ import annotations

import json
import csv
import yaml
import numpy as np
from pathlib import Path
from collections import defaultdict
import random


# -----------------------------------------------------------------------------
# Robust config loader bound to this package folder
# -----------------------------------------------------------------------------
def _base_dir() -> Path:
    """Resolves the directory that contains this script."""
    return Path(__file__).resolve().parent


def load_config(path: str | Path | None = None) -> dict:
    """
    Loads the global YAML configuration. If 'path' is not provided, it defaults
    to simulations/ga_benchmark/configs/global_config.yaml relative to this file.
    """
    if path is None:
        path = _base_dir() / "configs" / "global_config.yaml"
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# -----------------------------------------------------------------------------
# Graph generator: layered DAG with probabilistic edges forward in depth
# -----------------------------------------------------------------------------
def generate_dag(N: int, density: float, max_depth: int, rng: np.random.Generator):
    """
    Generates a layered DAG with N nodes (X0..X{N-1}) satisfying:
      • Exactly one sink (final good) located at the deepest layer.
      • At least one source exists.
      • Every node has a directed path to the final good.
      • Edges always go from lower to higher layers (acyclic).
    """
    # --- 1) Create full node set and assign layers ---------------------------
    all_nodes = [f"X{i}" for i in range(N)]
    layers: list[list[str]] = [[] for _ in range(max_depth)]
    for node in all_nodes:
        layer = int(rng.integers(0, max_depth))
        layers[layer].append(node)

    # Ensure the deepest layer is non-empty
    if not layers[-1]:
        # move one random node to the deepest layer
        donor_layer_idxs = [i for i in range(max_depth - 1) if layers[i]]
        src_li = int(rng.choice(donor_layer_idxs)) if donor_layer_idxs else 0
        node = layers[src_li].pop(0)
        layers[-1].append(node)

    # Choose a unique sink in the deepest layer
    final_good = str(rng.choice(layers[-1]))
    # Move any other nodes found in the deepest layer to the previous one,
    # so only 'final_good' remains at the last layer. This keeps edges strictly forward.
    others = [n for n in layers[-1] if n != final_good]
    if others:
        # If there is no previous layer (max_depth==1), put them in layer 0
        target_layer = max(0, max_depth - 2)
        layers[target_layer].extend(others)
        layers[-1] = [final_good]

    # --- 2) Sample forward edges between layers ------------------------------
    edges: list[tuple[str, str]] = []
    for li in range(max_depth - 1):
        for lj in range(li + 1, max_depth):
            for u in layers[li]:
                for v in layers[lj]:
                    if v == final_good and rng.random() < density:
                        edges.append((u, v))
                    elif v != final_good and rng.random() < density:
                        edges.append((u, v))

    # Remove any outgoing edges from the sink to keep it a sink
    edges = [(u, v) for (u, v) in edges if u != final_good]

    # --- 3) Ensure reachability to the final good ----------------------------
    # Build layer index map for forward-only additions
    layer_of = {n: li for li, layer_nodes in enumerate(layers) for n in layer_nodes}

    from collections import defaultdict, deque
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

    reachable = reachable_to_sink(edges)

    # For any node not yet reaching the sink, add a forward edge into any node
    # that already reaches the sink and is in a deeper layer (or the sink itself).
    for u in all_nodes:
        if u in reachable:
            continue
        # find candidates in deeper layers that already reach sink
        candidates = [v for v in reachable if layer_of[v] > layer_of[u]]
        if not candidates:
            # fallback: connect directly to the sink (always deeper)
            candidates = [final_good]
        v = str(rng.choice(candidates))
        edges.append((u, v))
        # update reachability incrementally
        reachable = reachable_to_sink(edges)

    # --- 4) Ensure at least one source (node with no incoming edges) ---------
    targets = {v for _, v in edges}
    sources = [n for n in all_nodes if n not in targets]
    if not sources:
        # cut one random incoming edge to create a source
        # prefer cutting from a non-sink destination to avoid breaking reachability
        non_sink_edges = [(u, v) for (u, v) in edges if v != final_good]
        cut_from = non_sink_edges if non_sink_edges else edges
        if cut_from:
            u, v = cut_from[int(rng.integers(0, len(cut_from)))]
            edges.remove((u, v))

    return edges




# -----------------------------------------------------------------------------
# Price matrices: monotone-on-average 3x3 blocks with multiplicative noise
# -----------------------------------------------------------------------------
def generate_price_matrices(
    N: int,
    start: float,
    step: float,
    size: int,
    noise_cfg: dict,
    rng: np.random.Generator,
):
    """
    Produces N square matrices (size x size). Matrix 'i' is centered at
    (start + i * step) with optional multiplicative noise, then re-centered
    to keep the target mean invariant (stabilizes sweeps).
    """
    matrices = []
    for i in range(N):
        base_value = start + i * step
        mat = np.full((size, size), base_value, dtype=float)

        if noise_cfg.get("enabled", False):
            std = float(noise_cfg.get("std", 0.05))
            dist = str(noise_cfg.get("distribution", "normal"))
            if dist == "normal":
                noise = rng.normal(1.0, std, size=(size, size))
            elif dist == "uniform":
                noise = rng.uniform(1.0 - std, 1.0 + std, size=(size, size))
            else:
                raise ValueError(f"Unknown noise distribution: {dist}")

            mat *= noise
            # Mean correction to preserve the intended global trend
            mean_target = base_value
            mat *= (mean_target / float(np.mean(mat)))

        matrices.append(mat.tolist())
    return matrices


# -----------------------------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------------------------
def main():
    cfg = load_config()
    graphs_cfg = cfg["graphs"]
    prices_cfg = cfg["prices"]

    rng = np.random.default_rng(graphs_cfg.get("generator_seed", None))

    # All outputs live under ./simulations/ga_benchmark/data/
    base = _base_dir()
    graphs_out = base / "data" / "graphs"
    prices_out = base / "data" / "prices"
    graphs_out.mkdir(parents=True, exist_ok=True)
    prices_out.mkdir(parents=True, exist_ok=True)

    # Central index that ties a graph with its corresponding price file
    env_index_path = base / "data" / "env_index.csv"

    with open(env_index_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["graph_id", "N", "density", "max_depth", "price_file"])

        graph_count = 0
        for N in graphs_cfg["sizes"]:
            for level in graphs_cfg["complexity_levels"]:
                name = level["name"]
                density = float(level["density"])
                max_depth = int(level["max_depth"])

                for rep in range(int(graphs_cfg["num_graphs_per_level"])):
                    gid = f"g{N}_{name}_{rep}"
                    edges = generate_dag(N, density, max_depth, rng)

                    graph_json = {
                        "graph_id": gid,
                        "N": N,
                        "density": density,
                        "max_depth": max_depth,
                        "edges": edges,
                    }
                    (graphs_out / f"{gid}.json").write_text(
                        json.dumps(graph_json, indent=2), encoding="utf-8"
                    )

                    matrices = generate_price_matrices(
                        N=N,
                        start=float(prices_cfg["base_value_start"]),
                        step=float(prices_cfg["base_value_step"]),
                        size=int(prices_cfg["matrix_size"]),
                        noise_cfg=prices_cfg.get("noise", {}),
                        rng=rng,
                    )
                    price_json = {
                        "price_id": f"prices_{gid}",
                        "graph_id": gid,
                        "N": N,
                        "type": prices_cfg["type"],
                        "step": prices_cfg["base_value_step"],
                        "noise": prices_cfg.get("noise", {}),
                        "matrices": matrices,
                    }
                    price_file = f"{gid}_prices.json"
                    (prices_out / price_file).write_text(
                        json.dumps(price_json, indent=2), encoding="utf-8"
                    )

                    writer.writerow([gid, N, density, max_depth, price_file])
                    graph_count += 1

    print(f"✅ Generated {graph_count} graphs, price matrices and index: {env_index_path}")


if __name__ == "__main__":
    main()

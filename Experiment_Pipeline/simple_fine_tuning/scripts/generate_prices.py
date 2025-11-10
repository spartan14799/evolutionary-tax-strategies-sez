#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Builds an agent-to-agent price tensor from a directed acyclic production graph stored in JSON.

Pipeline:
1) Load a graph definition from JSON (name + list of edges).
2) Compute a topological ordering and classify nodes as primary, intermediate, or final.
3) Compute scalar prices per good via recursive input-cost accumulation and markups.
4) Expand each scalar price into an (n_agents x n_agents) matrix with an optional
   row-wise markup for final goods and optional off-diagonal lognormal noise.
5) Persist the resulting tensor as .npy and write metadata as .json.
6) Optionally emit a combined environment JSON with graph and prices.

The command-line interface allows specifying pricing parameters and output naming.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def load_graph(json_path: Path) -> Tuple[str, List[Tuple[str, str]]]:
    """
    Loads a production graph from a JSON file.

    Args:
        json_path (Path): Absolute or relative path to the JSON file. The file
            must contain at least the key "edges", where edges is a list of
            two-element lists representing directed edges [u, v]. An optional
            "name" key may be provided.

    Returns:
        Tuple[str, List[Tuple[str, str]]]: A tuple containing:
            - The graph name as a string (falls back to "UnnamedGraph" if absent).
            - The list of directed edges as tuples (u, v).

    Raises:
        FileNotFoundError: If the JSON file does not exist.
        json.JSONDecodeError: If the file contents are not valid JSON.
        ValueError: If the JSON does not contain a non-empty "edges" list.
    """
    data = json.loads(json_path.read_text(encoding="utf-8"))
    name = str(data.get("name", "UnnamedGraph"))
    raw_edges = data.get("edges", [])
    edges = [tuple(e) for e in raw_edges]
    if not edges:
        raise ValueError("The JSON file does not contain a non-empty 'edges' list.")
    return name, edges


def topo_sort(nodes: List[str], edges: List[Tuple[str, str]]) -> List[str]:
    """
    Computes a topological ordering using Kahn's algorithm.

    Args:
        nodes (List[str]): The list of node identifiers present in the graph.
        edges (List[Tuple[str, str]]): The list of directed edges (u, v) where
            u -> v.

    Returns:
        List[str]: A topological ordering of the nodes such that all edges point
        forward in the order.

    Raises:
        ValueError: If a topological ordering cannot be found (the graph is not a DAG).
    """
    indeg: Dict[str, int] = {u: 0 for u in nodes}
    succ: Dict[str, List[str]] = {u: [] for u in nodes}
    for u, v in edges:
        indeg[v] = indeg.get(v, 0) + 1
        succ[u] = succ.get(u, [])
        succ[u].append(v)

    queue = [u for u in nodes if indeg[u] == 0]
    order: List[str] = []
    i = 0
    while i < len(queue):
        u = queue[i]
        i += 1
        order.append(u)
        for w in succ.get(u, []):
            indeg[w] -= 1
            if indeg[w] == 0:
                queue.append(w)

    if len(order) != len(nodes):
        raise ValueError("The graph appears to be cyclic; topological sort failed.")
    return order


def classify_nodes(
    nodes: List[str],
    edges: List[Tuple[str, str]]
) -> Tuple[Dict[str, str], Dict[str, List[str]], Dict[str, List[str]]]:
    """
    Classifies nodes by role and builds predecessor/successor lists.

    A node is classified as:
      - "primary"      if indegree == 0
      - "final"        if outdegree == 0
      - "intermediate" otherwise

    Args:
        nodes (List[str]): The list of node identifiers present in the graph.
        edges (List[Tuple[str, str]]): The list of directed edges (u, v) where
            u -> v.

    Returns:
        Tuple[Dict[str, str], Dict[str, List[str]], Dict[str, List[str]]]: A tuple with:
            - classes: mapping node -> role ("primary" | "intermediate" | "final")
            - preds: mapping node -> list of predecessor nodes
            - succ: mapping node -> list of successor nodes
    """
    preds: Dict[str, List[str]] = {u: [] for u in nodes}
    succ: Dict[str, List[str]] = {u: [] for u in nodes}
    for u, v in edges:
        succ[u].append(v)
        preds[v].append(u)

    classes: Dict[str, str] = {}
    for u in nodes:
        if len(preds[u]) == 0:
            classes[u] = "primary"
        elif len(succ[u]) == 0:
            classes[u] = "final"
        else:
            classes[u] = "intermediate"
    return classes, preds, succ


def build_price_tensor(
    nodes_order: List[str],
    preds: Dict[str, List[str]],
    classes: Dict[str, str],
    base: float,
    m1: float,
    m2: float,
    m3: float,
    n_agents: int,
    rng: np.random.Generator,
    noise_sigma: float = 0.0,
) -> np.ndarray:
    """
    Builds a price tensor of shape (n_goods, n_agents, n_agents) given a graph and pricing rules.

    The scalar price per good g is computed recursively as:
      - primary: base
      - intermediate: (1 + m1) * sum(price_of(pred) for pred in preds[g])
      - final: (1 + m2) * sum(price_of(pred) for pred in preds[g])

    Each scalar price is then expanded to an (n_agents x n_agents) matrix. For final goods,
    the first row is additionally multiplied by (1 + m3) to implement a row-wise markup.
    If noise_sigma > 0, off-diagonal entries receive multiplicative lognormal noise.

    The final tensor is rounded up (ceil) and clipped to integers >= 1.

    Args:
        nodes_order (List[str]): Nodes in topological order.
        preds (Dict[str, List[str]]): Mapping node -> list of predecessor nodes.
        classes (Dict[str, str]): Mapping node -> role label.
        base (float): Base price assigned to primary goods.
        m1 (float): Markup for intermediate goods.
        m2 (float): Markup for final goods.
        m3 (float): Additional row-wise markup for the first row of final goods matrices.
        n_agents (int): The number of agents per dimension in each price matrix.
        rng (np.random.Generator): Random generator for noise sampling.
        noise_sigma (float): Standard deviation of the lognormal noise. If 0.0, noise is disabled.

    Returns:
        np.ndarray: Integer tensor of shape (n_goods, n_agents, n_agents) with prices >= 1.
    """
    scalar_price: Dict[str, float] = {}

    def price_of(g: str) -> float:
        """Computes and caches the scalar price for node g."""
        if g in scalar_price:
            return scalar_price[g]
        cls = classes[g]
        if cls == "primary":
            val = base
        else:
            total = sum(price_of(p) for p in preds[g])
            if cls == "intermediate":
                val = (1.0 + m1) * total
            elif cls == "final":
                val = (1.0 + m2) * total
            else:
                raise ValueError(f"Unknown node class: {cls}")
        scalar_price[g] = float(val)
        return scalar_price[g]

    # Compute scalar prices in topological order for determinism
    for g in nodes_order:
        price_of(g)

    # Expand to n_agents x n_agents matrices and apply row-wise markup to finals
    T = np.zeros((len(nodes_order), n_agents, n_agents), dtype=float)
    for idx, g in enumerate(nodes_order):
        mat = np.full((n_agents, n_agents), scalar_price[g], dtype=float)
        if classes[g] == "final":
            mat[0, :] *= (1.0 + m3)
        T[idx] = mat

    # Apply off-diagonal lognormal multiplicative noise if requested
    if noise_sigma and noise_sigma > 0.0:
        for i in range(T.shape[0]):
            for r in range(n_agents):
                for c in range(n_agents):
                    if r == c:
                        continue  # preserve diagonal entries
                    factor = rng.lognormal(mean=0.0, sigma=noise_sigma)
                    T[i, r, c] *= factor

    # Ceil to integer-like prices and enforce lower bound
    T = np.ceil(T)
    T = np.clip(T, a_min=1, a_max=None).astype(int)
    return T


def main() -> None:
    """
    Entry point for the CLI that generates a price tensor and metadata files.

    Args:
        None

    Returns:
        None
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", type=Path, required=True, help="Path to the JSON graph file.")
    ap.add_argument("--out-dir", type=Path, required=True, help="Directory where outputs are written.")
    ap.add_argument("--name", type=str, default="FT2_Prices_V1", help="Base name for output files.")
    ap.add_argument("--base", type=float, default=10.0, help="Base price for primary goods.")
    ap.add_argument("--m1", type=float, default=0.25, help="Markup for intermediate goods.")
    ap.add_argument("--m2", type=float, default=0.50, help="Markup for final goods.")
    ap.add_argument("--m3", type=float, default=0.10, help="Additional row-wise markup for final goods (row 0).")
    ap.add_argument("--n-agents", type=int, default=3, help="Number of agents per dimension in each price matrix.")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    ap.add_argument("--noise-sigma", type=float, default=0.0, help="Stddev of lognormal multiplicative noise (off-diagonal).")
    ap.add_argument("--emit-env-json", action="store_true", help="If set, writes a combined environment JSON with graph+prices.")
    args = ap.parse_args()

    graph_name, edges = load_graph(args.graph)
    nodes = sorted(set([u for u, _ in edges] + [v for _, v in edges]))

    order = topo_sort(nodes, edges)
    classes, preds, _succ = classify_nodes(nodes, edges)

    rng = np.random.default_rng(args.seed)
    tensor = build_price_tensor(
        nodes_order=order,
        preds=preds,
        classes=classes,
        base=args.base,
        m1=args.m1,
        m2=args.m2,
        m3=args.m3,
        n_agents=args.n_agents,
        rng=rng,
        noise_sigma=args.noise_sigma,
    )

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    npy_path = out_dir / f"{args.name}_tensor.npy"
    meta_path = out_dir / f"{args.name}_meta.json"
    np.save(npy_path, tensor)

    meta = {
        "graph_file": str(args.graph.as_posix()),
        "graph_name": graph_name,
        "output_name": args.name,
        "nodes_topological": order,
        "node_index_map": {g: i for i, g in enumerate(order)},
        "n_agents": args.n_agents,
        "shape": list(tensor.shape),
        "params": {
            "base": args.base,
            "m1": args.m1,
            "m2": args.m2,
            "m3": args.m3,
            "seed": args.seed,
            "noise_sigma": args.noise_sigma
        }
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    if args.emit_env_json:
        env_json = {
            "name": f"{graph_name}__{args.name}",
            "graph": edges,
            "prices": tensor.tolist()
        }
        env_path = out_dir / f"{args.name}_env.json"
        env_path.write_text(json.dumps(env_json), encoding="utf-8")

    print(f"✅ Tensor saved to: {npy_path}")
    print(f"✅ Metadata saved to: {meta_path}")
    if args.emit_env_json:
        print(f"✅ Environment JSON saved to: {out_dir / (args.name + '_env.json')}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Builds a normalized environment dictionary for downstream wrappers from an environment JSON.

The input JSON is expected to have the following structure:
{
  "name": "...",
  "graph": [["A","D"], ...],
  "prices": [[[...], ...], ...]  # 3D tensor: (n_goods, n_agents, n_agents)
}

This loader:
1) Parses and validates the graph and the price tensor.
2) Derives the set of goods from the edge list (unique node labels).
3) Assembles a per-agent configuration block (agents_info) with a provided
   chart-of-accounts YAML path and a firm-related-goods list populated from the graph.
4) Returns a Python dictionary structured as:
   {
     "production_graph": List[Tuple[str, str]],
     "price_matrix": np.ndarray,                # shape (n_goods, n_agents, n_agents)
     "agents_info": Dict[str, Dict[str, Any]]
   }

Optionally, it can persist a JSON-normalized version where numpy arrays are serialized to lists.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


@dataclass(frozen=True)
class EnvDict:
    """
    Container for the normalized environment dictionary.

    Attributes:
        production_graph (List[Tuple[str, str]]): Directed edges (u, v).
        price_matrix (np.ndarray): Price tensor of shape (n_goods, n_agents, n_agents).
        agents_info (Dict[str, Dict[str, Any]]): Per-agent configuration block.
    """
    production_graph: List[Tuple[str, str]]
    price_matrix: np.ndarray
    agents_info: Dict[str, Dict[str, Any]]


def _extract_nodes_from_edges(edges: List[Tuple[str, str]]) -> List[str]:
    """
    Extracts a deterministic, sorted list of nodes from a directed edgelist.

    Args:
        edges (List[Tuple[str, str]]): Directed edges (u, v).

    Returns:
        List[str]: Sorted list of unique node identifiers.
    """
    nodes: set[str] = set()
    for u, v in edges:
        nodes.add(str(u))
        nodes.add(str(v))
    return sorted(nodes)


def _load_env_json(env_json_path: Path) -> Tuple[List[Tuple[str, str]], np.ndarray]:
    """
    Loads the environment JSON and converts the price tensor to a numpy array.

    Args:
        env_json_path (Path): Path to the environment JSON file.

    Returns:
        Tuple[List[Tuple[str, str]], np.ndarray]:
            A tuple with:
            - List of directed edges (u, v) as tuples of strings.
            - Price tensor with dtype=int and shape (n_goods, n_agents, n_agents).

    Raises:
        FileNotFoundError: If the file is missing.
        json.JSONDecodeError: If the file is not valid JSON.
        ValueError: If required keys are missing or tensor has invalid shape.
    """
    data = json.loads(env_json_path.read_text(encoding="utf-8"))

    raw_edges = data.get("graph")
    if not isinstance(raw_edges, list) or not raw_edges:
        raise ValueError("Key 'graph' must be a non-empty list of edges [u, v].")
    edges = [(str(u), str(v)) for (u, v) in raw_edges]

    raw_prices = data.get("prices")
    if raw_prices is None:
        raise ValueError("Key 'prices' is required in the environment JSON.")
    tensor = np.array(raw_prices, dtype=int)

    if tensor.ndim != 3:
        raise ValueError(f"Price tensor must be 3D; received shape={tensor.shape}.")
    if tensor.shape[1] != tensor.shape[2]:
        raise ValueError("Price tensor's last two dimensions must be square matrices (n_agents x n_agents).")

    return edges, tensor


def _build_agents_template(accounts_yaml_path: Path, n_agents: int) -> Dict[str, Dict[str, Any]]:
    """
    Builds a minimal agents_info template with stable defaults.

    Args:
        accounts_yaml_path (Path): Filesystem path to the chart of accounts YAML file.
        n_agents (int): Number of agents; defines valid range for price_mapping indices.

    Returns:
        Dict[str, Dict[str, Any]]: Dictionary keyed by agent name.

    Notes:
        - If n_agents < 3, price_mapping indices will be clamped to the range [0, n_agents-1].
        - This template can be extended as needed by the consuming system.
    """
    def _idx(i: int) -> int:
        return max(0, min(n_agents - 1, i))

    return {
        "MKT": {
            "type": "MKT",
            "inventory_strategy": "FIFO",
            "firm_related_goods": [],
            "income_statement_type": "standard",
            "accounts_yaml_path": str(accounts_yaml_path),
            "price_mapping": _idx(0),
        },
        "NCT": {
            "type": "NCT",
            "inventory_strategy": "FIFO",
            "firm_related_goods": [],
            "income_statement_type": "standard",
            "accounts_yaml_path": str(accounts_yaml_path),
            "price_mapping": _idx(1),
        },
        "ZF": {
            "type": "ZF",
            "inventory_strategy": "FIFO",
            "firm_related_goods": [],
            "income_statement_type": "standard",
            "accounts_yaml_path": str(accounts_yaml_path),
            "price_mapping": _idx(2),
        },
    }


def build_environment_dict(env_json_path: Path, accounts_yaml_path: Path) -> EnvDict:
    """
    Constructs an EnvDict from a combined environment JSON and a chart-of-accounts YAML path.

    Args:
        env_json_path (Path): Path to the environment JSON containing keys 'graph' and 'prices'.
        accounts_yaml_path (Path): Path to a chart-of-accounts YAML file that agents will reference.

    Returns:
        EnvDict: A structured container with production_graph, price_matrix, and agents_info.
    """
    edges, tensor = _load_env_json(env_json_path)
    goods = _extract_nodes_from_edges(edges)
    n_agents = int(tensor.shape[1])

    agents_info = _build_agents_template(accounts_yaml_path, n_agents)
    # Populate firm-related goods deterministically
    for cfg in agents_info.values():
        cfg["firm_related_goods"] = goods

    return EnvDict(
        production_graph=edges,
        price_matrix=tensor,
        agents_info=agents_info,
    )


def save_env_dict_json(env: EnvDict, output_path: Path) -> None:
    """
    Serializes EnvDict to a JSON file. The price tensor is written as nested lists.

    Args:
        env (EnvDict): Environment dictionary container to serialize.
        output_path (Path): Destination path for the JSON file.

    Returns:
        None
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "production_graph": env.production_graph,
        "price_matrix": env.price_matrix.tolist(),
        "agents_info": env.agents_info,
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    """
    CLI entry point for building and optionally saving a normalized environment dictionary.

    Args:
        None

    Returns:
        None
    """
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--env-json",
        type=Path,
        required=True,
        help="Path to the combined environment JSON (graph + prices).",
    )
    ap.add_argument(
        "--accounts-yaml",
        type=Path,
        required=True,
        help="Path to the chart-of-accounts YAML file to attach to agents.",
    )
    ap.add_argument(
        "--save-json",
        type=Path,
        default=None,
        help="If provided, saves the normalized env dict as JSON at this path.",
    )
    args = ap.parse_args()

    env = build_environment_dict(env_json_path=args.env_json, accounts_yaml_path=args.accounts_yaml)

    # Optional persistence
    if args.save_json is not None:
        save_env_dict_json(env, args.save_json)

    # Console summary
    print(f"Environment built successfully.")
    print(f"- Edges: {len(env.production_graph)}")
    print(f"- Goods (inferred): {len(_extract_nodes_from_edges(env.production_graph))}")
    print(f"- Price tensor shape: {tuple(env.price_matrix.shape)}")
    print(f"- Agents: {', '.join(env.agents_info.keys())}")
    if args.save_json is not None:
        print(f"- Saved normalized env JSON to: {args.save_json}")


if __name__ == "__main__":
    main()

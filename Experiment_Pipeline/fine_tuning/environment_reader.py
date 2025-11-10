from __future__ import annotations

from typing import Any, Dict, List, Tuple, Optional
import copy
import json

import numpy as np
import networkx as nx

import os 

#####################################################################################
# Test Suite reading, parsing and formatting for GA Wrappers
#####################################################################################


def create_base_suite_dict_from_json(json_path: str) -> Dict[str, Dict[str, Any]]:
    """
    Load and normalize environments from a JSON file.

    The function scans top-level keys that start with "env" (case-insensitive) and,
    for each environment, extracts:
      - a production graph represented as a list of directed edges (u, v)
      - a price matrix as a NumPy float array

    Non-environment keys (e.g., "used_params") are ignored.

    Args:
        json_path (str):
            Absolute or relative path to a JSON file whose structure is:
            {
              "env1": {
                "graph": [["X0", "X1"], ["X1", "X3"], ...],
                "prices": [[...], [...], ...]
              },
              "env2": { ... },
              "used_params": { ... }   # (ignored)
            }

    Returns:
        Dict[str, Dict[str, Any]]:
            A mapping:
            {
              "env1": {
                  "production_graph": List[Tuple[str, str]],  # list of edges (u, v)
                  "price_matrix": np.ndarray                  # shape (A, A) or similar
              },
              "env2": { ... }
            }

    Raises:
        FileNotFoundError:
            If `json_path` does not exist.
        ValueError:
            If no valid environment entries are discovered.
        json.JSONDecodeError:
            If the file is not valid JSON.

    Notes:
        - Edges are coerced to strings and filtered to pairs.
        - Price matrices are coerced to float dtype.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    env_suite: Dict[str, Dict[str, Any]] = {}
    for key, value in data.items():
        if not str(key).lower().startswith("env"):
            continue

        graph_in = value.get("graph", [])
        edges: List[Tuple[str, str]] = [
            (str(e[0]), str(e[1]))
            for e in graph_in
            if isinstance(e, (list, tuple)) and len(e) == 2
        ]

        price_matrix = np.array(value.get("prices", []), dtype=float)

        env_suite[str(key)] = {
            "production_graph": edges,
            "price_matrix": price_matrix,
        }

    if not env_suite:
        raise ValueError(f"No valid environment entries found in '{json_path}'")
    return env_suite


def _extract_nodes_from_edges_fast(edges: List[Tuple[str, str]]) -> List[str]:
    """
    Extract unique node labels from a list of edges using pure Python.

    Args:
        edges (List[Tuple[str, str]]): List of directed edges (u, v).

    Returns:
        List[str]: Sorted list of unique node labels.

    Notes:
        - Time complexity: O(E), where E is the number of edges.
        - This avoids constructing a NetworkX graph and is generally faster
          and lighter for large suites where only the node set is needed.
    """
    nodes: set[str] = set()
    for u, v in edges:
        nodes.add(u)
        nodes.add(v)
    # Sorting is optional; keep deterministic output for reproducibility
    return sorted(nodes)


def fill_env_dict(
    environment_dict: Dict[str, Any],
    agents_info: Dict[str, Dict[str, Any]],
    chart_of_accounts: str,
) -> Dict[str, Any]:
    """
    Populate a single environment with agent information.

    This function is intended to operate on a single environment entry
    (e.g., the value of `suite['env1']`) and attach a deep-copied
    `agents_info` block after enriching it with:
      - `accounts_yaml_path` set to the provided chart of accounts path
      - `firm_related_goods` populated with all goods (nodes) inferred from edges

    Args:
        environment_dict (Dict[str, Any]):
            Environment dictionary with at least:
            {
              "production_graph": List[Tuple[str, str]],
              "price_matrix": np.ndarray
            }
        agents_info (Dict[str, Dict[str, Any]]):
            Base agent configuration to attach; typical structure:
            {
              "MKT": {"type": "...", "price_mapping": 0, ...},
              "NCT": {"type": "...", "price_mapping": 1, ...},
              "ZF":  {"type": "...", "price_mapping": 2, ...}
            }
            This input is **not** mutated; a deep copy is made internally.
        chart_of_accounts (str):
            Filesystem path to the YAML chart of accounts to assign to each agent.
        use_networkx (bool, optional):
            If True, uses `networkx.from_edgelist(...).nodes()` to get goods.
            If False (default), uses a faster pure-Python extractor.
        validate_prices (bool, optional):
            If True (default), validates that `price_matrix` is 2-D and numeric.
        validate_price_mappings (bool, optional):
            If True (default), checks that each agent's `price_mapping` index
            falls within the first axis of `price_matrix`.

    Returns:
        Dict[str, Any]:
            A **new** environment dictionary with an added key:
            {
              "production_graph": ...,
              "price_matrix": ...,
              "agents_info": {
                  "MKT": {..., "accounts_yaml_path": chart_of_accounts,
                               "firm_related_goods": [goods...]},
                  "NCT": {...},
                  "ZF":  {...}
              }
            }

    Raises:
        KeyError:
            If required keys are missing in `environment_dict`.
        TypeError:
            If `price_matrix` is not array-like or not numeric when validation is on.
        ValueError:
            If validation of shapes or indices fails.

    Notes:
        - Uses deep copies to prevent cross-environment side effects when reusing
          the same `agents_info` template across all envs.
        - For node extraction, pure-Python is recommended for performance unless
          you specifically need a NetworkX graph object anyway.
    """
    if "production_graph" not in environment_dict:
        raise KeyError("environment_dict must contain 'production_graph'")
    if "price_matrix" not in environment_dict:
        raise KeyError("environment_dict must contain 'price_matrix'")

    edges: List[Tuple[str, str]] = environment_dict["production_graph"]
    price_matrix = environment_dict["price_matrix"]

    goods = _extract_nodes_from_edges_fast(edges)

    # Prepare agents (deep copy to avoid shared state)
    agents_out: Dict[str, Dict[str, Any]] = copy.deepcopy(agents_info)
    for agent_name, cfg in agents_out.items():
        cfg["accounts_yaml_path"] = chart_of_accounts
        cfg["firm_related_goods"] = goods


    # Return a NEW environment dict with agents attached
    out_env = dict(environment_dict)
    out_env["agents_info"] = agents_out
    return out_env


def populate_suite_agents_info(
    env_suite: Dict[str, Dict[str, Any]],
    base_agents_info: Dict[str, Dict[str, Any]],
    chart_of_accounts: str,
) -> Dict[str, Dict[str, Any]]:
    """
    Vectorized convenience wrapper: apply `fill_env_dict` to every environment.

    Args:
        env_suite (Dict[str, Dict[str, Any]]):
            Output of `create_base_suite_dict_from_json`, i.e.:
            {
              "env1": {"production_graph": [...], "price_matrix": np.ndarray},
              "env2": {...},
              ...
            }
        base_agents_info (Dict[str, Dict[str, Any]]):
            Template agents dictionary to be deep-copied per environment.
        chart_of_accounts (str):
            Filesystem path to the YAML chart of accounts to assign to each agent.
        use_networkx (bool, optional):
            Delegate to `fill_env_dict`. Defaults to False (faster pure-Python).
        validate_prices (bool, optional):
            Delegate to `fill_env_dict`. Defaults to True.
        validate_price_mappings (bool, optional):
            Delegate to `fill_env_dict`. Defaults to True.

    Returns:
        Dict[str, Dict[str, Any]]:
            A new suite dict with, for every env key:
            {
              "production_graph": ...,
              "price_matrix": ...,
              "agents_info": {...}
            }

    Notes:
        - Each environment gets an independent deep copy of `base_agents_info`.
        - Safe to reuse the same `base_agents_info` across many calls.
    """
    populated: Dict[str, Dict[str, Any]] = {}
    for env_key, env_val in env_suite.items():
        populated[env_key] = fill_env_dict(
            environment_dict=env_val,
            agents_info=base_agents_info,
            chart_of_accounts=chart_of_accounts,
        )
    return populated




def generate_suite_dict_from_json(json_path: str, 
                                  agents_info: Dict[str, Dict[str, Any]],
                                  chart_of_accounts: str) -> Dict[str, Dict[str, Any]]:
    
    base_dict = create_base_suite_dict_from_json(json_path)
    
    result = populate_suite_agents_info(
        env_suite=base_dict,
        base_agents_info=agents_info,
        chart_of_accounts=chart_of_accounts,
    )
    
    return result
    







# Test 

"""
if __name__ == "__main__":
    pass



JSON_BASENAME = "envsuite_10envs_20251018_1517.json"

exp_pipeline_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

json_env_path = os.path.join(
        exp_pipeline_dir, "test_suite", "output_test_suites", JSON_BASENAME
    )

ACCOUNTS_PATH = ".../chart_of_accounts.yaml"

BASE_AGENTS: Dict[str, Dict[str, Any]] = {
    "MKT": {
        "type": "MKT",
        "inventory_strategy": "FIFO",
        "firm_related_goods": [],
        "income_statement_type": "standard",
        "accounts_yaml_path": ACCOUNTS_PATH,
        "price_mapping": 0,
    },
    "NCT": {
        "type": "NCT",
        "inventory_strategy": "FIFO",
        "firm_related_goods": [],
        "income_statement_type": "standard",
        "accounts_yaml_path": ACCOUNTS_PATH,
        "price_mapping": 1,
    },
    "ZF": {
        "type": "ZF",
        "inventory_strategy": "FIFO",
        "firm_related_goods": [],
        "income_statement_type": "standard",
        "accounts_yaml_path": ACCOUNTS_PATH,
        "price_mapping": 2,
    },
}

result = generate_suite_dict_from_json(json_env_path, BASE_AGENTS, ACCOUNTS_PATH)


print(result)

"""
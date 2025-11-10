import time
import numpy as np
import networkx as nx
from typing import Dict, Any
import os

# --- GA and environment imports ---
from Experiment_Pipeline.fine_tuning.environment_reader import generate_suite_dict_from_json
from Experiment_Pipeline.fine_tuning.wrappers.pso_wrapper import run_pso_w2


def print_result_summary(result: Dict[str, Any], max_list_items: int = 3) -> None:
    """
    Pretty print summary of GA result dictionary with nested structures.
    Shows scalar values directly, lists with length and preview, and dicts with keys.
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
            keys = ", ".join(v.keys())
            print(f"{k:<22}: dict({len(v)}) keys=[{keys}]")
        else:
            print(f"{k:<22}: {type(v)}")


if __name__ == "__main__":

    JSON_BASENAME = "medium_suite.json"
    ACCOUNTS_FILENAME = "chart_of_accounts.yaml"

    # --- Resolve repository structure ---
    exp_pipeline_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    repo_root = os.path.abspath(os.path.join(exp_pipeline_dir, os.pardir))

    # --- Paths ---
    json_env_path = os.path.join(exp_pipeline_dir, "test_suite", "output_test_suites", JSON_BASENAME)
    accounts_path = os.path.join(repo_root, ACCOUNTS_FILENAME)

    # --- Check existence ---
    if not os.path.exists(json_env_path):
        raise FileNotFoundError(f"JSON environment suite not found: {json_env_path}")

    if not os.path.exists(accounts_path):
        raise FileNotFoundError(f"Chart of accounts file not found: {accounts_path}")

    print(f"[✔] Loaded JSON env suite from: {json_env_path}")
    print(f"[✔] Loaded Chart of Accounts from: {accounts_path}")

    # --- Base agent templates ---
    BASE_AGENTS: Dict[str, Dict[str, Any]] = {
        "MKT": {
            "type": "MKT",
            "inventory_strategy": "FIFO",
            "firm_related_goods": [],
            "income_statement_type": "standard",
            "accounts_yaml_path": None,
            "price_mapping": 0,
        },
        "NCT": {
            "type": "NCT",
            "inventory_strategy": "FIFO",
            "firm_related_goods": [],
            "income_statement_type": "standard",
            "accounts_yaml_path": None,
            "price_mapping": 1,
        },
        "ZF": {
            "type": "ZF",
            "inventory_strategy": "FIFO",
            "firm_related_goods": [],
            "income_statement_type": "standard",
            "accounts_yaml_path": None,
            "price_mapping": 2,
        },
    }

    # --- Build suite ---
    result = generate_suite_dict_from_json(json_env_path, BASE_AGENTS, accounts_path)

    # Pick one environment to test
    test_env = result["env1"]

    print("\n" + "-" * 90)
    print("Testing Joint GA Wrapper on env1")
    print("-" * 90)

    # --- Define hyperparameters ---
    base_hyperparams = {
        "generations": 10,
        "popsize": 10,
        "fix_last_gene": False,
        "seed": 42,
        "verbosity": 1,
        "log_every": 2,
        'k': 3,
        'p': 2
    }

    # Candidate parameters (these are the ones to be tuned)
    candidate_hp = {
        "c1": 0.20,
        "c2": 0.3,
        "w": 0.35,
    }

    # --- Run test ---
    test_result = run_pso_w2(test_env, base_hyperparams, candidate_hp, env_id="env1")


a = print_result_summary(test_result)

print('-------------------------- Best Genome ----------------------------------')

print(test_result['best_genome'])

print('-------------------------- Best Curve ----------------------------------')

print(test_result['curves']['best'])
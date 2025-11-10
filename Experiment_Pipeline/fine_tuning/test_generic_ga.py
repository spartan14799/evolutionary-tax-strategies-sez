import time
import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Any


import os 

from Experiment_Pipeline.fine_tuning.environment_reader import generate_suite_dict_from_json

from Experiment_Pipeline.fine_tuning.wrappers.generic_ga_wrapper import run_generic_ga_w2

if __name__ == "__main__":
    pass



JSON_BASENAME = "hard_suite.json"
ACCOUNTS_FILENAME = "chart_of_accounts.yaml"

# --- Resolve repository structure ---
# This script is inside Experiment_Pipeline/fine_tuning/
exp_pipeline_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

# Repository root: one level above Experiment_Pipeline/
repo_root = os.path.abspath(os.path.join(exp_pipeline_dir, os.pardir))

# --- Paths ---
json_env_path = os.path.join(
    exp_pipeline_dir, "test_suite", "output_test_suites", JSON_BASENAME
)

accounts_path = os.path.join(repo_root, ACCOUNTS_FILENAME)

# --- Check existence ---
if not os.path.exists(json_env_path):
    raise FileNotFoundError(f"JSON environment suite not found: {json_env_path}")

if not os.path.exists(accounts_path):
    raise FileNotFoundError(f"Chart of accounts file not found: {accounts_path}")

print(f"[✔] Loaded JSON env suite from: {json_env_path}")
print(f"[✔] Loaded Chart of Accounts from: {accounts_path}")




BASE_AGENTS: Dict[str, Dict[str, Any]] = {
    "MKT": {
        "type": "MKT",
        "inventory_strategy": "FIFO",
        "firm_related_goods": [],
        "income_statement_type": "standard",
        "accounts_yaml_path": None ,
        "price_mapping": 0,
    },
    "NCT": {
        "type": "NCT",
        "inventory_strategy": "FIFO",
        "firm_related_goods": [],
        "income_statement_type": "standard",
        "accounts_yaml_path":None ,
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

result = generate_suite_dict_from_json(json_env_path, BASE_AGENTS, accounts_path)


test_env = result['env1']

print('----------------------------------------------------------------------------------')
print('----------------------------------------------------------------------------------')
print('----------------------------------------------------------------------------------')
print('----------------------------------------------------------------------------------')
print('----------------------------------------------------------------------------------')

print(test_env)


base_hyperparams = {"generations": 60,
        "popsize": 60,'fix_last_gene': True,'seed': 42,'parent':14,'elitism':1,'seed':42,'verbosity':1,'log_every':1}

candidate_hp = {'mutpb': 0.05, 'cxpb': 0.7}


test = run_generic_ga_w2(test_env, base_hyperparams, candidate_hp)



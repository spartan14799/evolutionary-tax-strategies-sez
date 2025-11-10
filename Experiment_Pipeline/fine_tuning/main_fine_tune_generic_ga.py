# =============================================================================
# Fine-tuning main script
# =============================================================================

import os
import time
import numpy as np
from typing import Dict, Any

from Experiment_Pipeline.fine_tuning.environment_reader import generate_suite_dict_from_json
from Experiment_Pipeline.fine_tuning.fine_tuners.fine_tuner_generic_ga import run_fine_tune_generic_ga

# =============================================================================
# Setup
# =============================================================================

set_seed = 42
rng = np.random.default_rng(set_seed)

JSON_BASENAMES = ["hard_suite.json", "medium_suite.json", "easy_suite.json"]
ACCOUNTS_FILENAME = "chart_of_accounts.yaml"

exp_pipeline_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
repo_root = os.path.abspath(os.path.join(exp_pipeline_dir, os.pardir))


def assign_route(json_basename: str) -> str:
    return os.path.join(exp_pipeline_dir, "test_suite", "output_test_suites", json_basename)


json_env_paths = [assign_route(name) for name in JSON_BASENAMES]
accounts_path = os.path.join(repo_root, ACCOUNTS_FILENAME)

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

# =============================================================================
# Load environments
# =============================================================================

selected_envs = {}
for json_path, difficulty in zip(json_env_paths, ["Hard", "Medium", "Easy"]):
    print(f"[✔] Loading {difficulty} suite from: {json_path}")
    result = generate_suite_dict_from_json(json_path, BASE_AGENTS, accounts_path)
    env_key = rng.choice(list(result.keys()))
    selected_envs[difficulty.lower()] = result[env_key]
    print(f" → Selected {env_key} for {difficulty.lower()}.")

# =============================================================================
# Base hyperparameters and output path
# =============================================================================


base_hyperparams = {"generations": 60,
        "popsize": 60,
        'fix_last_gene': True,
        'seed': set_seed,
        #'parent':14,
        'elitism':1,
        'verbosity':1,
        'log_every':1}


output_csv = os.path.join(
    repo_root, "Experiment_Pipeline", "fine_tuning", "results", "fine_tune_results_generic_ga.csv"
)

# =============================================================================
# Run Fine-tuning and Measure Runtime
# =============================================================================

if __name__ == "__main__":
    start_time = time.time()

    results = run_fine_tune_generic_ga(
        selected_envs=selected_envs,
        base_hyperparams=base_hyperparams,
        output_path=output_csv,
        n_samples=100,
        n_jobs=4,
        seed=set_seed,
    )

    elapsed = time.time() - start_time
    hours, rem = divmod(elapsed, 3600)
    minutes, seconds = divmod(rem, 60)
    print("\n" + "=" * 70)
    print(f"✅ Fine-tuning completed in {int(hours)}h {int(minutes)}m {seconds:.2f}s")
    print(f"Results saved to: {output_csv}")
    print("=" * 70)

"""
EXPERIMENT PIPELINE
-------------------
Full end-to-end pipeline to:
  1. Parse configuration JSON (algorithms, environment, agents)
  2. Build environment database from graph JSON
  3. Parse environments and attach agent-specific info

Returns two dictionaries:
  - algorithms  → merged common + algorithm-specific parameters
  - envs        → environments with prices, links, agents, and budgets
"""

# =====================================================================
# Imports
# =====================================================================
from src.simulation.economy.order_book.utils.price_markup_generator import PriceMarkupGenerator
from src.simulation.economy.production_process.search_space import SearchSpace

import json
import math
import copy
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any, Union

from src.config_paths import resolve_chart_of_accounts_path


# =====================================================================
# 1. LOAD GRAPHS FROM JSON
# =====================================================================
def load_graphs_from_json(json_path: str) -> Dict[str, List[List[str]]]:
    """Load multiple graphs from a JSON file and return {graph_name: links}."""
    json_file = Path(json_path)
    if not json_file.exists():
        raise FileNotFoundError(f"JSON file not found at: {json_file}")

    with open(json_file, "r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise ValueError("Top level of the JSON must be a dictionary.")

    graphs_dict = {}
    for name, gdata in data.items():
        if "links" not in gdata:
            raise KeyError(f"Graph '{name}' lacks 'links' key.")
        graphs_dict[name] = gdata["links"]

    return graphs_dict


# =====================================================================
# 2. PRICE GENERATION
# =====================================================================
def generate_prices(
    links_list: List[Tuple[str, str]],
    base: float,
    m1: float,
    m2: float,
    m3: float,
    n_agents: int,
    perturb_std: float,
    ignored_agents: List[int],
    seed: int = 42,
) -> Dict[str, Dict[str, Any]]:
    """Generate normal and perturbed price tensors for each good/agent."""
    price_gen = PriceMarkupGenerator(links_list)
    price_gen.generate_all_price_tensors(base, m1, m2, m3, n_agents)
    price_gen.generate_perturbations(seed=seed, sigma=perturb_std, ignored_indices=ignored_agents)
    return {"normal_prices": price_gen.good_prices, "perturbed_prices": price_gen.perturbed_prices}


# =====================================================================
# 3. ESTIMATE COMPUTATIONAL BUDGET
# =====================================================================

def estimate_budget(
    links_list: List[Tuple[str, str]],
    alfa: float = 0.05,
    rho: float = 1.0,
    min_budget: int = 1500,
    max_budget: int = 10000,
) -> int:
    """
    Estimate computational budget that scales exponentially with number of links.
    Values below 1500 are raised to 1500; values above 10000 are capped at 10000.
    Intermediate values are left unchanged.
    """
    n_links = len(links_list)

    nodes = set([n for edge in links_list for n in edge])
    n_nodes = len(nodes)

    # Density uses our custom formula for DAG 

    density = (2* n_links) / (n_nodes * (n_nodes - 1))

    est_budget = min_budget * np.exp(alfa * (density ** rho))

    # Only clamp outside the interval
    if est_budget < min_budget:

        budget = 1500

    elif est_budget > max_budget:

        budget = max_budget
    else:
        budget = int(np.ceil(est_budget))

    return int(budget)


# =====================================================================
# 4. ASSEMBLE ENVIRONMENT RECORDS
# =====================================================================
def assemble_env_records(
    graphs_dict: Dict[str, List[Tuple[str, str]]],
    base: float,
    m1: float,
    m2: float,
    m3: float,
    n_agents: int,
    perturb_std: float,
    ignored_agents: List[int],
    fix_last_gene: bool = True,
    # Budget Calculation Variables
    alfa = 0.05,
    rho = 0.05,
    min_budget: int = 100,
    max_budget: int = 10000,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Create a list of environment records (standard + perturbed)."""
    environments = []
    env_id = 1
    for graph_name, links_list in graphs_dict.items():
        prices = generate_prices(links_list, base, m1, m2, m3, n_agents,
                                 perturb_std, ignored_agents, seed)
        
        # Budget Calculation
        budget = estimate_budget(links_list, alfa=alfa, rho=rho, min_budget=min_budget, max_budget=max_budget)

        environments.append({
            "env_id": env_id,
            "graph_type": graph_name,
            "env_name": f"{graph_name}_standard",
            "price_type": "standard",
            "links": links_list,
            "P": prices["normal_prices"],
            "budget": budget,
        })
        env_id += 1

        environments.append({
            "env_id": env_id,
            "graph_type": graph_name,
            "env_name": f"{graph_name}_perturbed",
            "price_type": "perturbed",
            "links": links_list,
            "P": prices["perturbed_prices"],
            "budget": budget,
        })
        env_id += 1
    return environments


# =====================================================================
# 5. EXPORT ENVIRONMENT DATABASE
# =====================================================================
def export_env_database(environments: List[Dict[str, Any]],
                        output_path: str = "environment_database.csv") -> pd.DataFrame:
    """Convert environment records to DataFrame and export to CSV."""
    df = pd.DataFrame(environments)
    df.to_csv(output_path, index=False, encoding="utf-8")
    return df


# =====================================================================
# 6. BUILD ENVIRONMENT DATABASE FROM JSON
# =====================================================================
def build_environment_database_from_json(
    json_path: str,
    base: float,
    m1: float,
    m2: float,
    m3: float,
    n_agents: int,
    perturb_std: float,
    ignored_agents: List[int],
    fix_last_gene: bool = True,

    # Variables for budget Calculation
    alfa = 0.05,
    rho = 0.05,
    min_budget: int = 100,
    max_budget: int = 10000,
    output_csv: str = "environment_database.csv",
    seed: int = 42,
) -> pd.DataFrame:
    """Full pipeline: graphs → prices → budgets → CSV."""
    graphs_dict = load_graphs_from_json(json_path)
    env_records = assemble_env_records(graphs_dict, base, m1, m2, m3,
                                       n_agents, perturb_std, ignored_agents,
                                       fix_last_gene, 
                                       alfa, rho, min_budget,
                                       max_budget, seed)
    df = export_env_database(env_records, output_csv)
    print(f" Environment database saved to: {output_csv}")
    return df


# =====================================================================
# 7. PARSE CONFIGURATION JSON
# =====================================================================
def parse_experiment_config(config_path: str) -> Dict[str, Dict[str, Any]]:
    """
    Parse the experiment configuration JSON into three structured dictionaries:
      - 'algorithms': merged common + algorithm-specific params
      - 'env': environment construction parameters (with yaml_name added)
      - 'agents': base agent definitions
    """
    config_path = Path(config_path)
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    # Core sections
    common = config.get("common", {})
    env_construction = config.get("env_construction", {}).copy()
    base_agents = config.get("BASE_AGENTS", {})

    # Extract yaml_name and inject it into env_construction
    chart_info = config.get("chart_of_accounts", {})
    yaml_name = chart_info.get("yaml_name", None)
    env_construction["yaml_name"] = str(
        resolve_chart_of_accounts_path(yaml_name, base_dir=config_path.parent)
    )

    # Detect algorithm blocks (exclude fixed sections)
    excluded = {"common", "env_construction", "chart_of_accounts", "BASE_AGENTS"}
    algorithm_params = {
        algo: {**common, **config[algo]}
        for algo in config.keys()
        if algo not in excluded
    }

    return {
        "algorithms": algorithm_params,
        "env": env_construction,
        "agents": base_agents,
    }


# =====================================================================
# 8. PARSE ENVIRONMENT DATABASE
# =====================================================================
def parse_environment_database(env_data: Union[str, pd.DataFrame]) -> Dict[str, Dict[str, Any]]:
    """
    Parse the CSV/DF into {env_name: {links, prices, budget}},
    handling both nested-dict and ndarray/list price formats.
    """
    if isinstance(env_data, str):
        df = pd.read_csv(env_data)
    else:
        df = env_data.copy()

    envs = {}
    for _, row in df.iterrows():
        env_name = row["env_name"]
        links = eval(row["links"]) if isinstance(row["links"], str) else row["links"]

        P = row["P"]
        # Handle stored price format (stringified JSON/dict or ndarray)
        if isinstance(P, str):
            P = eval(P)  # safely reconstruct original Python object

        # Case 1: nested dict (usual output)
        if isinstance(P, dict):
            goods = list(P.keys())
            first_good_val = next(iter(P.values()))
            # Case 1A: dict of dicts {good: {agent: price}}
            if isinstance(first_good_val, dict):
                agents = list(first_good_val.keys())
                prices = np.array([[P[g][a] for a in agents] for g in goods], dtype=float)
            # Case 1B: dict of lists (already numerical rows)
            else:
                prices = np.array(list(P.values()), dtype=float)

        # Case 2: already ndarray or list
        else:
            prices = np.array(P, dtype=float)

        envs[env_name] = {
            "links": links,
            "prices": prices,
            "budget": row["budget"],
        }

    return envs

# =====================================================================
# 9. ATTACH AGENTS TO ENVIRONMENTS
# =====================================================================
def attach_agents_to_envs(envs: Dict[str, Dict[str, Any]],
                          base_agents: Dict[str, Any],
                          accounts_yaml_path: str) -> Dict[str, Dict[str, Any]]:
    """Deep copy agents, assign YAML path, and populate firm_related_goods."""
    envs_full = {}
    for env_name, env in envs.items():
        links = env["links"]
        goods = sorted({n for e in links for n in e})

        agents_copy = copy.deepcopy(base_agents)
        for agent in agents_copy.values():
            agent["accounts_yaml_path"] = accounts_yaml_path
            agent["firm_related_goods"] = goods

        envs_full[env_name] = {**env, "agents_info": agents_copy}
    return envs_full


# =====================================================================
# 10. FULL WRAPPER PIPELINE
# =====================================================================
def full_environment_pipeline(graphs_json_path: str,
                              config_json_path: str,
                              output_csv: str = "environment_database.csv"):
    """
    1. Parse configuration JSON
    2. Build environment database
    3. Parse and enrich environments with agents
    """
    cfg = parse_experiment_config(config_json_path)
    env_cfg = cfg["env"]
    base_agents = cfg["agents"]

    df_envs = build_environment_database_from_json(
        json_path=graphs_json_path,
        base=env_cfg["base"],
        m1=env_cfg["m1"],
        m2=env_cfg["m2"],
        m3=env_cfg["m3"],
        n_agents=env_cfg["n_agents"],
        perturb_std=env_cfg["perturb_std"],
        ignored_agents=env_cfg["ignored_agents"],

        #Parameters for the budget calculation

        alfa =env_cfg["alfa"],
        rho = env_cfg["rho"],
        min_budget = env_cfg["min_budget"],
        max_budget = env_cfg["max_budget"],
        output_csv=output_csv,
        seed=env_cfg.get("seed", 42),
    )

    parsed_envs = parse_environment_database(df_envs)
    envs_with_agents = attach_agents_to_envs(parsed_envs, base_agents, env_cfg["yaml_name"])
    return cfg["algorithms"], envs_with_agents



# ============================================================================

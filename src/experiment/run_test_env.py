import os
import time
import random
import numpy as np
import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, Any

# --- Import all wrappers ---
from src.experiment.wrappers.flat_wrapper import run_flat_w1
from src.experiment.wrappers.generic_ga_wrapper import run_generic_ga_w1
from src.experiment.wrappers.joint_wrapper import run_joint_w1
from src.experiment.wrappers.random_search_wrapper import run_blindrandom_w1
from src.experiment.wrappers.pso_wrapper import run_pso_w1

# --- New Macro–Micro family wrappers ---
from src.experiment.wrappers.macro_micro_wrapper import run_macro_micro_ga_w1
from src.experiment.wrappers.macro_wrapper import run_macro_w1
from src.experiment.wrappers.micro_wrapper import run_micro_w1
from src.experiment.wrappers.recomb_wrapper import run_recomb_w1
from src.experiment.wrappers.no_crossover_wrapper import no_crossover_w1
from src.experiment.wrappers.mixed_generic import run_mixed_generic_w1


# ============================================================================
# Single run executor
# ============================================================================
def _execute_single_run(
    alg_name: str,
    wrapper,
    env_name: str,
    env_dict: Dict[str, Any],
    hp: Dict[str, Any],
    seed: int,
    run_id: int
) -> Dict[str, Any]:
    """Execute one algorithm run (isolated for parallel processing)."""
    pid = os.getpid()
    final_seed = (seed + pid + run_id * 7919) % (2**32 - 1)
    np.random.seed(final_seed)
    random.seed(final_seed)
    start_time = time.time()

    try:
        res = wrapper(
            graph_links=env_dict["links"],
            P=env_dict["prices"],
            agent_info_dict=env_dict["agents_info"],
            hp=hp,
        )
        elapsed = time.time() - start_time
        return {
            "env_name": env_name,
            "run": run_id,
            "algorithm": alg_name,
            "seed": final_seed,
            "best_curve": res.get("curves", {}).get("best", []),
            "best_value": res.get("best_utility", np.nan),
            "best_genome": res.get("best_genome", []),
            "all_best_genomes": res.get("all_best_genomes", []),
            "elapsed_sec": elapsed,
            "success": True,
        }
    except Exception as e:
        return {
            "env_name": env_name,
            "run": run_id,
            "algorithm": alg_name,
            "seed": final_seed,
            "error": str(e),
            "success": False,
        }


# ============================================================================
# Core routine: run_test_env
# ============================================================================
def run_test_env(
    env_name: str,
    seed: int,
    env_dict: Dict[str, Any],
    chart_of_accounts_path: str,
    runs: int,
    generations: int,
    path_output_db: str,
    configs: Dict[str, Dict[str, Any]],
    max_workers: int = 8
) -> pd.DataFrame:
    """
    Parallelized experimental test for all algorithms on a single environment.
    Automatically detects available algorithms from configs and runs only those.
    """

    # Initialize RNG and seeds
    np.random.seed(seed)
    seeds = np.random.randint(0, 10**6, size=runs)

    # Inject chart of accounts path into agents
    for ainfo in env_dict["agents_info"].values():
        ainfo["accounts_yaml_path"] = chart_of_accounts_path

    # Determine popsize
    budget = env_dict.get("budget", 10000)
    popsize = max(1, int(budget / generations))

    # Prepare DB path
    Path(path_output_db).parent.mkdir(parents=True, exist_ok=True)
    db_cols = [
        "env_name", "run", "algorithm", "seed", "best_curve", "best_value",
        "best_genome", "all_best_genomes", "elapsed_sec", "success"
    ]
    if Path(path_output_db).exists():
        db_df = pd.read_csv(path_output_db)
    else:
        db_df = pd.DataFrame(columns=db_cols)

    # ----------------------------------------------------------------------
    # Define available algorithm wrappers (expanded set)
    # ----------------------------------------------------------------------
    alg_map_all = {
        "flat": run_flat_w1,
        "generic": run_generic_ga_w1,
        "joint": run_joint_w1,
        "pso": run_pso_w1,
        "random": run_blindrandom_w1,
        "macro_micro": run_macro_micro_ga_w1,
        "macro": run_macro_w1,
        "micro": run_micro_w1,
        "recomb": run_recomb_w1,
        "no_crossover": no_crossover_w1,   
        "mixed_generic": run_mixed_generic_w1,
    }

    # Only keep algorithms that are present in configs
    available_in_config = {k: v for k, v in alg_map_all.items() if k in configs}

    if not available_in_config:
        raise ValueError(" ERROR No matching algorithms found between configs and wrappers!")

    print(f"\n Algorithms to run on {env_name}: {', '.join(available_in_config.keys())}")

    # ----------------------------------------------------------------------
    # Run all algorithms in parallel
    # ----------------------------------------------------------------------
    total_tasks = len(available_in_config) * runs
    completed = 0
    t0 = time.time()

    print(f" Starting {total_tasks} tasks on environment '{env_name}'...")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for alg_name, wrapper in available_in_config.items():
            base_hp = configs[alg_name].copy()
            base_hp["popsize"] = popsize
            base_hp["generations"] = generations

            for run_id, s in enumerate(seeds, 1):
                hp = base_hp.copy()
                hp["seed"] = int(s)
                futures.append(
                    executor.submit(
                        _execute_single_run,
                        alg_name, wrapper, env_name, env_dict, hp, int(s), run_id
                    )
                )

        for fut in as_completed(futures):
            completed += 1
            res = fut.result()

            if res["success"]:
                print(
                    f" {res['algorithm']} | {env_name} | Run {res['run']} | "
                    f"Best={res['best_value']:.4f} | Time={res['elapsed_sec']:.1f}s | seed={res['seed']}"
                )
            else:
                print(
                    f" {res['algorithm']} | {env_name} | Run {res['run']} failed: {res['error']} | seed={res['seed']}"
                )

            db_df = pd.concat([db_df, pd.DataFrame([res])], ignore_index=True)
            db_df.to_csv(path_output_db, index=False)

            elapsed_total = time.time() - t0
            print(
                f"Progress: {completed}/{total_tasks} ({100*completed/total_tasks:.1f}%) | "
                f"Elapsed: {elapsed_total:.1f}s",
                end="\r"
            )

    print(f"\n Finished {env_name}: {total_tasks} runs total in {time.time()-t0:.1f}s")
    db_df.to_csv(path_output_db, index=False)
    return db_df


# ============================================================================
# Multi-environment orchestrator
# ============================================================================
def run_all_envs(
    dicts_env: Dict[str, Dict[str, Any]],
    configs: Dict[str, Dict[str, Any]],
    chart_of_accounts_path: str,
    runs: int,
    generations: int,
    path_output_db: str,
    base_seed: int = 42,
    max_workers: int = 8,
) -> pd.DataFrame:
    """
    Execute all algorithms on all environments and record results.
    """
    print("\n Global experiment summary:")
    print(f"   → Algorithms detected: {', '.join(configs.keys())}")
    print(f"   → Total environments: {len(dicts_env)}")
    print(f"   → Runs per algorithm: {runs}")
    print(f"   → Generations: {generations}")
    print("-" * 60)

    all_dfs = []
    for i, (env_name, env_dict) in enumerate(dicts_env.items(), 1):
        env_seed = base_seed + i * 1000
        print(f"\n{'='*80}\n ENV {i}/{len(dicts_env)}: {env_name} | Seed: {env_seed}\n{'='*80}")
        df_env = run_test_env(
            env_name=env_name,
            seed=env_seed,
            env_dict=env_dict,
            chart_of_accounts_path=chart_of_accounts_path,
            runs=runs,
            generations=generations,
            path_output_db=path_output_db,
            configs=configs,
            max_workers=max_workers,
        )
        all_dfs.append(df_env)

    final_df = pd.concat(all_dfs, ignore_index=True)
    final_df.to_csv(path_output_db, index=False)
    print(f"\n All environments complete. Consolidated results: {path_output_db}")
    return final_df

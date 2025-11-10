# =============================================================================
# Fine-tuning runner for easy, medium, and hard difficulty scenarios
# =============================================================================

import os
import csv
import time
import json
import numpy as np
import multiprocessing as mp
from scipy.stats import qmc
from tqdm import tqdm
from typing import Dict, List, Tuple, Any

from Experiment_Pipeline.fine_tuning.wrappers.joint_wrapper import run_joint_w2


# =============================================================================
# 0. Auxiliary Functions
# =============================================================================

def parse_candidate_hp(params: Tuple[float, float, float]) -> Dict[str, Any]:
    """
    Parse a tuple of 3 hyperparameters into a candidate dictionary.

    Parameters
    ----------
    params : tuple(float, float, float)
        (sel_mutation, tail_mutation, parents_rate).

    Returns
    -------
    dict
        Dictionary with parsed hyperparameters.
    """
    return {
        "sel_mutation": params[0],
        "tail_mutation": params[1],
        "parents_rate": params[2],
    }


# =============================================================================
# 1. Candidate Evaluation
# =============================================================================

def test_candidate(
    candidate_tuple: Tuple[float, float, float],
    selected_envs: Dict[str, Dict[str, Any]],
    base_hyperparams: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Test a single GA hyperparameter candidate across all difficulty levels.

    Parameters
    ----------
    candidate_tuple : tuple(float, float, float)
        Candidate (sel_mutation, tail_mutation, parents_rate).
    selected_envs : dict
        Dictionary with environments for "easy", "medium", "hard".
    base_hyperparams : dict
        Common hyperparameters (e.g., generations, popsize, etc.).

    Returns
    -------
    dict
        {
          "candidate": (sel_mutation, tail_mutation, parents_rate),
          "results": {"easy": [...], "medium": [...], "hard": [...]}
        }
    """
    candidate_hp = parse_candidate_hp(candidate_tuple)
    print(
        f"\n[▶] Testing candidate: "
        f"sel_mutation={candidate_hp['sel_mutation']:.3f}, "
        f"tail_mutation={candidate_hp['tail_mutation']:.3f}, "
        f"parents_rate={candidate_hp['parents_rate']:.3f}"
    )

    difficulty_results: Dict[str, List[float]] = {}

    for diff, env in selected_envs.items():
        print(f"   → Running GA on {diff.capitalize()} environment...")
        try:
            res = run_joint_w2(
                env_dict=env,
                base_hyperparams=base_hyperparams,
                candidate_hyperparams=candidate_hp,
                env_id=diff,
            )

            curves = res.get("curves", {})
            best_curve = curves.get("best", [])

            difficulty_results[diff] = best_curve

            if best_curve:
                print(f"     [✔] {diff.capitalize()} | Best utility = {max(best_curve):.4f}")
            else:
                print(f"     [⚠] Empty curve returned.")

        except Exception as e:
            print(f"     [⚠] Error running GA on {diff}: {e}")
            difficulty_results[diff] = []

    return {"candidate": candidate_tuple, "results": difficulty_results}


# =============================================================================
# 1.5. Worker Function (for multiprocessing)
# =============================================================================

def worker_test_candidate(args: Tuple[Tuple[float, float, float], Dict[str, Any], Dict[str, Any]]) -> Dict[str, Any]:
    """Top-level wrapper for multiprocessing."""
    candidate_tuple, selected_envs, base_hyperparams = args
    return test_candidate(candidate_tuple, selected_envs, base_hyperparams)


# =============================================================================
# 2. Fine-tuning Routine
# =============================================================================

def run_fine_tune_joint(
    selected_envs: Dict[str, Dict[str, Any]],
    base_hyperparams: Dict[str, Any],
    output_path: str,
    n_samples: int = 100,
    n_jobs: int = 4,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Fine-tune GA hyperparameters (sel_mutation, tail_mutation, parents_rate)
    via Latin Hypercube sampling. Logs results continuously so partial progress
    is not lost.

    Parameters
    ----------
    selected_envs : dict
        Dictionary of environments ("easy", "medium", "hard").
    base_hyperparams : dict
        Base GA hyperparameters.
    output_path : str
        Path to CSV file for saving continuous logs.
    n_samples : int, optional
        Number of candidates to test. Default = 100.
    n_jobs : int, optional
        Number of parallel workers. Default = 4.
    seed : int, optional
        Random seed. Default = 42.

    Returns
    -------
    list of dict
        List of candidate results.
    """
    rng = np.random.default_rng(seed)
    sampler = qmc.LatinHypercube(d=3, seed=rng)
    candidate_grid = sampler.random(n=n_samples)

    # Scale from [0, 1] → [0.01, 0.99]
    candidate_grid = 0.01 + candidate_grid * (0.99 - 0.01)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if not os.path.exists(output_path):
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "sel_mutation", "tail_mutation", "parents_rate",
                "max_hard", "max_medium", "max_easy",
                "curve_hard", "curve_medium", "curve_easy",
                "timestamp_sec"
            ])

    # -------------------------------------------------------------
    # Log result (keep CSV format but include full curves serialized)
    # -------------------------------------------------------------
    def log_result(candidate_result: Dict[str, Any]) -> None:
        sel_mutation, tail_mutation, parents_rate = candidate_result["candidate"]
        results = candidate_result["results"]

        hard_curve = results.get("hard", [])
        med_curve = results.get("medium", [])
        easy_curve = results.get("easy", [])

        # JSON-serialize the curves
        hard_curve_str = json.dumps(hard_curve)
        med_curve_str = json.dumps(med_curve)
        easy_curve_str = json.dumps(easy_curve)

        # Max values for quick database indexing
        max_hard = max(hard_curve) if hard_curve else np.nan
        max_med = max(med_curve) if med_curve else np.nan
        max_easy = max(easy_curve) if easy_curve else np.nan

        with open(output_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                sel_mutation, tail_mutation, parents_rate,
                max_hard, max_med, max_easy,
                hard_curve_str, med_curve_str, easy_curve_str,
                time.time()
            ])

        print(
            f"[✔] Saved candidate (sel_mutation={sel_mutation:.3f}, "
            f"tail_mutation={tail_mutation:.3f}, parents_rate={parents_rate:.3f}) → "
            f"hard={max_hard:.3f}, medium={max_med:.3f}, easy={max_easy:.3f}"
        )

    # -------------------------------------------------------------
    # Run evaluations
    # -------------------------------------------------------------
    total = len(candidate_grid)
    results: List[Dict[str, Any]] = []
    print(f"\n[🚀] Starting fine-tune over {total} candidates using {n_jobs} workers...\n")

    with mp.Pool(processes=n_jobs) as pool:
        for candidate_result in tqdm(
            pool.imap_unordered(
                worker_test_candidate,
                [(tuple(c), selected_envs, base_hyperparams) for c in candidate_grid],
            ),
            total=total,
            desc="Evaluating candidates",
            ncols=90,
            dynamic_ncols=True,
            mininterval=2,
        ):
            results.append(candidate_result)
            log_result(candidate_result)

    print("\n[🏁] Fine-tuning complete. Results saved to:", output_path)
    return results

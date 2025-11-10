#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Runs all selected algorithms (Generic GA, Joint GA, PSO) over a fixed environment and seed set.

For each algorithm:
- loads the per-algorithm candidate list (Latin Hypercube samples),
- builds the environment dictionary once (shared across algorithms),
- dispatches each (candidate × seed) to the proper wrapper,
- collects curves (best/mean/median and std/variance if present),
- computes diagnostics (AUC, improvements, plateau, volatility),
- stores best genome and all genomes that reached the maximum (if exposed),
- writes a compact JSON result per run to <out_base>/<algo>/.

Optionally, it emits a per-algorithm CSV with core metrics and hyperparameters.

It supports sharding across multiple machines and multiprocessing on one host.
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

# --------------------------------------------------------------------------------------
# Ensure repository root is importable (so "Experiment_Pipeline" resolves)
# --------------------------------------------------------------------------------------
_THIS = Path(__file__).resolve()
# Layout assumption: .../FTZ_model_2.0/Experiment_Pipeline/fine_tuning_2/scripts/run_all.py
_REPO_ROOT = _THIS.parents[3]  # → .../FTZ_model_2.0
if str(_REPO_ROOT) not in sys.path:
    sys.path.append(str(_REPO_ROOT))

# --------------------------------------------------------------------------------------
# Local utilities and loaders (relative to this project layout)
# --------------------------------------------------------------------------------------
from Experiment_Pipeline.fine_tuning_2.scripts.ft2_env_loader import build_environment_dict
from Experiment_Pipeline.fine_tuning_2.scripts.utils_metrics import (
    auc_best_and_norm,
    curves_summary,
    per_generation_summary,
)

# Normalized wrappers for this pipeline
from Experiment_Pipeline.fine_tuning_2.wrappers.generic_wrapper import run_generic_w2
from Experiment_Pipeline.fine_tuning_2.wrappers.joint_wrapper import run_joint_w2
from Experiment_Pipeline.fine_tuning_2.wrappers.pso_wrapper import run_pso_w2


# ======================================================================================
# Helpers
# ======================================================================================

def _load_json(path: Path) -> Any:
    """
    Loads a JSON file.

    Args:
        path (Path): Path to the JSON file.

    Returns:
        Any: Parsed JSON object.
    """
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_plan_paths(plan_path: Path, plan: Dict[str, Any]) -> Tuple[Path, Path]:
    """
    Resolves env_json and accounts_yaml paths relative to the plan file.

    Args:
        plan_path (Path): Path to the plan JSON.
        plan (Dict[str, Any]): Parsed plan content.

    Returns:
        Tuple[Path, Path]: (env_json_path, accounts_yaml_path)
    """
    base = plan_path.parent
    env_json = (base / plan["env_json"]).resolve()
    accounts_yaml = (base / plan["accounts_yaml"]).resolve()
    return env_json, accounts_yaml


def _iter_tasks(
    seeds: List[int],
    candidates: List[Dict[str, Any]],
    num_shards: int,
    shard_idx: int,
) -> Iterable[Tuple[int, Dict[str, float], int]]:
    """
    Iterates (candidate_id, params_dict, seed) applying deterministic sharding.

    Args:
        seeds (List[int]): List of seeds.
        candidates (List[Dict[str, Any]]): Candidate list. Each item may be:
            {"id": int, "params": {...}} or {"candidate_id": int, "hparams": {...}}
            or directly a flat dict of params (an id will be assigned by position).
        num_shards (int): Number of shards (>= 1).
        shard_idx (int): Index of this shard in [0, num_shards - 1].

    Returns:
        Iterable[Tuple[int, Dict[str, float], int]]: (candidate_id, params, seed).
    """
    flat: List[Tuple[int, Dict[str, float]]] = []
    for i, c in enumerate(candidates):
        cid = c.get("id", c.get("candidate_id", i))
        params = c.get("params", c.get("hparams"))
        if params is None:
            params = {k: v for k, v in c.items() if k not in ("id", "candidate_id")}
        flat.append((int(cid), dict(params)))

    full: List[Tuple[int, Dict[str, float], int]] = []
    for cid, p in flat:
        for s in seeds:
            full.append((cid, p, int(s)))

    if num_shards <= 1:
        return full
    selected = []
    for idx, triplet in enumerate(full):
        if idx % num_shards == shard_idx:
            selected.append(triplet)
    return selected


def _ensure_mapping_env(env_obj) -> Dict[str, Any]:
    """
    Normalizes an environment object into a plain mapping expected by wrappers.

    Args:
        env_obj: Either a dict-like or an object with attributes
                 production_graph, price_matrix, agents_info.

    Returns:
        Dict[str, Any]: {"production_graph": ..., "price_matrix": ..., "agents_info": ...}

    Raises:
        TypeError: If the required fields are not available.
    """
    if isinstance(env_obj, dict):
        return env_obj
    # dataclass support
    try:
        from dataclasses import is_dataclass, asdict
        if is_dataclass(env_obj):
            d = asdict(env_obj)
            return {
                "production_graph": d["production_graph"],
                "price_matrix": d["price_matrix"],
                "agents_info": d["agents_info"],
            }
    except Exception:
        pass
    # attribute access fallback
    try:
        return {
            "production_graph": env_obj.production_graph,
            "price_matrix": env_obj.price_matrix,
            "agents_info": env_obj.agents_info,
        }
    except AttributeError:
        raise TypeError(
            "Environment must be a mapping with keys "
            "('production_graph','price_matrix','agents_info') "
            "or an object exposing those attributes."
        )


def _safe_dedup_genomes(all_best_genomes: Any) -> Any:
    """
    Deduplicates a list of genomes while preserving order. Returns the original
    object if it is not a list.

    Args:
        all_best_genomes (Any): Candidate genomes to deduplicate.

    Returns:
        Any: Deduplicated list of genomes or the original object.
    """
    if not isinstance(all_best_genomes, list):
        return all_best_genomes
    seen = set()
    uniq: List[List[Any]] = []
    for g in all_best_genomes:
        key = tuple(g) if isinstance(g, (list, tuple)) else (g,)
        if key not in seen:
            seen.add(key)
            uniq.append(list(g) if isinstance(g, (list, tuple)) else [g])
    return uniq


def _algo_run(
    algo: str,
    env_dict: Dict[str, Any],
    base_hp: Dict[str, Any],
    cand_hp: Dict[str, float],
    seed: int
) -> Dict[str, Any]:
    """
    Dispatches the run to the correct wrapper and builds a rich result payload.

    Args:
        algo (str): Algorithm name ("generic" | "joint" | "pso").
        env_dict (Dict[str, Any]): Environment dictionary for the wrappers.
        base_hp (Dict[str, Any]): Baseline hyperparameters (generations, popsize, etc.).
        cand_hp (Dict[str, float]): Candidate hyperparameters for this run.
        seed (int): Seed for reproducibility.

    Returns:
        Dict[str, Any]: Result payload containing metrics, genomes, curves and diagnostics.
    """
    hp = dict(base_hp)
    hp.update(cand_hp)
    hp["seed"] = int(seed)

    t0 = time.time()
    if algo == "generic":
        res = run_generic_w2(env_dict=env_dict, base_hyperparams=base_hp, candidate_hyperparams=hp, env_id="ft2")
    elif algo == "joint":
        res = run_joint_w2(env_dict=env_dict, base_hyperparams=base_hp, candidate_hyperparams=hp, env_id="ft2")
    elif algo == "pso":
        res = run_pso_w2(env_dict=env_dict, base_hyperparams=base_hp, candidate_hyperparams=hp, env_id="ft2")
    else:
        raise ValueError(f"Unsupported algo: {algo}")
    elapsed = time.time() - t0

    curves = res.get("curves", {}) or {}
    best_curve = curves.get("best", []) or []
    auc_raw, auc_norm = auc_best_and_norm(best_curve)
    diag = curves_summary(curves)
    per_gen = per_generation_summary(curves)
    best_final = float(best_curve[-1]) if best_curve else float("nan")

    best_genome = res.get("best_genome", None)
    all_best_genomes = _safe_dedup_genomes(res.get("all_best_genomes", None))

    meta = res.get("meta", {}) or {}
    budget = meta.get("budget", {}) or {}
    budget_triggered = bool(budget.get("triggered", False))
    stop_reason = str(budget.get("reason", ""))

    # --- Evals accounting (non-invasive) ---
    evals_total = -1

    # 1) Preferred: sum of per-generation evals if available
    if isinstance(meta.get("evals_per_gen"), list):
        try:
            evals_total = int(sum(int(x) for x in meta["evals_per_gen"]))
        except Exception:
            evals_total = -1

    # 2) Fallback: last value of a cumulative sequence if available
    if evals_total == -1 and isinstance(meta.get("evals_cum"), list) and meta["evals_cum"]:
        try:
            evals_total = int(meta["evals_cum"][-1])
        except Exception:
            evals_total = -1

    # 3) Final fallback: pull it from meta.budget.evals_total if provided by the wrapper
    if evals_total == -1:
        budget_obj = meta.get("budget", {}) or {}
        bt = budget_obj.get("evals_total", None)
        if isinstance(bt, (int, float)):
            evals_total = int(bt)


    raw_curves = {}
    for k in ("best", "mean", "median", "std", "variance"):
        v = curves.get(k)
        if isinstance(v, list):
            raw_curves[k] = v

    payload = {
        "algo": algo,
        "hparams": hp,
        "metrics": {
            "best_final": best_final,
            "auc_best": auc_raw,
            "auc_best_norm": auc_norm,
            "runtime_sec": float(meta.get("runtime_sec", elapsed)),
            "evals_total": int(evals_total),
            "budget_triggered": budget_triggered,
            "stop_reason": stop_reason,
        },
        "diagnostics": diag,
        "genomes": {
            "best_genome": best_genome,
            "all_best_genomes": all_best_genomes,
        },
        "raw": {"curves": raw_curves},
        "per_generation": per_gen,
        "meta": meta,
    }
    return payload


def _worker(args: Tuple[str, Dict[str, Any], Dict[str, Any], int, int, Dict[str, float], Path, bool]) -> None:
    """
    Worker function executed in parallel to run a single candidate × seed.

    Args:
        args (Tuple[str, Dict[str, Any], Dict[str, Any], int, int, Dict[str, float], Path, bool]):
            (algo, env_dict, base_hp, cid, seed, params, out_dir, overwrite)

    Returns:
        None
    """
    algo, env_dict, base_hp, cid, seed, params, out_dir, overwrite = args
    out_path = out_dir / f"{algo}_c{cid}_s{seed}.json"
    if out_path.exists() and not overwrite:
        return

    try:
        result = _algo_run(algo=algo, env_dict=env_dict, base_hp=base_hp, cand_hp=params, seed=seed)
        result["candidate_id"] = int(cid)
        result["seed"] = int(seed)
        out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception as e:
        error_obj = {
            "algo": algo,
            "candidate_id": int(cid),
            "seed": int(seed),
            "error": f"{type(e).__name__}: {e}",
        }
        out_path.write_text(json.dumps(error_obj, indent=2, ensure_ascii=False), encoding="utf-8")


def _emit_csv_summary(out_dir: Path, algo: str, csv_path: Path) -> None:
    """
    Scans JSON results under `out_dir` and emits a per-algorithm CSV with key metrics
    and hyperparameters. Columns are a superset across algorithms; unused columns
    are left blank.

    Args:
        out_dir (Path): Directory containing per-run JSON files.
        algo (str): Algorithm name ("generic" | "joint" | "pso").
        csv_path (Path): Output CSV path.

    Returns:
        None
    """
    rows: List[Dict[str, Any]] = []
    for p in sorted(out_dir.glob(f"{algo}_c*_s*.json")):
        try:
            obj = _load_json(p)
        except Exception:
            continue
        if "error" in obj:
            rows.append({
                "algo": algo,
                "candidate_id": obj.get("candidate_id"),
                "seed": obj.get("seed"),
                "error": obj.get("error"),
            })
            continue

        metrics = obj.get("metrics", {})
        hp = obj.get("hparams", {})
        row: Dict[str, Any] = {
            "algo": algo,
            "candidate_id": obj.get("candidate_id"),
            "seed": obj.get("seed"),
            "best_final": metrics.get("best_final"),
            "auc_best": metrics.get("auc_best"),
            "auc_best_norm": metrics.get("auc_best_norm"),
            "runtime_sec": metrics.get("runtime_sec"),
            "evals_total": metrics.get("evals_total"),
            "budget_triggered": metrics.get("budget_triggered"),
            "stop_reason": metrics.get("stop_reason"),
            # Generic GA hparams
            "cxpb": hp.get("cxpb"),
            "mutpb": hp.get("mutpb"),
            "mutation_rate": hp.get("mutation_rate"),
            # Joint GA hparams
            "parents_rate": hp.get("parents_rate"),
            "sel_mutation": hp.get("sel_mutation"),
            "tail_mutation": hp.get("tail_mutation"),
            # PSO hparams
            "c1": hp.get("c1"),
            "c2": hp.get("c2"),
            "w": hp.get("w"),
        }
        rows.append(row)

    if not rows:
        return

    fieldnames = [
        "algo", "candidate_id", "seed",
        "best_final", "auc_best", "auc_best_norm",
        "runtime_sec", "evals_total", "budget_triggered", "stop_reason",
        "cxpb", "mutpb", "mutation_rate",
        "parents_rate", "sel_mutation", "tail_mutation",
        "c1", "c2", "w",
        "error",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def _run_for_algo(
    algo: str,
    env_dict: Dict[str, Any],
    base_hp: Dict[str, Any],
    candidates_path: Path,
    out_dir: Path,
    seeds: List[int],
    n_jobs: int,
    num_shards: int,
    shard_idx: int,
    overwrite: bool,
    emit_csv: bool,
) -> None:
    """
    Executes the full grid (candidates × seeds) for a single algorithm.

    Args:
        algo (str): Algorithm name.
        env_dict (Dict[str, Any]): Shared environment dictionary for wrappers.
        base_hp (Dict[str, Any]): Baseline hyperparameters from the plan.
        candidates-dir (Path): Path to the candidates JSON file.
        candidates-file (File): Specific candidates JSON file to run.
        out_dir (Path): Output directory for this algorithm.
        seeds (List[int]): List of seeds to run.
        n_jobs (int): Parallel workers on this host.
        num_shards (int): Total shards to split the grid.
        shard_idx (int): Shard index for this process.
        overwrite (bool): Whether to overwrite existing run files.
        emit_csv (bool): Whether to emit a CSV summary after completion.
        verbosity: Level of verbosity for the wrappers.
    Returns:
        None
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    cand_raw = _load_json(candidates_path)
    if isinstance(cand_raw, dict) and "candidates" in cand_raw:
        candidates = cand_raw["candidates"]
    elif isinstance(cand_raw, list):
        candidates = cand_raw
    else:
        raise ValueError(f"Unrecognized candidates format in: {candidates_path}")

    grid = list(_iter_tasks(seeds=seeds, candidates=candidates, num_shards=int(num_shards), shard_idx=int(shard_idx)))
    work_items = [
        (algo, env_dict, base_hp, cid, seed, params, out_dir, bool(overwrite))
        for (cid, params, seed) in grid
    ]

    print(f"[INFO] {algo}: jobs={len(work_items)} | n_jobs={n_jobs} | shard {shard_idx}/{num_shards}")
    with mp.get_context("spawn").Pool(processes=int(n_jobs)) as pool:
        for _ in pool.imap_unordered(_worker, work_items):
            pass
    print(f"[INFO] {algo}: completed.")

    if emit_csv:
        csv_path = out_dir.parent / f"{algo}_summary.csv"
        _emit_csv_summary(out_dir=out_dir, algo=algo, csv_path=csv_path)
        print(f"[INFO] {algo}: CSV summary written to {csv_path}")


# ======================================================================================
# CLI
# ======================================================================================

def main() -> None:
    """
    CLI entrypoint.

    Args:
        None

    Returns:
        None
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", type=Path, required=True,
                    help="Plan JSON with seeds, env_json, accounts_yaml, and base budgets.")
    ap.add_argument("--candidates-dir", type=Path, required=True,
                    help="Directory containing candidates_{algo}.json files.")
    ap.add_argument("--candidates-file", type=str, default=None,
                help="Specific candidates JSON file to run (overrides default candidates_{algo}.json)")
    ap.add_argument("--out-base", type=Path, required=True,
                    help="Base output directory. Results will be written under <out-base>/<algo>/.")
    ap.add_argument("--algos", type=str, default="generic,joint,pso",
                    help="Comma-separated list of algorithms to run: subset of {generic,joint,pso}.")
    ap.add_argument("--n-jobs", type=int, default=max(1, mp.cpu_count() - 1),
                    help="Parallel workers on this host.")
    ap.add_argument("--num-shards", type=int, default=1,
                    help="Total shards to split each algorithm grid.")
    ap.add_argument("--shard-idx", type=int, default=0,
                    help="Shard index for this process [0..num-shards-1].")
    ap.add_argument("--overwrite", action="store_true",
                    help="Overwrite existing per-run JSON files.")
    ap.add_argument("--emit-csv", action="store_true",
                    help="Emit per-algorithm CSV summaries at the end.")
    ap.add_argument("--verbosity", type=int, default=1,
                help="Verbosity level for algorithm wrappers. 0 = silent.")
    args = ap.parse_args()

    plan = _load_json(args.plan)
    env_json, accounts_yaml = _resolve_plan_paths(args.plan, plan)
    seeds: List[int] = [int(s) for s in plan["seeds"]]

    # Build environment dictionary once and share it
    env_obj = build_environment_dict(env_json, accounts_yaml)
    env_dict = _ensure_mapping_env(env_obj)



    # Base hyperparameters from plan (shared across algorithms)
    base_hp: Dict[str, Any] = {
        "generations": int(plan.get("generations", 100)),
        "popsize": int(plan.get("popsize", 100)),
        "fix_last_gene": True,
        "seed": seeds[0] if seeds else 42,
        "verbosity": int(args.verbosity),
        "log_every": 0 if args.verbosity == 0 else 1,
        "evals_cap": int(plan.get("evals_cap", 10000)),
    }


    # Algorithm selection
    algos = [a.strip().lower() for a in str(args.algos).split(",") if a.strip()]
    valid = {"generic", "joint", "pso"}
    for a in algos:
        if a not in valid:
            raise ValueError(f"Unsupported algo in --algos: {a}")

    # Run each algorithm sequentially; inside each, parallelize across candidates × seeds
    for algo in algos:
        if args.candidates_file:
            cand_path = (args.candidates_dir / args.candidates_file).resolve()
        else:
            cand_path = (args.candidates_dir / f"candidates_{algo}.json").resolve()

        out_dir = (args.out_base / algo).resolve()
        print(f"[INFO] === {algo.upper()} ===")
        print(f"[INFO] Candidates: {cand_path}")
        print(f"[INFO] Output dir : {out_dir}")
        _run_for_algo(
            algo=algo,
            env_dict=env_dict,
            base_hp=base_hp,
            candidates_path=cand_path,
            out_dir=out_dir,
            seeds=seeds,
            n_jobs=int(args.n_jobs),
            num_shards=int(args.num_shards),
            shard_idx=int(args.shard_idx),
            overwrite=bool(args.overwrite),
            emit_csv=bool(args.emit_csv),
        )

    print("[INFO] All selected algorithms completed.")


if __name__ == "__main__":
    main()

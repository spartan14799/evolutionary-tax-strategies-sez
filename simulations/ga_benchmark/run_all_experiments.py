# simulations/ga_benchmark/run_all_experiments.py
# =============================================================================
# Master runner (FLAT | JOINT | EXHAUSTIVE) + METRICS with tidy folders
# =============================================================================

from __future__ import annotations
import sys, os, time, json, csv, shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional
import yaml, pandas as pd, numpy as np
from tqdm import tqdm

try:
    import ctypes
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
except Exception:
    pass

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from algorithms.ga.flat import run_ga_flat
from algorithms.ga.equivclass_joint import run_ga_equivclass_joint
from algorithms.ga.equivclass_exhaustive import run_ga_equivclass_exhaustive
from algorithms.ga.common import detect_prefix_layout_and_sizes

from simulations.ga_benchmark.environment_loader import generate_environment
from simulations.ga_benchmark.metrics import (
    compute_golden_best, summarize_run_against_golden,
    anytime_success, ecdf_time_to_hit, env_complexity_tags,
)

# ---------- utils ----------
def _load_yaml(path: str | Path | None = None) -> dict:
    if path is None:
        path = THIS_DIR / "configs" / "global_config.yaml"
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def _num_goods_from_edges(edges: List[Tuple[str, str]]) -> int:
    s = set(); [s.update(e) for e in edges]; return len(s)

def _extract_size_param(mapping: dict, size: int, default_val: int) -> int:
    try: return int(mapping.get(size, default_val))
    except Exception: return int(default_val)

def _make_exp_dir(root_results: Path, config_path: Path | None, cfg: dict) -> Path:
    tag_left = (config_path.stem if config_path else cfg.get("meta", {}).get("project_name", "experiment")).replace(" ", "_")
    tag = f"{tag_left}__{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    exp_dir = root_results / tag
    (exp_dir / "logs").mkdir(parents=True, exist_ok=True)
    (exp_dir / "aggregates").mkdir(parents=True, exist_ok=True)
    for sub in ("flat", "joint", "exhaustive"):  # subcarpetas por algoritmo
        (exp_dir / sub).mkdir(parents=True, exist_ok=True)
    # snapshot del YAML usado
    with open(exp_dir / "config_used.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
    # pointer 'latest'
    try:
        (root_results / "latest.txt").write_text(str(exp_dir), encoding="utf-8")
    except Exception:
        pass
    return exp_dir

def _log(log_file: Path, msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(line + "\n")

def _estimate_evals_per_combo(pop_tail: int, gens_tail: int, parents: int) -> int:
    """
    Estimación conservadora de evaluaciones por combinación en EXHAUSTIVE:
    - gen0 evalúa ~ pop_tail
    - cada gen reevalúa ≈ pop_tail (cota superior sencilla)
    """
    pop_tail = max(1, int(pop_tail))
    gens_tail = max(0, int(gens_tail))
    return pop_tail * (gens_tail + 1)

# ---------- main ----------
def run_all_experiments(config_path: str | Path | None = None) -> None:
    cfg = _load_yaml(config_path)
    config_path = Path(config_path) if config_path else None

    # paths
    results_root = Path(cfg["paths"]["results_dir"]).resolve()
    exp_dir = _make_exp_dir(results_root, config_path, cfg)
    log_file = exp_dir / "logs" / "benchmark.log"
    _log(log_file, "=== Benchmark started ===")

    # algos & seeds
    algo_list = [k.lower() for k in cfg.get("algorithms", {}).keys()]
    if not algo_list:
        _log(log_file, "No algorithms configured under 'algorithms'."); return
    seeds = cfg.get("output", {}).get("random_seeds", [11,17,23,42])

    # metrics defaults
    mcfg = cfg.get("metrics", {}) or {}
    epsilon = float(mcfg.get("epsilon", 1e-9))
    budgets_grid = list(map(int, mcfg.get("budgets_evals", [200,500,1000,3000,10000])))

    # budgets (global + per-algo override)
    bcfg = cfg.get("budgets", {}) or {}
    global_evals_cap = int(bcfg["evals_cap"]) if bcfg.get("evals_cap") else None
    global_time_limit_sec = (float(bcfg["time_limit_minutes"]) * 60.0) if bcfg.get("time_limit_minutes") else None
    budgets_per_algo = cfg.get("budgets_per_algo", {}) or {}

    # env selection policy
    env_cfg = cfg.get("environment", {}) or {}
    env_selection   = str(env_cfg.get("selection", "by_seed"))        # "by_seed" | "fixed_index" | "graph_id" | "all"
    env_fixed_index = env_cfg.get("fixed_index", None)
    env_graph_id    = env_cfg.get("graph_id", None)

    planned = len(algo_list) * len(seeds)
    if env_selection.lower() == "all":
        # si iteras todos los entornos dentro del runner, este "planned" es solo aproximado
        _log(log_file, f"Planned runs: {planned} (≈ {len(algo_list)} algo(s) × {len(seeds)} seed(s) × #env(s))")
    else:
        _log(log_file, f"Planned runs: {planned} ({len(algo_list)} algo(s) × {len(seeds)} seed(s) × 1 env(s))")

    runs_cache: List[Dict[str, Any]] = []
    env_seen: Dict[str, bool] = {}

    # Si quieres recorrer todos los entornos: crea la lista aquí.
    # Si no, el generate_environment con 'by_seed'/'fixed_index'/'graph_id' decide el par.
    env_ids_to_run: Optional[List[str]] = None
    if env_selection.lower() == "all":
        # Leer env_index.csv para listar graph_id (no dependemos de internals del loader)
        env_index_csv = (THIS_DIR / "data" / "env_index.csv")
        if not env_index_csv.exists():
            raise FileNotFoundError(f"No existe {env_index_csv}. Genera datos primero.")
        df_idx = pd.read_csv(env_index_csv)
        env_ids_to_run = df_idx["graph_id"].astype(str).tolist()

    # Bucle principal
    with tqdm(desc="Experiments", ncols=96) as pbar:
        # Iteramos envs según política
        if env_ids_to_run is None:
            # Modo 1 env (by_seed/fixed_index/graph_id) – el env se resuelve dentro del loop
            env_loop = [None]
        else:
            # Modo all envs – iteramos explícitamente por cada graph_id
            env_loop = env_ids_to_run

        for env_choice in env_loop:
            for algo_name in algo_list:
                algo_cfg = cfg["algorithms"].get(algo_name.upper(), {})

                per_algo_cfg = budgets_per_algo.get(algo_name.upper(), {}) or {}
                algo_evals_cap = per_algo_cfg.get("evals_cap", global_evals_cap)
                algo_time_limit_sec = (
                    float(per_algo_cfg["time_limit_minutes"]) * 60.0
                    if per_algo_cfg.get("time_limit_minutes") else global_time_limit_sec
                )

                algo_dir = exp_dir / algo_name  # subcarpeta para este algoritmo

                for seed in seeds:
                    out_json = algo_dir / (f"{env_choice}__seed_{seed:03d}.json" if env_choice else f"seed_{seed:03d}.json")
                    if out_json.exists() and out_json.stat().st_size > 0:
                        _log(log_file, f"⏩ Skipping {algo_name} (seed={seed}{' env='+env_choice if env_choice else ''}) — result exists.")
                        try:
                            res_loaded = json.loads(out_json.read_text(encoding="utf-8"))
                            env_id_loaded = res_loaded.get("meta", {}).get("env_id", f"env_{seed}")
                            runs_cache.append({
                                "env_id": env_id_loaded,
                                "algorithm": algo_name,
                                "seed": seed,
                                "result": res_loaded,
                                "runtime_sec": float(res_loaded.get("meta", {}).get("runtime_sec", np.nan)),
                            })
                        except Exception:
                            pass
                        pbar.update(1); continue

                    try:
                        start = time.time()
                        # ----- construir env -----
                        if env_choice is None:
                            ret = generate_environment(
                                seed=seed,
                                selection=env_selection,
                                fixed_index=env_fixed_index,
                                graph_id=env_graph_id,
                            )
                        else:
                            # modo "all": resolvemos por graph_id concreto
                            ret = generate_environment(
                                seed=seed,
                                selection="graph_id",
                                fixed_index=None,
                                graph_id=str(env_choice),
                            )

                        if len(ret) == 4:
                            agents, edges, pmatrix, env_meta = ret
                            env_id = env_meta.get("graph_id", f"env_{seed}")
                        else:
                            agents, edges, pmatrix = ret
                            env_id = f"env_{seed}"

                        # Size params
                        num_goods = _num_goods_from_edges(edges)
                        labels, sizes, _idx, _txb, L_min, info = detect_prefix_layout_and_sizes(edges, mode="graph")
                        K = int(sum(sizes)); L_used = max(int(L_min), K + 1)
                        pop = _extract_size_param(algo_cfg.get("popsize_by_size", {}), num_goods, 100)
                        gens= _extract_size_param(algo_cfg.get("generations_by_size", {}), num_goods, 50)

                        # Pretty log line
                        _log(
                            log_file,
                            f"▶ {algo_name.upper()} | env={env_id} | seed={seed} | N={num_goods} | pop={pop} | "
                            f"gens={gens} | caps=(evals={algo_evals_cap}, time={int(algo_time_limit_sec) if algo_time_limit_sec else None}s)"
                        )

                        # ----- run GA -----
                        if algo_name == "flat":
                            res = run_ga_flat(
                                production_graph=edges, pmatrix=pmatrix, agents_information=agents,
                                genome_shape=L_used, generations=gens, popsize=pop,
                                parents=max(1, int(algo_cfg.get("parents_fraction", 0.25) * pop)),
                                mutation_rate=float((algo_cfg.get("mutation_rate", [0.05])[0]
                                                    if isinstance(algo_cfg.get("mutation_rate", 0.05), list)
                                                    else algo_cfg.get("mutation_rate", 0.05))),
                                fix_last_gene=True, seed=seed, verbosity=1,
                                evals_cap=algo_evals_cap, time_limit_sec=algo_time_limit_sec,
                            )

                        elif algo_name == "joint":
                            res = run_ga_equivclass_joint(
                                production_graph=edges, pmatrix=pmatrix, agents_information=agents,
                                generations=gens, popsize=pop,
                                parents=max(1, int(algo_cfg.get("parents_fraction", 0.25) * pop)),
                                sel_mutation=float((algo_cfg.get("sel_mutation", [0.25])[0]
                                                    if isinstance(algo_cfg.get("sel_mutation", 0.25), list)
                                                    else algo_cfg.get("sel_mutation", 0.25))),
                                tail_mutation=float((algo_cfg.get("tail_mutation", [0.05])[0]
                                                    if isinstance(algo_cfg.get("tail_mutation", 0.05), list)
                                                    else algo_cfg.get("tail_mutation", 0.05))),
                                per_good_cap=algo_cfg.get("per_good_cap"),
                                max_index_probe=int(algo_cfg.get("max_index_probe", 3)),
                                fix_last_gene=True, seed=seed, verbosity=1,
                                evals_cap=algo_evals_cap, time_limit_sec=algo_time_limit_sec,
                            )

                        elif algo_name == "exhaustive":
                            # Parámetros específicos del tail
                            pop_tail = _extract_size_param(algo_cfg.get("population_tail_by_size", {}), num_goods, pop)
                            gens_tail= _extract_size_param(algo_cfg.get("generations_tail_by_size", {}), num_goods, gens)
                            parents_tail = max(1, int(algo_cfg.get("parents_fraction", 0.25) * pop_tail))

                            # Intento A: pasar budgets si la función ya los soporta
                            try:
                                res = run_ga_equivclass_exhaustive(
                                    production_graph=edges, pmatrix=pmatrix, agents_information=agents,
                                    generations=gens_tail, popsize=pop_tail,
                                    parents=parents_tail,
                                    mutation_rate=float((algo_cfg.get("mutation_rate_tail", [0.08])[0]
                                                        if isinstance(algo_cfg.get("mutation_rate_tail", 0.08), list)
                                                        else algo_cfg.get("mutation_rate_tail", 0.08))),
                                    fix_last_gene=True, seed=seed,
                                    max_combos=algo_cfg.get("max_combos_cap"),
                                    per_good_cap=algo_cfg.get("per_good_cap"),
                                    verbosity=1,
                                    evals_cap=algo_evals_cap,           # <-- si tu versión interna ya lo soporta, entra aquí
                                    time_limit_sec=algo_time_limit_sec, # <--
                                )
                            except TypeError:
                                # Intento B (fallback): adaptar presupuesto a max_combos efectivo
                                evals_per_combo_est = _estimate_evals_per_combo(pop_tail, gens_tail, parents_tail)
                                max_by_evals = None
                                if isinstance(algo_evals_cap, (int, float)) and algo_evals_cap and evals_per_combo_est > 0:
                                    max_by_evals = max(1, int(algo_evals_cap // evals_per_combo_est))

                                user_cap = algo_cfg.get("max_combos_cap", None)
                                if isinstance(user_cap, (int, float)) and user_cap:
                                    user_cap = int(user_cap)
                                else:
                                    user_cap = None

                                # min entre el cap del YAML y el derivado del presupuesto (si ambos existen)
                                if max_by_evals is not None and user_cap is not None:
                                    max_combos_effective = max(1, min(user_cap, max_by_evals))
                                elif max_by_evals is not None:
                                    max_combos_effective = max_by_evals
                                elif user_cap is not None:
                                    max_combos_effective = user_cap
                                else:
                                    # sin caps ni presupuesto: pon un límite razonable
                                    max_combos_effective = 200

                                _log(log_file, f"    (EXHAUSTIVE fallback) Using max_combos={max_combos_effective} "
                                               f"(est_evals_per_combo≈{evals_per_combo_est})")

                                res = run_ga_equivclass_exhaustive(
                                    production_graph=edges, pmatrix=pmatrix, agents_information=agents,
                                    generations=gens_tail, popsize=pop_tail,
                                    parents=parents_tail,
                                    mutation_rate=float((algo_cfg.get("mutation_rate_tail", [0.08])[0]
                                                        if isinstance(algo_cfg.get("mutation_rate_tail", 0.08), list)
                                                        else algo_cfg.get("mutation_rate_tail", 0.08))),
                                    fix_last_gene=True, seed=seed,
                                    max_combos=max_combos_effective,
                                    per_good_cap=algo_cfg.get("per_good_cap"),
                                    verbosity=1
                                )

                                # anotar en meta el adaptador de presupuesto
                                res.setdefault("meta", {})["exhaustive_budget_adapter"] = {
                                    "evals_cap": algo_evals_cap,
                                    "time_limit_sec": algo_time_limit_sec,
                                    "evals_per_combo_est": evals_per_combo_est,
                                    "max_combos_effective": max_combos_effective,
                                }

                        else:
                            raise ValueError(f"Unknown algorithm: {algo_name}")

                        # Meta enrich + persist per-run JSON
                        elapsed = time.time() - start
                        res.setdefault("meta", {})["runtime_sec"] = float(elapsed)
                        res["meta"]["env_id"] = env_id
                        out_json.write_text(json.dumps(res, indent=2), encoding="utf-8")
                        _log(log_file, f"✅ {algo_name.upper()} | env={env_id} | seed={seed} | runtime={elapsed:.2f}s")

                        # Cache for aggregation
                        runs_cache.append({
                            "env_id": env_id, "algorithm": algo_name, "seed": seed,
                            "result": res, "runtime_sec": float(elapsed),
                            "_complexity": {"edges": edges, "pmatrix": pmatrix, "K": K, "alphabet": len(info.get('alphabet',[0,1])) if isinstance(info, dict) else 2},
                        })

                    except Exception as e:
                        _log(log_file, f"❌ Error in {algo_name} | env={env_choice if env_choice else 'resolved'} | seed={seed}: {e}")

                    finally:
                        pbar.update(1)

    # ---------- aggregation ----------
    if not runs_cache:
        _log(log_file, "No runs to aggregate."); return

    golden = compute_golden_best(runs_cache, prefer_algo="exhaustive")

    runs_rows, curves_rows = [], []
    per_algo_hits: Dict[str, List[int]] = {}
    per_algo_hit_evals: Dict[str, List[int | None]] = {}
    env_seen_once: Dict[str, bool] = {}
    env_complex_csv = exp_dir / "aggregates" / "env_complexity.csv"

    for r in runs_cache:
        env_id = str(r["env_id"]); algo = r["algorithm"]; seed = int(r["seed"]); res = r["result"]

        # complexity (once per env)
        if env_id not in env_seen_once and "_complexity" in r:
            c = r["_complexity"]
            tags = env_complexity_tags(edges=c["edges"], pmatrix=c["pmatrix"], K=c["K"], alphabet_size=c["alphabet"])
            row = {"env_id": env_id}; row.update(tags)
            write_header = not env_complex_csv.exists()
            with open(env_complex_csv, "a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(row.keys()))
                if write_header: w.writeheader()
                w.writerow(row)
            env_seen_once[env_id] = True

        # hit/regret
        u_star = float(golden.get(env_id, float("-inf")))
        summ = summarize_run_against_golden(r, u_star, epsilon)

        runs_rows.append({
            "env_id": env_id, "algorithm": algo, "seed": seed,
            "golden_best": u_star, "best_final": summ["final_best"],
            "hit": summ["hit"], "hit_eval": summ["hit_eval"], "regret": summ["regret"],
            "evals_total": summ["evals_total"], "runtime_sec": r.get("runtime_sec", float("nan")),
        })

        best_curve = res.get("curves", {}).get("best", [])
        evals_cum = res.get("meta", {}).get("evals_cum")
        if evals_cum is None:
            per_gen = res.get("meta", {}).get("evals_per_gen", [])
            evals_cum = np.cumsum(np.asarray(per_gen, dtype=int)).tolist() if per_gen else list(range(1, len(best_curve)+1))
        for i, val in enumerate(best_curve):
            curves_rows.append({
                "env_id": env_id, "algorithm": algo, "seed": seed,
                "point": i, "evals_cum": int(evals_cum[i]) if i < len(evals_cum) else (i+1),
                "best_so_far": float(val),
            })

        per_algo_hits.setdefault(algo, []).append(int(summ["hit"]))
        per_algo_hit_evals.setdefault(algo, []).append(summ["hit_eval"])

    agg = exp_dir / "aggregates"
    pd.DataFrame([{"env_id": k, "golden_best": v} for k, v in golden.items()]).to_csv(agg / "golden_best.csv", index=False)
    runs_df = pd.DataFrame(runs_rows); runs_df.to_csv(agg / "runs.csv", index=False); runs_df.to_csv(agg / "summary.csv", index=False)
    pd.DataFrame(curves_rows).to_csv(agg / "curves.csv", index=False)
    pd.DataFrame(anytime_success(per_algo_hits, per_algo_hit_evals, budgets_grid)).to_csv(agg / "anytime.csv", index=False)
    ecdf_rows = [{"algorithm": a, "evals": int(x), "ecdf": float(y)} for a, he in per_algo_hit_evals.items() for x, y in ecdf_time_to_hit(he)]
    pd.DataFrame(ecdf_rows).to_csv(agg / "ecdf_tth.csv", index=False)

    _log(log_file, f"✅ Metrics written to: {agg}")
    print("\n✅ Experiment folder:", exp_dir)

if __name__ == "__main__":
    run_all_experiments()

# algorithms/ga/blind_random_search.py
# =============================================================================
# Random Search Baseline
# =============================================================================
# This module implements a budget-consistent random search baseline over
# binary genomes. Unlike a GA, it performs no selection, crossover, or mutation.
# Each generation simply re-samples an independent population of genomes
# and evaluates them with the same Economy-based objective used by the "flat"
# GA. The design mirrors the telemetry and logging of the flat GA to enable
# fair AUC-based comparisons under identical evaluation budgets.
#
# Key properties:
#   • Budget consistency: total evaluations are defined as
#         evals_total = popsize * (generations + 1)
#     (including generation 0), so AUC integrals computed over per-generation
#     curves are directly comparable to other algorithms using the same budget.
#   • API compatibility: the function signature and metadata fields align with
#     run_ga_flat, including seed handling, fix_last_gene enforcement on init,
#     and DEAP logbook statistics (fitness and bit-density chapters).
#   • Telemetry parity: verbosity and log_every produce the same terminal
#     outputs as the flat GA (per-generation best/mean/median), enabling
#     drop-in use within existing experiment harnesses and dashboards.
#   • Reproducibility: RNG seeding is centralized and affects both Python's
#     random and NumPy to ensure deterministic runs when a seed is provided.
#
# Limitations:
#   • No population carryover: each generation is an i.i.d. sample; diversity
#     and uniq metrics are diagnostic only and do not influence search.
#   • The "parents" and "mutation_rate" fields in meta are placeholders kept
#     for schema compatibility; they are fixed to 0 and 0.0 respectively.
# =============================================================================

from __future__ import annotations

from typing import Any, Dict, List, Callable

import random
import numpy as np
from deap import base, creator, tools

# Reuse helpers from the flat GA to keep identical evaluation and metrics.
from algorithms.ga.flat import (
    _ensure_deap_types,
    _init_binary_individual,
    _make_deap_evaluator,
    _extra_population_metrics,
)


def _clip_to_budget(
    popsize: int,
    generations: int,
    evals_cap: int | None,
) -> tuple[int, int, dict]:
    """
    Clips (popsize, generations) to respect an evaluation budget under the
    convention: evals_total = popsize * (generations + 1).

    Returns
    -------
    (popsize_eff, generations_eff, info_clip)
        info_clip contains diagnostic information about any clipping applied.
    """
    info = {
        "clipped": False,
        "reason": None,
        "popsize_before": int(popsize),
        "generations_before": int(generations),
    }
    if evals_cap is None:
        return int(popsize), int(generations), info

    popsize = max(1, int(popsize))
    generations = max(0, int(generations))

    # If the initial population alone exceeds the budget, shrink popsize
    # to the cap and force generations to zero.
    if popsize > evals_cap:
        info.update({"clipped": True, "reason": "popsize>evals_cap"})
        return int(evals_cap), 0, info

    # Max generations allowed with this popsize (including gen 0 in budget).
    # popsize * (G + 1) <= evals_cap  =>  G <= evals_cap // popsize - 1
    max_gens = max(0, int(evals_cap) // int(popsize) - 1)
    if generations > max_gens:
        info.update(
            {"clipped": True, "reason": "generations>max_allowed", "max_generations": max_gens}
        )
        generations = max_gens

    return int(popsize), int(generations), info


def run_random_search(
    production_graph,
    pmatrix,
    agents_information,
    genome_shape: int,
    # Requested order: popsize first, then generations.
    popsize: int = 30,
    generations: int = 15,
    # Compatibility with the general API.
    fix_last_gene: bool = True,
    seed: int | None = 42,
    verbosity: int = 1,
    log_every: int = 1,
    progress_cb: Callable[[int, float, float, float], None] | None = None,
    # Optional budget gates:
    evals_cap: int | None = None,
    time_limit_sec: float | None = None,
) -> Dict[str, Any]:
    """
    Random-search baseline with flat-GA style telemetry.

    This routine draws an independent population of binary genomes at each
    generation, evaluates them with the Economy objective, and records the
    same diagnostic statistics as run_ga_flat to support AUC-based comparisons
    under matched evaluation budgets.

    """
    import time

    t0 = time.time()

    def _budget_reason(evals_total: int) -> str | None:
        """Checks whether the evaluation or time budget has been exhausted."""
        if evals_cap is not None and evals_total >= int(evals_cap):
            return "evals"
        if time_limit_sec is not None and (time.time() - t0) >= float(time_limit_sec):
            return "time"
        return None

    # Seed both RNGs to guarantee deterministic population draws and fitnesses.
    # This mirrors the flat GA seeding policy for reproducible cross-method sweeps.
    random.seed(seed)
    np.random.seed(seed if seed is not None else None)
    _ensure_deap_types()

    L = int(genome_shape)

    # Clip (popsize, generations) to respect evals_cap while preserving popsize
    # whenever possible; report decisions in meta["budget"]["clip_info"].
    popsize_eff, generations_eff, clip_info = _clip_to_budget(popsize, generations, evals_cap)
    N = int(popsize_eff)
    G = int(generations_eff)

    # Minimal toolbox: only initialization and evaluation.
    toolbox = base.Toolbox()
    toolbox.register(
        "individual",
        _init_binary_individual,
        creator.Individual,
        length=L,
        fix_last_gene=bool(fix_last_gene),
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register(
        "evaluate",
        _make_deap_evaluator(production_graph, pmatrix, agents_information),
    )

    # Stats / logbook identical to flat for curve/AUC comparability.
    stats_fit = tools.Statistics(key=lambda ind: ind.fitness.values[0])
    stats_fit.register("avg", np.mean)
    stats_fit.register("std", np.std)
    stats_fit.register("med", np.median)
    stats_fit.register("min", np.min)
    stats_fit.register("max", np.max)

    stats_bits = tools.Statistics(key=lambda ind: sum(ind) / max(1, len(ind)))
    stats_bits.register("avg", np.mean)
    stats_bits.register("std", np.std)
    stats_bits.register("min", np.min)
    stats_bits.register("max", np.max)

    mstats = tools.MultiStatistics(fitness=stats_fit, bits=stats_bits)
    logbook = tools.Logbook()
    logbook.header = "gen", "evals", "fitness", "bits", "diversity", "uniq"
    logbook.chapters["fitness"].header = "min", "avg", "med", "std", "max"
    logbook.chapters["bits"].header = "min", "avg", "std", "max"

    best_hist: List[float] = []
    mean_hist: List[float] = []
    median_hist: List[float] = []
    evals_per_gen: List[int] = []

    budget_triggered = False
    budget_stop_reason: str | None = None

    # Track the global best across all generations; "all_best_genomes" returns
    # the final champion for schema parity with flat.
    global_best_val = -np.inf
    global_best_ind: creator.Individual | None = None

    # Generation 0: initial i.i.d. population draw and evaluation.
    # This counts toward the budget to keep AUC time axes aligned with flat.
    pop = toolbox.population(n=N)
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    evals_per_gen.append(len(fitnesses))

    record = mstats.compile(pop)
    record.update(_extra_population_metrics(pop))
    logbook.record(gen=0, evals=len(fitnesses), **record)

    vals0 = [ind.fitness.values[0] for ind in pop]
    b0 = float(np.max(vals0)) if vals0 else float("-inf")
    m0 = float(np.mean(vals0)) if vals0 else float("-inf")
    med0 = float(np.median(vals0)) if vals0 else float("-inf")
    best_hist.append(b0)
    mean_hist.append(m0)
    median_hist.append(med0)

    if b0 > global_best_val:
        argmax0 = int(np.argmax(vals0)) if vals0 else 0
        global_best_val = b0
        global_best_ind = creator.Individual(list(pop[argmax0])) if N > 0 else creator.Individual([0] * L)

    if verbosity and (log_every > 0):
        print(f"Gen 000: best={b0:.6f} | mean={m0:.6f} | median={med0:.6f}")
    if progress_cb:
        try:
            progress_cb(0, b0, m0, med0)
        except Exception:
            pass

    # Budget check after gen 0
    budget_stop_reason = _budget_reason(sum(evals_per_gen))
    if budget_stop_reason is not None:
        budget_triggered = True

    # Per-generation loop: re-sample fresh i.i.d. populations without inheritance.
    # No selection/crossover/mutation; statistics are purely descriptive.
    for gen in range(1, G + 1):
        if budget_triggered:
            break

        pop = toolbox.population(n=N-1)

        pop.append(creator.Individual(list(global_best_ind)))
        
        fitnesses = list(map(toolbox.evaluate, pop))
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit
        evals_per_gen.append(len(fitnesses))

        record = mstats.compile(pop)
        record.update(_extra_population_metrics(pop))
        logbook.record(gen=gen, evals=len(fitnesses), **record)

        vals = [ind.fitness.values[0] for ind in pop]
        b = float(np.max(vals)) if vals else float("-inf")
        m = float(np.mean(vals)) if vals else float("-inf")
        med = float(np.median(vals)) if vals else float("-inf")

        best_hist.append(b)
        mean_hist.append(m)
        median_hist.append(med)

        if b > global_best_val:
            argmax = int(np.argmax(vals)) if vals else 0
            global_best_val = b
            global_best_ind = creator.Individual(list(pop[argmax])) if N > 0 else creator.Individual([0] * L)

        if verbosity >= 1 and (gen % max(1, log_every) == 0 or gen == G):
            print(f"Gen {gen:03d}: best={b:.6f} | mean={m:.6f} | median={med:.6f}")
        if progress_cb:
            try:
                progress_cb(gen, b, m, med)
            except Exception:
                pass

        budget_stop_reason = _budget_reason(sum(evals_per_gen))
        if budget_stop_reason is not None:
            budget_triggered = True
            break

    # Finalize outputs
    if global_best_ind is None:
        global_best_ind = creator.Individual([0] * L)

    tie_tolerance = 1e-9  # kept for schema parity with flat
    evals_cum = np.cumsum(np.array(evals_per_gen, dtype=int)).tolist()
    runtime = time.time() - t0

    return {
        "best_genome": list(global_best_ind),
        "best_utility": float(global_best_val),
        "all_best_genomes": [list(global_best_ind)],
        "curves": {"best": best_hist, "mean": mean_hist, "median": median_hist},
        "meta": {
            "genome_shape": L,
            "generations": int(G),
            "popsize": int(N),
            # Fields kept for schema compatibility with flat:
            "parents": 0,
            "mutation_rate": 0.0,
            "fix_last_gene": bool(fix_last_gene),
            "seed": seed,
            "tie_tolerance": tie_tolerance,
            "evals_per_gen": evals_per_gen,
            "evals_cum": evals_cum,
            "runtime_sec": float(runtime),
            "budget": {
                "evals_cap": evals_cap,
                "time_limit_sec": time_limit_sec,
                "triggered": bool(budget_triggered),
                "reason": budget_stop_reason if budget_triggered else None,
                "evals_total": int(sum(evals_per_gen)),
                "time_total_sec": float(runtime),
                "clip_info": clip_info,
            },
        },
    }

# algorithms/ga/flat.py
# =============================================================================
# Purpose
# =============================================================================
# This module implements a baseline ("flat") Genetic Algorithm (GA) that evolves
# binary genomes (0/1) without leveraging domain-specific prefix structure.
#
# DEAP integration:
#   - Uses DEAP's creator/toolbox for types and operator registration.
#   - Preserves the original evolutionary policy (deterministic truncation /
#     elitist selection, one-point fixed-mid crossover, bit-flip mutation).
#   - Enforces the hard constraint genome[-1] == 1 when 'fix_last_gene=True'.
#   - Compiles DEAP Statistics / MultiStatistics into a Logbook each generation.
#   - Augments DEAP stats with custom metrics (mean Hamming diversity and count
#     of unique genomes), recorded alongside.
#
# Public API:
#   run_ga_flat(production_graph, pmatrix, agents_information, genome_shape, ...)
#     -> Dict with 'best_genome', 'best_utility', 'curves', and 'meta'.
#
# =============================================================================

from __future__ import annotations

from typing import Any, Dict, List, Tuple, Callable

import numpy as np
import random

from deap import base, creator, tools
from src.search_heuristics.common import deap_clone


from src.simulation.economy.economy import Economy


# =============================================================================
# DEAP type guards
# =============================================================================
def _ensure_deap_types() -> None:
    """
    Lazily creates DEAP 'creator' classes exactly once per process.
    Re-creating classes in DEAP raises, hence the defensive check.
    """
    try:
        _ = creator.FitnessMax
        _ = creator.Individual
    except AttributeError:
        # The flat GA maximizes the Economy's scalar utility.
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)


# =============================================================================
# Fitness evaluation (DEAP-compatible)
# =============================================================================
def _make_deap_evaluator(production_graph, pmatrix, agents_information):
    """
    Builds a DEAP-compatible evaluation function:
        evaluate(individual) -> (utility,)
    where 'utility' is obtained from Economy.get_reports()['utility'].
    """
    def evaluate(individual):
        u = Economy(
            production_graph=production_graph,
            pmatrix=pmatrix,
            agents_information=agents_information,
            genome=list(individual),
        ).get_reports().get("utility", 0.0)
        return (float(u),)
    return evaluate


# =============================================================================
# Initialization and variation operators (flat, no prefix semantics)
# =============================================================================
def _init_binary_individual(icls, length: int, fix_last_gene: bool):
    """
    Creates a binary individual (list of 0/1) of the given 'length'.
    The last bit is enforced to 1 when 'fix_last_gene' is True.
    """
    g = [random.randint(0, 1) for _ in range(int(length))]
    if fix_last_gene and length > 0:
        g[-1] = 1
    return icls(g)

def _mate_one_point_mid(ind1, ind2):
    """
    One-point crossover at the fixed midpoint. Child segments are swapped
    to mimic the original flat operator (deterministic cp = L//2).
    """
    L = len(ind1)
    if L <= 1:
        return ind1, ind2
    cp = L // 2
    ind1[cp:], ind2[cp:] = ind2[cp:], ind1[cp:]
    return ind1, ind2

def _mutate_bitflip_excl_last(ind, indpb: float, fix_last_gene: bool):
    """
    Bit-flip mutation over the whole genome except the last position
    when 'fix_last_gene' is True. The mutation is Bernoulli with 'indpb'
    per locus.
    """
    L = len(ind)
    last = L - 1 if fix_last_gene else L
    for j in range(last):
        if random.random() < indpb:
            ind[j] = 1 - ind[j]
    if fix_last_gene and L > 0:
        ind[-1] = 1
    return (ind,)

def _decorate_enforce_last_gene(toolbox: base.Toolbox, L: int, fix_last_gene: bool):
    """
    Post-operator decorator that re-enforces genome[-1] == 1 if requested.
    Guards against future operator changes that might touch the last bit.
    """
    if not fix_last_gene or L <= 0:
        return
    def keep_last_one(func):
        def wrapper(*args, **kwargs):
            offspring = func(*args, **kwargs)
            for child in offspring:
                if len(child) > 0:
                    child[-1] = 1
            return offspring
        return wrapper
    toolbox.decorate("mate", keep_last_one)
    toolbox.decorate("mutate", keep_last_one)


# =============================================================================
# Extra, non-DEAP statistics (Hamming diversity and unique genomes)
# =============================================================================
def _extra_population_metrics(pop: List[creator.Individual]) -> Dict[str, Any]:
    """
    Computes additional, non-trivial metrics that are not conveniently expressed
    via DEAP's Statistics aggregation:
      - 'diversity': mean pairwise Hamming distance over full genomes.
      - 'uniq': number of unique genomes (phenotypes) in the population.
    The implementation is O(N^2) and intended for moderate population sizes.
    """
    n = len(pop)
    if n == 0:
        return {"diversity": 0.0, "uniq": 0}
    # Uniques
    uniq = len({tuple(ind) for ind in pop})
    # Diversity
    if n < 2 or len(pop[0]) == 0:
        return {"diversity": 0.0, "uniq": int(uniq)}
    acc = 0.0
    pairs = 0
    for i in range(n):
        gi = np.asarray(pop[i], dtype=np.int8)
        for j in range(i + 1, n):
            gj = np.asarray(pop[j], dtype=np.int8)
            acc += float(np.mean(gi != gj))
            pairs += 1
    div = acc / pairs if pairs else 0.0
    return {"diversity": float(div), "uniq": int(uniq)}


# =============================================================================
# Public API
# =============================================================================
def run_ga_flat(
    production_graph,
    pmatrix,
    agents_information,
    genome_shape: int,
    generations: int = 15,
    popsize: int = 30,
    parents: int = 20,
    mutation_rate: float = 0.05,
    fix_last_gene: bool = True,
    seed: int | None = 42,
    verbosity: int = 1,
    log_every: int = 1,
    progress_cb: Callable[[int, float, float, float], None] | None = None,
    # --- Budget (optional) ---
    evals_cap: int | None = None,
    time_limit_sec: float | None = None,
) -> Dict[str, Any]:
    """
    Executes a baseline (flat) Genetic Algorithm (GA) using DEAP to maximize the
    scalar utility computed by the `Economy` class. The GA evolves binary genomes
    without domain-specific prefix semantics and enforces a fixed last gene when
    required.

    Parameters
    ----------
    production_graph : list[tuple[str, str]]
        Directed acyclic graph representing the production structure, where each
        edge (u, v) denotes an input-output relationship between goods.
    pmatrix : np.ndarray
        Price matrix of shape (n_goods, n_agents, n_agents), providing prices for
        each good and agent pair.
    agents_information : dict
        Dictionary containing agent-specific information required by the `Economy`
        class (e.g., names, types, tax rates, production capabilities).
    genome_shape : int
        Length of the binary genome to evolve.
    generations : int, optional (default=15)
        Number of generations for the evolutionary process.
    popsize : int, optional (default=30)
        Total number of individuals in the population.
    parents : int, optional (default=20)
        Number of elite individuals preserved between generations.
    mutation_rate : float, optional (default=0.05)
        Probability of flipping each gene during mutation.
    fix_last_gene : bool, optional (default=True)
        Whether to enforce genome[-1] = 1 for all individuals.
    seed : int or None, optional (default=42)
        Random seed for reproducibility.
    verbosity : int, optional (default=1)
        Level of printed progress (0 = silent, 1 = basic per-generation output).
    log_every : int, optional (default=1)
        Frequency (in generations) for logging and printing progress.
    progress_cb : callable or None, optional
        Optional callback function invoked as `progress_cb(gen, best, mean, median)`
        after each generation to report GA progress externally.
    evals_cap : int or None, optional
        Optional cap on the total number of fitness evaluations. If exceeded, the
        GA stops gracefully after completing the current generation.
    time_limit_sec : float or None, optional
        Optional wall-clock time limit (in seconds). If exceeded, the GA stops
        gracefully after the current generation.

    Returns
    -------
    dict
        Dictionary containing the best results and execution metadata:
        {
            "best_genome": list[int]
                Binary genome achieving the highest fitness (utility).
            "best_utility": float
                Fitness value (utility) of the best genome.
            "all_best_genomes": list[list[int]]
                All genomes achieving the best utility within tolerance.
            "curves": dict[str, list[float]]
                Evolutionary performance curves:
                {
                    "best": best utility per generation,
                    "mean": mean utility per generation,
                    "median": median utility per generation
                }
            "meta": dict
                Run metadata and diagnostics, including:
                {
                    "genome_shape": int,
                    "generations": int,
                    "popsize": int,
                    "parents": int,
                    "mutation_rate": float,
                    "fix_last_gene": bool,
                    "seed": int or None,
                    "tie_tolerance": float,
                    "evals_per_gen": list[int],
                    "evals_cum": list[int],
                    "runtime_sec": float,
                    "budget": {
                        "evals_cap": int or None,
                        "time_limit_sec": float or None,
                        "triggered": bool,
                        "reason": str or None,
                        "evals_total": int,
                        "time_total_sec": float
                    }
                }
        }

    Notes
    -----
    - Implements deterministic truncation (elitist) selection and one-point
      fixed-mid crossover.
    - Maintains population diversity metrics (mean Hamming distance and unique
      genomes) in the logbook.
    - Stops softly upon reaching evaluation or time budgets.
    - Designed for integration with larger experiment pipelines and fine-tuning
      procedures.
    """

    import time
    t0 = time.time()

    def _budget_reason(evals_total: int) -> str | None:
        if evals_cap is not None and evals_total >= int(evals_cap):
            return "evals"
        if time_limit_sec is not None and (time.time() - t0) >= float(time_limit_sec):
            return "time"
        return None

    # --- Seeding & types ---
    random.seed(seed)
    np.random.seed(seed if seed is not None else None)
    _ensure_deap_types()

    L = int(genome_shape)
    N = int(popsize)
    P = max(1, min(int(parents), N - 1))

    # --- Toolbox ---
    toolbox = base.Toolbox()
    toolbox.register("individual", _init_binary_individual, creator.Individual,
                     length=L, fix_last_gene=bool(fix_last_gene))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", _mate_one_point_mid)
    toolbox.register("mutate", _mutate_bitflip_excl_last,
                     indpb=float(mutation_rate), fix_last_gene=bool(fix_last_gene))
    toolbox.register("select", tools.selBest)
    toolbox.register("evaluate", _make_deap_evaluator(production_graph, pmatrix, agents_information))
    _decorate_enforce_last_gene(toolbox, L, fix_last_gene)

    # --- Population ---
    pop: List[creator.Individual] = toolbox.population(n=N)

    # --- Stats / logbook ---
    stats_fit = tools.Statistics(key=lambda ind: ind.fitness.values[0])
    stats_fit.register("avg", np.mean); stats_fit.register("std", np.std)
    stats_fit.register("med", np.median); stats_fit.register("min", np.min); stats_fit.register("max", np.max)
    stats_bits = tools.Statistics(key=lambda ind: sum(ind) / max(1, len(ind)))
    stats_bits.register("avg", np.mean); stats_bits.register("std", np.std)
    stats_bits.register("min", np.min); stats_bits.register("max", np.max)
    mstats = tools.MultiStatistics(fitness=stats_fit, bits=stats_bits)
    logbook = tools.Logbook()
    logbook.header = "gen", "evals", "fitness", "bits", "diversity", "uniq"
    logbook.chapters["fitness"].header = "min", "avg", "med", "std", "max"
    logbook.chapters["bits"].header = "min", "avg", "std", "max"

    best_hist, mean_hist, median_hist = [], [], []
    evals_per_gen: List[int] = []  # <- for metrics
    budget_triggered = False
    budget_reason = None

    # --- Gen 0 evaluation ---
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    evals_per_gen.append(len(fitnesses))  # record initial evals

    record = mstats.compile(pop)
    record.update(_extra_population_metrics(pop))
    logbook.record(gen=0, evals=len(fitnesses), **record)

    vals0 = [ind.fitness.values[0] for ind in pop]
    b0, m0, med0 = float(np.max(vals0)), float(np.mean(vals0)), float(np.median(vals0))
    best_hist.append(b0); mean_hist.append(m0); median_hist.append(med0)
    if verbosity and (log_every > 0):
        print(f"Gen 000: best={b0:.6f} | mean={m0:.6f} | median={med0:.6f}")
    if progress_cb:
        try: progress_cb(0, b0, m0, med0)
        except Exception: pass

    # Budget check after gen 0
    budget_reason = _budget_reason(sum(evals_per_gen))
    if budget_reason is not None:
        budget_triggered = True

    # --- Evolution loop ---
    gen = 0
    for gen in range(1, int(generations) + 1):
        if budget_triggered:
            break

        elites = list(map(deap_clone, toolbox.select(pop, P)))

        need = N - len(elites)
        offspring: List[creator.Individual] = []
        while len(offspring) < need:
            if len(elites) >= 2:
                p1, p2 = random.sample(elites, 2)
            else:
                p1 = elites[0]; p2 = deap_clone(elites[0])
            c1, c2 = deap_clone(p1), deap_clone(p2)
            c1, c2 = toolbox.mate(c1, c2)
            if hasattr(c1, "fitness"): del c1.fitness.values
            if hasattr(c2, "fitness"): del c2.fitness.values
            offspring.extend([c1, c2])
        offspring = offspring[:need]

        for mut in offspring:
            toolbox.mutate(mut)
            if hasattr(mut, "fitness"): del mut.fitness.values

        invalid = [ind for ind in offspring if not ind.fitness.valid]
        fits = list(map(toolbox.evaluate, invalid))
        for ind, fit in zip(invalid, fits):
            ind.fitness.values = fit

        # record evals for this generation
        evals_per_gen.append(len(invalid))

        pop[:] = elites + offspring

        record = mstats.compile(pop)
        record.update(_extra_population_metrics(pop))
        logbook.record(gen=gen, evals=len(invalid), **record)

        vals = [ind.fitness.values[0] for ind in pop]
        b, m, med = float(np.max(vals)), float(np.mean(vals)), float(np.median(vals))
        best_hist.append(b); mean_hist.append(m); median_hist.append(med)
        if verbosity >= 1 and (gen % max(1, log_every) == 0 or gen == int(generations)):
            print(f"Gen {gen:03d}: best={b:.6f} | mean={m:.6f} | median={med:.6f}")
        if progress_cb:
            try: progress_cb(gen, b, m, med)
            except Exception: pass

        # Budget check at end of generation
        budget_reason = _budget_reason(sum(evals_per_gen))
        if budget_reason is not None:
            budget_triggered = True
            break

    # --- Final selection / ties ---
    best_ind = tools.selBest(pop, 1)[0]
    best_val = float(best_ind.fitness.values[0])
    tie_tolerance = 1e-9
    vals = np.array([ind.fitness.values[0] for ind in pop], dtype=float)
    tie_idx = np.nonzero(np.isclose(vals, best_val, rtol=0.0, atol=tie_tolerance))[0]

    seen: set[Tuple[int, ...]] = set()
    all_best_genomes: List[List[int]] = []
    for idx in tie_idx:
        g_list = list(pop[idx])
        key = tuple(g_list)
        if key not in seen:
            seen.add(key)
            all_best_genomes.append(g_list)

    best_genome_list: List[int] = list(all_best_genomes[0]) if all_best_genomes else list(best_ind)

    # --- Meta & return ---
    evals_cum = np.cumsum(np.array(evals_per_gen, dtype=int)).tolist()
    runtime = time.time() - t0
    return {
        "best_genome": best_genome_list,
        "best_utility": best_val,
        "all_best_genomes": all_best_genomes if all_best_genomes else [list(best_ind)],
        "curves": {"best": best_hist, "mean": mean_hist, "median": median_hist},
        "meta": {
            "genome_shape": L,
            "generations": int(gen if not budget_triggered else gen),
            "popsize": N,
            "parents": P,
            "mutation_rate": float(mutation_rate),
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
                "reason": budget_reason,
                "evals_total": int(sum(evals_per_gen)),
                "time_total_sec": float(runtime),
            },
        },
    }

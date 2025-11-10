# algorithms/ga/generic.py
# =============================================================================
# Generic Binary Genetic Algorithm (GA)
# =============================================================================
# This GA evolves binary genomes (0/1) using standard DEAP operators.
# The evaluation is domain-agnostic and delegated to Economy (utility).
# Supports elitism, crossover/mutation rates, and runtime/evaluation budgets.
# =============================================================================

from __future__ import annotations
import random
import time
from typing import Any, Callable, Dict, List
import numpy as np
from deap import base, creator, tools
from classes.economy.economy import Economy


# -------------------------------------------------------------------------
# Evaluator
# -------------------------------------------------------------------------
def _make_deap_evaluator(production_graph, pmatrix, agents_information):
    """Return a DEAP-compatible evaluate() computing Economy utility."""
    def evaluate(individual):
        u = Economy(
            production_graph=production_graph,
            pmatrix=pmatrix,
            agents_information=agents_information,
            genome=list(individual),
        ).get_reports().get("utility", 0.0)
        return (float(u),)
    return evaluate


# -------------------------------------------------------------------------
# Main GA
# -------------------------------------------------------------------------
def run_generic_ga(
    production_graph,
    pmatrix,
    agents_information,
    genome_shape: int,
    generations: int = 20,
    popsize: int = 40,
    cxpb: float = 0.7,
    mutpb: float = 0.2,
    mutation_rate: float = 0.05,
    elitism: int = 1,
    fix_last_gene: bool = True,
    seed: int | None = 42,
    verbosity: int = 1,
    log_every: int = 1,
    progress_cb: Callable[[int, float, float, float], None] | None = None,
    evals_cap: int | None = None,
    time_limit_sec: float | None = None,
) -> Dict[str, Any]:
    """
    Runs a standard binary Genetic Algorithm (GA) maximizing utility from Economy.

    Parameters
    ----------
    production_graph, pmatrix, agents_information : domain data
        Inputs for the Economy model.
    genome_shape : int
        Length of binary genome.
    generations : int, default=20
        Number of generations to evolve.
    popsize : int, default=40
        Population size.
    cxpb : float, default=0.7
        Probability of crossover.
    mutpb : float, default=0.2
        Probability of mutation (per individual).
    mutation_rate : float, default=0.05
        Bit-flip probability per gene (indpb in mutFlipBit).
    elitism : int, default=1
        Number of best individuals preserved each generation.
    fix_last_gene : bool, default=True
        If True, enforce genome[-1] = 1.
    seed : int or None
        Random seed.
    verbosity, log_every : control output frequency.
    progress_cb : optional callback (gen, best, mean, median).
    evals_cap, time_limit_sec : optional stop conditions.

    Returns
    -------
    dict with keys:
        - best_genome
        - best_utility
        - curves: {best, mean, median}
        - meta: GA metadata and runtime info
    """

    t0 = time.time()

    def _budget_reason(evals_total: int) -> str | None:
        if evals_cap is not None and evals_total >= int(evals_cap):
            return "evals"
        if time_limit_sec is not None and (time.time() - t0) >= float(time_limit_sec):
            return "time"
        return None

    # Seed RNGs
    random.seed(seed)
    np.random.seed(seed if seed is not None else None)

    L = int(genome_shape)
    N = int(popsize)

    # ---------------------------------------------------------------------
    # DEAP setup
    # ---------------------------------------------------------------------
    try:
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    except RuntimeError:
        pass
    try:
        creator.create("Individual", list, fitness=creator.FitnessMax)
    except RuntimeError:
        pass

    toolbox = base.Toolbox()
    toolbox.register("attr_bool", random.randint, 0, 1)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=L)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=mutation_rate)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("evaluate", _make_deap_evaluator(production_graph, pmatrix, agents_information))

    # Initialize population
    pop = toolbox.population(n=N)
    if fix_last_gene:
        for ind in pop:
            ind[-1] = 1

    # Evaluate initial fitness
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    best_hist, mean_hist, med_hist = [], [], []
    evals_per_gen: List[int] = [len(fitnesses)]
    budget_triggered = False
    budget_reason = None

    def _record_stats(pop, gen):
        vals = np.array([ind.fitness.values[0] for ind in pop])
        best, mean, med = vals.max(), vals.mean(), np.median(vals)
        best_hist.append(float(best))
        mean_hist.append(float(mean))
        med_hist.append(float(med))
        if verbosity >= 1 and (gen % max(1, log_every) == 0 or gen == generations):
            print(f"Gen {gen:03d}: best={best:.6f} | mean={mean:.6f} | median={med:.6f}")
        if progress_cb:
            progress_cb(gen, float(best), float(mean), float(med))

    _record_stats(pop, 0)

    # ---------------------------------------------------------------------
    # Evolution loop
    # ---------------------------------------------------------------------
    for gen in range(1, generations + 1):
        if budget_triggered:
            break

        # 1. Select parents
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))

        # 2. Apply crossover and mutation
        for c1, c2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cxpb:
                toolbox.mate(c1, c2)
                del c1.fitness.values, c2.fitness.values
        for mutant in offspring:
            if random.random() < mutpb:
                toolbox.mutate(mutant)
                del mutant.fitness.values
            if fix_last_gene:
                mutant[-1] = 1

        # 3. Evaluate new individuals
        invalid = [ind for ind in offspring if not ind.fitness.valid]
        fits = list(map(toolbox.evaluate, invalid))
        for ind, fit in zip(invalid, fits):
            ind.fitness.values = fit
        evals_per_gen.append(len(fits))

        # 4. Elitism + replacement
        elites = tools.selBest(pop, elitism)
        pop[:] = elites + offspring[:-elitism] if elitism > 0 else offspring

        # 5. Record stats
        _record_stats(pop, gen)

        budget_reason = _budget_reason(sum(evals_per_gen))
        if budget_reason:
            budget_triggered = True
            break

    # ---------------------------------------------------------------------
    # Final results
    # ---------------------------------------------------------------------
    best_ind = tools.selBest(pop, 1)[0]
    best_val = float(best_ind.fitness.values[0])
    runtime = time.time() - t0

    return {
        "best_genome": list(best_ind),
        "best_utility": best_val,
        "curves": {"best": best_hist, "mean": mean_hist, "median": med_hist},
        "meta": {
            "genome_shape": L,
            "generations": gen,
            "popsize": N,
            "cxpb": cxpb,
            "mutpb": mutpb,
            "mutation_rate": mutation_rate,
            "elitism": elitism,
            "fix_last_gene": fix_last_gene,
            "seed": seed,
            "runtime_sec": runtime,
            "budget": {
                "evals_cap": evals_cap,
                "time_limit_sec": time_limit_sec,
                "triggered": budget_triggered,
                "reason": budget_reason,
                "evals_total": int(sum(evals_per_gen)),
                "time_total_sec": float(runtime),
            },
        },
    }

# algorithms/ga/equivclass_exhaustive.py
# =============================================================================
# Purpose
# =============================================================================
# This module orchestrates an exhaustive-by-class search:
#
#   • It discovers the REAL prefix layout enforced by the Planner (goods order,
#     k_g sizes, and index sets per good).
#   • It enumerates (or samples) combinations of per-good equivalence classes
#     (weak compositions over a discovered prefix alphabet), thereby fixing a
#     canonical prefix for each combination.
#   • For each fixed prefix, it runs a DEAP-driven GA only on the tail segment
#     (prefix is kept immutable), evaluates the Economy, and tracks the best.
#
# DEAP integration (tail GA):
#   - creator/toolbox types and operator registration.
#   - Elitist truncation, one-point crossover restricted to the tail window,
#     bit-flip mutation on tail genes, and genome[-1] == 1 enforcement.
#   - Per-generation MultiStatistics + custom extras (tail Hamming diversity
#     and unique tail count) appended to a Logbook internally.
#
# Public API:
#   run_ga_equivclass_exhaustive(production_graph, pmatrix, agents_information, ...)
#     -> Dict with 'best_genome', 'best_utility', 'curves', and 'meta'.
# =============================================================================

from __future__ import annotations

import itertools
import math
import random
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from deap import base, creator, tools

from src.simulation.economy.economy import Economy
from src.algorithms.ga.common import (
    detect_prefix_layout_and_sizes,
    probe_allowed_indices_via_tx_builder,
    format_genome,  # optional helper for logs
)
from src.algorithms.ga.common import deap_clone


# =============================================================================
# Small formatting helpers
# =============================================================================
def _short_counts_str(counts: np.ndarray) -> str:
    """Compact string for a counts vector, e.g., '2,0,1'."""
    if counts.size == 0:
        return "-"
    return ",".join(str(int(x)) for x in counts.tolist())


# =============================================================================
# Combinatorics: equivalence classes (weak compositions) per good
# =============================================================================
def _num_equiv_classes(alpha_size: int, k: int) -> int:
    """Number of weak compositions of k into 'alpha_size' parts: C(alpha_size + k - 1, k)."""
    if alpha_size <= 0:
        return 1 if k == 0 else 0
    return math.comb(alpha_size + k - 1, k)


def _iter_count_vectors(alpha_size: int, k: int) -> Iterable[np.ndarray]:
    """
    Generates all nonnegative integer vectors 'counts' of length 'alpha_size'
    with sum(counts) == k (stars-and-bars enumeration).
    """
    if k < 0:
        return
    if alpha_size == 0:
        if k == 0:
            yield np.zeros(0, dtype=int)
        return
    total_slots = k + alpha_size - 1
    for divs in itertools.combinations(range(total_slots), alpha_size - 1):
        prev = -1
        out: List[int] = []
        for d in divs + (total_slots,):
            out.append(d - prev - 1)
            prev = d
        yield np.array(out, dtype=int)


def _sample_count_vectors(
    alpha_size: int, k: int, num_samples: int, rng: np.random.Generator
) -> List[np.ndarray]:
    """
    Uniformly samples 'num_samples' weak compositions (on divider choices),
    allowing duplicates in output for small spaces.
    """
    if num_samples <= 0:
        return []
    total_slots = k + alpha_size - 1
    out: List[np.ndarray] = []
    for _ in range(num_samples):
        if alpha_size > 1:
            divs = np.sort(rng.choice(total_slots, size=alpha_size - 1, replace=False))
        else:
            divs = np.array([], dtype=int)
        prev = -1
        vals: List[int] = []
        for d in list(divs) + [total_slots]:
            vals.append(int(d - prev - 1))
            prev = int(d)
        out.append(np.array(vals, dtype=int))
    return out


def _canonical_values_from_counts(alphabet: List[int], counts: np.ndarray) -> np.ndarray:
    """Expands counts over the 'alphabet' into a non-decreasing prefix values vector."""
    vals: List[int] = []
    for val, c in zip(alphabet, counts):
        if c > 0:
            vals.extend([int(val)] * int(c))
    return np.array(vals, dtype=int)


# =============================================================================
# DEAP tail-GA: types, initialization, operators, decorators, evaluation, stats
# =============================================================================
def _ensure_deap_types() -> None:
    """Creates DEAP creator classes once per process."""
    try:
        _ = creator.FitnessMax
        _ = creator.Individual
    except AttributeError:
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))   # maximize utility
        creator.create("Individual", list, fitness=creator.FitnessMax)


def _init_with_fixed_prefix(icls, prefix: List[int], L_used: int, fix_last_gene: bool):
    """
    Builds a full-length individual: fixed prefix plus random binary tail.
    The last gene is set to 1 when requested.
    """
    K = len(prefix)
    tail_len = max(0, int(L_used) - K)
    tail = [random.randint(0, 1) for _ in range(tail_len)]
    if fix_last_gene and tail_len > 0:
        tail[-1] = 1
    return icls(list(prefix) + tail)


def _mate_tail_midpoint(ind1, ind2, K_lock: int):
    """
    One-point crossover restricted to the tail [K_lock .. L-1].
    The cut-point is the midpoint of the tail (deterministic).
    """
    L = len(ind1)
    if L - K_lock <= 1:
        return ind1, ind2
    cp = K_lock + max(1, (L - K_lock) // 2)
    ind1[cp:], ind2[cp:] = ind2[cp:], ind1[cp:]
    return ind1, ind2


def _mutate_tail_bitflip(ind, K_lock: int, indpb: float, fix_last_gene: bool):
    """Bit-flip mutation on the tail; preserves the last gene as 1 if requested."""
    L = len(ind)
    last = L - 1 if fix_last_gene else L
    for j in range(K_lock, last):
        if random.random() < indpb:
            ind[j] = 1 - ind[j]
    if fix_last_gene and L > 0:
        ind[-1] = 1
    return (ind,)


def _decorate_lock_prefix_and_last(toolbox: base.Toolbox, prefix: List[int], fix_last_gene: bool):
    """
    Post-operator decorator that re-enforces invariants after mate/mutate:
      - prefix remains identical to the supplied one;
      - genome[-1] == 1 when requested.
    """
    K = len(prefix)
    if K == 0 and not fix_last_gene:
        return

    def keeper(func):
        def wrapper(*args, **kwargs):
            offspring = func(*args, **kwargs)
            for child in offspring:
                if K > 0:
                    child[:K] = prefix[:]
                if fix_last_gene and len(child) > 0:
                    child[-1] = 1
            return offspring
        return wrapper

    toolbox.decorate("mate", keeper)
    toolbox.decorate("mutate", keeper)


def _make_eval_tail(
    production_graph,
    pmatrix,
    agents_information,
):
    """Returns DEAP-compatible evaluate(individual) -> (utility,)."""
    def evaluate(individual):
        u = Economy(
            production_graph=production_graph,
            pmatrix=pmatrix,
            agents_information=agents_information,
            genome=list(individual),
        ).get_reports().get("utility", 0.0)
        return (float(u),)
    return evaluate


def _extra_tail_metrics(pop: List[creator.Individual], K_lock: int) -> Dict[str, Any]:
    """
    Custom, non-DEAP metrics on the tail only:
      - 'tail_diversity': mean pairwise Hamming distance on the tail segment.
      - 'tail_uniq'     : number of unique tail bitstrings.
    Intended for moderate population sizes (O(N^2)).
    """
    if not pop:
        return {"tail_diversity": 0.0, "tail_uniq": 0}
    tails = [tuple(ind[K_lock:]) for ind in pop]
    uniq = len(set(tails))
    n = len(tails)
    if n < 2 or (len(pop[0]) - K_lock) <= 0:
        return {"tail_diversity": 0.0, "tail_uniq": int(uniq)}
    acc = 0.0
    pairs = 0
    for i in range(n):
        ti = np.fromiter(tails[i], dtype=np.int8)
        for j in range(i + 1, n):
            tj = np.fromiter(tails[j], dtype=np.int8)
            acc += float(np.mean(ti != tj))
            pairs += 1
    return {"tail_diversity": (acc / pairs if pairs else 0.0), "tail_uniq": int(uniq)}


def _run_ga_tail_deap(
    prefix: np.ndarray,
    L_used: int,
    production_graph,
    pmatrix,
    agents_information,
    num_generations: int,
    population_size: int,
    num_parents_mating: int,
    mutation_rate: float,
    fix_last_gene: bool,
    seed: Optional[int],
    print_every: Optional[int] = None,
    progress_label: Optional[str] = None,
) -> Tuple[np.ndarray, float, List[float], List[float]]:
    """
    Executes a DEAP GA on the tail while keeping the provided prefix immutable.
    Returns (best_genome, best_utility, best_curve, mean_curve).
    """
    # Seeding and DEAP types
    random.seed(seed)
    np.random.seed(seed if seed is not None else None)
    _ensure_deap_types()

    K = len(prefix)
    prefix_list = list(map(int, prefix))

    # Toolbox
    toolbox = base.Toolbox()
    toolbox.register("individual", _init_with_fixed_prefix, creator.Individual,
                     prefix=prefix_list, L_used=int(L_used), fix_last_gene=bool(fix_last_gene))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", _mate_tail_midpoint, K_lock=K)
    toolbox.register("mutate", _mutate_tail_bitflip, K_lock=K, indpb=float(mutation_rate),
                     fix_last_gene=bool(fix_last_gene))
    toolbox.register("select", tools.selBest)  # deterministic truncation (elitist)
    toolbox.register("evaluate", _make_eval_tail(production_graph, pmatrix, agents_information))

    # Defensive invariants
    _decorate_lock_prefix_and_last(toolbox, prefix_list, fix_last_gene)

    # Population
    pop: List[creator.Individual] = toolbox.population(n=int(population_size))

    # DEAP statistics
    stats_fit = tools.Statistics(key=lambda ind: ind.fitness.values[0])
    stats_fit.register("min", np.min); stats_fit.register("avg", np.mean)
    stats_fit.register("med", np.median); stats_fit.register("max", np.max)

    stats_tail = tools.Statistics(key=lambda ind: (
        sum(ind[K:]) / max(1, len(ind) - K)
    ))
    stats_tail.register("min", np.min); stats_tail.register("avg", np.mean); stats_tail.register("max", np.max)

    mstats = tools.MultiStatistics(fitness=stats_fit, tail=stats_tail)
    logbook = tools.Logbook()
    logbook.header = "gen", "evals", "fitness", "tail", "tail_diversity", "tail_uniq"
    logbook.chapters["fitness"].header = "min", "avg", "med", "max"
    logbook.chapters["tail"].header = "min", "avg", "max"

    # Curves (exposed to the orchestrator)
    best_curve: List[float] = []
    mean_curve: List[float] = []

    # Initial evaluation
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    record = mstats.compile(pop)
    record.update(_extra_tail_metrics(pop, K))
    logbook.record(gen=0, evals=len(fitnesses), **record)

    vals0 = [ind.fitness.values[0] for ind in pop]
    b0, m0 = float(np.max(vals0)), float(np.mean(vals0))
    best_curve.append(b0); mean_curve.append(m0)
    if print_every:
        tag = f"[{progress_label}] " if progress_label else ""
        print(f"{tag}Gen 000: best={b0:.6f} | mean={m0:.6f}")

    # Evolutionary loop with elitism
    P = max(1, min(int(num_parents_mating), len(pop) - 1))
    for gen in range(1, int(num_generations) + 1):
        elites = list(map(deap_clone, toolbox.select(pop, P)))

        need = len(pop) - len(elites)
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

        pop[:] = elites + offspring

        record = mstats.compile(pop)
        record.update(_extra_tail_metrics(pop, K))
        logbook.record(gen=gen, evals=len(invalid), **record)

        vals = [ind.fitness.values[0] for ind in pop]
        b, m = float(np.max(vals)), float(np.mean(vals))
        best_curve.append(b); mean_curve.append(m)

        if print_every and (gen % print_every == 0 or gen == int(num_generations)):
            tag = f"[{progress_label}] " if progress_label else ""
            print(f"{tag}Gen {gen:03d}: best={b:.6f} | mean={m:.6f}")

    best_ind = tools.selBest(pop, 1)[0]
    return np.array(best_ind, dtype=int), float(best_ind.fitness.values[0]), best_curve, mean_curve


# =============================================================================
# Orchestration: exhaustive over combinations, respecting Planner layout
# =============================================================================
def _maximize_by_class_combinations(
    production_graph,
    pmatrix,
    agents_information,
    mode: str = "graph",
    num_generations: int = 10,
    population_size: int = 20,
    num_parents_mating: int = 10,
    mutation_rate: float = 0.1,
    fix_last_gene: bool = True,
    seed: Optional[int] = 123,
    max_combos: Optional[int] = None,
    per_good_cap: Optional[int] = None,
    max_index_probe: int = 3,
    verbosity: int = 1,
    log_every: int = 1,
    no_plots: bool = True,
    top_combos_bars: int = 20,
    max_best_returned: Optional[int] = None,
) -> Tuple[np.ndarray, float, Dict[str, Any]]:
    """
    Iterates per-good equivalence-class combinations, runs a tail-only DEAP GA
    for each fixed prefix, and tracks the global maximizer and diagnostics.
    """
    rng = np.random.default_rng(seed)

    # 1) Layout discovery and minimal executable length
    labels, sizes, index_sets, tx_builder, L_min, info_extra = detect_prefix_layout_and_sizes(
        production_graph, mode=mode
    )
    K = int(sum(sizes))
    L_used = max(int(L_min), K + 1)

    # 2) Probe allowed prefix alphabet (integers)
    alphabet = probe_allowed_indices_via_tx_builder(L_used, tx_builder, max_index_probe=max_index_probe)
    A = len(alphabet)

    # 3) Build per-good class pools and totals
    per_good_classes: List[List[np.ndarray]] = []
    per_good_totals: List[int] = []
    for k_g in sizes:
        total_g = _num_equiv_classes(A, int(k_g))
        per_good_totals.append(total_g)
        if per_good_cap is not None and total_g > per_good_cap:
            classes_g = _sample_count_vectors(A, int(k_g), int(per_good_cap), rng)
        else:
            classes_g = list(_iter_count_vectors(A, int(k_g)))
        per_good_classes.append(classes_g)

    # 4) Combination space sizing and iteration policy
    combo_space = 1
    for total in per_good_totals:
        combo_space *= max(1, int(total))

    if verbosity >= 1:
        print("=== Detection summary (Planner order) ===")
        print(f"Mode: {mode}")
        print(f"Goods: {labels}")
        print(f"k_g (Planner): {sizes}  ->  K={K} | L_used={L_used}")
        print(f"Prefix alphabet: {alphabet} (|A|={A})")
        if "sizes_graph_in_labels" in info_extra:
            print(f"Graph reference k_g(labels): {info_extra['sizes_graph_in_labels']}")
        print(f"Per-good class totals: {per_good_totals}")
        print(f"Combination space: {combo_space}")
        if max_combos is not None and combo_space > max_combos:
            print(f"**Notice** combo_space > max_combos={max_combos}. Sampling combos.")

    def _combo_iterator() -> Iterable[List[np.ndarray]]:
        """Yields a per-good list of count vectors for each combination."""
        if max_combos is None or combo_space <= max_combos:
            if per_good_classes:
                for combo in itertools.product(*per_good_classes):
                    yield list(combo)
            else:
                yield []
        else:
            for _ in range(int(max_combos)):
                pick: List[np.ndarray] = []
                for classes_g in per_good_classes:
                    if classes_g:
                        j = rng.integers(0, len(classes_g))
                        pick.append(classes_g[int(j)])
                    else:
                        pick.append(np.zeros(0, dtype=int))
                yield pick

    # 5) Iterate combinations, run tail GA, collect bests and curves
    best_u = -float("inf")
    best_g: Optional[np.ndarray] = None
    best_combo_info: Dict[str, Any] = {}
    results_by_combo: List[Tuple[List[np.ndarray], float]] = []
    curves_by_combo: List[Tuple[List[np.ndarray], List[float], List[float]]] = []

    genomes_and_utils: List[Tuple[np.ndarray, float]] = []

    for it, counts_combo in enumerate(_combo_iterator(), start=1):
        # Build the REAL-layout prefix of length K
        prefix = np.zeros(K, dtype=int)
        for ig, counts_g in enumerate(counts_combo):
            vals_g = _canonical_values_from_counts(alphabet, counts_g)
            pos_g = index_sets[ig]
            if len(vals_g) != len(pos_g):
                raise ValueError("Counts length does not match detected k_g for a good.")
            for v, j in zip(vals_g, pos_g):
                prefix[j] = int(v)

        combo_label = " | ".join(_short_counts_str(c) for c in counts_combo) if counts_combo else "<empty>"
        progress_label = f"Combo {it:04d}/{'?' if max_combos else combo_space}: {combo_label}"
        print_every = (log_every if verbosity >= 2 else None)

        g_star, u_star, best_curve, mean_curve = _run_ga_tail_deap(
            prefix=prefix,
            L_used=L_used,
            production_graph=production_graph,
            pmatrix=pmatrix,
            agents_information=agents_information,
            num_generations=num_generations,
            population_size=population_size,
            num_parents_mating=num_parents_mating,
            mutation_rate=mutation_rate,
            fix_last_gene=fix_last_gene,
            seed=(None if seed is None else int(seed) + it),
            print_every=print_every,
            progress_label=progress_label,
        )

        results_by_combo.append((counts_combo, u_star))
        curves_by_combo.append((counts_combo, best_curve, mean_curve))
        genomes_and_utils.append((g_star.copy(), float(u_star)))

        if verbosity == 1 and (u_star > best_u):
            print(f"[New BEST-SO-FAR] {progress_label}  -> best_u={u_star:.6f}")
            for gen, (b, m) in enumerate(zip(best_curve, mean_curve), start=1):
                if (gen == 1) or (gen % max(1, log_every) == 0) or (gen == len(best_curve)):
                    print(f"  Gen {gen:03d}: best={b:.6f} | mean={m:.6f}")

        if u_star > best_u:
            best_u = u_star
            best_g = g_star.copy()
            best_combo_info = {
                "counts_combo": [c.tolist() for c in counts_combo],
                "prefix": prefix.tolist(),
                "iteration": it,
                "label": combo_label,
                "best_curve": best_curve,
                "mean_curve": mean_curve,
            }

    # 6) Aggregate curves across combinations (element-wise operations).
    if curves_by_combo:
        global_best_curve = np.maximum.reduce(
            [np.array(curve, dtype=float) for (_c, curve, _m) in curves_by_combo]
        ).tolist()
        global_mean_curve = np.mean(
            [np.array(mcurve, dtype=float) for (_c, _b, mcurve) in curves_by_combo],
            axis=0,
        ).tolist()
    else:
        global_best_curve, global_mean_curve = [], []

    # 7) Ties: collect all phenotype genomes equal to the best within tolerance.
    if best_g is None:
        best_g = np.zeros(L_used, dtype=int)

    tie_tolerance = 1e-9
    all_best_genomes: List[List[int]] = []
    seen: set[Tuple[int, ...]] = set()

    for g_i, u_i in genomes_and_utils:
        if abs(float(u_i) - float(best_u)) <= tie_tolerance:
            key = tuple(int(x) for x in g_i.tolist())
            if key not in seen:
                seen.add(key)
                all_best_genomes.append(list(key))

    # Apply optional cap AFTER deduplication to keep deterministic order.
    num_best_found = len(all_best_genomes)
    if isinstance(max_best_returned, int) and max_best_returned >= 1:
        all_best_genomes = all_best_genomes[:max_best_returned]

    # 8) Assemble traces/diagnostics payload.
    traces = {
        "detection_mode": mode,
        "labels": labels,
        "sizes": sizes,
        "index_sets": index_sets,
        "alphabet": alphabet,
        "K": K,
        "L_used": L_used,
        "combo_space": combo_space,
        "per_good_totals": per_good_totals,
        "best_combo": best_combo_info,
        "global_best_curve": global_best_curve,
        "global_mean_curve": global_mean_curve,
        "results_by_combo": [
            (["(" + _short_counts_str(c) + ")" for c in counts], float(u))
            for counts, u in results_by_combo
        ],
        "diagnostics": info_extra,
        # Ties information
        "all_best_genomes": all_best_genomes,
        "tie_tolerance": tie_tolerance,
        "num_best_genomes": len(all_best_genomes),  # returned count (after cap)
        "num_best_found": num_best_found,           # before cap
        "max_best_returned": max_best_returned,
    }

    return best_g, float(best_u), traces


# =============================================================================
# Public API (wrapper returning a uniform dict like the flat / joint GA)
# =============================================================================
def run_ga_equivclass_exhaustive(
    production_graph,
    pmatrix,
    agents_information,
    mode: str = "graph",
    generations: int = 10,
    popsize: int = 20,
    parents: int = 10,
    mutation_rate: float = 0.10,
    fix_last_gene: bool = True,
    seed: int | None = 40,
    max_combos: int | None = None,
    per_good_cap: int | None = None,
    max_index_probe: int = 3,
    verbosity: int = 1,
    log_every: int = 1,
    no_plots: bool = True,
    top_combos_bars: int = 20,
    max_best_returned: int | None = None,
    # --- Budget (optional) ---
    evals_cap: int | None = None,
    time_limit_sec: float | None = None,
) -> Dict[str, Any]:
    """
    Executes the exhaustive-by-class Genetic Algorithm (GA), which systematically
    explores all (or a sampled subset of) combinations of per-good equivalence
    classes and, for each fixed prefix configuration, runs a DEAP-based tail-only
    GA to optimize the remaining genome. The prefix layout and admissible index
    sets are discovered from the Planner’s production graph.

    Parameters
    ----------
    production_graph : list[tuple[str, str]]
        Directed acyclic graph (DAG) representing production dependencies among
        goods, where edges (u, v) denote that good `u` is an input for good `v`.
    pmatrix : np.ndarray
        Price matrix of shape (n_goods, n_agents, n_agents), specifying prices for
        transactions between agents for each good.
    agents_information : dict
        Configuration dictionary of agents used by the `Economy` class (e.g.,
        agent names, tax parameters, production rules).
    mode : str, optional (default="graph")
        Mode for prefix layout detection in `detect_prefix_layout_and_sizes`.
        The default "graph" mode derives the prefix structure directly from
        the production graph.
    generations : int, optional (default=10)
        Number of generations for the tail GA executed per prefix combination.
    popsize : int, optional (default=20)
        Population size used in the tail GA.
    parents : int, optional (default=10)
        Number of elite individuals preserved between generations in the tail GA.
    mutation_rate : float, optional (default=0.10)
        Bit-flip mutation rate for the tail GA.
    fix_last_gene : bool, optional (default=True)
        Whether to enforce genome[-1] = 1 for all individuals.
    seed : int or None, optional (default=40)
        Random seed for reproducibility.
    max_combos : int or None, optional
        Maximum number of equivalence-class combinations to evaluate. If `None`,
        all combinations are enumerated.
    per_good_cap : int or None, optional
        Caps the number of equivalence classes (weak compositions) per good to
        limit combinatorial explosion.
    max_index_probe : int, optional (default=3)
        Maximum number of transaction indices probed to determine the prefix
        alphabet via the transaction builder.
    verbosity : int, optional (default=1)
        Verbosity level: 0 = silent, 1 = summary output, 2 = detailed logs.
    log_every : int, optional (default=1)
        Frequency (in generations) for logging GA progress.
    no_plots : bool, optional (default=True)
        Placeholder flag for plot suppression in downstream pipelines.
    top_combos_bars : int, optional (default=20)
        Maximum number of top combinations shown in optional visual summaries.
    max_best_returned : int or None, optional
        Maximum number of best genomes to retain after deduplication.
    evals_cap : int or None, optional
        Optional cap on total evaluations across all runs. Included in metadata,
        though not strictly enforced in this version.
    time_limit_sec : float or None, optional
        Optional wall-clock time limit (seconds). Included in metadata, though
        not strictly enforced in this version.

    Returns
    -------
    dict
        Dictionary containing best results, curves, and metadata:

        {
            "best_genome": list[int]
                Genome achieving the highest utility (best prefix + optimized tail).
            "best_utility": float
                Highest utility value obtained during the exhaustive procedure.
            "all_best_genomes": list[list[int]]
                All genomes tied for best utility (within tolerance).
            "curves": dict[str, list[float]]
                Fitness evolution curves:
                {
                    "best_combo_best": best-fitness curve for the top combination,
                    "best_combo_mean": mean-fitness curve for the top combination,
                    "global_best": elementwise global best across combinations,
                    "global_mean": mean across all combinations
                }
            "meta": dict
                Run metadata and diagnostics:
                {
                    "labels": list[str]
                        Ordered list of goods detected by the Planner.
                    "sizes": list[int]
                        Per-good prefix segment sizes.
                    "K": int
                        Total prefix length.
                    "L_used": int
                        Total genome length (prefix + tail).
                    "alphabet": list[int]
                        Prefix alphabet (set of admissible indices).
                    "combo_space": int
                        Total number of class combinations explored (or sampled).
                    "tie_tolerance": float
                        Numerical tolerance for detecting fitness ties.
                    "num_best_genomes": int
                        Number of unique genomes achieving best utility.
                    "num_best_found": int
                        Total number of best genomes found before truncation.
                    "max_best_returned": int or None
                        Cap applied to number of returned best genomes.
                    "evals_per_gen": list[int]
                        Approximated evaluations per generation.
                    "evals_cum": list[int]
                        Cumulative evaluation counts.
                    "runtime_sec": float
                        Total runtime of the exhaustive search.
                    "budget": dict
                        Budget information (evaluation/time limits):
                        {
                            "evals_cap": int or None,
                            "time_limit_sec": float or None,
                            "triggered": bool,
                            "reason": str or None,
                            "evals_total": int,
                            "time_total_sec": float
                        }
                }

    Notes
    -----
    - The exhaustive GA decomposes the search into prefix (fixed equivalence-class
      combinations) and tail (optimized via DEAP) components.
    - Prefixes are generated using weak compositions of each good’s prefix size
      over the detected alphabet, constrained optionally by `per_good_cap`.
    - For each prefix, a tail-only GA optimizes remaining binary genes.
    - Tracks best utility globally and per prefix combination, computing both
      local (per combination) and global (aggregate) fitness curves.
    - Suitable for small search spaces or as a benchmark for Joint GA variants.
    """

    import time
    t0 = time.time()

    # Try to forward budget if the inner API supports it (harmless if ignored).
    best_g, best_u, traces = _maximize_by_class_combinations(
        production_graph=production_graph,
        pmatrix=pmatrix,
        agents_information=agents_information,
        mode=mode,
        num_generations=generations,
        population_size=popsize,
        num_parents_mating=parents,
        mutation_rate=mutation_rate,
        fix_last_gene=fix_last_gene,
        seed=seed,
        max_combos=max_combos,
        per_good_cap=per_good_cap,
        max_index_probe=max_index_probe,
        verbosity=verbosity,
        log_every=log_every,
        no_plots=no_plots,
        top_combos_bars=top_combos_bars,
        max_best_returned=max_best_returned,
        # If unsupported, Python will raise TypeError; in that case, call again without budget kwargs.
        evals_cap=evals_cap,
        time_limit_sec=time_limit_sec,
    )

    # If budget kwargs are not accepted, recall without them.
    # (Avoids breaking older internal implementations.)
    # noinspection PyBroadException
    try:
        pass
    except TypeError:
        best_g, best_u, traces = _maximize_by_class_combinations(
            production_graph=production_graph,
            pmatrix=pmatrix,
            agents_information=agents_information,
            mode=mode,
            num_generations=generations,
            population_size=popsize,
            num_parents_mating=parents,
            mutation_rate=mutation_rate,
            fix_last_gene=fix_last_gene,
            seed=seed,
            max_combos=max_combos,
            per_good_cap=per_good_cap,
            max_index_probe=max_index_probe,
            verbosity=verbosity,
            log_every=log_every,
            no_plots=no_plots,
            top_combos_bars=top_combos_bars,
            max_best_returned=max_best_returned,
        )

    curves = {
        "best_combo_best": traces.get("best_combo", {}).get("best_curve", []),
        "best_combo_mean": traces.get("best_combo", {}).get("mean_curve", []),
        "global_best": traces.get("global_best_curve", []),
        "global_mean": traces.get("global_mean_curve", []),
    }

    # Evaluation series (best effort): prefer explicit series; else approximate
    evals_per_gen = traces.get("evals_per_gen", [])
    if not evals_per_gen:
        # Fallback approximation: popsize at gen0 + popsize per generation
        L = int(generations) + 1
        P = int(popsize) if popsize else 1
        evals_per_gen = [P] + [P] * (L - 1)
    evals_cum = np.cumsum(np.array(evals_per_gen, dtype=int)).tolist()

    meta = {
        "labels": traces.get("labels"),
        "sizes": traces.get("sizes"),
        "K": traces.get("K"),
        "L_used": traces.get("L_used"),
        "alphabet": traces.get("alphabet"),
        "combo_space": traces.get("combo_space"),
        "tie_tolerance": traces.get("tie_tolerance"),
        "num_best_genomes": traces.get("num_best_genomes", 1),
        "num_best_found": traces.get("num_best_found", 1),
        "max_best_returned": traces.get("max_best_returned"),
        "evals_per_gen": evals_per_gen,
        "evals_cum": evals_cum,
        "runtime_sec": float(time.time() - t0),
        "budget": {
            "evals_cap": evals_cap,
            "time_limit_sec": time_limit_sec,
            "triggered": False if evals_cap is None and time_limit_sec is None
                         else (evals_cum[-1] >= (evals_cap or 10**18)),
            "reason": "evals" if (evals_cap and evals_cum[-1] >= evals_cap) else None,
            "evals_total": int(evals_cum[-1]) if evals_cum else 0,
            "time_total_sec": float(time.time() - t0),
        },
    }

    all_best_genomes = traces.get("all_best_genomes", [best_g.tolist()])
    return {
        "best_genome": list(best_g.tolist()),
        "best_utility": float(best_u),
        "all_best_genomes": all_best_genomes,
        "curves": curves,
        "meta": meta,
    }

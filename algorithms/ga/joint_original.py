# algorithms/ga/joint_original.py
# =============================================================================
# Purpose
# =============================================================================
# This module implements a unified Genetic Algorithm (GA) whose genotype embeds
# both: (i) per-good class selectors (indices into per-good equivalence-class
# pools) and (ii) a binary tail. At evaluation time, the phenotype provided to
# the Economy is obtained by expanding the selected classes into the REAL prefix
# positions (as discovered from the Planner’s transaction order) and appending
# the tail. The last gene is forcefully set to 1 when 'fix_last_gene=True'.
#
# DEAP integration:
#   - Uses creator/toolbox for types and operator registration.
#   - Keeps an elitist truncation regime with uniform crossover and split
#     mutation (selectors vs tail).
#   - Compiles MultiStatistics and enriches each record with custom metrics
#     (phenotype Hamming diversity and unique phenotype count).
#   - Tracks best utility per selector-combination for diagnostics.
#   - Enforces the last-gene invariant at operator-decoration level to keep
#     genotype/tail statistics consistent with phenotype constraints.
#
# Public API:
#   run_ga_equivclass_joint(production_graph, pmatrix, agents_information, ...)
#       -> Dict with 'best_genome', 'best_utility', 'all_best_genomes', 'curves',
#          'meta', and 'best_by_selectors'.
# =============================================================================

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import random

from deap import base, creator, tools

from classes.economy.economy import Economy
from algorithms.ga.common import (
    detect_prefix_layout_and_sizes,         # Planner-grounded prefix layout
    probe_allowed_indices_via_tx_builder,   # admissible integer alphabet
)
from algorithms.ga.common import deap_clone


# =============================================================================
# Combinatorics: per-good equivalence classes (weak compositions)
# =============================================================================
def _num_equiv_classes(alpha_size: int, k: int) -> int:
    """
    Returns the number of weak compositions of k into 'alpha_size' parts,
    i.e. the binomial coefficient C(alpha_size + k - 1, k).
    """
    if alpha_size <= 0:
        return 1 if k == 0 else 0
    import math
    return math.comb(alpha_size + k - 1, k)


def _iter_count_vectors(alpha_size: int, k: int):
    """
    Yields all non-negative integer vectors 'c' of length 'alpha_size' with
    sum(c) == k, via the classic stars-and-bars enumeration. Each vector is
    returned as a numpy array of dtype=int.
    """
    if k < 0:
        return
    if alpha_size == 0:
        if k == 0:
            yield np.zeros(0, dtype=int)
        return
    total_slots = k + alpha_size - 1
    from itertools import combinations
    for divs in combinations(range(total_slots), alpha_size - 1):
        prev = -1
        out: List[int] = []
        for d in divs + (total_slots,):
            out.append(d - prev - 1)
            prev = d
        yield np.array(out, dtype=int)


def _sample_count_vectors(alpha_size: int, k: int, num_samples: int, rng: np.random.Generator) -> List[np.ndarray]:
    """
    Uniformly samples 'num_samples' weak compositions (duplicates may occur for
    small spaces). Useful when per-good class spaces are very large.
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
    """
    Expands a counts vector over 'alphabet' into a non-decreasing value vector,
    e.g., alphabet=[0,1,2], counts=[2,0,1] -> [0,0,2].
    """
    vals: List[int] = []
    for val, c in zip(alphabet, counts):
        if c > 0:
            vals.extend([int(val)] * int(c))
    return np.array(vals, dtype=int)


def _build_class_pools(
    alphabet: List[int],
    sizes: Sequence[int],
    per_good_cap: Optional[int],
    rng: np.random.Generator,
) -> Tuple[List[List[np.ndarray]], List[int]]:
    """
    For each primary good g with size k_g, constructs a pool of equivalence
    classes (count vectors over 'alphabet' summing to k_g). Optionally caps
    pool size via random sampling. Returns the per-good pools and their sizes.
    """
    pools: List[List[np.ndarray]] = []
    pool_sizes: List[int] = []
    A = len(alphabet)
    for k_g in sizes:
        total_g = _num_equiv_classes(A, int(k_g))
        if per_good_cap is not None and total_g > per_good_cap:
            classes_g = _sample_count_vectors(A, int(k_g), int(per_good_cap), rng)
        else:
            classes_g = list(_iter_count_vectors(A, int(k_g)))
        pools.append(classes_g)
        pool_sizes.append(len(classes_g))
    return pools, pool_sizes


# =============================================================================
# Genotype → Phenotype mapping (selectors + tail  → full genome of length L_used)
# =============================================================================
def _phenotype_from_individual(
    indiv: Sequence[int],
    pools: List[List[np.ndarray]],
    index_sets: List[List[int]],
    alphabet: List[int],
    L_used: int,
    K: int,
    fix_last_gene: bool,
) -> np.ndarray:
    """
    Builds the full genome using the REAL prefix layout:
      - For each good g, maps selector index -> class counts -> expanded values,
        then writes those values at the Planner-detected positions for g.
      - Appends the tail segment starting at K.
    """
    G = len(pools)
    selectors = np.asarray(indiv[:G], dtype=int)
    tail = np.asarray(indiv[G:], dtype=int)

    # Prefix
    prefix = np.zeros(K, dtype=int)
    for ig in range(G):
        pool_g = pools[ig]
        sel = int(np.clip(selectors[ig], 0, len(pool_g) - 1))
        counts_g = pool_g[sel]
        vals_g = _canonical_values_from_counts(alphabet, counts_g)
        pos_g = index_sets[ig]
        if len(vals_g) != len(pos_g):
            raise ValueError(
                f"Counts length mismatch for good index {ig}: {len(vals_g)} != {len(pos_g)}"
            )
        for v, j in zip(vals_g, pos_g):
            prefix[j] = int(v)

    # Assemble full genome
    genome = np.zeros(L_used, dtype=int)
    if K > 0:
        genome[:K] = prefix
    tail_len = max(0, L_used - K)
    if tail_len > 0:
        genome[K:K + tail_len] = tail[:tail_len]

    if fix_last_gene and L_used > 0:
        genome[-1] = 1
    return genome


# =============================================================================
# DEAP: type guards, initialization, operators, and decorators
# =============================================================================
def _ensure_deap_types() -> None:
    """Creates DEAP creator classes once per process."""
    try:
        _ = creator.FitnessMax
        _ = creator.Individual
    except AttributeError:
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)


def _init_joint_individual(icls, G: int, tail_len: int, pool_sizes: List[int], fix_last_gene: bool):
    """
    Creates a joint individual: first G integers (selectors), followed by a
    binary tail of length 'tail_len'. The last tail bit is fixed to 1 if needed.
    """
    # selectors
    sel = [random.randint(0, max(0, pool_sizes[g] - 1)) for g in range(G)]
    # tail
    tail = [random.randint(0, 1) for _ in range(tail_len)]
    if fix_last_gene and tail_len > 0:
        tail[-1] = 1
    return icls(sel + tail)


def _mate_uniform(ind1, ind2):
    """
    Uniform crossover across the entire chromosome (selectors + tail). Since
    gene domains are discrete and inherited values remain admissible (selectors
    are copied, not re-sampled), domain consistency is preserved by design.
    """
    L = len(ind1)
    if L == 0:
        return ind1, ind2
    mask = [random.randint(0, 1) for _ in range(L)]
    for i in range(L):
        if mask[i]:
            ind1[i], ind2[i] = ind2[i], ind1[i]
    return ind1, ind2


def _mutate_joint(ind, G: int, pool_sizes: List[int], sel_mutation: float, tail_mutation: float, fix_last_gene: bool):
    """
    Applies two mutation regimes:
      - Selector genes (0..G-1): with probability 'sel_mutation', switch to a
        different valid class index (uniform among remaining options).
      - Tail genes (G..end): bit-flip with probability 'tail_mutation', except
        the last locus when 'fix_last_gene' is True.
    """
    L = len(ind)
    # selectors
    for g in range(G):
        if pool_sizes[g] > 1 and random.random() < sel_mutation:
            cur = int(ind[g])
            choices = list(range(pool_sizes[g]))
            choices.remove(cur)
            ind[g] = random.choice(choices)
    # tail
    tail_len = L - G
    if tail_len > 0:
        last = L - 1 if fix_last_gene else L
        for j in range(G, last):
            if random.random() < tail_mutation:
                ind[j] = 1 - ind[j]
        if fix_last_gene and L > 0:
            ind[-1] = 1
    return (ind,)


def _decorate_enforce_last_gene(toolbox: base.Toolbox, L_used: int, K: int, fix_last_gene: bool,
                                pools: List[List[np.ndarray]], index_sets: List[List[int]],
                                alphabet: List[int]):
    """
    Defensive decorator that:
      - re-enforces genotype[-1] == 1 (if requested) to keep genotype-tail
        statistics aligned with the phenotype constraint,
      - ensures the phenotype cache (if any) will be rebuilt by deleting it.
    Notes:
      - The parameters (L_used, K, pools, index_sets, alphabet) are intentionally
        accepted for future invariants and diagnostic extensions.
    """
    if not fix_last_gene:
        return

    def enforce_and_invalidate(func):
        def wrapper(*args, **kwargs):
            offspring = func(*args, **kwargs)
            for child in offspring:
                # Enforce last bit at genotype level to keep stats consistent:
                if len(child) > 0:
                    child[-1] = 1
                # Invalidate any cached phenotype attached to the individual:
                if hasattr(child, "_phenotype"):
                    delattr(child, "_phenotype")
            return offspring
        return wrapper

    toolbox.decorate("mate", enforce_and_invalidate)
    toolbox.decorate("mutate", enforce_and_invalidate)


# =============================================================================
# Evaluation wrapper (DEAP-compatible) with phenotype caching
# =============================================================================
def _make_joint_evaluator(
    production_graph,
    pmatrix,
    agents_information,
    pools: List[List[np.ndarray]],
    index_sets: List[List[int]],
    alphabet: List[int],
    L_used: int,
    K: int,
    fix_last_gene: bool,
):
    """
    Returns evaluate(individual) that:
      - builds (and caches) the phenotype genome on the individual as
        'individual._phenotype' to enable custom metrics without recomputation,
      - calls Economy(...) and returns (utility,) as required by DEAP.
    """
    def evaluate(individual):
        if not hasattr(individual, "_phenotype"):
            pheno = _phenotype_from_individual(
                individual, pools, index_sets, alphabet, L_used, K, fix_last_gene
            )
            setattr(individual, "_phenotype", pheno)
        else:
            pheno = getattr(individual, "_phenotype")

        u = Economy(
            production_graph=production_graph,
            pmatrix=pmatrix,
            agents_information=agents_information,
            genome=pheno.tolist(),
        ).get_reports().get("utility", 0.0)
        return (float(u),)
    return evaluate


# =============================================================================
# Custom (non-DEAP) population metrics on phenotypes
# =============================================================================
def _extra_population_metrics_phenotype(pop: List[creator.Individual]) -> Dict[str, Any]:
    """
    Computes custom metrics over full phenotypes cached on the individuals:
      - 'pheno_diversity': mean pairwise Hamming distance (O(N^2)).
      - 'pheno_uniq': number of unique phenotype genomes.
    Individuals lacking a cached phenotype are ignored; if fewer than 2 remain,
    metrics default to zeros with a meaningful unique count.
    """
    # Pull cached phenotypes if available
    phenos: List[np.ndarray] = []
    for ind in pop:
        ph = getattr(ind, "_phenotype", None)
        if ph is not None:
            phenos.append(np.asarray(ph, dtype=np.int8))

    n = len(phenos)
    if n == 0:
        return {"pheno_diversity": 0.0, "pheno_uniq": 0}
    if n == 1 or phenos[0].size == 0:
        return {"pheno_diversity": 0.0, "pheno_uniq": 1}

    # Uniques
    uniq = len({tuple(p.tolist()) for p in phenos})

    # Diversity
    acc = 0.0
    pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            acc += float(np.mean(phenos[i] != phenos[j]))
            pairs += 1
    div = acc / pairs if pairs else 0.0
    return {"pheno_diversity": float(div), "pheno_uniq": int(uniq)}


# =============================================================================
# Public API
# =============================================================================
def run_joint_original(
    production_graph,
    pmatrix,
    agents_information,
    mode: str = "graph",
    generations: int = 50,
    popsize: int = 50,
    parents: int = 20,
    sel_mutation: float = 0.25,
    tail_mutation: float = 0.05,
    per_good_cap: Optional[int] = None,
    max_index_probe: int = 3,
    fix_last_gene: bool = True,
    seed: Optional[int] = 44,
    verbosity: int = 1,
    log_every: int = 1,
    # --- Budget (optional) ---
    evals_cap: int | None = None,
    time_limit_sec: float | None = None,
) -> Dict[str, Any]:
    """
    Executes the joint equivalence-class Genetic Algorithm (GA) that combines
    per-good class selectors with a binary tail. The algorithm evolves compact
    mixed-type genotypes that are expanded into full phenotypic genomes before
    evaluation through the `Economy` model.

    Parameters
    ----------
    production_graph : list[tuple[str, str]]
        Directed acyclic graph describing the production network, where each edge
        (u, v) denotes a dependency between goods.
    pmatrix : np.ndarray
        Price matrix of shape (n_goods, n_agents, n_agents), defining transaction
        prices between agents for each good.
    agents_information : dict
        Agent configuration dictionary required by the `Economy` class, specifying
        names, tax regimes, production abilities, etc.
    mode : str, optional (default="graph")
        Mode for prefix layout detection via `detect_prefix_layout_and_sizes`. The
        "graph" mode builds the prefix directly from the production graph topology.
    generations : int, optional (default=50)
        Number of evolutionary generations.
    popsize : int, optional (default=50)
        Number of individuals in the population.
    parents : int, optional (default=20)
        Number of elite individuals preserved across generations.
    sel_mutation : float, optional (default=0.25)
        Mutation probability applied to selector genes (integer indices).
    tail_mutation : float, optional (default=0.05)
        Mutation probability applied to tail genes (binary segment).
    per_good_cap : int or None, optional
        Optional cap limiting the number of equivalence classes (weak compositions)
        per good. If `None`, all possible combinations are generated.
    max_index_probe : int, optional (default=3)
        Maximum number of index probes used to infer admissible transaction indices
        via the transaction builder.
    fix_last_gene : bool, optional (default=True)
        Whether to enforce the last gene of the genome as 1 (ensuring consistent
        phenotype constraints).
    seed : int or None, optional (default=44)
        Random seed for reproducibility.
    verbosity : int, optional (default=1)
        Verbosity level for logging and summary output (0 = silent).
    log_every : int, optional (default=1)
        Frequency of generation-level logging (e.g., print every n generations).
    evals_cap : int or None, optional
        Maximum number of total fitness evaluations allowed before early stopping.
        When exceeded, the algorithm terminates after completing the current generation.
    time_limit_sec : float or None, optional
        Wall-clock time limit in seconds for early stopping.

    Returns
    -------
    dict
        Dictionary with results, performance curves, and metadata:

        {
            "best_genome": list[int]
                Phenotypic genome (expanded from selectors + tail) that achieved
                the highest utility.
            "best_utility": float
                Maximum utility value obtained during the run.
            "all_best_genomes": list[list[int]]
                All unique phenotypic genomes achieving the best utility (within tolerance).
            "curves": dict[str, list[float]]
                Fitness evolution curves across generations:
                {
                    "best": best fitness per generation,
                    "mean": average fitness per generation,
                    "median": median fitness per generation
                }
            "meta": dict
                Metadata and diagnostics of the run:
                {
                    "labels": list[str]
                        Ordered list of goods detected by the Planner.
                    "sizes": list[int]
                        Per-good prefix segment sizes.
                    "K": int
                        Total prefix length (sum of sizes).
                    "L_used": int
                        Full genome length (prefix + tail).
                    "alphabet": list[int]
                        Alphabet of admissible transaction indices for prefix construction.
                    "pool_sizes": list[int]
                        Number of equivalence classes (count vectors) per good.
                    "genome_internal": dict
                        Internal structure with number of selectors and tail length.
                    "generations": int
                        Number of generations effectively completed.
                    "popsize": int
                        Population size.
                    "parents": int
                        Number of elites retained.
                    "sel_mutation": float
                        Selector mutation rate.
                    "tail_mutation": float
                        Tail mutation rate.
                    "fix_last_gene": bool
                        Whether last gene enforcement was active.
                    "seed": int or None
                        Random seed used.
                    "tie_tolerance": float
                        Tolerance threshold for identifying ties in fitness.
                    "num_best_genomes": int
                        Count of genomes achieving best fitness.
                    "evals_per_gen": list[int]
                        Number of fitness evaluations per generation.
                    "evals_cum": list[int]
                        Cumulative evaluation counts.
                    "runtime_sec": float
                        Total runtime in seconds.
                    "budget": dict
                        Budget control and termination information:
                        {
                            "evals_cap": int or None,
                            "time_limit_sec": float or None,
                            "triggered": bool,
                            "reason": str or None,
                            "evals_total": int,
                            "time_total_sec": float
                        }
                }
            "best_by_selectors": list[tuple[str, float]]
                Sorted list of tuples (selector_combination, best_utility) showing
                the best performance achieved per selector configuration.
        }

    Notes
    -----
    - Genotype structure: `[selectors | binary_tail]`, where selector genes index
      equivalence classes of per-good allocations.
    - Phenotypes are built by expanding selected equivalence classes into real
      prefix positions, then appending the binary tail.
    - Maintains phenotype-level metrics: mean Hamming diversity and count of
      unique phenotypes in the population.
    - Supports early stopping by evaluation or time budgets.
    - Intended for fine-tuning experiments and large-scale environment suites
      using equivalence-class representations.
    """

    import time
    t0 = time.time()
    def _budget_reason(evals_total: int) -> str | None:
        if evals_cap is not None and evals_total >= int(evals_cap):
            return "evals"
        if time_limit_sec is not None and (time.time() - t0) >= float(time_limit_sec):
            return "time"
        return None

    random.seed(seed); np.random.seed(seed if seed is not None else None)
    _ensure_deap_types()

    labels, sizes, index_sets, tx_builder, L_min, _ = detect_prefix_layout_and_sizes(
        production_graph, mode=mode
    )
    K = int(sum(sizes))
    L_used = max(int(L_min), K + 1)
    alphabet = probe_allowed_indices_via_tx_builder(L_used, tx_builder, max_index_probe=max_index_probe)
    pools, pool_sizes = _build_class_pools(alphabet, sizes, per_good_cap, np.random.default_rng(seed))
    G = len(pools)
    tail_len = max(0, L_used - K)

    if verbosity >= 1:
        print("=== Joint GA detection summary ===")
        print(f"Goods (Planner order): {labels}")
        print(f"k_g per good: {sizes}  ->  K={K} | L_used={L_used}")
        print(f"Alphabet for prefix: {alphabet}  (|A|={len(alphabet)})")
        print(f"Per-good pool sizes: {pool_sizes}  (≈ product {int(np.prod(pool_sizes)) if pools else 1})")
        print(f"Internal chromosome: selectors={G}, tail={tail_len} (total={G + tail_len})")

    toolbox = base.Toolbox()
    toolbox.register("individual", _init_joint_individual, creator.Individual,
                     G=G, tail_len=tail_len, pool_sizes=pool_sizes, fix_last_gene=bool(fix_last_gene))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", _mate_uniform)
    toolbox.register("mutate", _mutate_joint, G=G, pool_sizes=pool_sizes,
                     sel_mutation=float(sel_mutation), tail_mutation=float(tail_mutation),
                     fix_last_gene=bool(fix_last_gene))
    toolbox.register("select", tools.selBest)
    toolbox.register("evaluate", _make_joint_evaluator(
        production_graph, pmatrix, agents_information, pools, index_sets, alphabet, L_used, K, fix_last_gene
    ))
    _decorate_enforce_last_gene(toolbox, L_used, K, fix_last_gene, pools, index_sets, alphabet)

    pop: List[creator.Individual] = toolbox.population(n=int(popsize))

    stats_fit = tools.Statistics(key=lambda ind: ind.fitness.values[0])
    stats_fit.register("min", np.min); stats_fit.register("avg", np.mean)
    stats_fit.register("med", np.median); stats_fit.register("max", np.max)
    stats_tail = tools.Statistics(key=lambda ind: (sum(ind[G:]) / max(1, len(ind) - G)))
    stats_tail.register("min", np.min); stats_tail.register("avg", np.mean); stats_tail.register("max", np.max)

    mstats = tools.MultiStatistics(fitness=stats_fit, tail=stats_tail)
    logbook = tools.Logbook()
    logbook.header = "gen", "evals", "fitness", "tail", "pheno_diversity", "pheno_uniq"
    logbook.chapters["fitness"].header = "min", "avg", "med", "max"
    logbook.chapters["tail"].header = "min", "avg", "max"

    best_curve: List[float] = []; mean_curve: List[float] = []; median_curve: List[float] = []
    best_by_selectors: Dict[Tuple[int, ...], float] = {}
    evals_per_gen: List[int] = []
    budget_triggered = False; budget_reason = None

    # --- Gen 0 ---
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses): ind.fitness.values = fit
    evals_per_gen.append(len(fitnesses))

    for ind in pop:
        ksel = tuple(int(x) for x in ind[:G]); val = float(ind.fitness.values[0])
        if val > best_by_selectors.get(ksel, -float("inf")): best_by_selectors[ksel] = val

    record = mstats.compile(pop)
    record.update(_extra_population_metrics_phenotype(pop))
    logbook.record(gen=0, evals=len(fitnesses), **record)

    vals0 = [ind.fitness.values[0] for ind in pop]
    b0, m0, med0 = float(np.max(vals0)), float(np.mean(vals0)), float(np.median(vals0))
    best_curve.append(b0); mean_curve.append(m0); median_curve.append(med0)
    if verbosity and (log_every > 0):
        print(f"Gen 000: best={b0:.6f} | mean={m0:.6f} | median={med0:.6f}")

    budget_reason = _budget_reason(sum(evals_per_gen))
    if budget_reason is not None: budget_triggered = True

    # --- Loop ---
    gen = 0
    for gen in range(1, int(generations) + 1):
        if budget_triggered: break

        elites = list(map(deap_clone, toolbox.select(pop, max(1, min(int(parents), len(pop) - 1)))))

        need = len(pop) - len(elites)
        offspring: List[creator.Individual] = []
        while len(offspring) < need:
            if len(elites) >= 2: p1, p2 = random.sample(elites, 2)
            else: p1 = elites[0]; p2 = deap_clone(elites[0])
            c1, c2 = deap_clone(p1), deap_clone(p2)
            c1, c2 = toolbox.mate(c1, c2)
            if hasattr(c1, "fitness"): del c1.fitness.values
            if hasattr(c2, "fitness"): del c2.fitness.values
            if hasattr(c1, "_phenotype"): delattr(c1, "_phenotype")
            if hasattr(c2, "_phenotype"): delattr(c2, "_phenotype")
            offspring.extend([c1, c2])
        offspring = offspring[:need]

        for mut in offspring:
            toolbox.mutate(mut)
            if hasattr(mut, "fitness"): del mut.fitness.values
            if hasattr(mut, "_phenotype"): delattr(mut, "_phenotype")

        invalid = [ind for ind in offspring if not ind.fitness.valid]
        fits = list(map(toolbox.evaluate, invalid))
        for ind, fit in zip(invalid, fits): ind.fitness.values = fit

        evals_per_gen.append(len(invalid))

        pop[:] = elites + offspring

        for ind in pop:
            ksel = tuple(int(x) for x in ind[:G]); val = float(ind.fitness.values[0])
            if val > best_by_selectors.get(ksel, -float("inf")): best_by_selectors[ksel] = val

        record = mstats.compile(pop)
        record.update(_extra_population_metrics_phenotype(pop))
        logbook.record(gen=gen, evals=len(invalid), **record)

        vals = [ind.fitness.values[0] for ind in pop]
        b, m, med = float(np.max(vals)), float(np.mean(vals)), float(np.median(vals))
        best_curve.append(b); mean_curve.append(m); median_curve.append(med)
        if verbosity >= 1 and (gen % max(1, log_every) == 0 or gen == int(generations)):
            print(f"Gen {gen:03d}: best={b:.6f} | mean={m:.6f} | median={med:.6f}")

        budget_reason = _budget_reason(sum(evals_per_gen))
        if budget_reason is not None: budget_triggered = True; break

    vals = np.array([ind.fitness.values[0] for ind in pop], dtype=float)
    best_val = float(np.max(vals))
    tie_tolerance = 1e-9
    tie_idx = np.nonzero(np.isclose(vals, best_val, rtol=0.0, atol=tie_tolerance))[0]

    seen: set[Tuple[int, ...]] = set()
    all_best_genomes: List[List[int]] = []
    for idx in tie_idx:
        ind = pop[idx]
        ph = getattr(ind, "_phenotype", None)
        if ph is None:
            ph = _phenotype_from_individual(ind, pools, index_sets, alphabet, L_used, K, fix_last_gene)
        key = tuple(int(x) for x in ph.tolist())
        if key not in seen:
            seen.add(key); all_best_genomes.append(list(key))

    best_genome_list: List[int]
    if all_best_genomes:
        best_genome_list = list(all_best_genomes[0])
    else:
        best_ind = tools.selBest(pop, 1)[0]
        best_ph = getattr(best_ind, "_phenotype", None)
        if best_ph is None:
            best_ph = _phenotype_from_individual(best_ind, pools, index_sets, alphabet, L_used, K, fix_last_gene)
        best_genome_list = list(best_ph.tolist())

    best_by_selectors_list = sorted(
        [("("+",".join(map(str, k))+")", float(v)) for k, v in best_by_selectors.items()],
        key=lambda x: x[1], reverse=True
    )

    evals_cum = np.cumsum(np.array(evals_per_gen, dtype=int)).tolist()
    runtime = time.time() - t0
    return {
        "best_genome": best_genome_list,
        "best_utility": best_val,
        "all_best_genomes": all_best_genomes if all_best_genomes else [list(best_genome_list)],
        "curves": {"best": best_curve, "mean": mean_curve, "median": median_curve},
        "meta": {
            "labels": labels,
            "sizes": list(map(int, sizes)),
            "K": K, "L_used": L_used,
            "alphabet": alphabet, "pool_sizes": pool_sizes,
            "genome_internal": {"selectors": G, "tail_len": tail_len},
            "generations": int(gen if not budget_triggered else gen),
            "popsize": int(popsize), "parents": int(parents),
            "sel_mutation": float(sel_mutation), "tail_mutation": float(tail_mutation),
            "fix_last_gene": bool(fix_last_gene), "seed": seed,
            "tie_tolerance": tie_tolerance,
            "num_best_genomes": len(all_best_genomes) if all_best_genomes else 1,
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
        "best_by_selectors": best_by_selectors_list,
    }
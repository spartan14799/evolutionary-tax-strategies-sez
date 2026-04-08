# algorithms/ga/macro_micro.py

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Callable

import numpy as np

from typing import Dict, Any

import random

import math

import time

from deap import base, creator, tools

from scipy.stats import norm

from src.algorithms.ga.common import (
    detect_prefix_layout_and_sizes,         # Planner-grounded prefix layout
    probe_allowed_indices_via_tx_builder,   # admissible integer alphabet
)
from src.algorithms.ga.common import deap_clone
from src.simulation.economy.economy import Economy



# algorithms/ga/macro_micro.py
# =============================================================================
# Purpose
# =============================================================================
# This module implements a two–stage (“Macro→Micro”) Genetic Algorithm (GA)
# for mixed–domain chromosomes whose genotype is
#     [ selectors (integer indices into per–good class pools) | binary tail ].
# At evaluation time, the phenotype provided to the Economy is obtained by
# expanding the selected per–good equivalence classes into the REAL prefix
# positions (as discovered from the Planner’s transaction order) and then
# appending the binary tail. The last locus can be forcefully set to 1 when
# 'fix_last_gene=True' to keep phenotype constraints consistent.
#
# Variation pipeline:
#   - MACRO crossover (one–point) at the selector|tail boundary (index G),
#     applied with probability 'p_macro'.
#   - MICRO crossover (segmented n–point) applied independently within the
#     selectors and within the tail, with densities 'lambda_in' and 'lambda_out',
#     and overall application probability 'p_micro'.
#   - Mutation is split by domain:
#       * Selectors: Gaussian step sizes over bounded integers, calibrated via
#         P(|Δ| ≥ τ) = p_min where τ = tau_policy(U_g) and U_g is the per–gene
#         domain width (pool_sizes[g] - 1). Steps are clipped to [0, U_g].
#       * Tail: standard bit–flip with probability 'tail_mutation_prob'.
#
# Phenotype and diagnostics:
#   - A cache–aware evaluator builds and stores the expanded phenotype on each
#     individual ('_phenotype'), enabling phenotype–level diagnostics without
#     recomputation (e.g., mean pairwise Hamming diversity and unique counts).
#   - A defensive decorator re–enforces the last–gene invariant after mate/mutate
#     and invalidates stale phenotype caches as needed.
#
# DEAP integration:
#   - Provides creator/toolbox setup for individuals, population, mate, mutate,
#     and evaluate (DEAP–compatible).
#   - Includes a statistics factory (MultiStatistics) to track fitness and tail
#     density, mirroring the JOINT implementation, and extends logs with
#     phenotype–level metrics.
#   - Uses an elitism + random non–elite parent sampling scheme that prevents
#     double counting elites in the mating pool (exploration–friendly).
#
# Planner coupling:
#   - Detects the REAL prefix layout and sizes from the production graph.
#   - Probes admissible integer alphabets via the transaction builder.
#   - Builds per–good equivalence–class pools (weak compositions) with optional
#     capping for tractability.
#
# Public API:
#   run_ga_macro_micro(production_graph, pmatrix, agents_information, ...)
#       -> Dict with 'best_genome', 'best_utility', 'all_best_genomes', 'curves',
#          'meta', and 'best_by_selectors'.
# =============================================================================




# =============================================================================
# Combinatorics: per-good equivalence classes (weak compositions)
# =============================================================================
def _num_equiv_classes(alpha_size: int, k: int) -> int:
    """
    Compute the number of weak compositions of k into 'alpha_size' parts,
    i.e., C(alpha_size + k - 1, k).

    Parameters
    ----------
    alpha_size : int
        Size of the discrete alphabet (|A|).
    k : int
        Number of positions to fill for the good (k_g).

    Returns
    -------
    int
        Number of weak compositions.

    Edge cases
    ----------
    - If alpha_size <= 0: return 1 if k == 0 else 0.
    """
    if alpha_size <= 0:
        return 1 if k == 0 else 0
    return math.comb(alpha_size + k - 1, k)


def _iter_count_vectors(alpha_size: int, k: int):
    """
    Yield all count vectors c ∈ ℕ^{alpha_size} with sum(c) = k
    via stars-and-bars.

    Yields
    ------
    np.ndarray
        A counts vector (dtype=int) of length 'alpha_size'.
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


def _sample_count_vectors(
    alpha_size: int,
    k: int,
    num_samples: int,
    rng: np.random.Generator,
) -> List[np.ndarray]:
    """
    Uniformly sample 'num_samples' weak compositions (with possible duplicates).

    Useful when the full space is very large and we want to cap per-good pools.
    """
    if num_samples <= 0:
        return []
    total_slots = k + alpha_size - 1
    out: List[np.ndarray] = []
    for _ in range(num_samples):
        if alpha_size > 1:
            divs = np.sort(
                rng.choice(total_slots, size=alpha_size - 1, replace=False)
            )
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
    Expand a counts vector over 'alphabet' into a non-decreasing value vector.

    Example
    -------
    alphabet=[0,1,2], counts=[2,0,1]  →  [0,0,2]
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
    For each primary good (with size k_g), build a pool of equivalence classes:
      - Each class is a counts vector over 'alphabet' summing to k_g.
      - Optionally cap the pool via uniform sampling.

    Parameters
    ----------
    alphabet : list[int]
        Discrete admissible values for REAL prefix indices.
    sizes : Sequence[int]
        k_g per good (number of positions/units for each good).
    per_good_cap : Optional[int]
        If not None and the total number of weak compositions exceeds
        this cap, sample 'per_good_cap' classes uniformly at random.
    rng : np.random.Generator
        Random generator for sampling.

    Returns
    -------
    pools : list[list[np.ndarray]]
        Per-good list of class count vectors.
    pool_sizes : list[int]
        Sizes of each per-good pool (after capping if any).
    """
    pools: List[List[np.ndarray]] = []
    pool_sizes: List[int] = []
    a_size = len(alphabet)

    for k_g in sizes:
        total_g = _num_equiv_classes(a_size, int(k_g))
        if per_good_cap is not None and total_g > per_good_cap:
            classes_g = _sample_count_vectors(a_size, int(k_g), int(per_good_cap), rng)
        else:
            classes_g = list(_iter_count_vectors(a_size, int(k_g)))
        pools.append(classes_g)
        pool_sizes.append(len(classes_g))

    return pools, pool_sizes


# =============================================================================
# Genotype → Phenotype mapping
# =============================================================================
def _phenotype_from_individual(
    indiv: Sequence[int],
    pools: List[List[np.ndarray]],
    index_sets: List[List[int]],
    alphabet: List[int],
    l_used: int,
    k_prefix: int,
    fix_last_gene: bool,
) -> np.ndarray:
    """
    Build the full phenotype genome using the REAL prefix layout.

    Steps
    -----
    1) Split the genotype into:
         - selectors: one integer per good (index into that good's class pool)
         - tail: remaining binary segment (appended after the REAL prefix)
    2) For each good g:
         selector index --> counts vector --> canonical values over 'alphabet'
         then write those values into the Planner-detected positions 'index_sets[g]'.
    3) Append tail starting at position k_prefix.
    4) Optionally enforce last gene == 1.

    Parameters
    ----------
    indiv : Sequence[int]
        Genotype: [selectors | tail].
    pools : list[list[np.ndarray]]
        Per-good pools of class counts.
    index_sets : list[list[int]]
        For each good g, indices in the REAL prefix where its values go.
    alphabet : list[int]
        Discrete alphabet used to expand counts into values.
    l_used : int
        Total used genome length (prefix + tail) in the phenotype.
    k_prefix : int
        REAL prefix length (sum of per-good sizes).
    fix_last_gene : bool
        Whether to force the last phenotype locus to 1.

    Returns
    -------
    np.ndarray
        Phenotype genome (length == l_used), dtype=int.

    Raises
    ------
    ValueError
        If a counts→values expansion length doesn't match its index set.
    """
    g_count = len(pools)
    selectors = np.asarray(indiv[:g_count], dtype=int)
    tail = np.asarray(indiv[g_count:], dtype=int)

    # Build REAL prefix from per-good selections
    prefix = np.zeros(k_prefix, dtype=int)
    for ig in range(g_count):
        pool_g = pools[ig]
        if not pool_g:
            continue
        sel = int(np.clip(selectors[ig], 0, len(pool_g) - 1))
        counts_g = pool_g[sel]
        vals_g = _canonical_values_from_counts(alphabet, counts_g)
        pos_g = index_sets[ig]
        if len(vals_g) != len(pos_g):
            raise ValueError(
                f"Counts length mismatch for good index {ig}: "
                f"{len(vals_g)} != {len(pos_g)}"
            )
        for v, j in zip(vals_g, pos_g):
            prefix[j] = int(v)

    # Assemble phenotype genome
    genome = np.zeros(l_used, dtype=int)
    if k_prefix > 0:
        genome[:k_prefix] = prefix

    tail_len = max(0, l_used - k_prefix)
    if tail_len > 0:
        genome[k_prefix : k_prefix + tail_len] = tail[:tail_len]

    if fix_last_gene and l_used > 0:
        genome[-1] = 1

    return genome


# =============================================================================
# DEAP: type guards, initialization, and decorators
# =============================================================================
def _ensure_deap_types() -> None:
    """
    Creates DEAP creator classes once per process.

    Ensures the presence of:
      - creator.FitnessMax  (single-objective maximization)
      - creator.Individual  (list-based individual with that fitness)
    """
    try:
        _ = creator.FitnessMax  # type: ignore[attr-defined]
        _ = creator.Individual  # type: ignore[attr-defined]
    except AttributeError:
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)


def _init_joint_individual(
    icls,
    G: int,
    tail_len: int,
    pool_sizes: List[int],
    fix_last_gene: bool,
    rng_py: Optional[random.Random] = None,
):
    """
    Initialize a joint individual for the macro_micro GA:
      - First G integer selector genes (indices into per-good class pools).
      - Followed by a binary tail of length 'tail_len'.

    The last tail bit is fixed to 1 when 'fix_last_gene' is True.

    Parameters
    ----------
    icls : type
        The individual class constructor (usually creator.Individual).
    G : int
        Number of selector genes (one per good with equivalence-class pool).
    tail_len : int
        Length of the binary tail segment.
    pool_sizes : List[int]
        For each good g ∈ {0..G-1}, the size of its class pool (≥ 1).
    fix_last_gene : bool
        Whether to force the last locus of the genotype to 1.

    Returns
    -------
    creator.Individual
        A newly initialized individual.
    """
    if rng_py is None:
        rng_py = random.Random()
    
    # Selectors: uniform over valid class indices per good.
    sel = [rng_py.randint(0, max(0, pool_sizes[g] - 1)) for g in range(G)]

    # Tail: Bernoulli(0.5) per locus.
    tail = [rng_py.randint(0, 1) for _ in range(tail_len)]
    if fix_last_gene and tail_len > 0:
        tail[-1] = 1

    return icls(sel + tail)


def _decorate_enforce_last_gene(
    toolbox: base.Toolbox,
    L_used: int,
    K: int,
    fix_last_gene: bool,
    pools: List[List[np.ndarray]],
    index_sets: List[List[int]],
    alphabet: List[int],
):
    """
    Defensive decorator that:
      - Re-enforces genotype[-1] == 1 (if requested) to keep genotype-tail
        statistics aligned with the phenotype constraint,
      - Ensures the phenotype cache (if any) will be rebuilt by deleting it.

    Notes
    -----
    The parameters (L_used, K, pools, index_sets, alphabet) are accepted for
    future invariants/diagnostics; they are unused here but kept for API parity.
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

        # Preserve metadata for nicer introspection
        wrapper.__name__ = getattr(func, "__name__", "enforced_" + str(func))
        wrapper.__doc__ = getattr(func, "__doc__", None)
        return wrapper

    toolbox.decorate("mate", enforce_and_invalidate)
    toolbox.decorate("mutate", enforce_and_invalidate)


def _safe_fitness_value(ind) -> float:
    """
    Returns a comparable scalar fitness value for an individual.

    If the individual has no valid fitness (e.g., not evaluated yet),
    returns -inf to ensure it never gets selected as elite by accident.
    """
    try:
        if hasattr(ind, "fitness") and ind.fitness.valid:
            # Single-objective: take the first component
            return float(ind.fitness.values[0])
    except Exception:
        pass
    return float("-inf")


# =============================================================================
# DEAP: type guards, initialization, operators, and decorators
# =============================================================================

# =============================================================================
# Selection: Parents pool and elites
# =============================================================================
def _resolve_elite_count(pop_size: int, parents: int, elite_fraction: float) -> int:
    """
    Computes a robust elite count using the same philosophy as JOINT:
    - Base rule: floor(parents * elite_fraction)
    - Guarantees:
        * At least 1 elite (to preserve best-so-far),
        * Never more than pop_size - 1 (keeps room for offspring),
        * Never more than parents - 1 (keeps room for non-elite parents).
    """
    if pop_size <= 1:
        return 0  # degenerate population: no space to preserve elites meaningfully
    base = int(math.floor(parents * max(0.0, float(elite_fraction))))
    # Ensure feasible bounds relative to population and parent budget
    upper_cap = max(0, min(pop_size - 1, max(0, parents - 1)))
    return max(1, min(base, upper_cap)) if parents > 0 else 0


# Parents pool with elites + tournament over non-elites
def select_elite_and_tournament(
    pop: List[creator.Individual],
    parents: int,
    elite_fraction: float,
    tourn_size: int = 3,
    clone_fn: Optional[Callable] = None,
) -> Tuple[List[creator.Individual], List[creator.Individual], List[creator.Individual]]:
    """
    Splits the population into (elites, non_elites_tournament_parents, non_elites_all)
    according to an 'elitism + tournament among non-elites' scheme.

    This selector mirrors the replacement logic used elsewhere in this codebase
    (preserve elites; breed from a parents pool) but replaces uniform random sampling
    among non-elites with DEAP's tournament selection on the non-elite subset only.
    Elites are cloned to avoid double counting and to preserve them unchanged.

    Parameters
    ----------
    pop : list[creator.Individual]
        Current population (evaluated or partially evaluated).
    parents : int
        Target size for the parent pool (elites + additional non-elites).
        May be larger than the non-elite pool; the function caps safely.
    elite_fraction : float
        Fraction of 'parents' to allocate to elitism. Internally robustified
        via `_resolve_elite_count` to maintain at least one elite and to keep
        room for offspring and non-elite parents.
    tourn_size : int, optional (default=3)
        Tournament size for DEAP's `tools.selTournament`. Higher values increase
        selection pressure among non-elites; must be ≥ 2 for meaningful tournaments.
    clone_fn : callable or None, optional
        Cloning function for elite preservation, e.g., `deap_clone`.
        If None, defaults to `deap_clone`.

    Returns
    -------
    elites : list[creator.Individual]
        A cloned list of elite individuals to be carried unchanged into the next generation.
    non_elites_tournament_parents : list[creator.Individual]
        Tournament-selected parents from the non-elite subset (no elites are present here).
        May be empty if no non-elites are available or if `parents <= elite_count`.
    non_elites_all : list[creator.Individual]
        All non-elite individuals (references to the original population), useful
        for diagnostics or alternative downstream strategies.

    Notes
    -----
    - Individuals without valid fitness are assigned fitness = -inf via `_safe_fitness_value`
      and never enter the elite set unless the population is entirely invalid.
    - Elites are not allowed to participate in the tournament; this prevents elite
      over-representation in the mating pool and preserves exploration capacity.
    - The number of tournament-selected parents is capped by the size of the non-elite pool.
    - DEAP's tournament uses Python's global RNG; seeding is already managed by the runner.
    """
    if clone_fn is None:
        clone_fn = deap_clone

    n = len(pop)
    if n == 0 or parents <= 0:
        return [], [], pop[:]  # nothing to do

    # Stable ranking by scalar fitness (descending); ties preserve order
    ranked = sorted(pop, key=_safe_fitness_value, reverse=True)

    # Robust elite count consistent with JOINT philosophy
    elite_count = _resolve_elite_count(pop_size=n, parents=parents, elite_fraction=elite_fraction)

    # Elite references (no clones yet) and non-elites set
    elite_refs = ranked[:elite_count]
    elite_ids = {id(ind) for ind in elite_refs}
    non_elites_all = [ind for ind in pop if id(ind) not in elite_ids]

    # Clone elites for safe carry-over
    elites = [clone_fn(ind) for ind in elite_refs]

    # Determine parent shortfall to be filled from non-elites via tournament
    parents_needed = max(0, min(int(parents) - elite_count, len(non_elites_all)))

    if parents_needed > 0 and len(non_elites_all) > 0:
        tsize = max(2, int(tourn_size))
        non_elites_tournament_parents = tools.selTournament(
            non_elites_all, k=parents_needed, tournsize=tsize
        )
    else:
        non_elites_tournament_parents = []

    return elites, non_elites_tournament_parents, non_elites_all

# Parents pool with elites + random over non-elites
def select_elite_and_random(
    pop: List[creator.Individual],
    parents: int,
    elite_fraction: float,
    clone_fn: Optional[Callable] = None,
    rng: Optional[random.Random] = None,
) -> Tuple[List[creator.Individual], List[creator.Individual], List[creator.Individual]]:
    """
    Splits the population into (elites, non_elites_random_parents, non_elites_all)
    according to an 'elitism + random among non-elites' scheme.

    The procedure mirrors JOINT's replacement logic (preserve elites,
    breed from a parent pool) but replaces tournament selection for the
    non-elite portion with uniform random sampling WITHOUT replacement.

    Parameters
    ----------
    pop : list[creator.Individual]
        Current population (evaluated or partially evaluated).
    parents : int
        Target size for the parent pool (elites + additional non-elites).
        May be larger than the non-elite pool; the function caps safely.
    elite_fraction : float
        Fraction of 'parents' to allocate to elitism. Internally robustified
        to maintain at least one elite and to keep room for offspring.
    clone_fn : callable or None, optional
        Function used to clone individuals for elite preservation, e.g. `deap_clone`.
        If None, uses `tools.clone`.
    rng : random.Random or None, optional
        RNG instance for reproducibility. If None, uses the global `random`.

    Returns
    -------
    elites : list[creator.Individual]
        A cloned list of elite individuals to be carried unchanged into the next generation.
    non_elites_random_parents : list[creator.Individual]
        Randomly sampled (without replacement) non-elite parents to complete the parent pool.
        May be empty if no non-elites are available or if `parents <= elite_count`.
    non_elites_all : list[creator.Individual]
        All non-elite individuals (references to the original population), useful
        for diagnostics or alternative downstream strategies.

    Notes
    -----
    - Individuals lacking a valid fitness are treated as having fitness = -inf and will
      not appear among elites unless the entire population is invalid (in which case
      elites fall back to the best available ordering).
    - The function avoids duplicating elites in the non-elite parent sample.
    - Sampling among non-elites is WITHOUT replacement to prevent over-counting.
    - If the non-elite pool is smaller than the requested number of additional parents,
      the function returns as many as available (no replacement fallback to elites),
      mirroring the conservative behavior used in JOINT.
    - This selection reduces selection pressure vs. tournaments, favoring exploration.
      If stronger pressure is desired while keeping randomness, consider rank-based
      stochastic universal sampling or biased sampling with probability ∝ rank.
    """
    if clone_fn is None:
        clone_fn = deap_clone
    if rng is None:
        rng = random

    n = len(pop)
    if n == 0 or parents <= 0:
        return [], [], pop[:]  # nothing to do

    # Stable sort by fitness (descending); ties keep relative order to avoid bias.
    # Use (fitness, tie_breaker) if stricter determinism across equal fitness is needed.
    ranked = sorted(pop, key=_safe_fitness_value, reverse=True)

    # Compute elite count with robust caps (keeps at least one elite when feasible)
    elite_count = _resolve_elite_count(pop_size=n, parents=parents, elite_fraction=elite_fraction)

    # Select references (no cloning yet) and build non-elite pool
    elite_refs = ranked[:elite_count]
    elite_ids = {id(ind) for ind in elite_refs}
    non_elites_all = [ind for ind in pop if id(ind) not in elite_ids]

    # Clone elites for safe carry-over (prevents later in-place variation from mutating them)
    elites = [clone_fn(ind) for ind in elite_refs]

    # Determine how many additional parents are needed from the non-elite pool
    parents_needed = max(0, min(int(parents) - elite_count, len(non_elites_all)))

    # Random sampling among non-elites WITHOUT replacement
    if parents_needed > 0:
        # rng.sample raises ValueError if k > len(pop); guarded by min() above
        non_elites_random_parents = rng.sample(non_elites_all, k=parents_needed)
    else:
        non_elites_random_parents = []

    return elites, non_elites_random_parents, non_elites_all

def _select_elites_for_carry(
    pop: List[creator.Individual],
    elite_fraction: float,
    min_elites: int = 1,
    clone_fn: Optional[Callable] = None,
) -> List[creator.Individual]:
    """
    Select and CLONE the elite set to carry into the next generation.
    The count is derived from 'elite_fraction' on the current population size,
    but is robustified to:
      - keep at least 'min_elites' (default = 1),
      - never exceed len(pop) - 1 (leave room for offspring).

    Notes
    -----
    - Uses the same scalar fitness ordering as elsewhere (_safe_fitness_value).
    - Cloning avoids in-place variation from mutating the preserved elites.
    """
    if clone_fn is None:
        clone_fn = deap_clone

    n = len(pop)
    if n <= 1:
        return [clone_fn(pop[0])] if n == 1 else []

    # Base on population, not on 'parents' (pairwise mode ignores parent pool).
    base = int(math.floor(max(0.0, float(elite_fraction)) * n))
    # Robust caps
    upper = max(0, n - 1)
    ecount = max(int(min_elites), min(base, upper))

    ranked = sorted(pop, key=_safe_fitness_value, reverse=True)
    return [clone_fn(ind) for ind in ranked[:ecount]]

# =============================================================================
# Macro crossover: one-point at the selector|tail boundary (index G)
# =============================================================================
def _mate_macro_boundary(ind1, ind2, G: int, fix_last_gene: bool):
    """
    Perform a one-point crossover **exactly** at the selector|tail boundary.

    Genotype layout:
        ind = [ selectors(0..G-1) | tail(G..L-1) ]

    This operator swaps the **suffix starting at G** between the two parents,
    producing two offspring:
        child1 = [sel_1 | tail_2]
        child2 = [sel_2 | tail_1]

    The operator works **in place** (DEAP convention) and returns (ind1, ind2).

    Parameters
    ----------
    ind1, ind2 : list-like
        Parent individuals to be modified in place.
    G : int
        Number of selector genes (boundary index in genotype space).
        The binary tail begins at position G.
    fix_last_gene : bool
        Whether the last locus must be set to 1. This function enforces it
        defensively for edge cases; in normal operation, a decorator is also
        applied at toolbox registration time to keep invariants.

    Returns
    -------
    tuple
        (ind1, ind2) after in-place crossover.

    Edge-case handling
    ------------------
    - If len(ind1) != len(ind2), the operator swaps tails up to the minimum
      common length and leaves any extra trailing genes untouched (robustness).
      In this framework, individuals should have equal length; the behavior
      here is defensive.
    - If G <= 0 (no selectors), swapping the suffix from 0 means swapping the
      entire genomes (equivalent to exchanging parents).
    - If G >= L (no tail), there is nothing to swap; parents are returned as-is.
    - If fix_last_gene is True and L > 0, the last locus is re-enforced to 1.

    Notes
    -----
    - This macro operator is intentionally simple and deterministic in its cut
      position to compose cleanly with a subsequent micro–crossover that will
      operate *within* segments.
    """
    # Defensive: equal-length is expected; handle gracefully otherwise.
    L1 = len(ind1)
    L2 = len(ind2)
    if L1 == 0 or L2 == 0:
        # Degenerate genomes: nothing to do.
        return ind1, ind2

    L = min(L1, L2)
    cut = max(0, min(int(G), L))  # clamp G to [0, L]

    # If cut == L, there is no tail to swap; return as-is.
    if cut >= L:
        # Still enforce invariant if requested.
        if fix_last_gene and L1 > 0:
            ind1[-1] = 1
        if fix_last_gene and L2 > 0:
            ind2[-1] = 1
        return ind1, ind2

    # Swap suffixes [cut: ] in place.
    tail1 = ind1[cut:L1]  # full remainder of ind1
    tail2 = ind2[cut:L2]  # full remainder of ind2

    # Replace in place; respect original lengths independently.
    ind1[cut:L1] = tail2[: L1 - cut]
    ind2[cut:L2] = tail1[: L2 - cut]

    # Enforce last-gene invariant defensively (a decorator should also do this).
    if fix_last_gene:
        if len(ind1) > 0:
            ind1[-1] = 1
        if len(ind2) > 0:
            ind2[-1] = 1

    # Any phenotype cache attached to individuals becomes stale after crossover.
    if hasattr(ind1, "_phenotype"):
        delattr(ind1, "_phenotype")
    if hasattr(ind2, "_phenotype"):
        delattr(ind2, "_phenotype")

    return ind1, ind2

# =============================================================================
# Micro crossover: segmented n-point over [selectors|tail]
# =============================================================================
def _mate_micro_segmented(
    ind1,
    ind2,
    G: int,
    lambda_in: float,
    lambda_out: float,
    fix_last_gene: bool,
    rng: Optional[random.Random] = None,
):
    """
    Perform segmented n-point crossover on two offspring:
      - An n-point crossover within the selector segment [0..G-1],
      - An independent n-point crossover within the tail segment [G..L-1].

    The number of crossover points in each segment is determined by:
        n_in  = max(1, floor(lambda_in  * L_in))
        n_out = max(1, floor(lambda_out * L_out))
    where L_in = G and L_out = (L - G). Each set of points is sampled uniformly
    at random without replacement from the interior indices of its segment.

    IMPORTANT:
        - If G <= 1 (i.e., at most one selector), the selector segment is skipped,
            but the tail segment is still recombined if its length ≥ 2.
        - If a segment length is < 2, that segment is skipped.

    Parameters
    ----------
    ind1, ind2 : list-like
        Two offspring to be modified in place.
    G : int
        Number of selector genes. Tail starts at index G.
    lambda_in : float
        Density parameter for the number of crossover points in the selector segment.
        Typical range [0, 1]. Values are clamped to be non-negative; the resulting
        n_in is also upper-bounded by (L_in - 1).
    lambda_out : float
        Density parameter for the number of crossover points in the tail segment.
        Typical range [0, 1]. Values are clamped to be non-negative; the resulting
        n_out is also upper-bounded by (L_out - 1).
    fix_last_gene : bool
        Whether to enforce the invariant 'last locus == 1' after crossover.
        A separate decorator usually enforces this as well; this function
        re-enforces it defensively.
    rng : random.Random or None, optional
        RNG used for reproducible point sampling. Defaults to the global `random`.

    Returns
    -------
    tuple
        (ind1, ind2) after in-place crossover.

    Notes
    -----
    - The multi-point crossover alternates ownership of contiguous segments
      delimited by the sampled cut points (standard n-point scheme).
    - Phenotype caches attached to individuals are invalidated at the end.
    - If lengths differ (unexpected), the operator uses the minimum common length
      for safety and leaves extra trailing genes untouched.
    """
    if rng is None:
        rng = random

    L1, L2 = len(ind1), len(ind2)
    if L1 == 0 or L2 == 0:
        return ind1, ind2

    # Work with the common prefix of both genomes (defensive robustness).
    L = min(L1, L2)
    G = max(0, min(int(G), L))  # clamp boundary to [0, L]

    # Helper: n-point crossover on a half-open interval [start, end)
    def _npoint_inplace(seq1, seq2, start: int, end: int, n_points: int, rng_: random.Random):
        """Apply an n-point crossover in-place on [start, end)."""
        seg_len = end - start
        if seg_len < 2 or n_points <= 0:
            return

        # Interior cut positions: choose from {start+1, ..., end-1}
        max_points = max(0, seg_len - 1)
        n_eff = max(1, min(int(n_points), max_points))
        # Sample without replacement and sort
        cuts = sorted(rng_.sample(range(start + 1, end), k=n_eff))

        # Build segment boundaries: start | cuts... | end
        bounds = [start] + cuts + [end]

        # Alternate swapping on odd-indexed segments
        for idx in range(len(bounds) - 1):
            a, b = bounds[idx], bounds[idx + 1]
            if (idx % 2) == 1:  # swap odd segments
                # Swap slice [a:b] in place
                tmp = seq1[a:b]
                seq1[a:b] = seq2[a:b]
                seq2[a:b] = tmp

    # Compute lengths of segments
    L_in = G
    L_out = L - G

    # Crossover in SELECTOR segment
    if L_in >= 2:
        lam_in = max(0.0, float(lambda_in))
        n_in = max(1, int(math.floor(lam_in * L_in)))
        n_in = min(n_in, L_in - 1)
        if n_in > 0:
            _npoint_inplace(ind1, ind2, start=0, end=G, n_points=n_in, rng_=rng)

    # Crossover in TAIL segment
    if L_out >= 2:
        lam_out = max(0.0, float(lambda_out))
        n_out = max(1, int(math.floor(lam_out * L_out)))
        n_out = min(n_out, L_out - 1)
        if n_out > 0:
            _npoint_inplace(ind1, ind2, start=G, end=L, n_points=n_out, rng_=rng)

    # Defensive invariant + phenotype cache invalidation
    if fix_last_gene:
        if len(ind1) > 0:
            ind1[-1] = 1
        if len(ind2) > 0:
            ind2[-1] = 1
    if hasattr(ind1, "_phenotype"):
        delattr(ind1, "_phenotype")
    if hasattr(ind2, "_phenotype"):
        delattr(ind2, "_phenotype")

    return ind1, ind2

# =============================================================================
# Mutation rates resolver (simple defaults + explicit overrides)
# =============================================================================
def _resolve_mutation_rates_simple(
    G: int,
    L_used: int,
    K: int,
    sel_mutation: Optional[float],
    tail_mutation: Optional[float],
) -> Tuple[float, float]:
    """
    Resolve per-gene mutation probabilities for selectors and tail with
    a minimal precedence consistent with Una-May's guidance:

      1) If both explicit overrides are provided, use them as-is.
      2) If only one override is provided, use it and set the other to the
         baseline 1 / (G + (L_used - K)).
      3) If none are provided, use the baseline for both segments.

    Returned values are clamped to [0, 1].

    Parameters
    ----------
    G : int
        Number of selector genes (head segment length).
    L_used : int
        Full phenotype length (prefix + tail).
    K : int
        REAL prefix length.
    sel_mutation : float or None
        Explicit per-gene probability for selector mutation.
    tail_mutation : float or None
        Explicit per-bit probability for tail bit-flip.

    Returns
    -------
    (p_sel, p_tail) : tuple[float, float]
        Final per-gene probabilities in [0, 1].
    """
    tail_len = max(0, L_used - K)
    total_len = max(1, G + tail_len)
    baseline = 1.0 / float(total_len)

    if sel_mutation is not None and tail_mutation is not None:
        ps, pt = float(sel_mutation), float(tail_mutation)
    elif sel_mutation is not None:
        ps, pt = float(sel_mutation), baseline
    elif tail_mutation is not None:
        ps, pt = baseline, float(tail_mutation)
    else:
        ps, pt = baseline, baseline

    # Clamp to [0, 1]
    ps = min(1.0, max(0.0, ps))
    pt = min(1.0, max(0.0, pt))
    return ps, pt


def _sigma_from_pmin(tau: float, p_min: float) -> float:
    """
    Compute σ such that  P(|Δ| ≥ τ) = p_min  under Δ ~ N(0, σ²).
    Uses the exact relation:  Φ(τ/σ) = 1 - p_min/2  ⇒  σ = τ / Φ^{-1}(1 - p_min/2).

    Edge cases are handled defensively:
      - If p_min <= 0, returns a very small σ (practically no movement).
      - If p_min >= 1, returns a very large σ (very frequent large moves).
      - Guards against z ≈ 0.

    Parameters
    ----------
    tau : float
        Threshold for a “significant” step size (in absolute value).
    p_min : float
        Target probability P(|Δ| ≥ τ) in (0, 1).

    Returns
    -------
    float
        σ > 0
    """
    # Clamp p_min to a safe open interval
    p = float(min(max(p_min, 1e-12), 1.0 - 1e-12))
    z = float(norm.ppf(1.0 - p / 2.0))
    # Guard z -> 0
    if abs(z) < 1e-12:
        # If p very close to 1, z→0+, σ should blow up; cap to a large value
        return float(1e6)
    sigma = float(tau / z)
    # Minimal positive σ to avoid degenerate draws
    return max(sigma, 1e-12)

def make_tau_policy(percent: float, min_tau: int = 1) -> Callable[[int], int]:
    """
    Returns a tau_policy based on a percentage of the integer domain [0, U].
        τ = max(1, round(tau_percent * U))

    Ej: tau_percent=0.05 -> 5% of U.
    """
    percent = float(percent)
    min_tau = int(min_tau)
    def _policy(U: int) -> int:
        if U <= 0:
            return min_tau
        return max(min_tau, int(round(percent * float(U))))
    return _policy

def _default_tau_policy(U: int) -> int:
    """
    Default τ policy for integer selectors with range [0, U].
    - For small ranges, ensure τ>=1 (i.e., seek ±1 as 'significant').
    - For larger ranges, scale smoothly with U.

    This version uses a simple, smooth rule:
        τ = max(1, round(0.01 * U))
    which aligns with the “~1% significant change” heuristic.

    Parameters
    ----------
    U : int
        Upper bound of the integer domain [0, U].

    Returns
    -------
    int
        τ >= 1
    """
    if U <= 0:
        return 1
    return max(1, int(round(0.01 * float(U))))


def _sample_integer_delta(
    U: int,
    p_min: float,
    tau_policy: Callable[[int], int],
    rng: np.random.Generator,
    force_nonzero: bool = True,
    max_resamples: int = 8,
) -> int:
    """
    Draw an integer step Δ for a bounded integer gene x ∈ [0, U].

    Steps
    -----
    1) Compute τ = tau_policy(U)  (e.g., ≈1% of U, but ≥1).
    2) Compute σ from P(|Δ|≥τ) = p_min via σ = τ / Φ^{-1}(1 - p_min/2).
    3) Draw Δ_cont ~ N(0, σ²), round to nearest int ⇒ Δ = round(Δ_cont).
    4) Optionally avoid Δ=0 by resampling a few times and, if still 0, forcing ±1.

    Parameters
    ----------
    U : int
        Upper bound of the gene domain [0, U].
    p_min : float
        Target probability of a 'significant' change: P(|Δ| ≥ τ).
    tau_policy : Callable[[int], int]
        Function mapping U → τ (in integer units).
    rng : np.random.Generator
        Random number generator.
    force_nonzero : bool, optional
        If True, tries to avoid Δ=0 (resampling/biasing). Default True.
    max_resamples : int, optional
        Max resamples to avoid Δ=0 before forcing ±1. Default 8.

    Returns
    -------
    int
        Integer step Δ (unclipped).
    """
    U = int(max(0, U))
    tau = int(max(1, int(tau_policy(U))))
    sigma = _sigma_from_pmin(float(tau), float(p_min))

    # Draw and round
    delta = int(round(rng.normal(loc=0.0, scale=sigma)))

    if force_nonzero and delta == 0:
        # Resample a few times to dodge zero steps
        for _ in range(int(max_resamples)):
            delta = int(round(rng.normal(loc=0.0, scale=sigma)))
            if delta != 0:
                break
        if delta == 0:
            # Fall back to ±1, biased by which side has “room” if desired
            # Here: unbiased ±1
            delta = 1 if rng.random() < 0.5 else -1

    return delta


def _mutate_selectors_gaussian(
    ind,
    G: int,
    pool_sizes: List[int],
    sel_mutation_prob: float,
    p_min: float,
    tau_policy: Callable[[int], int] = _default_tau_policy,
    rng: Optional[np.random.Generator] = None,
) -> None:
    """
    Mutate the first G selector genes (indices into per-good pools) using
    Gaussian step sizes calibrated by P(|Δ|≥τ)=p_min on a per-gene range.

    Each selector g lives in [0, U_g] with U_g = pool_sizes[g]-1.
    If pool_sizes[g] ≤ 1, the gene is skipped (no admissible moves).

    Parameters
    ----------
    ind : list[int] or creator.Individual
        Individual whose selector segment will be mutated in place.
    G : int
        Number of selector genes at the head of the chromosome.
    pool_sizes : list[int]
        Per-good pool sizes. Domain is [0, pool_sizes[g]-1].
    sel_mutation_prob : float
        Per-gene probability of attempting a selector mutation.
    p_min : float
        Target probability of a 'significant' change (see _sample_integer_delta).
    tau_policy : Callable[[int], int], optional
        Policy mapping U → τ. Default is ~1% of U (but ≥1).
    rng : np.random.Generator, optional
        RNG. If None, uses a default_rng().
    """
    if rng is None:
        rng = np.random.default_rng()

    for g in range(int(G)):
        if g >= len(ind):
            break
        U_g = int(max(0, pool_sizes[g] - 1))
        if U_g <= 0:
            continue  # nothing to mutate
        if rng.uniform() < float(sel_mutation_prob):
            x = int(ind[g])
            delta = _sample_integer_delta(U_g, p_min, tau_policy, rng, force_nonzero=True)
            x_new = int(x + delta)
            # Clip to domain [0, U_g]
            x_new = 0 if x_new < 0 else (U_g if x_new > U_g else x_new)
            ind[g] = x_new


def _mutate_tail_bitflip(
    ind,
    G: int,
    tail_mutation_prob: float,
    fix_last_gene: bool,
    rng: Optional[np.random.Generator] = None,
) -> None:
    """
    Bit-flip mutation on the binary tail (positions G..end-1).
    Respects 'fix_last_gene' by skipping the very last locus when True.

    Parameters
    ----------
    ind : list[int] or creator.Individual
        Individual mutated in place.
    G : int
        Offset where the tail starts.
    tail_mutation_prob : float
        Per-bit probability of flipping.
    fix_last_gene : bool
        If True and len(ind)>0, the last gene is kept at 1.
    """
    if rng is None:
        rng = np.random.default_rng()

    L = len(ind)
    if L <= G:
        return
    last = L - 1 if fix_last_gene else L
    for j in range(G, last):
        if rng.uniform() < float(tail_mutation_prob):
            ind[j] = 1 - int(ind[j])
    if fix_last_gene and L > 0:
        ind[-1] = 1


def mutate_macro_micro_scipy(
    ind,
    G: int,
    pool_sizes: List[int],
    sel_mutation_prob: float,
    tail_mutation_prob: float,
    p_min: float,
    tau_policy: Callable[[int], int] = _default_tau_policy,
    fix_last_gene: bool = True,
    rng: Optional[np.random.Generator] = None,
):
    """
    Unified mutation operator (DEAP-compatible) for macro_micro using SciPy.

    Applies:
      1) Selector (integer) mutation with Gaussian steps calibrated by
         P(|Δ| ≥ τ) = p_min, where τ = tau_policy(U_g), U_g = pool_sizes[g]-1.
      2) Binary tail bit-flip with probability 'tail_mutation_prob'.
      3) Enforces last-gene invariant if requested.

    Parameters
    ----------
    ind : list[int] or creator.Individual
        Individual to mutate in place.
    G : int
        Number of selector genes.
    pool_sizes : list[int]
        Per-good pool sizes (domain [0, pool_sizes[g]-1]).
    sel_mutation_prob : float
        Per-gene probability of attempting a selector mutation.
    tail_mutation_prob : float
        Per-bit probability of flipping in the tail.
    p_min : float
        Target probability of 'significant' move size in selectors.
    tau_policy : Callable[[int], int], optional
        Policy U → τ. Default ≈1% of range (but ≥1).
    fix_last_gene : bool, optional
        If True, forces last gene to 1.
    rng : np.random.Generator, optional
        RNG instance.

    Returns
    -------
    tuple
        (ind,) as required by DEAP.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Mutate selectors (bounded integers)
    _mutate_selectors_gaussian(
        ind=ind,
        G=int(G),
        pool_sizes=pool_sizes,
        sel_mutation_prob=float(sel_mutation_prob),
        p_min=float(p_min),
        tau_policy=tau_policy,
        rng=rng,
    )

    # Mutate tail (binary)
    _mutate_tail_bitflip(
        ind=ind,
        G=int(G),
        tail_mutation_prob=float(tail_mutation_prob),
        fix_last_gene=bool(fix_last_gene),
        rng=rng,
    )

    # Invalidate DEAP fitness/phenotype caches if present
    if hasattr(ind, "fitness"):
        try:
            del ind.fitness.values
        except Exception:
            pass
    if hasattr(ind, "_phenotype"):
        try:
            delattr(ind, "_phenotype")
        except Exception:
            pass

    return (ind,)


# =============================================================================
# Evaluation wrapper (DEAP-compatible) with phenotype caching for macro_micro
# =============================================================================
def _make_macro_micro_evaluator(
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
    Builds a DEAP-compatible evaluate(individual) function that:
      1) Constructs and caches the phenotype genome on the individual as
         'individual._phenotype' (to avoid repeated recomputation).
      2) Calls Economy(...) and returns a single-objective fitness tuple (utility,).

    The phenotype expands the selector part into REAL prefix values using the
    equivalence-class pools and then appends the binary tail. The last-gene
    invariant can be enforced by operators and decorators upstream.
    """
    def evaluate(individual):
        ph = getattr(individual, "_phenotype", None)
        if ph is None:
            ph = _phenotype_from_individual(
                individual, pools, index_sets, alphabet, L_used, K, fix_last_gene
            )
            setattr(individual, "_phenotype", ph)

        u = Economy(
            production_graph=production_graph,
            pmatrix=pmatrix,
            agents_information=agents_information,
            genome=ph.tolist(),
        ).get_reports().get("utility", 0.0)

        return (float(u),)
    return evaluate


# =============================================================================
# Custom population metrics computed on cached phenotypes
# =============================================================================
def _extra_population_metrics_phenotype(pop: List[creator.Individual]) -> Dict[str, Any]:
    """
    Computes phenotype-level diagnostics over the population using cached genomes:
      - 'pheno_diversity': mean pairwise Hamming distance in [0, 1].
      - 'pheno_uniq'     : count of unique phenotype genomes.

    Individuals lacking a cached phenotype are ignored. If fewer than two
    phenotypes are available, diversity is zero and uniqueness is set accordingly.
    """
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

    uniq = len({tuple(p.tolist()) for p in phenos})

    acc = 0.0
    pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            acc += float(np.mean(phenos[i] != phenos[j]))
            pairs += 1
    div = acc / pairs if pairs else 0.0
    return {"pheno_diversity": float(div), "pheno_uniq": int(uniq)}


# =============================================================================
# Toolbox builder with probabilistic macro/micro crossover
# =============================================================================
def build_toolbox_macro_micro(
    G: int,
    K: int,
    L_used: int,
    pool_sizes: List[int],
    fix_last_gene: bool,
    # crossover (micro)
    lambda_in: float = 0.25,
    lambda_out: float = 0.50,
    # NEW: probabilities to apply each crossover stage
    p_macro: float = 1.00,
    p_micro: float = 1.00,
    # mutation
    sel_mutation_prob: float = 0.25,
    tail_mutation_prob: float = 0.05,
    p_min: float = 0.50,
    tau_policy: Callable[[int], int] = _default_tau_policy,
    # evaluator (DEAP-compatible): evaluate(ind) -> (fitness,)
    evaluator_fn: Callable = None,
    # New: random number generators
    rng_py: Optional[random.Random] = None,
    rng_np: Optional[np.random.Generator] = None,
):
    """
    Registers in a DEAP Toolbox all building blocks for the macro→micro GA:

    - Individual/population initializers (selectors + binary tail).
    - Crossover pipeline 'mate': optional MACRO at boundary G (prob p_macro),
      followed by optional MICRO segmented n-point crossover (prob p_micro).
    - Mutation: SciPy-calibrated Gaussian steps on selectors + bit-flip on tail.
    - Evaluate: user-injected callable (mandatory).
    - Decorator: enforces the 'last gene == 1' invariant after mate/mutate.

    Parameters
    ----------
    G, K, L_used : int
        Internal genome structure: G selector loci, REAL prefix of size K,
        and total phenotype length L_used.
    pool_sizes : list[int]
        Per-good equivalence-class pool sizes (domain for each selector).
    fix_last_gene : bool
        Enforces the last locus of the genome to 1 after operators.
    lambda_in, lambda_out : float
        Densities of crossover points for the MICRO operator in selector and tail segments.
    p_macro, p_micro : float
        Probabilities in [0,1] to apply MACRO and MICRO crossover stages, respectively.
        If a stage is skipped, offspring are just clones from parents up to that point.
    sel_mutation_prob, tail_mutation_prob : float
        Per-gene mutation probabilities for selectors and tail bits.
    p_min : float
        Target probability for a “significant” selector jump: P(|Δ| ≥ τ).
    tau_policy : Callable[[int], int]
        Maps selector domain U to the significance threshold τ.
    evaluator_fn : Callable
        Function evaluate(ind) -> (fitness,). Mandatory.

    Returns
    -------
    toolbox : deap.base.Toolbox
        Toolbox with 'individual', 'population', 'mate', 'mutate', 'evaluate' registered.
    """

    if rng_py is None:
        rng_py = random.Random()
    if rng_np is None:
        rng_np = np.random.default_rng()

    _ensure_deap_types()

    toolbox = base.Toolbox()

    # --- Individuals and population ---
    tail_len = max(0, L_used - K)
    toolbox.register(
        "individual",
        _init_joint_individual,
        icls=creator.Individual,
        G=int(G),
        tail_len=int(tail_len),
        pool_sizes=pool_sizes,
        fix_last_gene=bool(fix_last_gene),
        rng_py=rng_py,
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # --- Atomic crossover operators (registered but used internally) ---
    toolbox.register(
        "mate_macro",
        _mate_macro_boundary,
        G=int(G),
        fix_last_gene=bool(fix_last_gene),
    )
    toolbox.register(
        "mate_micro",
        _mate_micro_segmented,
        G=int(G),
        lambda_in=float(lambda_in),
        lambda_out=float(lambda_out),
        fix_last_gene=bool(fix_last_gene),
        rng=rng_py,
    )

    # --- Probabilistic crossover pipeline ---
    p_macro = float(max(0.0, min(1.0, p_macro)))
    p_micro = float(max(0.0, min(1.0, p_micro)))

    def mate_macro_micro(ind1, ind2):
        """
        Produces two offspring by cloning the parents and then:
          - With prob p_macro: applies MACRO crossover at boundary G.
          - With prob p_micro: applies MICRO segmented n-point crossover.
        Order is MACRO → MICRO. Stages can be independently skipped by probability.
        """
        c1, c2 = deap_clone(ind1), deap_clone(ind2)

        # Stage 1: MACRO at boundary G
        if rng_py.random() < p_macro:
            c1, c2 = toolbox.mate_macro(c1, c2)

        # Stage 2: MICRO within segments
        if rng_py.random() < p_micro:
            c1, c2 = toolbox.mate_micro(c1, c2)

        return c1, c2

    toolbox.register("mate", mate_macro_micro)

    # --- Mutation (SciPy-based for selectors) ---
    toolbox.register(
        "mutate",
        mutate_macro_micro_scipy,
        G=int(G),
        pool_sizes=pool_sizes,
        sel_mutation_prob=float(sel_mutation_prob),
        tail_mutation_prob=float(tail_mutation_prob),
        p_min=float(p_min),
        tau_policy=tau_policy,
        fix_last_gene=bool(fix_last_gene),
        rng=rng_np,
    )

    # --- Evaluator (mandatory) ---
    if evaluator_fn is None:
        raise ValueError("evaluator_fn must be provided: evaluate(ind) -> (fitness,)")
    toolbox.register("evaluate", evaluator_fn)

    # --- Defensive invariant: enforce last gene and invalidate phenotype cache ---
    _decorate_enforce_last_gene(
        toolbox, L_used=L_used, K=K, fix_last_gene=fix_last_gene,
        pools=[], index_sets=[], alphabet=[]
    )

    return toolbox


# =============================================================================
# Optional: statistics + logbook factory (useful before the loop)
# =============================================================================
def make_mstats_and_logbook(G: int):
    """
    Constructs a MultiStatistics object (fitness and tail density) and a Logbook
    configured with headers consistent with JOINT, ready to be used in the loop.

    Returns
    -------
    mstats : deap.tools.MultiStatistics
        Tracks min/avg/med/max of fitness and min/avg/max of tail density.
    logbook : deap.tools.Logbook
        Pre-configured logbook with chapters 'fitness' and 'tail'.
    """
    stats_fit = tools.Statistics(key=lambda ind: ind.fitness.values[0])
    stats_fit.register("min", np.min)
    stats_fit.register("avg", np.mean)
    stats_fit.register("med", np.median)
    stats_fit.register("max", np.max)

    stats_tail = tools.Statistics(key=lambda ind: (sum(ind[G:]) / max(1, len(ind) - G)))
    stats_tail.register("min", np.min)
    stats_tail.register("avg", np.mean)
    stats_tail.register("max", np.max)

    mstats = tools.MultiStatistics(fitness=stats_fit, tail=stats_tail)
    logbook = tools.Logbook()
    logbook.header = "gen", "evals", "fitness", "tail", "pheno_diversity", "pheno_uniq"
    logbook.chapters["fitness"].header = "min", "avg", "med", "max"
    logbook.chapters["tail"].header = "min", "avg", "max"
    return mstats, logbook



def _decorate_enforce_last_gene_ops(toolbox: base.Toolbox, fix_last_gene: bool, op_names: Sequence[str]):
    """
    Apply the 'last gene == 1' invariant (and phenotype cache invalidation)
    to every operator name in 'op_names' that exists in the toolbox.
    """
    if not fix_last_gene:
        return

    def enforce_and_invalidate(func):
        def wrapper(*args, **kwargs):
            offspring = func(*args, **kwargs)
            # 'offspring' may be a tuple/list of individuals or a single pair
            # Normalize to iterable of individuals.
            seq = offspring
            if not isinstance(seq, (list, tuple)):
                seq = [offspring]
            for child in seq:
                # Some DEAP ops return tuples (c1, c2); dive one level if needed.
                if isinstance(child, (list, tuple)) and child and hasattr(child[0], "__iter__"):
                    for g in child:
                        if len(g) > 0:
                            g[-1] = 1
                        if hasattr(g, "_phenotype"):
                            delattr(g, "_phenotype")
                else:
                    if len(child) > 0:
                        child[-1] = 1
                    if hasattr(child, "_phenotype"):
                        delattr(child, "_phenotype")
            return offspring
        wrapper.__name__ = getattr(func, "__name__", "enforced_" + str(func))
        wrapper.__doc__ = getattr(func, "__doc__", None)
        return wrapper

    for name in op_names:
        if hasattr(toolbox, name):
            toolbox.decorate(name, enforce_and_invalidate)


# =============================================================================
# Public API: High-level runner for the Macro→Micro GA
# =============================================================================
def run_ga_macro_micro(
    # --- Evaluation model inputs (domain-specific, passed to Economy) ---
    production_graph,
    pmatrix,
    agents_information,
    # --- Prefix detection / admissible alphabet discovery ---
    mode: str = "graph",
    per_good_cap: Optional[int] = None,
    max_index_probe: int = 3,
    # --- Evolutionary budget / loop controls ---
    generations: int = 50,
    popsize: int = 50,
    parents: int = 20,
    elite_fraction: float = 0.25,
    # --- Parent selection policy ---
    tourn_size: int = 3,
    parent_selection: str = "tournament", # <- "tournament" | "random"
    mating_selection: str = "pool",         # <- "pool" | "pairwise_tournament"
    # --- Variation operators: crossover (macro + micro) and mutation ---
    lambda_in: float = 0.25,
    lambda_out: float = 0.50,
    p_macro: float = 1.00,
    p_micro: float = 1.00,
    sel_mutation: Optional[float] = None,
    tail_mutation: Optional[float] = None,
    p_min: float = 0.30,
    tau_policy: Callable[[int], int] = _default_tau_policy,
    tau_percent: Optional[float] = None,  # ← tau_policy = make_tau_policy_from_percent
    fix_last_gene: bool = True,
    # --- Reproducibility, logging, and early stop ---
    seed: Optional[int] = 44,
    verbosity: int = 1,
    log_every: int = 1,
    evals_cap: Optional[int] = None,
    time_limit_sec: Optional[float] = None,
) -> dict:
    """
    Execute the Macro→Micro Genetic Algorithm (GA) that composes:
      (i) a **macro** one-point crossover at the selector|tail boundary (index G),
     (ii) a **micro** segmented n-point crossover applied *within* selectors and tail,
    followed by a split-domain mutation rule (Gaussian integer steps for selectors,
    bit-flip for the binary tail). The genotype is
        [ selectors (G genes) | binary tail ]
    while the evaluated phenotype expands the selectors into the REAL prefix
    according to per-good equivalence-class pools inferred from the Planner.

    The runner includes:
      - **Planner-grounded prefix detection** and **alphabet probing** to constrain
        admissible integer values.
      - **Equivalence-class pool construction** per primary good (weak compositions),
        optionally capped for tractability via uniform sampling.
      - **DEAP** toolbox assembly with **probabilistic crossover stages** `p_macro`
        and `p_micro` (either stage may be skipped stochastically).
      - A **cache-aware evaluator** that stores the phenotype on each individual
        (`_phenotype`) to enable phenotype-level diagnostics without recomputation.
      - A tournament-based **parent selection policy**.
      - A conservative **elitism + tournament/random non-elite parent sampling** strategy that
        avoids double-counting elites in the parent pool (exploration-friendly).
      - Detailed **run-time statistics** mirroring the JOINT implementation
        (fitness, tail density) plus phenotype-level diagnostics
        (mean pairwise Hamming diversity and unique phenotype count).
      - **Early-stop watchdogs** on total evaluations and wall-clock time.

    Parameters
    ----------
    production_graph : Any
        Production DAG or equivalent structure accepted by `Economy(...)`.
    pmatrix : np.ndarray
        Price tensor/matrix used by the evaluation model.
    agents_information : dict
        Agent configuration consumed by `Economy(...)`.
    mode : str, optional (default="graph")
        Mode for `detect_prefix_layout_and_sizes(...)` to infer the REAL prefix.
    per_good_cap : int or None, optional
        Upper bound on the number of equivalence classes per good. When None,
        all weak compositions are enumerated (may be large).
    max_index_probe : int, optional (default=3)
        Number of transaction-index probes for alphabet inference.
    generations : int, optional (default=50)
        Number of evolutionary generations (not counting generation 0 evaluation).
    popsize : int, optional (default=50)
        Number of individuals in the population.
    parents : int, optional (default=20)
        Target size of the parent pool per generation (elites + random non-elites).
    elite_fraction : float, optional (default=0.25)
        Fraction of `parents` preserved as elites. Internally capped to keep at
        least one slot for offspring and at least one non-elite parent when feasible.
    tourn_size : int, optional (default=3)
        Tournament size used when `parent_selection="tournament"`. Larger values
        increase selection pressure among non-elites.
    parent_selection : {"tournament", "random"}, optional (default="tournament")
        Policy for selecting non-elite parents. "tournament" uses DEAP's tournament
        selector over non-elites; "random" reuses the existing uniform sampling
        without replacement.
    mating_selection : {"pool","pairwise_tournament"}
        Policy for selecting parents for crossover. "pool" uses the parent pool, 
        "pairwise_tournament" uses a tournament over all the individuals.
    lambda_in : float, optional (default=0.25)
        Density for the number of micro crossover points in the selectors segment.
    lambda_out : float, optional (default=0.50)
        Density for the number of micro crossover points in the tail segment.
    p_macro : float, optional (default=1.00)
        Probability of applying the MACRO crossover stage at boundary G.
    p_micro : float, optional (default=1.00)
        Probability of applying the MICRO segmented crossover stage.
    sel_mutation : float, optional (default=None)
        Per-gene mutation probability for selector indices (0..G-1). If None, the probability
        is set to 1/G+(L_used-K), where K is the REAL prefix length.
    tail_mutation : float, optional (default=None)
        Per-bit mutation probability for the binary tail (G..end-1). If None, the probability
        is set to 1/G+(L_used-K), where K is the REAL prefix length.
    p_min : float, optional (default=0.30)
        Calibration target for selector step sizes: for a selector domain [0, U],
        σ is selected so that P(|Δ| ≥ τ) = p_min with τ = tau_policy(U).
    tau_policy : Callable[[int], int], optional
        Maps domain width U to the “significant step” threshold τ (≥ 1).
    tau_percent : float, optional (default=None)
        Determines a tau_policy based on a percentage of the domain width.
    fix_last_gene : bool, optional (default=True)
        If True, the last genome locus is enforced to 1 by decorators/operators.
    seed : int or None, optional (default=44)
        Seed for `random` and `numpy` RNGs. When None, global RNG state is used.
    verbosity : int, optional (default=1)
        0 = silent, 1 = per-generation summary (every `log_every`), >1 = verbose.
    log_every : int, optional (default=1)
        Print frequency for generation-level summaries.
    evals_cap : int or None, optional
        Early-stop cap on cumulative fitness evaluations (≥ 1).
    time_limit_sec : float or None, optional
        Early-stop cap on wall-clock seconds (≥ 0.0).

    Returns
    -------
    dict
        A structured report with core outputs and diagnostics:

        {
          "best_genome": list[int],                  # one representative best phenotype
          "best_utility": float,                     # best fitness value found
          "all_best_genomes": list[list[int]],       # all unique phenotypes at the tie
          "curves": {                                # learning curves
              "best":   list[float],
              "mean":   list[float],
              "median": list[float],
          },
          "meta": {                                  # run metadata and telemetry
              "labels": list[str],
              "sizes": list[int],
              "K": int,
              "L_used": int,
              "alphabet": list[int],
              "pool_sizes": list[int],
              "genome_internal": {"selectors": int, "tail_len": int},
              "generations": int,                    # executed generations
              "popsize": int,
              "parents": int,
              "elite_fraction": float,
              "tourn_size": int,
              "parent_selection": str,
              "lambda_in": float,
              "lambda_out": float,
              "p_macro": float,
              "p_micro": float,
              "sel_mutation": float,
              "tail_mutation": float,
              "p_min": float,
              "tau_policy": int,
              "tau_percent": float or None,
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
                  "reason": str or None,             # "evals" | "time" | None
                  "evals_total": int,
                  "time_total_sec": float
              }
          },
          "best_by_selectors": list[tuple[str, float]]  # diagnostics: best utility per selector tuple
        }

    Design Notes
    ------------
    * The macro operator exchanges the entire tail between parents at the boundary;
      the micro operator then recombines *within* segments. The MACRO→MICRO order
      yields a clean coarse→fine recombination pipeline. The probabilities `p_macro`
      and `p_micro` modulate exploration: setting them below 1.0 stochastically
      skips the corresponding stage (offspring remain closer to parents).
    * Selector mutation scales with domain width via `tau_policy` and `p_min`,
      ensuring that “significant” moves occur with a controlled probability
      even when per-good selector spaces differ in size.
    * The last-gene invariant is enforced at operator/decorator level to keep
      genotype statistics consistent with phenotype constraints.
    """

    # -------------------------------
    # 0) Early-stop watchdog helpers
    # -------------------------------
    t0 = time.time()

    def _budget_reason(evals_total: int) -> Optional[str]:
        """Return 'evals' or 'time' if a budget limit is reached, else None."""
        if evals_cap is not None and evals_total >= int(evals_cap):
            return "evals"
        if time_limit_sec is not None and (time.time() - t0) >= float(time_limit_sec):
            return "time"
        return None

    # ---------------------------
    # 1) Reproducibility seeding
    # ---------------------------
    random.seed(seed)
    np.random.seed(seed if seed is not None else None)
    _ensure_deap_types()

    rng_py = random.Random(seed)
    rng_np = np.random.default_rng(seed)

    # -------------------------------------------------------------
    # 2) Detect REAL prefix layout & infer admissible integer values
    # -------------------------------------------------------------
    labels, sizes, index_sets, tx_builder, L_min, _ = detect_prefix_layout_and_sizes(
        production_graph, mode=mode
    )
    K = int(sum(sizes))
    # Ensure at least one tail bit so the last-gene invariant has a locus
    L_used = max(int(L_min), K + 1)

    alphabet = probe_allowed_indices_via_tx_builder(
        L_used, tx_builder, max_index_probe=max_index_probe
    )

    # Build per-good equivalence-class pools (weak compositions)
    pools, pool_sizes = _build_class_pools(
        alphabet=alphabet,
        sizes=sizes,
        per_good_cap=per_good_cap,
        rng=rng_np,
    )
    G = len(pools)
    tail_len = max(0, L_used - K)

    # ----------------------------
    # 3) Resolve mutation probabilities before any logging that references them
    # ----------------------------
    p_sel, p_tail = _resolve_mutation_rates_simple(
        G=G,
        L_used=L_used,
        K=K,
        sel_mutation=sel_mutation,
        tail_mutation=tail_mutation,
    )

    # Detection summary
    if verbosity >= 1:
        print("=== Macro→Micro GA detection summary ===")
        print(f"Goods (Planner order)       : {labels}")
        print(f"k_g per good                : {sizes}  ->  K={K} | L_used={L_used}")
        print(f"Alphabet for prefix         : {alphabet}  (|A|={len(alphabet)})")
        prod_est = int(np.prod(pool_sizes)) if pools else 1
        print(f"Per-good pool sizes         : {pool_sizes}  (≈ product {prod_est})")
        print(f"Internal chromosome         : selectors={G}, tail={tail_len} (total={G + tail_len})")
        print(
            "Operators                   : "
            f"p_macro={p_macro}, p_micro={p_micro}, "
            f"λ_in={lambda_in}, λ_out={lambda_out}, tourn_size={tourn_size}, "
            f"sel_mut={p_sel}, tail_mut={p_tail} , mating={mating_selection}"
        )

    # ----------------------------
    # 4) Evaluator (cache-aware phenotype build)
    # ----------------------------
    evaluator_fn = _make_macro_micro_evaluator(
        production_graph=production_graph,
        pmatrix=pmatrix,
        agents_information=agents_information,
        pools=pools,
        index_sets=index_sets,
        alphabet=alphabet,
        L_used=L_used,
        K=K,
        fix_last_gene=fix_last_gene,
    )

    # ----------------------------
    # 5) Build the DEAP toolbox with resolved mutation probabilities
    # ----------------------------
    tau_policy_local = make_tau_policy(tau_percent) if tau_percent is not None else tau_policy

    toolbox = build_toolbox_macro_micro(
        G=G, K=K, L_used=L_used,
        pool_sizes=pool_sizes,
        fix_last_gene=fix_last_gene,
        lambda_in=lambda_in, lambda_out=lambda_out,
        p_macro=p_macro, p_micro=p_micro,
        sel_mutation_prob=p_sel,
        tail_mutation_prob=p_tail,
        p_min=p_min,
        tau_policy=tau_policy_local, 
        evaluator_fn=evaluator_fn,
        rng_py=rng_py,
        rng_np=rng_np,
    )

    # ----------------------------
    # 6) Initialize the population
    # ----------------------------
    pop: List[creator.Individual] = toolbox.population(n=int(popsize))

    # ----------------------------
    # 7) Statistics & logbook setup
    # ----------------------------
    mstats, logbook = make_mstats_and_logbook(G=G)

    best_curve: List[float] = []
    mean_curve: List[float] = []
    median_curve: List[float] = []
    best_by_selectors: dict[Tuple[int, ...], float] = {}
    evals_per_gen: List[int] = []
    budget_triggered = False

    # -------------------------------------
    # 8) Generation 0: evaluate & summarize
    # -------------------------------------
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    evals_per_gen.append(len(fitnesses))

    # Diagnostics: best utility per selector configuration
    for ind in pop:
        ksel = tuple(int(x) for x in ind[:G])
        val = float(ind.fitness.values[0])
        if val > best_by_selectors.get(ksel, -float("inf")):
            best_by_selectors[ksel] = val

    record = mstats.compile(pop)
    record.update(_extra_population_metrics_phenotype(pop))
    logbook.record(gen=0, evals=len(fitnesses), **record)

    vals0 = [ind.fitness.values[0] for ind in pop]
    b0, m0, med0 = float(np.max(vals0)), float(np.mean(vals0)), float(np.median(vals0))
    best_curve.append(b0)
    mean_curve.append(m0)
    median_curve.append(med0)

    if verbosity >= 1 and (log_every > 0):
        print(f"Gen 000: best={b0:.6f} | mean={m0:.6f} | median={med0:.6f} | "
              f"pheno_div={record.get('pheno_diversity', 0):.4f} | uniq={record.get('pheno_uniq', 0)}")

    if _budget_reason(sum(evals_per_gen)) is not None:
        budget_triggered = True

    # ================================================================
    # 9) Main Evolutionary Loop: Elitism + Parent/Mating Selection
    # ================================================================
    for gen in range(1, int(generations) + 1):
        if budget_triggered:
            break

        if str(mating_selection).lower() == "pairwise_tournament":
            # ---- NEW MODE: pairwise tournament mating (ignores 'parents') ----
            # 9.1) Elites are cloned and carried over (independent from parent pool size).
            elites = _select_elites_for_carry(
                pop=pop,
                elite_fraction=float(elite_fraction),
                min_elites=1,
                clone_fn=deap_clone,
            )
            elite_count = len(elites)

            # 9.2) Reproduction: each parent is chosen by an independent tournament.
            #      This matches the canonical GA approach: "draw mom by a tournament,
            #      draw dad by another tournament", then mate → mutate → evaluate.
            offspring: List[creator.Individual] = []
            need = len(pop) - elite_count

            while len(offspring) < need:
                # Single-winner tournaments (k=1) return a list with one selected ind.
                p1 = tools.selTournament(pop, k=1, tournsize=max(2, int(tourn_size)))[0]
                p2 = tools.selTournament(pop, k=1, tournsize=max(2, int(tourn_size)))[0]

                c1, c2 = deap_clone(p1), deap_clone(p2)
                c1, c2 = toolbox.mate(c1, c2)  # macro→micro; decorators enforce invariants

                # Invalidate any stale fitness/phenotype caches before mutation/eval.
                for c in (c1, c2):
                    if hasattr(c, "fitness"):
                        try:
                            del c.fitness.values
                        except Exception:
                            pass
                    if hasattr(c, "_phenotype"):
                        try:
                            delattr(c, "_phenotype")
                        except Exception:
                            pass

                offspring.extend([c1, c2])

            # Trim in case we overfilled by one pair.
            offspring = offspring[:need]

            # 9.3) Mutation + evaluation of invalid individuals.
            for mut in offspring:
                toolbox.mutate(mut)
                if hasattr(mut, "fitness"):
                    try:
                        del mut.fitness.values
                    except Exception:
                        pass
                if hasattr(mut, "_phenotype"):
                    try:
                        delattr(mut, "_phenotype")
                    except Exception:
                        pass

            invalid = [ind for ind in offspring if not ind.fitness.valid]
            fits = list(map(toolbox.evaluate, invalid))
            for ind, fit in zip(invalid, fits):
                ind.fitness.values = fit
            evals_per_gen.append(len(invalid))

            # 9.4) Replacement: elites + offspring.
            pop[:] = elites + offspring

        else:
            # ---- EXISTING MODE: "pool" (uses 'parents' + parent_selection policy) ----
            sel_policy = str(parent_selection).strip().lower()
            if sel_policy == "tournament":
                elites, cand_parents, non_elites_all = select_elite_and_tournament(
                    pop=pop,
                    parents=int(parents),
                    elite_fraction=float(elite_fraction),
                    tourn_size=int(tourn_size),
                    clone_fn=deap_clone,
                )
            else:
                elites, cand_parents, non_elites_all = select_elite_and_random(
                    pop=pop,
                    parents=int(parents),
                    elite_fraction=float(elite_fraction),
                    clone_fn=deap_clone,
                    rng=rng_py,
                )

            elite_count = len(elites)

            # Parent pool (avoid empty cases; allow exploration via non-elite mix).
            if cand_parents:
                parents_pool = elites + cand_parents
            else:
                parents_pool = elites[:] if elites else [deap_clone(pop[0])]

            # 9.2) Reproduction from the parent pool using the registered mate pipeline.
            offspring: List[creator.Individual] = []
            need = len(pop) - elite_count
            while len(offspring) < need:
                if len(parents_pool) >= 2:
                    p1, p2 = rng_py.sample(parents_pool, 2)
                else:
                    # Degenerate fallback: selfing
                    p1 = parents_pool[0]
                    p2 = deap_clone(parents_pool[0])

                c1, c2 = deap_clone(p1), deap_clone(p2)
                c1, c2 = toolbox.mate(c1, c2)

                for c in (c1, c2):
                    if hasattr(c, "fitness"):
                        try:
                            del c.fitness.values
                        except Exception:
                            pass
                    if hasattr(c, "_phenotype"):
                        try:
                            delattr(c, "_phenotype")
                        except Exception:
                            pass

                offspring.extend([c1, c2])

            offspring = offspring[:need]

            # 9.3) Mutation + evaluation
            for mut in offspring:
                toolbox.mutate(mut)
                if hasattr(mut, "fitness"):
                    try:
                        del mut.fitness.values
                    except Exception:
                        pass
                if hasattr(mut, "_phenotype"):
                    try:
                        delattr(mut, "_phenotype")
                    except Exception:
                        pass

            invalid = [ind for ind in offspring if not ind.fitness.valid]
            fits = list(map(toolbox.evaluate, invalid))
            for ind, fit in zip(invalid, fits):
                ind.fitness.values = fit
            evals_per_gen.append(len(invalid))

            # 9.4) Replacement
            pop[:] = elites + offspring

        # Diagnostics: refresh best-by-selectors
        for ind in pop:
            ksel = tuple(int(x) for x in ind[:G])
            val = float(ind.fitness.values[0])
            if val > best_by_selectors.get(ksel, -float("inf")):
                best_by_selectors[ksel] = val

        # 9.5) Logging & curves
        record = mstats.compile(pop)
        record.update(_extra_population_metrics_phenotype(pop))
        logbook.record(gen=gen, evals=len(invalid), **record)

        vals = [ind.fitness.values[0] for ind in pop]
        b, m, med = float(np.max(vals)), float(np.mean(vals)), float(np.median(vals))
        best_curve.append(b)
        mean_curve.append(m)
        median_curve.append(med)

        if verbosity >= 1 and (gen % max(1, log_every) == 0 or gen == int(generations)):
            print(f"Gen {gen:03d}: best={b:.6f} | mean={m:.6f} | median={med:.6f} | "
                  f"pheno_div={record.get('pheno_diversity', 0):.4f} | uniq={record.get('pheno_uniq', 0)}")

        reason = _budget_reason(sum(evals_per_gen))
        if reason is not None:
            budget_triggered = True
            break

    # ------------------------------------------------------------
    # 10) Aggregation: best phenotype(s) and structured report
    # ------------------------------------------------------------
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
            ph = _phenotype_from_individual(
                ind, pools, index_sets, alphabet, L_used, K, fix_last_gene
            )
        key = tuple(int(x) for x in ph.tolist())
        if key not in seen:
            seen.add(key)
            all_best_genomes.append(list(key))

    if all_best_genomes:
        best_genome_list = list(all_best_genomes[0])
    else:
        # Fallback: compute phenotype from the fittest individual
        best_ind = tools.selBest(pop, 1)[0]
        best_ph = getattr(best_ind, "_phenotype", None)
        if best_ph is None:
            best_ph = _phenotype_from_individual(
                best_ind, pools, index_sets, alphabet, L_used, K, fix_last_gene
            )
        best_genome_list = list(best_ph.tolist())
        all_best_genomes = [best_genome_list]

    best_by_selectors_list = sorted(
        [("("+",".join(map(str, k))+")", float(v)) for k, v in best_by_selectors.items()],
        key=lambda x: x[1],
        reverse=True,
    )

    evals_cum = np.cumsum(np.array(evals_per_gen, dtype=int)).tolist()
    runtime = time.time() - t0

    return {
        "best_genome": best_genome_list,
        "best_utility": best_val,
        "all_best_genomes": all_best_genomes,
        "curves": {"best": best_curve, "mean": mean_curve, "median": median_curve},
        "meta": {
            "labels": labels,
            "sizes": list(map(int, sizes)),
            "K": K,
            "L_used": L_used,
            "alphabet": alphabet,
            "pool_sizes": pool_sizes,
            "genome_internal": {"selectors": G, "tail_len": tail_len},
            "generations": int(gen if not budget_triggered else gen),
            "popsize": int(popsize),
            "parents": int(parents),
            "elite_fraction": float(elite_fraction),
            "tourn_size": int(tourn_size),
            "parent_selection": parent_selection,
            "mating_selection": mating_selection,
            "lambda_in": float(lambda_in),
            "lambda_out": float(lambda_out),
            "p_macro": float(p_macro),
            "p_micro": float(p_micro),
            "sel_mutation": float(p_sel),
            "tail_mutation": float(p_tail),
            "p_min": float(p_min),
            "tau_policy": ("percent" if tau_percent is not None else "default"),
            "tau_percent": (float(tau_percent) if tau_percent is not None else None),
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
                "reason": _budget_reason(sum(evals_per_gen)),
                "evals_total": int(sum(evals_per_gen)),
                "time_total_sec": float(runtime),
            },
        },
        "best_by_selectors": best_by_selectors_list,
    }

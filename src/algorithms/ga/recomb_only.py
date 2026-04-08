# algorithms/ga/recomb_only.py
# =============================================================================
# Purpose
# =============================================================================
# This module implements a Genetic Algorithm (GA) variant that uses a single
# crossover operator: gene-wise uniform recombination across the *entire*
# mixed-domain chromosome:
#
#     genotype = [ selectors (integer indices into per-good class pools) | binary tail ]
#
# For each gene i, the child inherits gene i from parent1 with probability
# `p_recomb` and from parent2 with probability (1 - p_recomb). No macro/micro
# staged crossover is used. Mutation is identical to the macro_micro GA:
#   - Selectors (bounded integers): Gaussian step sizes calibrated via
#     P(|Δ| ≥ τ) = p_min with τ = tau_policy(U_g) and U_g = pool_sizes[g] - 1
#     (SciPy-based).
#   - Tail (binary): standard bit-flip with probability `tail_mutation_prob`.
#
# Selection mirrors the macro_micro runner:
#   - Elitism + *uniform random sampling without replacement* among non-elites
#     to complete the parent pool. Elites may be used repeatedly as parents
#     when forming pairs during reproduction.
#
# Phenotype construction expands the selected per-good equivalence classes into
# the REAL prefix positions (as detected by the Planner) and appends the binary
# tail. A defensive decorator enforces the last-gene invariant (1) after mate
# and mutate, and invalidates phenotype caches.
#
# Public API:
#   run_ga_recomb(...)
#       -> Dict with 'best_genome', 'best_utility', 'all_best_genomes', 'curves',
#          'meta', and 'best_by_selectors'.
# =============================================================================

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import math
import random
import time

import numpy as np
from deap import base, creator, tools
from scipy.stats import norm

from src.algorithms.ga.common import (
    detect_prefix_layout_and_sizes,         # Planner-grounded prefix layout
    probe_allowed_indices_via_tx_builder,   # admissible integer alphabet
    _resolve_mutation_rates_simple
)
from src.algorithms.ga.common import deap_clone
from src.simulation.economy.economy import Economy


# =============================================================================
# Combinatorics: per-good equivalence classes (weak compositions)
# =============================================================================
def _num_equiv_classes(alpha_size: int, k: int) -> int:
    """
    Compute the number of weak compositions of k into 'alpha_size' parts,
    i.e., the binomial coefficient C(alpha_size + k - 1, k).

    Edge cases
    ----------
    - If alpha_size <= 0: returns 1 if k == 0 else 0.
    """
    if alpha_size <= 0:
        return 1 if k == 0 else 0
    import math as _math
    return _math.comb(alpha_size + k - 1, k)


def _iter_count_vectors(alpha_size: int, k: int):
    """
    Yield all non-negative integer vectors c of length 'alpha_size' such that
    sum(c) == k, via stars-and-bars enumeration.

    Yields
    ------
    np.ndarray
        Count vector of dtype=int and length 'alpha_size'.
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
    Useful when the full class space is very large and must be capped.
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

    Returns
    -------
    pools : list[list[np.ndarray]]
        Per-good list of class count vectors.
    pool_sizes : list[int]
        Sizes of each per-good pool (after capping if any).
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
# Genotype → Phenotype mapping
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
    Build the full phenotype using the Planner-detected REAL prefix layout.

    Steps
    -----
    1) Split genotype into:
         - selectors: one integer per good, indexing the good's class pool
         - tail: remaining binary segment appended after the REAL prefix
    2) For each good:
         selector -> counts -> canonical values over 'alphabet'
         then write those values into positions 'index_sets[g]'.
    3) Append tail starting at position K.
    4) Optionally enforce last gene == 1.

    Returns
    -------
    np.ndarray
        Phenotype genome (length L_used), dtype=int.

    Raises
    ------
    ValueError
        If an expanded counts vector length does not match its index set.
    """
    G = len(pools)
    selectors = np.asarray(indiv[:G], dtype=int)
    tail = np.asarray(indiv[G:], dtype=int)

    # REAL prefix from per-good selections
    prefix = np.zeros(K, dtype=int)
    for ig in range(G):
        pool_g = pools[ig]
        if not pool_g:
            continue
        sel = int(np.clip(selectors[ig], 0, len(pool_g) - 1))
        counts_g = pool_g[sel]
        vals_g = _canonical_values_from_counts(alphabet, counts_g)
        pos_g = index_sets[ig]
        if len(vals_g) != len(pos_g):
            raise ValueError(f"Counts length mismatch for good {ig}: {len(vals_g)} != {len(pos_g)}")
        for v, j in zip(vals_g, pos_g):
            prefix[j] = int(v)

    # Assemble phenotype genome
    genome = np.zeros(L_used, dtype=int)
    if K > 0:
        genome[:K] = prefix

    tail_len = max(0, L_used - K)
    if tail_len > 0:
        genome[K : K + tail_len] = tail[:tail_len]

    if fix_last_gene and L_used > 0:
        genome[-1] = 1

    return genome


# =============================================================================
# DEAP: type guards, initialization, decorators
# =============================================================================
def _ensure_deap_types() -> None:
    """
    Create DEAP creator classes once per process:

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
    Initialize an individual:
      - First G integer selector genes (indices into per-good class pools).
      - Followed by a binary tail of length 'tail_len'.
      - The last tail bit is fixed to 1 when 'fix_last_gene' is True.
    """
    if rng_py is None:
        rng_py = random.Random()

    sel = [rng_py.randint(0, max(0, pool_sizes[g] - 1)) for g in range(G)]
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
      - Ensures any cached phenotype attached to the individual is invalidated.
    """
    if not fix_last_gene:
        return

    def enforce_and_invalidate(func):
        def wrapper(*args, **kwargs):
            offspring = func(*args, **kwargs)
            for child in offspring:
                if len(child) > 0:
                    child[-1] = 1
                if hasattr(child, "_phenotype"):
                    delattr(child, "_phenotype")
            return offspring

        wrapper.__name__ = getattr(func, "__name__", "enforced_" + str(func))
        wrapper.__doc__ = getattr(func, "__doc__", None)
        return wrapper

    toolbox.decorate("mate", enforce_and_invalidate)
    toolbox.decorate("mutate", enforce_and_invalidate)


def _safe_fitness_value(ind) -> float:
    """
    Return a comparable scalar fitness value. If invalid/missing, return -inf
    so the individual never gets selected as elite by accident.
    """
    try:
        if hasattr(ind, "fitness") and ind.fitness.valid:
            return float(ind.fitness.values[0])
    except Exception:
        pass
    return float("-inf")


# =============================================================================
# Selection helpers (elitism + random among non-elites)
# =============================================================================
def _resolve_elite_count(pop_size: int, parents: int, elite_fraction: float) -> int:
    """
    Compute an elite count consistent with space/budget constraints:
      - Base: floor(parents * elite_fraction)
      - At least 1 when feasible,
      - No more than pop_size - 1 (leave room for offspring),
      - No more than parents - 1 (leave room for non-elite parents).
    """
    if pop_size <= 1:
        return 0
    base = int(math.floor(parents * max(0.0, float(elite_fraction))))
    upper_cap = max(0, min(pop_size - 1, max(0, parents - 1)))
    return max(1, min(base, upper_cap)) if parents > 0 else 0


# =============================================================================
# Selection: Parents pool with elites + tournament over non-elites
# =============================================================================
def select_elite_and_tournament(
    pop: List[creator.Individual],
    parents: int,
    elite_fraction: float,
    tourn_size: int = 3,
    clone_fn: Optional[Callable] = None,
) -> Tuple[List[creator.Individual], List[creator.Individual], List[creator.Individual]]:
    """
    Splits the population into (elites, non_elites_tournament_parents, non_elites_all)
    using an 'elitism + tournament among non-elites' scheme.

    Elites are cloned and preserved unchanged. Additional parents are selected
    by DEAP's tournament selector restricted to the non-elite subset only, which
    prevents elite over-representation in the mating pool.

    Parameters
    ----------
    pop : list[creator.Individual]
        Current population (evaluated or partially evaluated).
    parents : int
        Target size for the parent pool (elites + additional non-elites).
    elite_fraction : float
        Fraction of 'parents' allocated to elitism; internally robustified by
        `_resolve_elite_count` to maintain feasibility.
    tourn_size : int, optional (default=3)
        Tournament size for `tools.selTournament`; must be ≥ 2 for meaningful pressure.
    clone_fn : callable or None, optional
        Cloner used to preserve elites, e.g., `deap_clone`. Defaults to `deap_clone`.

    Returns
    -------
    elites : list[creator.Individual]
        Cloned elite individuals to carry over unchanged.
    non_elites_tournament_parents : list[creator.Individual]
        Parents selected from the non-elite pool by tournament (no elites here).
    non_elites_all : list[creator.Individual]
        All non-elite individuals (original references), useful for diagnostics.

    Notes
    -----
    - Individuals with invalid fitness are treated as having -inf fitness by
      `_safe_fitness_value`, preventing accidental elite status.
    - The number of tournament parents is capped by the size of the non-elite pool.
    """
    if clone_fn is None:
        clone_fn = deap_clone

    n = len(pop)
    if n == 0 or parents <= 0:
        return [], [], pop[:]

    # Stable sort by scalar fitness (descending).
    ranked = sorted(pop, key=_safe_fitness_value, reverse=True)

    # Robust elite count consistent with population/parent budget.
    elite_count = _resolve_elite_count(pop_size=n, parents=parents, elite_fraction=elite_fraction)

    # Build elite/non-elite partitions.
    elite_refs = ranked[:elite_count]
    elite_ids = {id(ind) for ind in elite_refs}
    non_elites_all = [ind for ind in pop if id(ind) not in elite_ids]

    # Clone elites for safe carry-over.
    elites = [clone_fn(ind) for ind in elite_refs]

    # Tournament among non-elites only.
    parents_needed = max(0, min(int(parents) - elite_count, len(non_elites_all)))
    if parents_needed > 0 and len(non_elites_all) > 0:
        tsize = max(2, int(tourn_size))
        non_elites_tournament_parents = tools.selTournament(
            non_elites_all, k=parents_needed, tournsize=tsize
        )
    else:
        non_elites_tournament_parents = []

    return elites, non_elites_tournament_parents, non_elites_all


def select_elite_and_random(
    pop: List[creator.Individual],
    parents: int,
    elite_fraction: float,
    clone_fn: Optional[Callable] = None,
    rng: Optional[random.Random] = None,
) -> Tuple[List[creator.Individual], List[creator.Individual], List[creator.Individual]]:
    """
    Split the population into (elites, non_elites_random_parents, non_elites_all)
    using an 'elitism + random among non-elites' scheme.

    Notes
    -----
    - Individuals lacking a valid fitness are treated as -inf.
    - Sampling among non-elites is WITHOUT replacement to avoid over-counting.
    """
    if clone_fn is None:
        clone_fn = deap_clone
    if rng is None:
        rng = random

    n = len(pop)
    if n == 0 or parents <= 0:
        return [], [], pop[:]

    ranked = sorted(pop, key=_safe_fitness_value, reverse=True)
    elite_count = _resolve_elite_count(pop_size=n, parents=parents, elite_fraction=elite_fraction)

    elite_refs = ranked[:elite_count]
    elite_ids = {id(ind) for ind in elite_refs}
    non_elites_all = [ind for ind in pop if id(ind) not in elite_ids]

    elites = [clone_fn(ind) for ind in elite_refs]

    parents_needed = max(0, min(int(parents) - elite_count, len(non_elites_all)))
    if parents_needed > 0:
        rnd_parents = rng.sample(non_elites_all, k=parents_needed)
    else:
        rnd_parents = []

    return elites, rnd_parents, non_elites_all

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
# Recombination-only crossover: gene-wise uniform across full chromosome
# =============================================================================
def _mate_uniform_recomb(
    ind1,
    ind2,
    p_recomb: float,
    fix_last_gene: bool,
    rng: Optional[random.Random] = None,
):
    """
    Perform gene-wise uniform recombination across the entire chromosome
    (selectors + tail). For each locus i:
        child1[i] <- ind1[i] with prob p_recomb, else ind2[i]
        child2[i] <- ind2[i] with prob p_recomb, else ind1[i]

    This operator works *in place* on (ind1, ind2) and returns the pair.

    Parameters
    ----------
    ind1, ind2 : list-like
        Parents to be modified in place (DEAP convention).
    p_recomb : float
        Probability in [0,1] of inheriting the allele from parent1 at each locus.
    fix_last_gene : bool
        If True and genome length > 0, enforce last locus == 1 for both children.
    rng : random.Random or None
        RNG for Bernoulli draws. Defaults to Python's global RNG.

    Returns
    -------
    tuple
        (ind1, ind2) after in-place recombination.
    """
    if rng is None:
        rng = random

    L1, L2 = len(ind1), len(ind2)
    if L1 == 0 or L2 == 0:
        return ind1, ind2

    L = min(L1, L2)
    p = float(max(0.0, min(1.0, p_recomb)))

    # Swap-based implementation: with probability (1 - p), exchange alleles.
    q = 1.0 - p
    for i in range(L):
        if rng.random() < q:
            ind1[i], ind2[i] = ind2[i], ind1[i]

    # Defensive invariant and cache invalidation.
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
# Mutation (SciPy-calibrated for selectors) — identical to macro_micro
# =============================================================================
def _sigma_from_pmin(tau: float, p_min: float) -> float:
    """
    Compute σ such that  P(|Δ| ≥ τ) = p_min  under Δ ~ N(0, σ²).
    Uses the exact relation:  Φ(τ/σ) = 1 - p_min/2  ⇒  σ = τ / Φ^{-1}(1 - p_min/2).

    Edge cases are handled defensively.
    """
    p = float(min(max(p_min, 1e-12), 1.0 - 1e-12))
    z = float(norm.ppf(1.0 - p / 2.0))
    if abs(z) < 1e-12:
        return float(1e6)
    sigma = float(tau / z)
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
    Default τ policy for integer selectors with domain [0, U].
    τ = max(1, round(0.01 * U))  →  “~1% significant change”.
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
    1) τ = tau_policy(U)  (≥1)
    2) σ from P(|Δ|≥τ) = p_min
    3) Δ_cont ~ N(0, σ²), round → Δ
    4) Optionally avoid Δ=0 by resampling; if still 0, force ±1.
    """
    U = int(max(0, U))
    tau = int(max(1, int(tau_policy(U))))
    sigma = _sigma_from_pmin(float(tau), float(p_min))

    delta = int(round(rng.normal(loc=0.0, scale=sigma)))
    if force_nonzero and delta == 0:
        for _ in range(int(max_resamples)):
            delta = int(round(rng.normal(loc=0.0, scale=sigma)))
            if delta != 0:
                break
        if delta == 0:
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
    """
    if rng is None:
        rng = np.random.default_rng()

    for g in range(int(G)):
        if g >= len(ind):
            break
        U_g = int(max(0, pool_sizes[g] - 1))
        if U_g <= 0:
            continue
        if rng.uniform() < float(sel_mutation_prob):
            x = int(ind[g])
            delta = _sample_integer_delta(U_g, p_min, tau_policy, rng, force_nonzero=True)
            x_new = int(x + delta)
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
    Bit-flip mutation on the binary tail (positions G..end-1). Respects
    'fix_last_gene' by skipping the last locus when True.
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


def mutate_recomb_only_scipy(
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
    Unified mutation operator (DEAP-compatible) used by this recombination-only GA:

    1) Selector (integer) mutation with Gaussian steps calibrated by
       P(|Δ| ≥ τ) = p_min, τ = tau_policy(U_g), U_g = pool_sizes[g]-1.
    2) Binary tail bit-flip with probability 'tail_mutation_prob'.
    3) Enforces last-gene invariant if requested.
    """
    if rng is None:
        rng = np.random.default_rng()

    _mutate_selectors_gaussian(
        ind=ind,
        G=int(G),
        pool_sizes=pool_sizes,
        sel_mutation_prob=float(sel_mutation_prob),
        p_min=float(p_min),
        tau_policy=tau_policy,
        rng=rng,
    )
    _mutate_tail_bitflip(
        ind=ind,
        G=int(G),
        tail_mutation_prob=float(tail_mutation_prob),
        fix_last_gene=bool(fix_last_gene),
        rng=rng,
    )

    # Invalidate caches if present
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
# Evaluation wrapper (cache-aware phenotype build)
# =============================================================================
def _make_recomb_only_evaluator(
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
    Build a DEAP-compatible evaluate(individual) function that:
      1) Constructs and caches the phenotype genome on the individual as
         'individual._phenotype' (avoid repeated recomputation).
      2) Calls Economy(...) and returns a single-objective fitness tuple (utility,).
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
# Population metrics computed on cached phenotypes
# =============================================================================
def _extra_population_metrics_phenotype(pop: List[creator.Individual]) -> Dict[str, Any]:
    """
    Compute phenotype-level diagnostics over the population using cached genomes:
      - 'pheno_diversity': mean pairwise Hamming distance in [0, 1].
      - 'pheno_uniq'     : count of unique phenotype genomes.

    Individuals lacking a cached phenotype are ignored. If fewer than two
    phenotypes are available, diversity is zero and uniqueness reflects available items.
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
# Toolbox builder (recombination-only crossover)
# =============================================================================
def build_toolbox_recomb_only(
    G: int,
    K: int,
    L_used: int,
    pool_sizes: List[int],
    fix_last_gene: bool,
    # recombination
    p_recomb: float = 0.5,
    # mutation
    sel_mutation_prob: float = 0.25,
    tail_mutation_prob: float = 0.05,
    p_min: float = 0.50,
    tau_policy: Callable[[int], int] = _default_tau_policy,
    # evaluator (DEAP-compatible): evaluate(ind) -> (fitness,)
    evaluator_fn: Callable = None,
    # RNGs
    rng_py: Optional[random.Random] = None,
    rng_np: Optional[np.random.Generator] = None,
):
    """
    Register in a DEAP Toolbox all building blocks for the recombination-only GA:

    - Individual/population initializers (selectors + binary tail).
    - Crossover 'mate': gene-wise uniform recombination with probability `p_recomb`
      to inherit each locus from parent1 (else from parent2).
    - Mutation: SciPy-calibrated Gaussian steps on selectors + bit-flip on tail.
    - Evaluate: user-injected callable (mandatory).
    - Decorator: enforces the 'last gene == 1' invariant after mate/mutate.
    """
    if rng_py is None:
        rng_py = random.Random()
    if rng_np is None:
        rng_np = np.random.default_rng()

    _ensure_deap_types()
    toolbox = base.Toolbox()

    # Individuals and population
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

    # Recombination-only crossover
    p_recomb = float(max(0.0, min(1.0, p_recomb)))

    def mate_uniform(ind1, ind2):
        c1, c2 = deap_clone(ind1), deap_clone(ind2)
        c1, c2 = _mate_uniform_recomb(c1, c2, p_recomb=p_recomb, fix_last_gene=fix_last_gene, rng=rng_py)
        return c1, c2

    toolbox.register("mate", mate_uniform)

    # Mutation (SciPy-based for selectors)
    toolbox.register(
        "mutate",
        mutate_recomb_only_scipy,
        G=int(G),
        pool_sizes=pool_sizes,
        sel_mutation_prob=float(sel_mutation_prob),
        tail_mutation_prob=float(tail_mutation_prob),
        p_min=float(p_min),
        tau_policy=tau_policy,
        fix_last_gene=bool(fix_last_gene),
        rng=rng_np,
    )

    # Evaluator (mandatory)
    if evaluator_fn is None:
        raise ValueError("evaluator_fn must be provided: evaluate(ind) -> (fitness,)")
    toolbox.register("evaluate", evaluator_fn)

    # Defensive invariant decorator
    _decorate_enforce_last_gene(
        toolbox, L_used=L_used, K=K, fix_last_gene=fix_last_gene,
        pools=[], index_sets=[], alphabet=[]
    )

    return toolbox


# =============================================================================
# Statistics + logbook factory
# =============================================================================
def make_mstats_and_logbook(G: int):
    """
    Construct a MultiStatistics (fitness and tail density) and a Logbook with
    headers consistent with other GA variants in this suite.
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


# =============================================================================
# Public API: High-level runner (Recombination-only GA)
# =============================================================================
def run_ga_recomb_only(
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
    # --- NEW: parent selection policy (mirrors macro_micro) ---
    tourn_size: int = 3,                     # tournament size among non-elites
    parent_selection: str = "tournament",    # "tournament" or "random"
    mating_selection: str = "pool",          # "pool" or "pairwise_tournament"
    # --- Variation operators: recombination-only crossover + mutation ---
    p_recomb: float = 0.50,          # gene-wise prob of inheriting from parent1
    sel_mutation: Optional[float] = None,
    tail_mutation: Optional[float] = None,
    p_min: float = 0.30,             # calibration for selector step sizes
    tau_policy: Callable[[int], int] = _default_tau_policy,
    tau_percent: Optional[float] = None,  # ← tau_policy = make_tau_policy_from_percent
    fix_last_gene: bool = True,
    # --- Reproducibility, logging, and early stop ---
    seed: Optional[int] = 44,
    verbosity: int = 1,
    log_every: int = 1,
    evals_cap: Optional[int] = None,
    time_limit_sec: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Executes a recombination-only Genetic Algorithm (GA) over mixed-domain chromosomes
    (selectors + binary tail) with optional tournament-based parent selection among
    non-elites. The procedure expands selector genes into a Planner-grounded REAL
    prefix to build phenotypes, evaluates fitness via `Economy`, and tracks
    phenotype-level diagnostics throughout the run.

    Overview
    --------
    Chromosome layout:
        genotype = [ selectors (G integer genes) | binary tail ]
    Phenotype construction:
        selectors index per-good equivalence-class pools (weak compositions) and are
        expanded into the REAL prefix positions detected from the production graph;
        the binary tail is appended after the prefix. The last locus can be enforced
        to 1 to respect downstream invariants.

    Variation:
        - Recombination: gene-wise uniform across the entire chromosome. For each
        locus, the allele is inherited from parent1 with probability `p_recomb`,
        otherwise from parent2.
        - Mutation: split by domain. Selector genes apply Gaussian step sizes on
        bounded integers calibrated via P(|Δ| ≥ τ) = p_min with τ = tau_policy(U_g),
        U_g = pool_sizes[g] - 1 (steps are clipped to [0, U_g]); tail genes use
        standard bit-flip with probability `tail_mutation`.

    Selection and replacement:
        - Elitism preserves a fraction of the parent budget, robustly capped to
        keep room for both offspring and non-elite parents.
        - Non-elite parents are obtained either by tournament among non-elites
        (`parent_selection="tournament"`, size `tourn_size`) or by uniform random
        sampling without replacement (`parent_selection="random"`). You can also use
        a pure tournament crossover policy.
        - Elites are cloned and never participate in the non-elite parent selector,
        avoiding double counting in the mating pool.

    Detections and constraints:
        - REAL prefix layout and sizes are detected from the production graph.
        - The admissible integer alphabet for prefix indices is inferred via the
        transaction builder.
        - Per-good equivalence-class pools (weak compositions) are enumerated or
        uniformly down-sampled via `per_good_cap` for tractability.

    Parameters
    ----------
    production_graph : Any
        Production DAG or equivalent structure accepted by `Economy`.
    pmatrix : np.ndarray
        Price tensor/matrix consumed by `Economy`.
    agents_information : Dict[str, Any]
        Agent configuration consumed by `Economy`.
    mode : str, default "graph"
        Mode for prefix detection (passed to `detect_prefix_layout_and_sizes`).
    per_good_cap : Optional[int], default None
        Maximum number of equivalence classes (weak compositions) per good; when set
        and the exact space exceeds this cap, classes are uniformly sampled.
    max_index_probe : int, default 3
        Number of probes used to infer admissible transaction indices for the prefix.
    generations : int, default 50
        Number of evolutionary generations (excluding generation 0 evaluation).
    popsize : int, default 50
        Population size.
    parents : int, default 20
        Target parent pool size per generation (elites + non-elite parents).
    elite_fraction : float, default 0.25
        Fraction of `parents` allocated to elitism; internally robustified to keep
        at least one offspring and at least one non-elite parent when feasible.
    tourn_size : int, default 3
        Tournament size among non-elites when `parent_selection="tournament"`.
    parent_selection : {"tournament", "random"}, default "tournament"
        Policy for selecting non-elite parents.
    mating_selection : {"pool","pairwise_tournament"}
        Policy for selecting parents for crossover. "pool" uses the parent pool, 
        "pairwise_tournament" uses a tournament over all the individuals.
    p_recomb : float, default 0.50
        Gene-wise probability of inheriting an allele from parent1 during uniform
        recombination; inheritance from parent2 occurs with probability (1 - p_recomb).
    sel_mutation : float, optional (default=None)
        Per-gene mutation probability for selector indices (0..G-1). If None, the probability
        is set to 1/G+(L_used-K), where K is the REAL prefix length.
    tail_mutation : float, optional (default=None)
        Per-bit mutation probability for the binary tail (G..end-1). If None, the probability
        is set to 1/G+(L_used-K), where K is the REAL prefix length.
    p_min : float, default 0.30
        Calibration target for selector step sizes: P(|Δ| ≥ τ) = p_min with
        τ = tau_policy(U_g), U_g = pool_sizes[g] - 1.
    tau_policy : Callable[[int], int], default _default_tau_policy
        Maps selector domain width U to the significance threshold τ (≥ 1).
    tau_percent : float, optional (default=None)
        Determines a tau_policy based on a percentage of the domain width.
    fix_last_gene : bool, default True
        Enforces last locus == 1 after operators (decorator-level invariant).
    seed : Optional[int], default 44
        Seed for `random` and `numpy` RNGs (reproducibility).
    verbosity : int, default 1
        0 = silent; 1 = per-generation summary (every `log_every`); >1 = verbose.
    log_every : int, default 1
        Print frequency (in generations) when `verbosity >= 1`.
    evals_cap : Optional[int], default None
        Early-stop cap on total fitness evaluations; when reached, the run terminates
        after completing the current generation.
    time_limit_sec : Optional[float], default None
        Early-stop cap on wall-clock time in seconds.

    Returns
    -------
    Dict[str, Any]
        Structured report with solutions, learning curves, metadata, and diagnostics:
        {
        "best_genome": List[int],                 # representative best phenotype
        "best_utility": float,                    # best fitness value
        "all_best_genomes": List[List[int]],      # unique best phenotypes at tie
        "curves": {
            "best": List[float],
            "mean": List[float],
            "median": List[float],
        },
        "meta": {
            "labels": List[str],
            "sizes": List[int],
            "K": int,
            "L_used": int,
            "alphabet": List[int],
            "pool_sizes": List[int],
            "genome_internal": {"selectors": int, "tail_len": int},
            "generations": int,
            "popsize": int,
            "parents": int,
            "elite_fraction": float,
            "tourn_size": int,
            "parent_selection": str,
            "p_recomb": float,
            "sel_mutation": float,
            "tail_mutation": float,
            "p_min": float,
            "tau_policy": int,
            "tau_percent": float or None,
            "fix_last_gene": bool,
            "seed": Optional[int],
            "tie_tolerance": float,
            "evals_per_gen": List[int],
            "evals_cum": List[int],
            "runtime_sec": float,
            "budget": {
                "evals_cap": Optional[int],
                "time_limit_sec": Optional[float],
                "triggered": bool,
                "reason": Optional[str],          # "evals" | "time" | None
                "evals_total": int,
                "time_total_sec": float
            }
        },
        "best_by_selectors": List[Tuple[str, float]]
        }
    """
    t0 = time.time()

    def _budget_reason(evals_total: int) -> Optional[str]:
        if evals_cap is not None and evals_total >= int(evals_cap):
            return "evals"
        if time_limit_sec is not None and (time.time() - t0) >= float(time_limit_sec):
            return "time"
        return None

    # Seeding
    random.seed(seed)
    np.random.seed(seed if seed is not None else None)
    _ensure_deap_types()

    rng_py = random.Random(seed)
    rng_np = np.random.default_rng(seed)

    # Detect REAL prefix & infer admissible integer values
    labels, sizes, index_sets, tx_builder, L_min, _ = detect_prefix_layout_and_sizes(
        production_graph, mode=mode
    )
    K = int(sum(sizes))
    # Ensure at least one tail bit so the last-gene invariant has a locus
    L_used = max(int(L_min), K + 1)

    alphabet = probe_allowed_indices_via_tx_builder(
        L_used, tx_builder, max_index_probe=max_index_probe
    )

    # Build per-good equivalence-class pools
    pools, pool_sizes = _build_class_pools(
        alphabet=alphabet,
        sizes=sizes,
        per_good_cap=per_good_cap,
        rng=rng_np,
    )
    G = len(pools)
    tail_len = max(0, L_used - K)

    # --- Resolve per-locus mutation probabilities (align with macro_micro) ---
    p_sel, p_tail = _resolve_mutation_rates_simple(
        G=G,
        L_used=L_used,
        K=K,
        sel_mutation=sel_mutation,      # may be prob in [0,1] or expected count >= 1
        tail_mutation=tail_mutation,    # idem
    )


    if verbosity >= 1:
        print("=== Recombination-only GA detection summary ===")
        print(f"Goods (Planner order)       : {labels}")
        print(f"k_g per good                : {sizes}  ->  K={K} | L_used={L_used}")
        print(f"Alphabet for prefix         : {alphabet}  (|A|={len(alphabet)})")
        prod_est = int(np.prod(pool_sizes)) if pools else 1
        print(f"Per-good pool sizes         : {pool_sizes}  (≈ product {prod_est})")
        print(f"Internal chromosome         : selectors={G}, tail={tail_len} (total={G + tail_len})")
        print(f"Operators                   : p_recomb={p_recomb}, sel_mut={p_sel}, "
            f"tail_mut={p_tail}, p_min={p_min}, mating={mating_selection}")
        
    # Evaluator
    evaluator_fn = _make_recomb_only_evaluator(
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

    tau_policy_local = make_tau_policy(tau_percent) if tau_percent is not None else tau_policy

    # Toolbox
    toolbox = build_toolbox_recomb_only(
        G=G, K=K, L_used=L_used,
        pool_sizes=pool_sizes,
        fix_last_gene=fix_last_gene,
        p_recomb=p_recomb,
        sel_mutation_prob=p_sel,
        tail_mutation_prob=p_tail,
        p_min=p_min, tau_policy=tau_policy_local,
        evaluator_fn=evaluator_fn,
        rng_py=rng_py, rng_np=rng_np,
    )

    # Initialize population
    pop: List[creator.Individual] = toolbox.population(n=int(popsize))

    # Statistics & logbook
    mstats, logbook = make_mstats_and_logbook(G=G)

    best_curve: List[float] = []
    mean_curve: List[float] = []
    median_curve: List[float] = []
    best_by_selectors: Dict[Tuple[int, ...], float] = {}
    evals_per_gen: List[int] = []
    budget_triggered = False

    # Generation 0: evaluate & summarize
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    evals_per_gen.append(len(fitnesses))

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
    # Main Evolutionary Loop: Elitism + Parent Selection Policy
    # ================================================================
    gens_completed = 0  # Tracks how many generations actually completed

    for gen in range(1, int(generations) + 1):
        if budget_triggered:
            break
        gens_completed = gen

        if str(mating_selection).lower() == "pairwise_tournament":
            # ---- Pairwise tournament mating (ignores 'parents') ----
            # 1) Select elites to carry over; independent of the parent-pool size.
            elites = _select_elites_for_carry(
                pop=pop,
                elite_fraction=float(elite_fraction),
                min_elites=1,
                clone_fn=deap_clone,
            )
            elite_count = len(elites)

            # 2) Reproduction via independent tournaments for each parent; then mate.
            offspring: List[creator.Individual] = []
            need = len(pop) - elite_count

            while len(offspring) < need:
                # Single-winner tournaments (k=1) return a list with one selected individual.
                p1 = tools.selTournament(pop, k=1, tournsize=max(2, int(tourn_size)))[0]
                p2 = tools.selTournament(pop, k=1, tournsize=max(2, int(tourn_size)))[0]

                c1, c2 = deap_clone(p1), deap_clone(p2)
                # Uniform recombination across full chromosome; decorators enforce invariants.
                c1, c2 = toolbox.mate(c1, c2)

                # Invalidate any stale caches prior to mutation/evaluation.
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

            # Trim in case an extra pair overflowed the requirement.
            offspring = offspring[:need]

            # 3) Mutation and evaluation of invalid individuals.
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

            # 4) Replacement: elites + offspring.
            pop[:] = elites + offspring

            # 5) Diagnostics and logging (mirrors the "pool" branch for parity).
            for ind in pop:
                ksel = tuple(int(x) for x in ind[:G])
                val = float(ind.fitness.values[0])
                if val > best_by_selectors.get(ksel, -float("inf")):
                    best_by_selectors[ksel] = val

            record = mstats.compile(pop)
            record.update(_extra_population_metrics_phenotype(pop))
            logbook.record(gen=gen, evals=len(invalid), **record)

            vals = [ind.fitness.values[0] for ind in pop]
            b, m, med = float(np.max(vals)), float(np.mean(vals)), float(np.median(vals))
            best_curve.append(b)
            mean_curve.append(m)
            median_curve.append(med)

            if verbosity >= 1 and (gen % max(1, log_every) == 0 or gen == int(generations)):
                print(
                    f"Gen {gen:03d}: best={b:.6f} | mean={m:.6f} | median={med:.6f} | "
                    f"pheno_div={record.get('pheno_diversity', 0):.4f} | uniq={record.get('pheno_uniq', 0)}"
                )

            reason = _budget_reason(sum(evals_per_gen))
            if reason is not None:
                budget_triggered = True
                break

        else:
            # ---- Parent-pool mode ("pool"): uses 'parents' + parent_selection policy ----
            # 1) Build elites + non-elite parents according to the configured policy.
            sel_policy = str(parent_selection).strip().lower()
            if sel_policy == "tournament":
                elites, cand_parents, _non_elites_all = select_elite_and_tournament(
                    pop=pop,
                    parents=int(parents),
                    elite_fraction=float(elite_fraction),
                    tourn_size=int(tourn_size),
                    clone_fn=deap_clone,
                )
            else:
                elites, cand_parents, _non_elites_all = select_elite_and_random(
                    pop=pop,
                    parents=int(parents),
                    elite_fraction=float(elite_fraction),
                    clone_fn=deap_clone,
                    rng=rng_py,
                )

            elite_count = len(elites)

            # Defensive parent pool construction.
            parents_pool = elites + cand_parents if cand_parents else (
                elites[:] if elites else [deap_clone(pop[0])]
            )

            # 2) Reproduction via uniform recombination to fill the offspring set.
            offspring: List[creator.Individual] = []
            need = len(pop) - elite_count
            while len(offspring) < need:
                if len(parents_pool) >= 2:
                    p1, p2 = rng_py.sample(parents_pool, 2)
                else:
                    p1 = parents_pool[0]
                    p2 = deap_clone(parents_pool[0])

                c1, c2 = deap_clone(p1), deap_clone(p2)
                c1, c2 = toolbox.mate(c1, c2)  # Invariants enforced by decorators.

                # Explicit cache invalidation (defensive).
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

            # 3) Mutation and evaluation of invalid individuals.
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

            # 4) Replacement: elites + offspring.
            pop[:] = elites + offspring

            # 5) Diagnostics and logging.
            for ind in pop:
                ksel = tuple(int(x) for x in ind[:G])
                val = float(ind.fitness.values[0])
                if val > best_by_selectors.get(ksel, -float("inf")):
                    best_by_selectors[ksel] = val

            record = mstats.compile(pop)
            record.update(_extra_population_metrics_phenotype(pop))
            logbook.record(gen=gen, evals=len(invalid), **record)

            vals = [ind.fitness.values[0] for ind in pop]
            b, m, med = float(np.max(vals)), float(np.mean(vals)), float(np.median(vals))
            best_curve.append(b)
            mean_curve.append(m)
            median_curve.append(med)

            if verbosity >= 1 and (gen % max(1, log_every) == 0 or gen == int(generations)):
                print(
                    f"Gen {gen:03d}: best={b:.6f} | mean={m:.6f} | median={med:.6f} | "
                    f"pheno_div={record.get('pheno_diversity', 0):.4f} | uniq={record.get('pheno_uniq', 0)}"
                )

            reason = _budget_reason(sum(evals_per_gen))
            if reason is not None:
                budget_triggered = True
                break
    # ------------------------------------------------------------
    # Aggregation: best phenotype(s) and structured report
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
            "p_recomb": float(p_recomb),
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

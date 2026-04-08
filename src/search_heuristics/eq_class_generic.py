# algorithms/ga/eq_class_generic.py
# =============================================================================
# Equivalence-Class Genetic Algorithm (Selectors + Binary Tail)
# =============================================================================
# This module implements a mixed-type GA tailored to DAG-shaped production
# processes. The genotype is split into:
#
#   [ integer selectors for source goods | binary tail ]
#
# - For each input (a source node in the DAG), the number of directed paths
#   from that input to the final good defines the *cardinality* P_i of a
#   local encoding. The corresponding genotype gene is an integer k_i ∈ [0, P_i].
#   At evaluation time, k_i is deterministically expanded to a length-P_i
#   binary vector with exactly k_i ones (canonical "left-packed" encoding).
#
# - For non-input nodes that can reach the final good, the total number of
#   paths across them is summed. That sum, plus one extra bit, defines the
#   length of the binary tail. The last bit can be forcefully set to 1.
#
# The phenotype fed to `Economy` is the concatenation of:
#   - the expanded binary blocks for all inputs (in topological order),
#   - followed by the binary tail (in genotype order).
#
# The GA uses DEAP operators (two-point crossover and a mixed mutation) while
# maintaining the "last gene equals 1" invariant when requested.
# =============================================================================

from __future__ import annotations

import random
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
from deap import base, creator, tools

from src.simulation.economy.economy import Economy


# =============================================================================
# Graph utilities
# =============================================================================
def _ensure_dag(graph_like) -> nx.DiGraph:
    """
    Coerces the input into a NetworkX DiGraph and validates acyclicity.

    Parameters
    ----------
    graph_like : Any
        Either a networkx.DiGraph instance or an edge-like iterable suitable
        for constructing a DiGraph.

    Returns
    -------
    nx.DiGraph
        A copy (or newly constructed) directed acyclic graph.

    Raises
    ------
    ValueError
        If the resulting graph is not a DAG.
    """
    if isinstance(graph_like, nx.DiGraph):
        dag = graph_like.copy()
    else:
        dag = nx.DiGraph(graph_like)
    if not nx.is_directed_acyclic_graph(dag):
        raise ValueError("production_graph must be a directed acyclic graph (DAG).")
    return dag


def _detect_final_good(dag: nx.DiGraph, final_good: Optional[Any]) -> Any:
    """
    Determines the final good (sink) to be optimized towards.

    Strategy
    --------
    - If `final_good` is provided, it is validated and returned.
    - Otherwise, if there is a unique sink, it is used.
    - If there are multiple sinks, the one appearing last in a topological
      ordering is chosen (stable heuristic).

    Parameters
    ----------
    dag : nx.DiGraph
        Production DAG.
    final_good : Any or None
        Optional explicit node to treat as the final good.

    Returns
    -------
    Any
        The chosen final good node.

    Raises
    ------
    ValueError
        If `final_good` is not present in the DAG, or the DAG has no sinks.
    """
    if final_good is not None:
        if final_good not in dag:
            raise ValueError("Specified `final_good` is not a node in `production_graph`.")
        return final_good

    sinks = [n for n in dag.nodes if dag.out_degree(n) == 0]
    if len(sinks) == 0:
        raise ValueError("DAG has no sink nodes; cannot infer a final good.")
    if len(sinks) == 1:
        return sinks[0]

    topo = list(nx.topological_sort(dag))
    order = {n: i for i, n in enumerate(topo)}
    sinks.sort(key=lambda n: order[n], reverse=True)
    return sinks[0]


def _reachable_to_target(dag: nx.DiGraph, target: Any) -> set:
    """
    Computes the set of nodes that can reach `target`.

    Parameters
    ----------
    dag : nx.DiGraph
        Production DAG.
    target : Any
        Target node.

    Returns
    -------
    set
        Nodes with at least one path to `target`, including `target` itself.
    """
    rev = dag.reverse(copy=False)
    return nx.descendants(rev, target) | {target}


def _count_paths_to_target(dag: nx.DiGraph, target: Any) -> Dict[Any, int]:
    """
    Counts the number of directed paths from every node to `target`.

    Notes
    -----
    This is dynamic programming over a reverse topological order:
    - paths[target] = 1 (empty path to itself).
    - paths[u] = sum(paths[v] for v in succ(u)) when `u` reaches `target`.
    - paths[u] = 0 for nodes that cannot reach `target`.

    Parameters
    ----------
    dag : nx.DiGraph
        Production DAG.
    target : Any
        Target node.

    Returns
    -------
    Dict[Any, int]
        Map: node -> number of directed paths from node to `target`.
    """
    topo = list(nx.topological_sort(dag))
    reach = _reachable_to_target(dag, target)
    paths: Dict[Any, int] = {n: 0 for n in dag.nodes}

    for n in reversed(topo):
        if n not in reach:
            paths[n] = 0
        elif n == target:
            paths[n] = 1
        else:
            paths[n] = sum(paths[s] for s in dag.successors(n))
    return paths


def _topo_inputs_and_others(dag: nx.DiGraph, target: Any) -> Tuple[List[Any], List[Any]]:
    """
    Splits nodes into inputs (in-degree 0) and non-inputs, both filtered to those
    that can reach the target, and returns them in topological order.

    Parameters
    ----------
    dag : nx.DiGraph
        Production DAG.
    target : Any
        Target node.

    Returns
    -------
    (List[Any], List[Any])
        (inputs_in_topological_order, others_in_topological_order)
    """
    topo = list(nx.topological_sort(dag))
    inputs = [n for n in topo if dag.in_degree(n) == 0]
    others = [n for n in topo if n not in inputs]
    reach = _reachable_to_target(dag, target)
    inputs = [n for n in inputs if n in reach]
    others = [n for n in others if n in reach]
    return inputs, others


# =============================================================================
# Genotype → Phenotype mapping
# =============================================================================
def _expand_input_integers_to_bits(input_ints: List[int], input_path_counts: List[int]) -> List[int]:
    """
    Expands per-input integers k_i into binary blocks of size P_i with exactly
    k_i ones, using a deterministic canonical encoding.

    Encoding
    --------
    For an input with P_i paths and gene value k_i ∈ [0, P_i], the expansion is:
        [1, 1, ..., 1, 0, 0, ..., 0]  (k_i ones followed by P_i - k_i zeros)

    Parameters
    ----------
    input_ints : List[int]
        Integer genes for inputs, ordered topologically.
    input_path_counts : List[int]
        Path count P_i for each input, in the same order.

    Returns
    -------
    List[int]
        Concatenated binary vector for all input blocks.
    """
    out: List[int] = []
    for k, P in zip(input_ints, input_path_counts):
        P = int(max(0, P))
        k = int(max(0, min(P, k)))
        out.extend([1] * k)
        out.extend([0] * (P - k))
    return out


def _phenotype_from_genotype(
    individual: List[int],
    num_inputs: int,
    input_path_counts: List[int],
    fix_last_gene: bool,
) -> List[int]:
    """
    Builds the full phenotype (binary genome) from the mixed genotype.

    Steps
    -----
    1) Reads the integer prefix of length `num_inputs`.
    2) Expands it to the left-packed binary blocks using `input_path_counts`.
    3) Appends the raw binary tail.
    4) Re-enforces the last bit to 1 when `fix_last_gene=True`.

    Returns
    -------
    List[int]
        Full binary phenotype.
    """
    input_ints = [int(x) for x in individual[:num_inputs]]
    prefix_bits = _expand_input_integers_to_bits(input_ints, input_path_counts)
    tail_bits = [int(x) for x in individual[num_inputs:]]
    genome_bits = prefix_bits + tail_bits
    if fix_last_gene and genome_bits:
        genome_bits[-1] = 1
    return genome_bits


# =============================================================================
# DEAP integration: type guards, initialization, operators
# =============================================================================
def _ensure_deap_types() -> None:
    """
    Ensures DEAP `creator` classes exist for Fitness and Individual.
    This function is idempotent across repeated imports/uses.
    """
    try:
        _ = creator.FitnessMax
    except Exception:
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    try:
        _ = creator.Individual
    except Exception:
        creator.create("Individual", list, fitness=creator.FitnessMax)


def _init_individual(
    icls,
    num_inputs: int,
    input_path_counts: List[int],
    tail_len: int,
    fix_last_gene: bool,
):
    """
    Initializes a mixed-type individual:
      - Prefix: `num_inputs` integer genes, k_i ∈ [0, P_i].
      - Tail  : `tail_len` binary genes; the last one may be forced to 1.

    Parameters are pre-bound via `toolbox.register`.
    """
    # Prefix (integer selectors)
    prefix = [random.randint(0, max(0, int(P))) for P in input_path_counts[:num_inputs]]
    # Tail (binary)
    tail = [random.randint(0, 1) for _ in range(int(tail_len))]
    if fix_last_gene and tail:
        tail[-1] = 1
    return icls(prefix + tail)


def _mate_two_point(ind1, ind2):
    """
    Two-point crossover applied to the full mixed chromosome.
    Domain consistency is preserved because integer genes remain integers and
    binary genes remain binary by value exchange.
    """
    tools.cxTwoPoint(ind1, ind2)
    return ind1, ind2


def _mutate_mixed(
    ind,
    num_inputs: int,
    input_path_counts: List[int],
    selector_indpb: float,
    bit_indpb: float,
    fix_last_gene: bool,
):
    """
    Mixed mutation operator.

    - Integer prefix (0 .. num_inputs-1):
        With probability `selector_indpb` per gene, either:
          * ±1 bounded step (50%), or
          * uniform resample in [0, P_i] (50%).
        When P_i = 0, the gene is clamped to 0.

    - Binary tail (num_inputs .. end):
        With probability `bit_indpb` per bit, flip 0↔1, except the final locus
        when `fix_last_gene=True`.

    The last bit is re-enforced to 1 when `fix_last_gene=True`.
    """
    # Integer selectors
    for i in range(num_inputs):
        if random.random() < selector_indpb:
            P = int(max(0, input_path_counts[i]))
            if P <= 0:
                ind[i] = 0
            else:
                if random.random() < 0.5:
                    step = random.choice([-1, 1])
                    ind[i] = int(max(0, min(P, int(ind[i]) + step)))
                else:
                    ind[i] = random.randint(0, P)

    # Binary tail
    last_idx = len(ind) - 1
    for j in range(num_inputs, len(ind)):
        if fix_last_gene and j == last_idx:
            continue
        if random.random() < bit_indpb:
            ind[j] = 1 - int(ind[j])

    if fix_last_gene and len(ind) > 0:
        ind[-1] = 1
    return (ind,)


# =============================================================================
# Evaluator (DEAP-compatible)
# =============================================================================
def _make_deap_evaluator(
    production_graph,
    pmatrix,
    agents_information,
    num_inputs: int,
    input_path_counts: List[int],
    fix_last_gene: bool,
):
    """
    Builds a DEAP-compatible evaluator that maps the mixed genotype to the
    binary phenotype required by `Economy`.

    Mapping steps
    -------------
    1) Read the prefix of `num_inputs` integers (k_i).
    2) Expand each k_i to a length-P_i binary block with k_i ones.
    3) Concatenate all input blocks and append the raw binary tail.
    4) Enforce the final bit to 1 when `fix_last_gene=True`.
    5) Call `Economy(..., genome=<phenotype_bits>)` and return (utility,).
    """
    def evaluate(individual):
        genome_bits = _phenotype_from_genotype(
            individual, num_inputs=num_inputs, input_path_counts=input_path_counts, fix_last_gene=fix_last_gene
        )
        u = Economy(
            production_graph=production_graph,
            pmatrix=pmatrix,
            agents_information=agents_information,
            genome=genome_bits,
        ).get_reports().get("utility", 0.0)
        return (float(u),)
    return evaluate


# =============================================================================
# Public API
# =============================================================================
def run_eq_class_generic_ga(
    production_graph,
    pmatrix,
    agents_information,
    *,
    # Graph / mapping
    final_good: Optional[Any] = None,
    # Evolution
    generations: int = 20,
    popsize: int = 40,
    cxpb: float = 0.7,
    mutpb: float = 0.2,
    mutation_rate: float = 0.05,         # bit flip prob. for tail genes
    selector_mutation_rate: float = 0.25,  # per-gene prob. for integer selectors
    elitism: int = 1,
    fix_last_gene: bool = True,
    seed: Optional[int] = 42,
    verbosity: int = 1,
    log_every: int = 1,
    progress_cb: Callable[[int, float, float, float], None] | None = None,
    # Budgets
    evals_cap: Optional[int] = None,
    time_limit_sec: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Runs the equivalence-class GA producing a mixed genotype tailored to a
    production DAG and evaluating phenotypes through `Economy`.

    Genotype layout
    ---------------
    [ integer selectors for inputs | binary tail ]

    - The integer selector for input i is in [0, P_i], where P_i is the number
      of directed paths from that input to the final good.
    - The binary tail length equals the sum of path counts for all non-input
      nodes that can reach the final good, plus 1 extra bit.
    - The last bit can be enforced to 1 via `fix_last_gene`.

    Phenotype layout (fed to Economy)
    ---------------------------------
    Concatenation of:
      1) Expanded input blocks: for each input i, a length-P_i block with k_i
         ones in a canonical left-packed arrangement.
      2) The binary tail, in genotype order.
      3) Final bit re-enforcement to 1 if requested.

    Parameters
    ----------
    production_graph : Any
        DAG describing the production network. Either a `networkx.DiGraph` or a
        structure suitable to build one (e.g., edge list).
    pmatrix : Any
        Price or parameter matrix forwarded to `Economy`.
    agents_information : Any
        Agent configuration forwarded to `Economy`.
    final_good : Any, optional
        Explicit final good (sink). If None, a sink is inferred as documented.
    generations : int, default=20
        Number of evolutionary generations.
    popsize : int, default=40
        Population size.
    cxpb : float, default=0.7
        Crossover probability per mating pair.
    mutpb : float, default=0.2
        Mutation probability per individual.
    mutation_rate : float, default=0.05
        Per-bit flip probability in the binary tail (indpb).
    selector_mutation_rate : float, default=0.25
        Per-gene mutation probability for integer selectors (prefix).
    elitism : int, default=1
        Number of top individuals copied unchanged to the next generation.
    fix_last_gene : bool, default=True
        When True, the genotype's last locus and phenotype's last bit are set
        (or reset) to 1 after variation and before evaluation.
    seed : int or None, default=42
        Random seed for reproducibility.
    verbosity : int, default=1
        Logging level (>=1 prints progress).
    log_every : int, default=1
        Print frequency in generations.
    progress_cb : callable or None
        Optional callback receiving (gen, best, mean, median) each time stats
        are recorded.
    evals_cap : int or None
        Optional cap on total fitness evaluations (early stop).
    time_limit_sec : float or None
        Optional wall-clock time limit in seconds (early stop).

    Returns
    -------
    Dict[str, Any]
        {
          "best_genome": List[int],      # best *phenotype* found (expanded)
          "best_utility": float,         # corresponding utility
          "all_best_genomes": List[List[int]],  # all unique phenotypes at best utility
          "best_by_selectors": List[Tuple[str, float]],  # "(k1,k2,...)", best_utility
          "curves": { "best": [...], "mean": [...], "median": [...] },
          "meta": {
              "inputs_topo": List[Any],
              "others_topo": List[Any],
              "input_path_counts": List[int],
              "tail_len": int,
              "genotype_len": int,
              "generations": int,
              "popsize": int,
              "cxpb": float,
              "mutpb": float,
              "mutation_rate_bits": float,
              "selector_mutation_rate": float,
              "elitism": int,
              "fix_last_gene": bool,
              "seed": int or None,
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
    - This routine returns the *phenotype* of the best individual (expanded).
      It also aggregates all unique maximizing phenotypes found during the run.
    - Nodes that cannot reach the final good are ignored for path aggregation.
    """
    t0 = time.time()

    def _budget_reason(evals_total: int) -> Optional[str]:
        if evals_cap is not None and evals_total >= int(evals_cap):
            return "evals"
        if time_limit_sec is not None and (time.time() - t0) >= float(time_limit_sec):
            return "time"
        return None

    # RNG seeding
    random.seed(seed)
    np.random.seed(seed if seed is not None else None)

    # Graph analysis
    dag = _ensure_dag(production_graph)
    tgt = _detect_final_good(dag, final_good)
    paths = _count_paths_to_target(dag, tgt)
    inputs, others = _topo_inputs_and_others(dag, tgt)

    input_path_counts = [int(paths[n]) for n in inputs]
    # Tail length: sum of paths across relevant non-inputs + 1 extra bit
    tail_len = int(sum(int(paths[n]) for n in others) + 1)

    num_inputs = len(inputs)
    genotype_len = int(num_inputs + tail_len)

    if verbosity >= 1:
        print("=== eq_class_generic_ga | Mapping summary ===")
        print(f"Final good: {tgt!r}")
        print(f"Inputs (topological): {inputs}")
        print(f"P_i per input: {input_path_counts}  -> expanded_prefix_bits={sum(input_path_counts)}")
        print(f"Non-inputs considered: {others}")
        print(f"tail_len = sum_paths(non-inputs) + 1 = {tail_len}")
        print(f"genotype_len = {genotype_len} = inputs({num_inputs}) + tail({tail_len})")

    # DEAP toolbox
    _ensure_deap_types()
    toolbox = base.Toolbox()

    toolbox.register(
        "individual",
        _init_individual,
        creator.Individual,
        num_inputs=num_inputs,
        input_path_counts=input_path_counts,
        tail_len=tail_len,
        fix_last_gene=bool(fix_last_gene),
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", _mate_two_point)
    toolbox.register(
        "mutate",
        _mutate_mixed,
        num_inputs=num_inputs,
        input_path_counts=input_path_counts,
        selector_indpb=float(selector_mutation_rate),
        bit_indpb=float(mutation_rate),
        fix_last_gene=bool(fix_last_gene),
    )
    toolbox.register(
        "evaluate",
        _make_deap_evaluator(
            production_graph,
            pmatrix,
            agents_information,
            num_inputs=num_inputs,
            input_path_counts=input_path_counts,
            fix_last_gene=bool(fix_last_gene),
        ),
    )

    # Initialize population
    pop = toolbox.population(n=int(popsize))

    # --- Global best tracking over the entire run (expanded phenotypes) ---
    tie_tolerance = 1e-9
    best_val_global = -float("inf")
    best_phenos_set: set[Tuple[int, ...]] = set()

    def _update_global_best(individual) -> None:
        nonlocal best_val_global
        val = float(individual.fitness.values[0])
        ph = _phenotype_from_genotype(
            individual,
            num_inputs=num_inputs,
            input_path_counts=input_path_counts,
            fix_last_gene=bool(fix_last_gene),
        )
        key = tuple(int(x) for x in ph)
        if val > best_val_global + tie_tolerance:
            best_val_global = val
            best_phenos_set.clear()
            best_phenos_set.add(key)
        elif abs(val - best_val_global) <= tie_tolerance:
            best_phenos_set.add(key)

    # Evaluate generation 0
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
        _update_global_best(ind)

    best_hist: List[float] = []
    mean_hist: List[float] = []
    med_hist: List[float] = []
    evals_per_gen: List[int] = [len(fitnesses)]
    budget_triggered = False
    budget_reason = None

    # Track best-by-selectors (keys are selector tuples)
    best_by_selectors: Dict[Tuple[int, ...], float] = {}
    for ind in pop:
        sel_key = tuple(int(x) for x in ind[:num_inputs])
        val = float(ind.fitness.values[0])
        if val > best_by_selectors.get(sel_key, -float("inf")):
            best_by_selectors[sel_key] = val

    def _record(population, gen):
        vals = np.array([ind.fitness.values[0] for ind in population], dtype=float)
        b, m, md = float(vals.max()), float(vals.mean()), float(np.median(vals))
        best_hist.append(b)
        mean_hist.append(m)
        med_hist.append(md)
        if verbosity >= 1 and (gen % max(1, log_every) == 0 or gen == generations):
            print(f"Gen {gen:03d}: best={b:.6f} | mean={m:.6f} | median={md:.6f}")
        if progress_cb:
            progress_cb(gen, b, m, md)

    _record(pop, 0)

    # Evolution loop
    for gen in range(1, int(generations) + 1):
        if budget_triggered:
            break

        # Selection: tournament with cloning
        offspring = tools.selTournament(pop, len(pop), tournsize=3)
        offspring = list(map(toolbox.clone, offspring))

        # Crossover
        for c1, c2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cxpb:
                toolbox.mate(c1, c2)
                if hasattr(c1, "fitness"):
                    del c1.fitness.values
                if hasattr(c2, "fitness"):
                    del c2.fitness.values

        # Mutation (per individual)
        for child in offspring:
            if random.random() < mutpb:
                toolbox.mutate(child)
                if hasattr(child, "fitness"):
                    del child.fitness.values

        # Defensive re-enforcement of the last genotype bit, if requested
        if fix_last_gene:
            for child in offspring:
                if len(child) > 0:
                    child[-1] = 1

        # Evaluate invalid offspring
        invalid = [ind for ind in offspring if not ind.fitness.valid]
        fits = list(map(toolbox.evaluate, invalid))
        for ind, fit in zip(invalid, fits):
            ind.fitness.values = fit
            _update_global_best(ind)
        evals_per_gen.append(len(invalid))

        # Elitism + replacement
        elites = tools.selBest(pop, int(max(0, min(int(elitism), len(pop)))))
        pop[:] = elites + offspring[: max(0, len(pop) - len(elites))]

        # Update best-by-selectors with current population
        for ind in pop:
            sel_key = tuple(int(x) for x in ind[:num_inputs])
            val = float(ind.fitness.values[0])
            if val > best_by_selectors.get(sel_key, -float("inf")):
                best_by_selectors[sel_key] = val

        _record(pop, gen)

        budget_reason = _budget_reason(sum(evals_per_gen))
        if budget_reason:
            budget_triggered = True
            break

    # Results (global across the entire run)
    if best_phenos_set:
        all_best_genomes = [list(p) for p in sorted(best_phenos_set)]
        best_genome_list = list(all_best_genomes[0])
        best_val = float(best_val_global)
    else:
        # Fallback: expand current best if no global was recorded (should not happen)
        best_ind = tools.selBest(pop, 1)[0]
        best_genome_list = _phenotype_from_genotype(
            best_ind, num_inputs=num_inputs, input_path_counts=input_path_counts, fix_last_gene=bool(fix_last_gene)
        )
        all_best_genomes = [list(best_genome_list)]
        best_val = float(best_ind.fitness.values[0])

    # Format best_by_selectors: list of ("(k1,k2,...)", best_utility) sorted desc
    best_by_selectors_list = sorted(
        [("("+",".join(map(str, k))+")", float(v)) for k, v in best_by_selectors.items()],
        key=lambda x: x[1], reverse=True
    )

    runtime = time.time() - t0

    return {
        "best_genome": best_genome_list,  # phenotype (expanded)
        "best_utility": best_val,
        "all_best_genomes": all_best_genomes,  # all maximizing phenotypes (expanded)
        "best_by_selectors": best_by_selectors_list,
        "curves": {"best": best_hist, "mean": mean_hist, "median": med_hist},
        "meta": {
            "inputs_topo": inputs,
            "others_topo": others,
            "input_path_counts": input_path_counts,
            "tail_len": tail_len,
            "genotype_len": int(num_inputs + tail_len),
            "generations": gen if 'gen' in locals() else 0,
            "popsize": int(popsize),
            "cxpb": float(cxpb),
            "mutpb": float(mutpb),
            "mutation_rate_bits": float(mutation_rate),
            "selector_mutation_rate": float(selector_mutation_rate),
            "elitism": int(elitism),
            "fix_last_gene": bool(fix_last_gene),
            "seed": seed,
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

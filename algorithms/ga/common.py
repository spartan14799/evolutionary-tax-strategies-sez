# algorithms/ga/common.py
# =============================================================================
# Purpose
# =============================================================================
# This module centralizes utilities shared by multiple Genetic Algorithm (GA)
# implementations in the project. It avoids duplication and ensures that
# each GA can rely on a consistent, well-tested set of helpers.
#
# Responsibilities:
#   1) Project bootstrap: expose ROOT_DIR so other modules can find resources.
#   2) Graph helpers: transform inputs into a directed graph, derive primary
#      goods, identify the final good (sink), and compute path counts in DAGs.
#   3) Planner/Economy bridges: construct a transaction builder callable that
#      runs the real Planner, and normalize its outputs into a stable format.
#   4) Genome utilities for layout discovery:
#        - Calibrate minimal genome length L that the Planner accepts.
#        - Detect the *real* prefix layout and sizes per primary good by
#          probing the Planner’s transaction sequence.
#        - Probe the alphabet (valid integer indices) that the Planner accepts
#          in the prefix positions.
#
# These building blocks are independent of any particular GA strategy (flat,
# exhaustive-by-class, joint with selectors+tail). GA implementations can import
# and compose them to build robust search procedures over the same domain logic.
#
# The module is intentionally free of global imports to heavyweight domain
# classes; domain imports (Planner, ProductionGraph, etc.) are performed lazily
# inside functions to reduce import-time coupling and circular dependencies.
# =============================================================================

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np


# =============================================================================
# Project bootstrap
# =============================================================================
# ROOT_DIR points to the repository root (…/FTZ_MODEL_2.0/). This is useful
# for resolving resource files (e.g., YAML configurations) without relying on
# relative working directories.
ROOT_DIR = Path(__file__).resolve().parents[2]


# =============================================================================
# Graph utilities
# =============================================================================
def _as_digraph(graph) -> "Any":
    """
    Returns a networkx.DiGraph from either an existing DiGraph or an edge list.

    The function is defensive:
      - If 'graph' is already a DiGraph, it is returned as-is.
      - If 'graph' is an iterable of (u, v) pairs, it is wrapped into a DiGraph.
      - If networkx is not available or something unexpected happens, the given
        'graph' is returned unchanged; callers that require a DiGraph should
        import networkx and assert types themselves.

    Parameters
    ----------
    graph : networkx.DiGraph | Iterable[Tuple[Hashable, Hashable]]
        Either a directed graph instance or an edge list.

    Returns
    -------
    networkx.DiGraph | Any
        A DiGraph when possible, or the original object otherwise.
    """
    try:
        import networkx as nx
        if isinstance(graph, nx.DiGraph):
            return graph
        G = nx.DiGraph()
        G.add_edges_from(graph)
        return G
    except Exception:
        # Fallback: return the input unchanged to avoid hard dependency at import time.
        return graph


def derive_primary_goods(graph) -> Set[str]:
    """
    Computes the set of 'primary goods' as nodes with in-degree == 0.

    The function accepts either a networkx.DiGraph or an edge list. When
    networkx is not available, a simple in-degree map is computed manually.

    Returns
    -------
    Set[str]
        Node labels (as strings) with zero in-degree.
    """
    try:
        import networkx as nx
        if isinstance(graph, nx.DiGraph):
            return {str(n) for n, deg in graph.in_degree() if deg == 0}
    except Exception:
        pass

    indeg: Dict[str, int] = {}
    nodes: Set[str] = set()
    for u, v in graph:
        u, v = str(u), str(v)
        nodes.add(u)
        nodes.add(v)
        indeg[v] = indeg.get(v, 0) + 1
        indeg.setdefault(u, indeg.get(u, 0))
    return {n for n in nodes if indeg.get(n, 0) == 0}


def derive_final_good(graph, allow_many: bool = False) -> str:
    """
    Returns the 'final good' (a sink with out-degree == 0). By default,
    enforces uniqueness and raises if multiple sinks exist.

    Parameters
    ----------
    graph : networkx.DiGraph | Iterable[Tuple[str, str]]
    allow_many : bool
        If False (default), a single sink is required. If True, the first sink
        in iteration order is returned.

    Returns
    -------
    str
        The identifier of the final good (sink).

    Raises
    ------
    ValueError
        If there is no sink or multiple sinks when allow_many=False.
    """
    import networkx as nx  # explicit dependency here; this requires networkx.
    G = _as_digraph(graph)
    if not isinstance(G, nx.DiGraph):
        # In strict usage this should not happen; calling code can handle it.
        raise ValueError("A networkx.DiGraph is required to derive the final good.")
    sinks = [str(n) for n in G.nodes if G.out_degree(n) == 0]
    if not sinks:
        raise ValueError("No sinks found in the graph.")
    if not allow_many and len(sinks) != 1:
        raise ValueError(f"Expected a single sink, found {len(sinks)}: {sinks}")
    return sinks[0]


def count_paths_to_target_dag(graph, target: str) -> Dict[str, int]:
    """
    Counts the number of simple directed paths from each node to 'target'
    in a Directed Acyclic Graph (DAG). This is often used as a reference
    for how many times a primary good might be 'needed' on the way to the
    final product, independent of the Planner’s execution order.

    Notes
    -----
    - Requires networkx and that 'graph' is a DAG.
    - Returns 0 for nodes from which 'target' is unreachable.

    Returns
    -------
    Dict[str, int]
        Map node -> number of simple paths to 'target'.
    """
    import networkx as nx
    G = _as_digraph(graph)
    if not isinstance(G, nx.DiGraph):
        raise ValueError("A networkx.DiGraph is required for path counting.")
    if not nx.is_directed_acyclic_graph(G):
        raise ValueError("The graph must be a DAG.")

    paths = {str(n): 0 for n in G.nodes}
    target = str(target)
    paths[target] = 1

    # Reverse topological order ensures that when visiting 'u', all successors
    # have already accumulated their path counts.
    for u in reversed(list(nx.topological_sort(G))):
        if str(u) == target:
            continue
        total = 0
        for _, v in G.out_edges(u):
            total += paths[str(v)]
        paths[str(u)] = total
    return paths


# =============================================================================
# Planner/Economy bridges
# =============================================================================
def normalize_transactions(tx_out: Any) -> List[Tuple[str, str, str, str]]:
    """
    Normalizes the Planner output to a uniform list of 4-tuples:
        (buyer, seller, action, good)

    The function is robust to:
      - returns in the form (tx_list,) (i.e., a single-element tuple),
      - entries that are not lists/tuples or have unexpected shapes.

    Returns
    -------
    List[Tuple[str, str, str, str]]
        Normalized list of transactions. Values are strings.
    """
    # Flatten singletons like (tx_list,) into tx_list.
    while isinstance(tx_out, tuple) and len(tx_out) == 1:
        tx_out = tx_out[0]
    # Convert general tuples to list for iteration.
    if isinstance(tx_out, tuple):
        tx_out = list(tx_out)

    norm: List[Tuple[str, str, str, str]] = []
    if not isinstance(tx_out, list):
        return norm

    for item in tx_out:
        if not isinstance(item, (list, tuple)) or len(item) < 4:
            continue
        _, party, action, good = item
        action = str(action)
        good = str(good)
        buyer = seller = ""

        # The Planner’s schema indicates:
        #   - "Buy": party is (buyer, seller)
        #   - "Produce": party is (actor,), stored into 'buyer' for convenience
        if action == "Buy":
            if isinstance(party, (list, tuple)):
                if len(party) >= 1:
                    buyer = str(party[0])
                if len(party) >= 2:
                    seller = str(party[1])
        elif action == "Produce":
            if isinstance(party, (list, tuple)) and len(party) >= 1:
                buyer = str(party[0])  # use 'buyer' as 'actor' container

        norm.append((buyer, seller, action, good))
    return norm


def make_transactions_builder(production_graph) -> Callable[[np.ndarray], List[Tuple[str, str, str, str]]]:
    """
    Returns a callable 'builder(genome) -> normalized_transactions' that executes
    the real Planner for a given genome and converts its output to a stable form.

    The builder lazily constructs:
        ProductionGraph -> ProductionProcess -> Planner
    and reuses that Planner instance across calls for performance.

    Parameters
    ----------
    production_graph : networkx.DiGraph | Iterable[Tuple[str, str]]
        The production graph (as DiGraph or edge list).

    Returns
    -------
    Callable[[np.ndarray], List[Tuple[str, str, str, str]]]
        A function that, for any integer vector 'genome', returns the normalized
        transaction list produced by Planner.execute_plan(genome).
    """
    # Lazy domain imports to avoid heavy imports at module import-time.
    from classes.economy.production_process.production_graph import ProductionGraph
    from classes.economy.production_process.production_process import ProductionProcess
    from classes.planner.planner import Planner

    try:
        import networkx as nx
        G = production_graph if isinstance(production_graph, nx.DiGraph) else nx.DiGraph(production_graph)
    except Exception:
        # If networkx is not present, assume the domain classes accept raw edges.
        G = production_graph

    pgraph = ProductionGraph(G)
    pproc = ProductionProcess(pgraph)
    planner = Planner(pproc)

    def builder(genome: np.ndarray) -> List[Tuple[str, str, str, str]]:
        # Ensure plain Python ints for domain code.
        g = [int(x) for x in genome]
        return normalize_transactions(planner.execute_plan(g))

    return builder


# =============================================================================
# Minimal genome length calibration and prefix layout detection
# =============================================================================
def _parse_required_steps_from_error(msg: str) -> Optional[int]:
    """
    Extracts an integer requirement from error messages like:
      "... at least X steps ..."
    Returns None if no such pattern is found.
    """
    import re
    m = re.search(r"at\s+least\s+(\d+)", msg, flags=re.IGNORECASE)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            return None
    return None


def calibrate_min_len_via_builder(
    builder: Callable[[np.ndarray], Any],
    base_L: int = 4,
    cap: int = 4096,
) -> int:
    """
    Increases genome length L until the given 'builder' stops raising
    "at least X steps" style errors. This discovers a valid minimal length
    that the Planner accepts for the current graph/problem.

    Parameters
    ----------
    builder : Callable[[np.ndarray], Any]
        The transactions builder produced by make_transactions_builder.
    base_L : int
        Initial length to probe. A sensible default is (number_of_primary_goods + 1).
    cap : int
        Safety cap to avoid infinite loops on misconfigured environments.

    Returns
    -------
    int
        The minimal acceptable genome length L_min.

    Raises
    ------
    RuntimeError
        If no valid L is found up to 'cap'.
    """
    L = max(1, int(base_L))
    while L <= cap:
        probe = np.zeros(L, dtype=int)
        try:
            _ = builder(probe)
            return L
        except Exception as err:
            # Prefer pattern-guided jumps when the planner hints an explicit lower bound.
            req = _parse_required_steps_from_error(str(err))
            if req is not None:
                L = max(L + 1, req)
                continue
            # Fall back to conservative increments for common validation errors,
            # but surface unexpected exceptions to avoid masking real issues.
            if isinstance(err, ValueError):
                L = L + 1
                continue
            raise
    raise RuntimeError(f"Could not find a minimal genome length <= {cap}.")


def probe_allowed_indices_via_tx_builder(
    L_used: int,
    tx_builder: Callable[[np.ndarray], Any],
    max_index_probe: int = 16,
) -> List[int]:
    """
    Probes which integer values are accepted by the Planner when used as
    constant genomes of length L_used. For idx in [0 .. max_index_probe-1],
    it attempts to execute the Planner with a genome filled with 'idx'.
    Those indices that do not raise are considered allowed.

    This is a simple yet practical way to discover the effective alphabet
    for prefix genes (e.g., the set of valid action codes).

    Returns
    -------
    List[int]
        Sorted list of accepted integer indices. Defaults to [0, 1] if none
        can be inferred (conservative binary fallback).
    """
    allowed: List[int] = []
    for idx in range(max_index_probe):
        g = np.full(L_used, idx, dtype=int)
        try:
            _ = tx_builder(g)
            allowed.append(idx)
        except Exception:
            continue
    return sorted(set(allowed)) if allowed else [0, 1]


def detect_prefix_layout_and_sizes(
    production_graph,
    mode: str = "graph",
) -> Tuple[List[str], List[int], List[List[int]], Callable[[np.ndarray], Any], int, Dict[str, Any]]:
    """
    Detects the *real* prefix layout that the Planner enforces, along with the
    length contributions (k_g) for each primary good. The detection is based on
    a probe execution and inspection of the resulting transaction sequence,
    rather than solely on static graph properties.

    Returned tuple:
      - labels:     primary goods in order of first appearance in real
                    MKT-purchase transactions observed in the probe run.
      - sizes:      k_g per good (how many prefix positions the Planner uses
                    for that good), in the same order as 'labels'.
      - index_sets: for each good, the exact prefix indices [0..K-1] assigned
                    by the Planner (respecting the *observed* order).
      - tx_builder: the reusable transactions builder.
      - L_min:      minimal acceptable genome length (via calibration).
      - info_extra: diagnostics dictionary; includes (when mode='graph')
                    reference sizes computed from the DAG path counts.

    Parameters
    ----------
    production_graph : networkx.DiGraph | Iterable[Tuple[str, str]]
        Production graph.
    mode : {'graph', 'transactions'}
        If 'graph', diagnostic info may include graph-derived sizes as a
        reference. The actual layout always follows the Planner’s transactions.

    Returns
    -------
    (labels, sizes, index_sets, tx_builder, L_min, info_extra)
    """
    primary_all = sorted(derive_primary_goods(production_graph))
    tx_builder = make_transactions_builder(production_graph)

    # Calibrate a minimal acceptable length for the probe.
    base_L = max(2, len(primary_all) + 1)
    L_min = calibrate_min_len_via_builder(tx_builder, base_L=base_L)

    # Probe with a zero genome to extract the *real* purchase sequence from MKT
    # of primary goods. This reflects the Planner’s actual execution order.
    probe = np.zeros(L_min, dtype=int)
    seq_goods: List[str] = []
    try:
        tx = tx_builder(probe)
        for _, seller, action, good in tx:
            if action == "Buy" and seller == "MKT" and good in primary_all:
                seq_goods.append(good)
    except Exception:
        # If the Planner raises for the zero-probe (unlikely after calibration),
        # seq_goods remains empty and sizes will be zeros; callers can handle it.
        pass

    # Build the prefix layout: group positions by good, preserving order
    # of first appearance from the observed sequence.
    positions_by_good: Dict[str, List[int]] = {}
    for pos, g in enumerate(seq_goods):
        positions_by_good.setdefault(g, []).append(pos)

    labels = [g for g in dict.fromkeys(seq_goods).keys()]  # order of first appearance
    sizes_planner = [len(positions_by_good[g]) for g in labels]
    index_sets = [positions_by_good[g] for g in labels]
    K_planner = int(sum(sizes_planner))

    info_extra: Dict[str, Any] = {
        "K_planner": K_planner,
        "labels": labels,
        "sizes_planner": sizes_planner,
    }

    # Optional: provide graph-based reference sizes (DAG path counts to sink).
    if mode == "graph" and primary_all:
        try:
            final_good = derive_final_good(production_graph)
            paths = count_paths_to_target_dag(production_graph, final_good)
            sizes_graph_full = {g: int(max(0, paths.get(g, 0))) for g in primary_all}
            info_extra["sizes_graph_full"] = sizes_graph_full
            sizes_graph_in_labels = [sizes_graph_full.get(g, 0) for g in labels]
            info_extra["sizes_graph_in_labels"] = sizes_graph_in_labels
            # Note: if sum of graph-based sizes != K_planner, the Planner’s
            # observed layout prevails. This mismatch is informative only.
        except Exception:
            # Graph diagnostics are best-effort; proceed silently on failure.
            pass

    return labels, sizes_planner, index_sets, tx_builder, L_min, info_extra


# =============================================================================
# Optional convenience: tiny helpers often handy in notebooks / logs
# =============================================================================
def format_genome(g: Sequence[int]) -> str:
    """
    Formats an integer genome as a compact string: [0,1,1,0,...].
    """
    return "[" + ",".join(str(int(x)) for x in g) + "]"


def load_graph_yaml(path: str | Path) -> List[Tuple[str, str]]:
    """
    Convenience loader for YAML graphs of the form:
        edges:
          - [A, B]
          - [B, C]
          ...
    This helper is optional (useful for experiments/config-driven runs).

    Returns
    -------
    List[Tuple[str, str]]
        Edge list as (u, v) string tuples.
    """
    import yaml
    p = Path(path)
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    edges = data.get("edges", [])
    return [(str(u), str(v)) for (u, v) in edges]


    # --- DEAP compatibility: safe clone for Individuals --------------------------
def deap_clone(ind):
    """
    Returns a fresh copy of a DEAP Individual, preserving its fitness values.
    Works even when deap.tools.clone is unavailable.
    """
    # Re-instantiates same class from the list payload
    new_ind = type(ind)(ind)
    # Preserve fitness values so elites remain valid without re-evaluation
    try:
        new_ind.fitness.values = tuple(ind.fitness.values)
    except Exception:
        # If no fitness, leave as-is (it will be evaluated later)
        pass
    return new_ind



def _resolve_mutation_rates_simple(
    G: int,
    L_used: int,
    K: int,
    sel_mutation: Optional[float] = None,
    tail_mutation: Optional[float] = None,
    same_rates: bool = True,
) -> Tuple[float, float]:
    """
    Si same_rates=True → comportarse como macro_micro:
      - baseline = 1 / (G + tail_len)
      - si no hay overrides → p_sel = p_tail = baseline
      - si hay un override → el otro = baseline
      - si hay dos overrides → se respetan tal cual (macro_micro hace esto)

    Si same_rates=False → usa defaults segmentados con caps (los que ya tenías).
    Además, cualquier override en [0,1] es prob; si x>=1 se interpreta como λ
    (conteo esperado) y se convierte a prob por locus λ/N.
    """
    G = int(max(0, G))
    tail_len = max(0, int(L_used) - int(K))

    # --- Utilitarios ---
    def _as_prob(x: Optional[float], n_locus: int, p_default: float) -> float:
        if x is None:
            return p_default
        xv = float(x)
        if xv < 0:
            return p_default
        if xv <= 1.0:
            return max(0.0, min(1.0, xv))
        if n_locus <= 0:
            return 0.0
        return max(0.0, min(1.0, xv / float(n_locus)))

    # === Rama 1: macro_micro (mismas tasas por baseline global) ===
    if same_rates:
        total = max(1, G + tail_len)
        baseline = 1.0 / float(total)

        if sel_mutation is None and tail_mutation is None:
            ps = pt = baseline
        elif sel_mutation is not None and tail_mutation is None:
            ps = _as_prob(sel_mutation, max(G, 1), baseline)
            pt = baseline
        elif sel_mutation is None and tail_mutation is not None:
            pt = _as_prob(tail_mutation, max(tail_len, 1), baseline)
            ps = baseline
        else:
            # En macro_micro, si ambos vienen dados, se respetan tal cual.
            ps = _as_prob(sel_mutation, max(G, 1), baseline)
            pt = _as_prob(tail_mutation, max(tail_len, 1), baseline)

        return float(max(0.0, min(1.0, ps))), float(max(0.0, min(1.0, pt)))

    # === Rama 2: tus defaults segmentados con caps (como tenías) ===
    p_sel_default = min(0.25, max(0.02, 1.0 / max(G, 1))) if G > 0 else 0.0
    p_tail_default = min(0.10, max(0.005, 1.0 / max(tail_len, 10))) if tail_len > 0 else 0.0

    ps = _as_prob(sel_mutation, G, p_sel_default)
    pt = _as_prob(tail_mutation, tail_len, p_tail_default)
    return float(max(0.0, min(1.0, ps))), float(max(0.0, min(1.0, pt)))

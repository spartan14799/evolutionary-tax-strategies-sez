# validate_prefix_invariance.py
# -*- coding: utf-8 -*-
"""
Validates that the utility is invariant to permuting the binary decisions in the genome.

Auto-detection:
  * --mode graph: goods count = number of primary->final paths (DAG)
  * --mode transactions: goods count = number of 'MKT' purchases of each primary good (probe)

Equivalence Classes:
  * --equiv prefix: freely permutes positions 0..K-1 (original behavior)
  * --equiv per-good: one class per primary good -> only permutes the genome positions
    assigned to that good (mapped using the Planner to respect the actual order).

Returns 0 (OK) / 1 (FAIL). The user must replace the example block in __main__ with their actual data.
"""

import argparse
import math
import numpy as np
from pathlib import Path
import sys
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from collections import Counter

# =============================================================================
# Project bootstrap
# =============================================================================
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(1, str(ROOT_DIR))

# =============================================================================
# Domain imports (User's code)
# =============================================================================
from src.simulation.economy.economy import Economy
from src.simulation.economy.production_process.production_graph import ProductionGraph
from src.simulation.economy.production_process.production_process import ProductionProcess
from src.simulation.planner.planner import Planner
from src.config_paths import get_default_chart_of_accounts_path


# =============================================================================
# Helpers: Graph
# =============================================================================
def derive_primary_goods(graph) -> Set[str]:
    """Returns goods with in-degree 0 (accepts nx.DiGraph or list of edges)."""
    try:
        import networkx as nx
        if isinstance(graph, nx.DiGraph):
            return {n for n, deg in graph.in_degree() if deg == 0}
    except Exception:
        pass
    indeg: Dict[str, int] = {}
    nodes: Set[str] = set()
    for u, v in graph:
        nodes.add(u); nodes.add(v)
        indeg[v] = indeg.get(v, 0) + 1
        indeg.setdefault(u, indeg.get(u, 0))
    return {n for n in nodes if indeg.get(n, 0) == 0}

def _as_digraph(graph):
    import networkx as nx
    if isinstance(graph, nx.DiGraph):
        return graph
    G = nx.DiGraph()
    G.add_edges_from(graph)
    return G

def derive_final_good(graph, allow_many: bool = False) -> str:
    """Returns the sink (out-degree 0). Requires a single one by default."""
    import networkx as nx
    G = _as_digraph(graph)
    sinks = [n for n in G.nodes if G.out_degree(n) == 0]
    if not sinks:
        raise ValueError("No sinks found in the graph.")
    if not allow_many and len(sinks) != 1:
        raise ValueError(f"Expected a single sink, found {len(sinks)}: {sinks}")
    return sinks[0]

def count_paths_to_target_dag(graph, target: str) -> Dict[str, int]:
    """# simple directed paths from node -> target in DAG."""
    import networkx as nx
    G = _as_digraph(graph)
    if not nx.is_directed_acyclic_graph(G):
        raise ValueError("The graph must be a DAG.")
    paths = {n: 0 for n in G.nodes}
    paths[target] = 1
    for u in reversed(list(nx.topological_sort(G))):
        if u == target:
            continue
        total = 0
        for _, v in G.out_edges(u):
            total += paths[v]
        paths[u] = total
    return paths


# =============================================================================
# Planner / Economy bridges
# =============================================================================
def normalize_transactions(tx_out: Any) -> List[Tuple[str, str, str, str]]:
    """Converts Planner.execute_plan() output to (buyer, seller, action, good)."""
    while isinstance(tx_out, tuple) and len(tx_out) == 1:
        tx_out = tx_out[0]
    if isinstance(tx_out, tuple):
        tx_out = list(tx_out)
    norm: List[Tuple[str, str, str, str]] = []
    if not isinstance(tx_out, list):
        return norm
    for item in tx_out:
        if not isinstance(item, (list, tuple)) or len(item) < 4:
            continue
        _, party, action, good = item
        action = str(action); good = str(good)
        buyer = seller = ""
        if action == "Buy":
            if isinstance(party, (list, tuple)):
                if len(party) >= 1: buyer = str(party[0])
                if len(party) >= 2: seller = str(party[1])
        elif action == "Produce":
            if isinstance(party, (list, tuple)) and len(party) >= 1:
                buyer = str(party[0])
        norm.append((buyer, seller, action, good))
    return norm

def make_transactions_builder(production_graph) -> Callable[[np.ndarray], List[Tuple[str, str, str, str]]]:
    """Builds the actual transaction builder using the user's Planner."""
    try:
        import networkx as nx
        G = production_graph if isinstance(production_graph, nx.DiGraph) else nx.DiGraph(production_graph)
    except Exception:
        G = production_graph
    pgraph = ProductionGraph(G)
    pp = ProductionProcess(pgraph)
    planner = Planner(pp)
    def builder(genome: np.ndarray) -> List[Tuple[str, str, str, str]]:
        g = [int(x) for x in genome]
        return normalize_transactions(planner.execute_plan(g))
    return builder

def _filter_mkt_primary_buys(tx: List[Tuple[str, str, str, str]], primary: Set[str]) -> List[Tuple[str, str, str, str]]:
    return [(b, s, a, g) for (b, s, a, g) in tx if a == "Buy" and s == "MKT" and g in primary]

def count_MKT_primary_buys(transactions: List[Tuple[str, str, str, str]], primary: Set[str]) -> int:
    """k_tx = # of 'MKT' purchases of primary goods."""
    return sum(1 for b, s, a, g in transactions if a == "Buy" and s == "MKT" and g in primary)

def count_MKT_primary_buys_by_good(transactions: List[Tuple[str, str, str, str]], primary: Set[str]) -> Dict[str, int]:
    """Count of 'MKT' purchases, grouped by primary good."""
    d: Dict[str, int] = {}
    for _, s, a, g in transactions:
        if a == "Buy" and s == "MKT" and g in primary:
            d[g] = d.get(g, 0) + 1
    return dict(sorted(d.items()))

def parse_required_steps_from_error(msg: str) -> Optional[int]:
    import re
    m = re.search(r"at\s+least\s+(\d+)", msg, flags=re.IGNORECASE)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            return None
    return None

def calibrate_min_len_via_builder(builder: Callable[[np.ndarray], Any], base_L: int = 4, cap: int = 4096) -> int:
    """Increments L until the builder no longer raises 'at least X steps'."""
    L = max(1, int(base_L))
    while L <= cap:
        probe = np.zeros(L, dtype=int)
        try:
            _ = builder(probe)
            return L
        except ValueError as e:
            req = parse_required_steps_from_error(str(e))
            L = max(L + 1, req) if req is not None else (L + 1)
    raise RuntimeError(f"Failed to find minimum L <= {cap}.")

def detect_k_via_graph_and_L(production_graph) -> Tuple[int, int, Callable[[np.ndarray], Any]]:
    """k_graph via paths; L_min via actual builder."""
    primary = derive_primary_goods(production_graph)
    final_good = derive_final_good(production_graph)
    paths = count_paths_to_target_dag(production_graph, final_good)
    k_graph = sum(paths.get(p, 0) for p in primary)
    tx_builder = make_transactions_builder(production_graph)
    L_min = calibrate_min_len_via_builder(tx_builder, base_L=max(2, k_graph + 1))
    return k_graph, L_min, tx_builder

def detect_k_via_transactions_and_L(production_graph) -> Tuple[int, int, Callable[[np.ndarray], Any]]:
    """k_tx via transactions; L_min via actual builder."""
    primary = derive_primary_goods(production_graph)
    tx_builder = make_transactions_builder(production_graph)
    L_min = calibrate_min_len_via_builder(tx_builder, base_L=max(2, len(primary) + 1))
    probe = np.zeros(L_min, dtype=int)
    try:
        tx = tx_builder(probe)
        k_tx = count_MKT_primary_buys(tx, primary)
    except Exception:
        k_tx = len(primary)
    return k_tx, L_min, tx_builder


# =============================================================================
# Segment detection (per-good) + INDEX tracing with Planner
# =============================================================================
def detect_segments_via_graph(production_graph) -> Tuple[List[str], List[int]]:
    """Segments per primary good using DAG paths: size_i = paths[primary_i]."""
    primary = sorted(list(derive_primary_goods(production_graph)))
    final_good = derive_final_good(production_graph)
    paths = count_paths_to_target_dag(production_graph, final_good)
    labels, sizes = [], []
    for g in primary:
        c = int(paths.get(g, 0))
        if c > 0:
            labels.append(g)
            sizes.append(c)
    return labels, sizes

def detect_segments_via_transactions(production_graph) -> Tuple[List[str], List[int], Callable[[np.ndarray], Any]]:
    """Segments per primary good using observed MKT purchases in a probe with Planner."""
    primary = sorted(list(derive_primary_goods(production_graph)))
    tx_builder = make_transactions_builder(production_graph)
    labels, sizes = [], []
    try:
        # Minimum length to execute Planner
        L_min = calibrate_min_len_via_builder(tx_builder, base_L=max(2, len(primary) + 1))
        probe = np.zeros(L_min, dtype=int)
        tx = tx_builder(probe)
        by_good = count_MKT_primary_buys_by_good(tx, set(primary))
        for g in primary:
            c = int(by_good.get(g, 0))
            if c > 0:
                labels.append(g)
                sizes.append(c)
    except Exception:
        labels = list(primary)
        sizes = [1 for _ in labels]
    return labels, sizes, tx_builder

def _multiset_diff(a: List[Tuple[str, str, str, str]],
                   b: List[Tuple[str, str, str, str]]) -> Counter:
    """Returns multiset A-B (positive counts only)."""
    ca, cb = Counter(a), Counter(b)
    return ca - cb

def map_primary_index_sets_with_planner(
    L: int,
    K: int,
    builder: Callable[[np.ndarray], List[Tuple[str, str, str, str]]],
    primary_goods: Set[str],
    target_counts: Dict[str, int],
    fix_last_gene: bool = True,
) -> Dict[str, List[int]]:
    """
    Traces which genome indices generate MKT purchases of each primary good:
    by activating one gene at a time (one-hot) in 0..K-1 and measuring the transaction Δ.

    Returns {good: [indices]} with length per good ≈ target_counts[good].
    If a good falls short, it is padded with free indices from the prefix.
    """
    # Base genome
    g0 = np.zeros(L, dtype=int)
    if fix_last_gene:
        g0[-1] = 1

    try:
        tx0 = builder(g0)
    except Exception as e:
        raise RuntimeError(f"Could not build base transactions for tracing: {e}")

    base_mkt = _filter_mkt_primary_buys(tx0, primary_goods)
    need = dict(target_counts)  # remaining
    out: Dict[str, List[int]] = {g: [] for g in target_counts.keys()}

    assigned_any: Set[int] = set()
    prefix_scan = min(K, L)  # assumes the prefix contains the primary good purchases

    for i in range(prefix_scan):
        gi = g0.copy()
        gi[i] = 1
        try:
            txi = builder(gi)
        except Exception:
            continue

        mkti = _filter_mkt_primary_buys(txi, primary_goods)
        delta = _multiset_diff(mkti, base_mkt)  # new purchases by activating i

        # assign this index to the goods where a positive delta appears
        for (buyer, seller, action, good), cnt in delta.items():
            if cnt <= 0 or good not in need:
                continue
            to_take = min(cnt, max(0, need[good]))
            for _ in range(to_take):
                if i not in assigned_any:
                    out[good].append(i)
                    assigned_any.add(i)
                need[good] -= 1
                if need[good] <= 0:
                    need[good] = 0

        # early stop if all goods are satisfied
        if all(v <= 0 for v in need.values()):
            break

    # Padding in case of missing indices for a good
    if any(v > 0 for v in need.values()):
        free = [i for i in range(prefix_scan) if i not in assigned_any]
        it = iter(free)
        for g, rem in list(need.items()):
            while rem > 0:
                try:
                    i = next(it)
                except StopIteration:
                    break
                out[g].append(i)
                rem -= 1
            need[g] = rem

    return {g: sorted(ixs) for g, ixs in out.items()}

# =============================================================================
# Permutations
# =============================================================================
def sample_prefix_permutations(k: int, max_permutations: int, seed: Optional[int] = None) -> List[np.ndarray]:
    """
    Returns permutations of 0..k-1:
    - If k! <= max_permutations -> ALL.
    - Otherwise, a random sample of size max_permutations.
    """
    import itertools
    rng = np.random.default_rng(seed)
    if k <= 1:
        return [np.array(np.arange(k), dtype=int)]
    total = math.factorial(k)
    if total <= max_permutations:
        return [np.array(p, dtype=int) for p in itertools.permutations(range(k))]
    perms = set()
    while len(perms) < max_permutations:
        perm = tuple(rng.permutation(k).tolist())
        perms.add(perm)
    return [np.array(p, dtype=int) for p in perms]

def sample_block_permutations_by_sizes(sizes: List[int],
                                       max_permutations: int,
                                       seed: Optional[int] = None) -> List[List[np.ndarray]]:
    """
    Returns a list of combinations of permutations per block (not flattened):
    [[block_0_perm, block_1_perm, ...], ...]
    If ∏(s_i!) <= max_permutations -> all; otherwise, samples.
    """
    import itertools
    rng = np.random.default_rng(seed)
    sizes = [int(s) for s in sizes if s > 0]
    if not sizes:
        return [[]]
    if all(s <= 1 for s in sizes):
        return [[np.arange(s, dtype=int) for s in sizes]]

    total = 1
    for s in sizes:
        total *= math.factorial(s)

    def perms_for_size(s: int):
        if s <= 1:
            return [np.arange(s, dtype=int)]
        return [np.array(p, dtype=int) for p in itertools.permutations(range(s))]

    if total <= max_permutations:
        per_block = [perms_for_size(s) for s in sizes]
        return [list(combo) for combo in itertools.product(*per_block)]

    # Sampling
    seen: Set[Tuple[Tuple[int, ...], ...]] = set()
    out: List[List[np.ndarray]] = []
    while len(out) < max_permutations:
        combo: List[np.ndarray] = []
        key_parts: List[Tuple[int, ...]] = []
        for s in sizes:
            if s <= 1:
                p = np.arange(s, dtype=int)
            else:
                p = np.array(rng.permutation(s), dtype=int)
            combo.append(p)
            key_parts.append(tuple(p.tolist()))
        key = tuple(key_parts)
        if key in seen:
            continue
        seen.add(key)
        out.append(combo)
    return out

def apply_indexset_perm(g: np.ndarray, index_sets: List[List[int]], block_perms: List[np.ndarray]) -> np.ndarray:
    """Permutes in-place the values of g within each index set according to block_perms."""
    gp = g.copy()
    for idxs, p in zip(index_sets, block_perms):
        if len(idxs) <= 1:
            continue
        arr_idx = np.array(idxs, dtype=int)
        gp[arr_idx] = gp[arr_idx][p]
    return gp


# =============================================================================
# DIAGNOSTICS: base vs. permuted transaction comparator
# =============================================================================
def _count_by_good_mkt_primary(tx: List[Tuple[str, str, str, str]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for _, _, _, g in tx:
        counts[g] = counts.get(g, 0) + 1
    return dict(sorted(counts.items()))

def _print_tx_sample(label: str, tx_list: List[Tuple[str, str, str, str]], max_rows: int) -> None:
    print(f"    {label} (max {max_rows}):")
    for i, t in enumerate(tx_list[:max_rows], start=1):
        print(f"      {i:>3}: {t}")
    if len(tx_list) > max_rows:
        print(f"      ... (+{len(tx_list) - max_rows} more)")

def diagnose_non_invariance(g_base: np.ndarray,
                            g_perm: np.ndarray,
                            production_graph,
                            tx_builder: Callable[[np.ndarray], List[Tuple[str, str, str, str]]],
                            primary: Set[str],
                            explain_max: int = 40) -> None:
    """Prints a diagnosis focused on MKT purchases of primary goods."""
    try:
        tx_base = tx_builder(g_base)
        tx_perm = tx_builder(g_perm)
    except Exception as e:
        print(f"  [Diag] Failed to get transactions: {e}")
        return

    mkt_base = _filter_mkt_primary_buys(tx_base, primary)
    mkt_perm = _filter_mkt_primary_buys(tx_perm, primary)

    c_base = Counter(mkt_base)
    c_perm = Counter(mkt_perm)

    print("  [Diag] MKT purchases of primary goods:")
    print(f"    total base={len(mkt_base)} | perm={len(mkt_perm)}")
    print(f"    summary by good (base): {_count_by_good_mkt_primary(mkt_base)}")
    print(f"    summary by good (perm): {_count_by_good_mkt_primary(mkt_perm)}")

    if c_base == c_perm:
        print("    ⇒ The multiset of MKT purchases of primaries is EQUAL.")
        print("      The utility difference stems from something else (prices, sequence, non-primaries, etc.).")
        return

    only_in_perm = list((c_perm - c_base).elements())
    only_in_base = list((c_base - c_perm).elements())
    print("    ⇒ CHANGES detected in MKT purchases of primaries.")
    _print_tx_sample("only in perm", only_in_perm, explain_max)
    _print_tx_sample("only in base", only_in_base, explain_max)


# =============================================================================
# Main verification
# =============================================================================
def calculate_fitness(genome: np.ndarray, production_graph, pmatrix, agents_information) -> float:
    econ = Economy(production_graph=production_graph,
                   pmatrix=pmatrix,
                   agents_information=agents_information,
                   genome=genome.tolist())
    rep = econ.get_reports()
    return float(rep.get("utility", 0.0))

def verify_prefix_invariance(production_graph,
                              pmatrix,
                              agents_information,
                              k: Optional[int],
                              L: Optional[int],
                              mode: str,
                              trials: int,
                              max_permutations: int,
                              tol_abs: float,
                              seed: Optional[int],
                              fix_last_gene: bool = True,
                              explain: bool = False,
                              explain_max: int = 40,
                              explain_per_trial: int = 1,
                              equiv: str = "prefix") -> bool:
    """
    Returns True if ALL comparisons satisfy |Δu| <= tol_abs.

    equiv:
      - 'prefix': permutes 0..K-1 as a single block
      - 'per-good': builds indices per good (with Planner) and permutes only within those indices
    """
    if seed is not None:
        np.random.seed(seed)

    tx_builder: Callable[[np.ndarray], List[Tuple[str, str, str, str]]] = make_transactions_builder(production_graph)
    primary_goods = derive_primary_goods(production_graph)

    # --- Detect K or counts per good ---
    if equiv == "prefix":
        if k is None or L is None:
            if mode == "transactions":
                k_auto, L_min, _ = detect_k_via_transactions_and_L(production_graph)
            else:
                k_auto, L_min, _ = detect_k_via_graph_and_L(production_graph)
            if k is None:
                k = k_auto
            if L is None:
                L = max(L_min, k + 1)
        else:
            # Ensure Planner executes
            _ = calibrate_min_len_via_builder(tx_builder, base_L=max(2, k + 1))
    else:
        # per-good -> requires counts per good
        if mode == "transactions":
            labels, sizes_tx, tx_builder2 = detect_segments_via_transactions(production_graph)
            tx_builder = tx_builder2  # uses the detected actual builder
        else:
            labels, sizes_tx = detect_segments_via_graph(production_graph)
        K = sum(sizes_tx)
        if K == 0:
            print("[Info] No units of primary goods detected (>0). Trivial invariance.")
            return True

        # Minimum length and final L
        L_min = calibrate_min_len_via_builder(tx_builder, base_L=max(2, K + 1))
        if L is None:
            L = max(L_min, K + 1)

        # target map per good
        target_counts = {g: c for g, c in zip(labels, sizes_tx)}

        # --- Exact INDEX tracing with Planner in the prefix 0..K-1 ---
        index_map = map_primary_index_sets_with_planner(
            L=L, K=K, builder=tx_builder, primary_goods=primary_goods,
            target_counts=target_counts, fix_last_gene=fix_last_gene
        )
        # Normalize segment order and sizes
        labels = [g for g in labels if len(index_map.get(g, [])) > 0]
        index_sets: List[List[int]] = [sorted(index_map[g]) for g in labels]
        sizes_tx = [len(ixs) for ixs in index_sets]
        K = sum(sizes_tx)

        print(f"[Info] indices per good (equiv='per-good'):")
        for g, ixs in zip(labels, index_sets):
            print(f"   - {g}: idx={ixs} (n={len(ixs)})")
        if k is not None and k != K:
            print(f"[Warning] Ignoring --k={k} in 'per-good' mode; using K=∑n={K}.")
        k = K  # for building base genomes

    print("\n=== Invariance Validation (equiv='{}') ===".format(equiv))
    print(f"[Info] mode={mode} | k={k} | L={L} | trials={trials} | max_permutations={max_permutations} | tol={tol_abs:.2e}")

    if k is None or k <= 1:
        print("[Result] K <= 1 ⇒ trivial invariance.")
        return True

    rng = np.random.default_rng(seed)
    global_max_dev = 0.0
    global_worst = None
    all_ok = True

    for t in range(1, trials + 1):
        g0 = np.zeros(L, dtype=int)
        if fix_last_gene:
            g0[-1] = 1
        # Randomly fill the K prefix to avoid bias (even if they are binary)
        g0[:k] = rng.integers(0, 2, size=k)

        u0 = calculate_fitness(g0, production_graph, pmatrix, agents_information)

        if equiv == "prefix":
            perms = sample_prefix_permutations(k, max_permutations=max_permutations, seed=rng.integers(1, 10**9))
            combos = [(None, [p]) for p in perms]  # None index_sets (not used)
        else:
            sizes = [len(ix) for ix in index_sets]
            block_perms = sample_block_permutations_by_sizes(sizes, max_permutations=max_permutations,
                                                             seed=rng.integers(1, 10**9))
            combos = [(index_sets, bp) for bp in block_perms]

        max_dev = 0.0
        worst_local = None
        explained = 0

        for pi, (idx_sets, block_perm) in enumerate(combos, start=1):
            if equiv == "prefix":
                gp = g0.copy()
                gp[:k] = gp[:k][block_perm[0]]
                perm_pretty = block_perm[0].tolist()
            else:
                gp = apply_indexset_perm(g0, idx_sets, block_perm)
                perm_pretty = [p.tolist() for p in block_perm]

            up = calculate_fitness(gp, production_graph, pmatrix, agents_information)
            dev = abs(up - u0)

            if dev > max_dev:
                worst_local = (g0.copy(), gp.copy(), u0, up, perm_pretty, idx_sets if equiv == "per-good" else None)
                max_dev = dev

            if dev > tol_abs:
                all_ok = False
                print(f"[Trial {t}] ❌ NOT invariant (perm #{pi}): |Δu|={dev:.3e} > tol={tol_abs:.1e}")
                print(f"     u_base={u0:.12f}, u_perm={up:.12f}")
                if equiv == "prefix":
                    print(f"     perm={perm_pretty}")
                else:
                    print(f"     block_perms={perm_pretty}  # follows the order of index_sets indicated above")

                if explain and explained < explain_per_trial:
                    diagnose_non_invariance(g0, gp, production_graph, tx_builder, primary_goods,
                                            explain_max=explain_max)
                    explained += 1

        print(f"[Trial {t}] max |Δu| = {max_dev:.3e}")
        if max_dev > global_max_dev:
            global_max_dev = max_dev
            global_worst = worst_local

    status = "✅ OK" if all_ok else "❌ FAIL"
    print(f"\n[Summary] {status} | worst |Δu| = {global_max_dev:.3e}  (tol={tol_abs:.1e})")
    if global_worst and not all_ok:
        g_base_w, g_perm_w, u_base_w, u_perm_w, perm_worst, idx_sets_w = global_worst
        print(f"  Worst case → u_base={u_base_w:.12f}, u_perm={u_perm_w:.12f}")
        if equiv == "prefix":
            print(f"  perm={perm_worst}")
        else:
            print(f"  block_perms={perm_worst}")
            if idx_sets_w:
                print(f"  index_sets={idx_sets_w}")
            print("  [Worst case Diag]")
            diagnose_non_invariance(g_base_w, g_perm_w, production_graph, tx_builder, primary_goods,
                                     explain_max=explain_max)
    return all_ok


# =============================================================================
# CLI
# =============================================================================
def parse_args():
    p = argparse.ArgumentParser(description="Validates invariance to permutations of the prefix or per primary good (binary).")
    p.add_argument("--mode", choices=["graph", "transactions"], default="graph",
                   help="How to auto-detect K or counts per good: 'graph' (paths) or 'transactions' (MKT purchases).")
    p.add_argument("--equiv", choices=["prefix", "per-good"], default="per-good",
                   help="Type of equivalence class(es): 'prefix' (one) or 'per-good' (one per primary good).")
    p.add_argument("--k", type=int, default=None, help="Fixed k (only equiv=prefix). If omitted, it is auto-detected.")
    p.add_argument("--L", type=int, default=None, help="Genome length L. If omitted, it is calibrated.")
    p.add_argument("--trials", type=int, default=3, help="Base genomes to test.")
    p.add_argument("--max-permutations", type=int, default=200, help="Max. permutations per trial.")
    p.add_argument("--tol", type=float, default=1e-9, help="Absolute tolerance in utility.")
    p.add_argument("--seed", type=int, default=123, help="Global seed.")
    p.add_argument("--no-fix-last", action="store_true", help="Do not force the last gene to 1.")
    # Diagnostics
    p.add_argument("--explain", action="store_true",
                   help="Prints transaction diagnostics when invariance fails.")
    p.add_argument("--explain-max", type=int, default=40,
                   help="Maximum number of sample rows per list in the diagnosis (default: 40).")
    p.add_argument("--explain-per-trial", type=int, default=1,
                   help="Maximum number of diagnostics per trial (default: 1).")
    return p.parse_args()


# =============================================================================
# Demo / Template
# =============================================================================
if __name__ == "__main__":
    args = parse_args()

    # ------- EXAMPLE (The user must replace this with their actual graph/matrix/agents) -------
    graph_info = [
    ("A", "I"), ("B", "I"),
    ("C", "J"), ("D", "J"),
    ("E", "K"), ("F", "K"),
    ("G", "L"), ("H", "L"),

    ("I", "M"), ("I", "N"), ("I", "O"),
    ("J", "M"), ("J", "O"), ("J", "N"),
    ("K", "N"), ("K", "P"), ("K", "Q"),
    ("L", "O"), ("L", "Q"), ("L", "P"),

    ("M", "R"), 
    ("N", "S"),
    ("O", "S"),
    ("P", "R"), ("P", "T"),
    ("Q", "T"),


    ("R", "U"), 
    ("S", "U"), ("S", "V"),
    ("T", "V"),

    ("U", "W"), 
    ("V", "W")
]

    price_matrix = np.array([
    [[10, 1, 1], [11, 1, 1], [12, 1, 1]],
    [[2, 2, 2], [2, 2, 2], [2, 2, 2]],
    [[3, 3, 3], [3, 3, 3], [3, 3, 3]],
    [[4, 4, 4], [4, 4, 4], [4, 4, 4]],
    [[5, 5, 5], [5, 5, 5], [5, 5, 5]],
    [[6, 6, 6], [6, 6, 6], [6, 6, 6]],
    [[7, 7, 7], [7, 7, 7], [7, 7, 7]],
    [[8, 8, 8], [8, 8, 8], [8, 8, 8]],
    [[9, 9, 9], [9, 9, 9], [9, 9, 9]],
    [[10, 10, 10], [10, 10, 10], [10, 10, 10]],
    [[11, 11, 11], [11, 11, 11], [11, 11, 11]],
    [[12, 12, 12], [12, 12, 12], [12, 12, 12]],
    [[13, 13, 13], [13, 13, 13], [13, 13, 13]],
    [[14, 14, 14], [14, 14, 14], [14, 14, 14]],
    [[15, 15, 15], [15, 15, 15], [15, 15, 15]],
    [[16, 16, 16], [16, 16, 16], [16, 16, 16]],
    [[17, 17, 17], [17, 17, 17], [17, 17, 17]],
    [[18, 18, 18], [18, 18, 18], [18, 18, 18]],
    [[19, 19, 19], [19, 19, 19], [19, 19, 19]],
    [[20, 20, 20], [20, 20, 20], [20, 20, 20]],
    [[21, 21, 21], [21, 21, 21], [21, 21, 21]],
    [[22, 22, 22], [22, 22, 22], [22, 22, 22]],
    [[23, 23, 23], [23, 23, 23], [23, 23, 23]]
], dtype=int)

    goods_list = sorted({n for u, v in graph_info for n in (u, v)})
    accounts_path = get_default_chart_of_accounts_path()
    agents_info = {
        "MKT": {"type":"MKT","inventory_strategy":"FIFO","firm_related_goods":goods_list,
                "income_statement_type":"standard","accounts_yaml_path":accounts_path,"price_mapping":0},
        "NCT": {"type":"NCT","inventory_strategy":"FIFO","firm_related_goods":goods_list,
                "income_statement_type":"standard","accounts_yaml_path":accounts_path,"price_mapping":1},
        "ZF":  {"type":"ZF","inventory_strategy":"FIFO","firm_related_goods":goods_list,
                "income_statement_type":"standard","accounts_yaml_path":accounts_path,"price_mapping":2}
    }
    # ----------------------------------------------------------------------

    ok = verify_prefix_invariance(
        production_graph=graph_info,
        pmatrix=price_matrix,
        agents_information=agents_info,
        k=args.k,
        L=args.L,
        mode=args.mode,
        trials=args.trials,
        max_permutations=args.max_permutations,
        tol_abs=args.tol,
        seed=args.seed,
        fix_last_gene=(not args.no_fix_last),
        explain=args.explain,
        explain_max=args.explain_max,
        explain_per_trial=args.explain_per_trial,
        equiv=args.equiv
    )
    sys.exit(0 if ok else 1)

# -*- coding: utf-8 -*-
"""
ga_equivclass_selector_tail.py

GA UNIFICADO con dos-partes de genoma:
  - Parte A (selectores): para cada bien primario g, un índice que elige una
    clase de equivalencia (multiconjunto) entre las posibles para k_g.
  - Parte B (tail): bits libres (excepto último si se fija en 1).

Fenotipo:
  - A partir de los selectores -> bloques canónicos por bien (valores no-decrecientes
    sobre el alfabeto permitido) colocados en las posiciones REALES del prefijo
    según el ORDEN DEL PLANNER (detectado por sonda).
  - Tail se copia a partir de K; último gen opcionalmente fijado a 1.

Ventajas:
  - Evita el producto cartesiano explosivo de “muchos GA”.
  - Mantiene el respeto estricto del orden del Planner.
  - Muy paralelizable dentro del GA (fitness por individuo).

Incluye:
  - Logs por generación (best/mean/median)
  - Reporte de detección (labels, k_g, alfabeto, #clases por bien)
  - Gráficas: (1) best/mean/median, (2) barra top combinaciones (selectores más aptos)

Ejemplo al final con tu Grafo y Matriz.
"""

import argparse
import math
import itertools
import numpy as np
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple
import sys
import matplotlib.pyplot as plt

# =============================================================================
# Project bootstrap
# =============================================================================
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(1, str(ROOT_DIR))

# =============================================================================
# Domain imports (tu código real)
# =============================================================================
from src.simulation.economy.economy import Economy
from src.simulation.economy.production_process.production_graph import ProductionGraph
from src.simulation.economy.production_process.production_process import ProductionProcess
from src.simulation.planner.planner import Planner


# =============================================================================
# Utilidades de grafo
# =============================================================================
def _as_digraph(graph):
    import networkx as nx
    if isinstance(graph, nx.DiGraph):
        return graph
    G = nx.DiGraph()
    G.add_edges_from(graph)
    return G

def derive_primary_goods(graph) -> Set[str]:
    """Bienes con in-degree 0 (acepta nx.DiGraph o lista de aristas)."""
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

def derive_final_good(graph, allow_many: bool = False) -> str:
    """Devuelve el sink (out-degree 0). Exige único por defecto."""
    import networkx as nx
    G = _as_digraph(graph)
    sinks = [n for n in G.nodes if G.out_degree(n) == 0]
    if not sinks:
        raise ValueError("No hay sinks en el grafo.")
    if not allow_many and len(sinks) != 1:
        raise ValueError(f"Se esperaba un único sink, hay {len(sinks)}: {sinks}")
    return sinks[0]

def count_paths_to_target_dag(graph, target: str) -> Dict[str, int]:
    """# caminos dirigidos simples nodo -> target en DAG."""
    import networkx as nx
    G = _as_digraph(graph)
    if not nx.is_directed_acyclic_graph(G):
        raise ValueError("El grafo debe ser un DAG.")
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

def format_genome(g: np.ndarray) -> str:
    return "[" + ",".join(str(int(x)) for x in g) + "]"

def short_counts_str(counts: np.ndarray) -> str:
    return "-" if counts.size == 0 else ",".join(str(int(x)) for x in counts.tolist())


# =============================================================================
# Planner / Economy bridges
# =============================================================================
def normalize_transactions(tx_out: Any) -> List[Tuple[str, str, str, str]]:
    """Convierte salida de Planner.execute_plan() a (buyer, seller, action, good)."""
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
    """Builder real de transacciones usando tu Planner."""
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

def parse_required_steps_from_error(msg: str) -> Optional[int]:
    import re
    m = re.search(r"at\s+least\s+(\d+)", msg, flags=re.IGNORECASE)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            return None
    return None

def calibrate_min_len_via_builder(builder: Callable[[np.ndarray], Any],
                                  base_L: int = 4,
                                  cap: int = 4096) -> int:
    """Incrementa L hasta que el builder deje de lanzar 'at least X steps'."""
    L = max(1, int(base_L))
    while L <= cap:
        probe = np.zeros(L, dtype=int)
        try:
            _ = builder(probe)
            return L
        except ValueError as e:
            req = parse_required_steps_from_error(str(e))
            L = max(L + 1, req) if req is not None else (L + 1)
    raise RuntimeError(f"No se encontró L mínimo <= {cap}.")


# =============================================================================
# Detección del LAYOUT del prefijo (orden real del Planner) + tamaños por bien
# =============================================================================
def detect_prefix_layout_and_sizes(production_graph,
                                   mode: str = "graph") -> Tuple[List[str], List[int], List[List[int]], Callable[[np.ndarray], Any], int, Dict[str, Any]]:
    """
    Retorna:
      labels: bienes primarios en orden de primera aparición en la secuencia real
      sizes:  k_g según cuántas compras MKT de cada bien aparecen (Planner)
      index_sets: posiciones del prefijo (0..K-1) asociadas a cada bien g, respetando
                  el **orden de aparición en las transacciones**.
      tx_builder: builder real
      L_min: longitud mínima para ejecutar Planner
      info_extra: diagnóstico (incluye tamaños por grafo como referencia si mode='graph')
    """
    primary_all = sorted(derive_primary_goods(production_graph))
    tx_builder = make_transactions_builder(production_graph)
    base_L = max(2, len(primary_all) + 1)
    L_min = calibrate_min_len_via_builder(tx_builder, base_L=base_L)

    probe = np.zeros(L_min, dtype=int)
    seq_goods: List[str] = []
    try:
        tx = tx_builder(probe)
        for _, seller, action, good in tx:
            if action == "Buy" and seller == "MKT" and good in primary_all:
                seq_goods.append(good)
    except Exception:
        pass

    positions_by_good: Dict[str, List[int]] = {}
    for pos, g in enumerate(seq_goods):
        positions_by_good.setdefault(g, []).append(pos)

    labels = [g for g in dict.fromkeys(seq_goods).keys()]
    sizes_planner = [len(positions_by_good[g]) for g in labels]
    index_sets = [positions_by_good[g] for g in labels]
    K_planner = sum(sizes_planner)

    info_extra: Dict[str, Any] = {"K_planner": K_planner, "labels": labels, "sizes_planner": sizes_planner}

    if mode == "graph" and primary_all:
        final_good = derive_final_good(production_graph)
        paths = count_paths_to_target_dag(production_graph, final_good)
        sizes_graph_full = {g: int(max(0, paths.get(g, 0))) for g in primary_all}
        info_extra["sizes_graph_full"] = sizes_graph_full
        sizes_graph_in_labels = [sizes_graph_full.get(g, 0) for g in labels]
        info_extra["sizes_graph_in_labels"] = sizes_graph_in_labels
        if sum(sizes_graph_in_labels) != K_planner:
            print("**Aviso**: suma k_g (grafo) != K (Planner). Se usará el layout del Planner.")
    return labels, sizes_planner, index_sets, tx_builder, L_min, info_extra


# =============================================================================
# Detección de alfabeto permitido para el prefijo
# =============================================================================
def probe_allowed_indices_via_tx_builder(L_used: int,
                                         tx_builder: Callable[[np.ndarray], Any],
                                         max_index_probe: int = 16) -> List[int]:
    """
    Sonda: mete genomas constantes==idx y acepta los idx que no hacen fallar al Planner.
    """
    allowed = []
    for idx in range(max_index_probe):
        g = np.full(L_used, idx, dtype=int)
        try:
            _ = tx_builder(g)
            allowed.append(idx)
        except Exception:
            continue
    return sorted(set(allowed)) if allowed else [0, 1]


# =============================================================================
# Clases de equivalencia por bien (multiconjuntos de tamaño k_g)
# =============================================================================
def num_equiv_classes(alpha_size: int, k: int) -> int:
    """#clases = C(alpha_size + k - 1, k)."""
    if alpha_size <= 0:
        return 1 if k == 0 else 0
    return math.comb(alpha_size + k - 1, k)

def iter_count_vectors(alpha_size: int, k: int) -> Iterable[np.ndarray]:
    """
    Genera TODOS los vectores de conteo c ∈ N^alpha_size con sum(c)=k.
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
        counts = []
        for d in divs + (total_slots,):
            counts.append(d - prev - 1)
            prev = d
        yield np.array(counts, dtype=int)

def sample_count_vectors(alpha_size: int, k: int, num_samples: int, rng: np.random.Generator) -> List[np.ndarray]:
    """Muestrea clases al azar (uniforme) si no deseas todas."""
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
        counts = []
        for d in list(divs) + [total_slots]:
            counts.append(int(d - prev - 1))
            prev = int(d)
        out.append(np.array(counts, dtype=int))
    return out

def canonical_values_from_counts(alphabet: List[int], counts: np.ndarray) -> np.ndarray:
    """Vector ordenado (no-decreciente) con el multiconjunto de 'counts' sobre 'alphabet'."""
    vals = []
    for val, c in zip(alphabet, counts):
        if c > 0:
            vals.extend([int(val)] * int(c))
    return np.array(vals, dtype=int)


# =============================================================================
# GA — representación, operadores y evaluación
# =============================================================================
def build_class_pools(alphabet: List[int],
                      labels: List[str],
                      sizes: List[int],
                      per_good_cap: Optional[int],
                      rng: np.random.Generator) -> Tuple[List[List[np.ndarray]], List[int]]:
    """
    Para cada bien g crea su repositorio de clases (vectores 'counts' de tamaño |A|).
    Si per_good_cap se especifica y es menor que #total de clases, se muestrea.
    """
    pools: List[List[np.ndarray]] = []
    pool_sizes: List[int] = []
    A = len(alphabet)
    for k_g in sizes:
        total_g = num_equiv_classes(A, k_g)
        if per_good_cap is not None and total_g > per_good_cap:
            classes_g = sample_count_vectors(A, k_g, per_good_cap, rng)
        else:
            classes_g = list(iter_count_vectors(A, k_g))
        pools.append(classes_g)
        pool_sizes.append(len(classes_g))
    return pools, pool_sizes

def phenotype_from_individual(indiv: np.ndarray,
                              pools: List[List[np.ndarray]],
                              index_sets: List[List[int]],
                              alphabet: List[int],
                              L_used: int,
                              K: int,
                              fix_last_gene: bool) -> np.ndarray:
    """
    indiv = [ selectors (len=G) | tail (len=L_used-K) ]
    Construye el GENOMA completo que se le pasa a Economy.
    """
    G = len(pools)
    selectors = indiv[:G].astype(int)
    tail = indiv[G:].astype(int)

    # armar prefijo vacío y llenarlo con los valores canónicos por bien
    prefix = np.zeros(K, dtype=int)
    for ig in range(G):
        pool_g = pools[ig]
        sel = int(np.clip(selectors[ig], 0, len(pool_g) - 1))
        counts_g = pool_g[sel]
        vals_g = canonical_values_from_counts(alphabet, counts_g)
        pos_g = index_sets[ig]
        # seguridad
        if len(vals_g) != len(pos_g):
            raise ValueError(f"Counts y k_g no coinciden (bien idx={ig}): {len(vals_g)} != {len(pos_g)}")
        # asignar respetando el orden REAL del Planner
        for v, j in zip(vals_g, pos_g):
            prefix[j] = int(v)

    genome = np.zeros(L_used, dtype=int)
    if K > 0:
        genome[:K] = prefix
    if (L_used - K) > 0:
        last_tail_len = L_used - K
        genome[K:K + last_tail_len] = tail[:last_tail_len]
    if fix_last_gene and L_used > 0:
        genome[-1] = 1
    return genome

def init_population(G: int,
                    tail_len: int,
                    pool_sizes: List[int],
                    popsize: int,
                    fix_last_gene: bool,
                    seed: Optional[int]) -> np.ndarray:
    rng = np.random.default_rng(seed)
    pop = np.zeros((popsize, G + tail_len), dtype=int)
    # selectores: índice en [0..pool_sizes[g]-1]
    for i in range(popsize):
        for g in range(G):
            pop[i, g] = rng.integers(0, pool_sizes[g])
        # tail binario
        if tail_len > 0:
            pop[i, G:G + tail_len] = rng.integers(0, 2, size=tail_len, dtype=int)
            if fix_last_gene:
                pop[i, G + tail_len - 1] = 1
    return pop

def evaluate_population(pop: np.ndarray,
                        pools: List[List[np.ndarray]],
                        index_sets: List[List[int]],
                        alphabet: List[int],
                        L_used: int,
                        K: int,
                        production_graph,
                        pmatrix,
                        agents_information,
                        fix_last_gene: bool) -> Tuple[np.ndarray, List[Tuple[Tuple[int, ...], float]]]:
    fitness = np.zeros(pop.shape[0], dtype=float)
    # para estadísticos por combinación de selectores
    sel_stats: Dict[Tuple[int, ...], float] = {}
    for i in range(pop.shape[0]):
        indiv = pop[i, :]
        genome = phenotype_from_individual(indiv, pools, index_sets, alphabet, L_used, K, fix_last_gene)
        u = Economy(production_graph=production_graph,
                    pmatrix=pmatrix,
                    agents_information=agents_information,
                    genome=genome.tolist()).get_reports().get("utility", 0.0)
        fitness[i] = float(u)
        sel_key = tuple(int(x) for x in indiv[:len(pools)])
        best_so_far = sel_stats.get(sel_key, -float("inf"))
        if u > best_so_far:
            sel_stats[sel_key] = float(u)
    sel_list = [(k, v) for k, v in sel_stats.items()]
    return fitness, sel_list

def select_parents(pop: np.ndarray, fit: np.ndarray, num_parents: int) -> np.ndarray:
    order = np.argsort(fit)[::-1]
    return pop[order[:num_parents], :].copy()

def crossover_uniform(parents: np.ndarray,
                      offspring_size: Tuple[int, int],
                      G: int,
                      rng: np.random.Generator) -> np.ndarray:
    """
    Uniform crossover: para cada gen toma p1/p2 con prob 0.5.
    Mantiene dominios: selectores siguen siendo enteros, tail binario.
    """
    offspring = np.empty(offspring_size, dtype=int)
    for i in range(offspring_size[0]):
        p1 = parents[i % parents.shape[0], :]
        p2 = parents[(i + 1) % parents.shape[0], :]
        mask = rng.integers(0, 2, size=offspring_size[1], dtype=int)
        child = np.where(mask == 1, p1, p2).astype(int)
        offspring[i, :] = child
    return offspring

def mutate_population(pop: np.ndarray,
                      pool_sizes: List[int],
                      sel_mutation: float,
                      tail_mutation: float,
                      fix_last_gene: bool,
                      rng: np.random.Generator) -> np.ndarray:
    """
    - Mutación en selectores: con prob sel_mutation, cambia a otro índice válido.
    - Mutación en tail: bitflip con prob tail_mutation (excepto último si se fija).
    """
    mutated = pop.copy()
    G = len(pool_sizes)
    tail_len = mutated.shape[1] - G
    for i in range(mutated.shape[0]):
        # selectores
        for g in range(G):
            if rng.random() < sel_mutation and pool_sizes[g] > 1:
                cur = int(mutated[i, g])
                # elige un índice distinto
                choices = list(range(pool_sizes[g]))
                choices.remove(cur)
                mutated[i, g] = rng.choice(choices)
        # tail
        if tail_len > 0:
            last = tail_len - 1 if fix_last_gene else tail_len
            for j in range(last):
                if rng.random() < tail_mutation:
                    pos = G + j
                    mutated[i, pos] = 1 - mutated[i, pos]
            if fix_last_gene:
                mutated[i, G + tail_len - 1] = 1
    return mutated


# =============================================================================
# GA principal (unificado)
# =============================================================================
def run_joint_ga(production_graph,
                 pmatrix,
                 agents_information,
                 mode: str = "graph",           # etiqueta informativa
                 generations: int = 50,
                 popsize: int = 50,
                 parents: int = 20,
                 sel_mutation: float = 0.25,
                 tail_mutation: float = 0.05,
                 per_good_cap: Optional[int] = None,
                 max_index_probe: int = 16,
                 fix_last_gene: bool = True,
                 seed: Optional[int] = 44,
                 verbosity: int = 1,
                 log_every: int = 1,
                 no_plots: bool = False,
                 top_combos_bars: int = 20) -> Tuple[np.ndarray, float, Dict[str, Any]]:
    rng = np.random.default_rng(seed)

    # 1) Layout y tamaños (orden REAL del Planner)
    labels, sizes, index_sets, tx_builder, L_min, info_extra = \
        detect_prefix_layout_and_sizes(production_graph, mode=mode)
    K = int(sum(sizes))
    L_used = max(L_min, K + 1)

    # 2) Alfabeto válido para el prefijo
    alphabet = probe_allowed_indices_via_tx_builder(L_used, tx_builder, max_index_probe=max_index_probe)
    A = len(alphabet)

    # 3) Repositorios de clases por bien
    pools, pool_sizes = build_class_pools(alphabet, labels, sizes, per_good_cap, rng)
    G = len(pools)
    tail_len = max(0, L_used - K)

    print("=== Detección (orden Planner) ===")
    print(f"Bienes (orden de aparición): {labels}")
    print(f"k_g por bien: {sizes}  -> K={K}  | L_usado={L_used}")
    print(f"Alfabeto prefijo: {alphabet} (|A|={A})")
    print(f"Tamaño pool por bien: {pool_sizes}  (producto potencial ~ {np.prod(pool_sizes) if pool_sizes else 1})")
    print(f"Genoma interno = [selectores={G}] + [tail={tail_len}]  -> total={G + tail_len}")

    # 4) Inicialización
    pop = init_population(G, tail_len, pool_sizes, popsize, fix_last_gene, seed)
    best_curve: List[float] = []
    mean_curve: List[float] = []
    median_curve: List[float] = []
    # Para barras: mejor utilidad observada por combinación de selectores
    best_by_selectors: Dict[Tuple[int, ...], float] = {}

    # 5) Evolución
    for gen in range(1, generations + 1):
        fit, sel_list = evaluate_population(pop, pools, index_sets, alphabet, L_used, K,
                                            production_graph, pmatrix, agents_information, fix_last_gene)
        b = float(np.max(fit)); m = float(np.mean(fit)); med = float(np.median(fit))
        best_curve.append(b); mean_curve.append(m); median_curve.append(med)

        # acumular mejores por combinación de selectores
        for ksel, val in sel_list:
            cur = best_by_selectors.get(ksel, -float("inf"))
            if val > cur:
                best_by_selectors[ksel] = val

        if verbosity >= 1 and (gen == 1 or gen % log_every == 0 or gen == generations):
            print(f"Gen {gen:03d}: best={b:.6f} | mean={m:.6f} | median={med:.6f}")

        parents_mat = select_parents(pop, fit, parents)
        offspring = crossover_uniform(parents_mat, (pop.shape[0] - parents_mat.shape[0], pop.shape[1]), G, rng)
        offspring = mutate_population(offspring, pool_sizes, sel_mutation, tail_mutation, fix_last_gene, rng)

        # Reemplazo con elitismo
        pop[:parents_mat.shape[0], :] = parents_mat
        pop[parents_mat.shape[0]:, :] = offspring

    # 6) Resultado final
    fit, _sel_list = evaluate_population(pop, pools, index_sets, alphabet, L_used, K,
                                         production_graph, pmatrix, agents_information, fix_last_gene)
    idx = int(np.argmax(fit))
    indiv_best = pop[idx, :].copy()
    best_u = float(fit[idx])

    # Fenotipo ganador
    best_genome = phenotype_from_individual(indiv_best, pools, index_sets, alphabet, L_used, K, fix_last_gene)

    # Gráficas
    if not no_plots:
        if best_curve:
            plt.figure()
            plt.plot(best_curve, label="Best")
            if mean_curve:
                plt.plot(mean_curve, label="Mean")
            
            if median_curve:
                plt.plot(median_curve, label="Median")

        
        plt.title("Joint GA — Best/Mean/Median")
        plt.xlabel("Generación"); plt.ylabel("Utilidad")
        plt.grid(True); plt.legend(); plt.tight_layout(); plt.show()

        if best_by_selectors:
            ordered = sorted(best_by_selectors.items(), key=lambda x: x[1], reverse=True)
            top = ordered[:top_combos_bars]
            labels_bar = ["(" + ",".join(map(str, k)) + ")" for k, _ in top]
            values_bar = [v for _, v in top]
            plt.figure()
            x = np.arange(len(values_bar))
            plt.bar(x, values_bar)
            plt.xticks(x, labels_bar, rotation=90)
            plt.title(f"Top {len(values_bar)} combinaciones de selectores (mejor utilidad observada)")
            plt.xlabel("Selectores (índices por bien)"); plt.ylabel("Utilidad")
            plt.tight_layout(); plt.show()

    traces = {
        "labels": labels,
        "sizes": sizes,
        "index_sets": index_sets,
        "alphabet": alphabet,
        "K": K,
        "L_used": L_used,
        "pool_sizes": pool_sizes,
        "best_curve": best_curve,
        "mean_curve": mean_curve,
        "median_curve": median_curve,
        "best_by_selectors": sorted(
            [("("+",".join(map(str,k))+")", float(v)) for k, v in best_by_selectors.items()],
            key=lambda x: x[1], reverse=True
        ),
        "diagnostics": info_extra
    }
    return best_genome, best_u, traces


# =============================================================================
# CLI / Demo
# =============================================================================
def parse_args():
    p = argparse.ArgumentParser(description="GA unificado con selectores de clases de equivalencia (por bien) + tail binario (respeta el orden del Planner).")
    p.add_argument("--mode", choices=["graph", "transactions"], default="graph",
                   help="Etiqueta de detección (el layout SIEMPRE se toma del Planner; 'graph' solo reporta referencia).")
    p.add_argument("--generations", type=int, default=15)
    p.add_argument("--popsize", type=int, default=30)
    p.add_argument("--parents", type=int, default=15)
    p.add_argument("--sel-mutation", type=float, default=0.15, dest="sel_mutation")
    p.add_argument("--tail-mutation", type=float, default=0.05, dest="tail_mutation")
    p.add_argument("--per-good-cap", type=int, default=None,
                   help="Si se define y #clases por bien lo excede, se muestrean 'per_good_cap' clases para ese bien.")
    p.add_argument("--max-index-probe", type=int, default=16,
                   help="Máximo valor entero a probar al detectar el alfabeto del prefijo.")
    p.add_argument("--seed", type=int, default=40)
    p.add_argument("--no-fix-last", action="store_true", help="No fijar el último gen a 1.")
    p.add_argument("--verbosity", type=int, choices=[0,1,2], default=1)
    p.add_argument("--log-every", type=int, default=1)
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--top-combos-bars", type=int, default=20)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # --------- EJEMPLO con tu grafo y matriz ---------
    Grafo = [
  ("A", "B"),
  ("B", "C"),
  ("B", "D"),
  ("B", "E"),
  ("C", "E"),
  ("D", "E"),
  ("E", "F"),
]

    Matriz = np.array([
    [[7, 4, 8], [5, 7, 3], [7, 8, 5]],
    [[8, 5, 7], [7, 9, 3], [7, 8, 5]],
    [[10, 5, 11], [7, 9, 4], [10, 11, 8]],
    [[11, 7, 15], [7, 13, 4], [10, 17, 11]],
    [[18, 8, 15], [12, 15, 5], [16, 15, 13]],
    [[18, 12, 21], [14, 20, 7], [23, 25, 16]]
], dtype=int)

    goods_list = sorted({n for u, v in Grafo for n in (u, v)})
    accounts_path = ROOT_DIR / "chart_of_accounts.yaml"
    agents_info = {
        "MKT": {"type": "MKT", "inventory_strategy": "FIFO", "firm_related_goods": goods_list,
                "income_statement_type": "standard", "accounts_yaml_path": accounts_path, "price_mapping": 0},
        "NCT": {"type": "NCT", "inventory_strategy": "FIFO", "firm_related_goods": goods_list,
                "income_statement_type": "standard", "accounts_yaml_path": accounts_path, "price_mapping": 1},
        "ZF":  {"type": "ZF",  "inventory_strategy": "FIFO", "firm_related_goods": goods_list,
                "income_statement_type": "standard", "accounts_yaml_path": accounts_path, "price_mapping": 2},
    }

    best_g, best_u, traces = run_joint_ga(
        production_graph=Grafo,
        pmatrix=Matriz,
        agents_information=agents_info,
        mode=args.mode,
        generations=args.generations,
        popsize=args.popsize,
        parents=args.parents,
        sel_mutation=args.sel_mutation,
        tail_mutation=args.tail_mutation,
        per_good_cap=args.per_good_cap,
        max_index_probe=args.max_index_probe,
        fix_last_gene=(not args.no_fix_last),
        seed=args.seed,
        verbosity=args.verbosity,
        log_every=max(1, args.log_every),
        no_plots=args.no_plots,
        top_combos_bars=args.top_combos_bars
    )

    print("\n=== Resultado final (Joint GA: selectores + tail) ===")
    print(f"Bienes (orden Planner): {traces['labels']}")
    print(f"k_g: {traces['sizes']}  -> K={traces['K']}  L={traces['L_used']}")
    print(f"Pool por bien: {traces['pool_sizes']}")
    print(f"\nMejor utilidad: {best_u:.6f}")
    print(f"Mejor genoma (fenotipo): {format_genome(best_g)}")

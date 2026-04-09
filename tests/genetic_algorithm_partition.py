# -*- coding: utf-8 -*-
"""
Maximización por COMBINACIÓN de clases de equivalencia (una por bien primario),
corriendo un GA del TAIL para cada combinación de representantes del prefijo.

Esta versión **respeta el orden del Planner**:
- Detecta el layout del prefijo leyendo la secuencia real de compras a MKT
  de bienes primarios en una ejecución sonda.
- Para cada bien g obtiene las posiciones exactas del prefijo que le pertenecen.
- Al construir un prefijo para una combinación de clases, asigna los valores
  en **esas posiciones** (no en bloques contiguos).

Incluye:
- Logs por generación (configurables).
- Reporte de detección (labels, k_g, alfabeto, #clases por bien, #combos).
- Gráficas: (1) mejor combinación (best vs mean), (2) global (máx/mean),
            (3) top-N combinaciones por utilidad.

Por defecto hace BÚSQUEDA EXHAUSTIVA del espacio de combinaciones. Puedes
limitar con --max-combos o --per-good-cap.
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
from src.config_paths import get_default_chart_of_accounts_path


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
    """
    Convierte salida de Planner.execute_plan() a (buyer, seller, action, good).
    Soporta returns con coma final: (tx_list,).
    """
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
                buyer = str(party[0])  # usamos 'buyer' como 'actor'
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
# Detección de layout del prefijo (orden real del Planner) + tamaños por bien
# =============================================================================
def detect_prefix_layout_and_sizes(production_graph,
                                   mode: str = "graph") -> Tuple[List[str], List[int], List[List[int]], Callable[[np.ndarray], Any], int, Dict[str, Any]]:
    """
    Retorna:
      labels: bienes primarios en orden de primera aparición en la secuencia real
      sizes:  k_g = # posiciones del prefijo que pertenecen a cada bien (según Planner)
      index_sets: posiciones del prefijo (0..K-1) asociadas a cada bien g, respetando
                  el **orden de aparición en las transacciones**.
      tx_builder: builder real
      L_min: longitud mínima para ejecutar Planner
      info_extra: dict con diagnóstico (incluye tamaños 'graph' si mode=='graph')

    Nota: incluso en mode='graph' usamos la **secuencia real** del Planner para
    fijar el layout y K; los tamaños por grafo se reportan solo como referencia.
    """
    primary_all = sorted(derive_primary_goods(production_graph))
    tx_builder = make_transactions_builder(production_graph)
    # L mínimo compatible
    base_L = max(2, len(primary_all) + 1)
    L_min = calibrate_min_len_via_builder(tx_builder, base_L=base_L)

    # Sonda y extracción de secuencia real de compras MKT de bienes primarios
    probe = np.zeros(L_min, dtype=int)
    seq_goods: List[str] = []
    try:
        tx = tx_builder(probe)
        for _, seller, action, good in tx:
            if action == "Buy" and seller == "MKT" and good in primary_all:
                seq_goods.append(good)
    except Exception:
        pass

    # Construir layout por Planner: posiciones 0..K-1 etiquetadas por bien
    positions_by_good: Dict[str, List[int]] = {}
    for pos, g in enumerate(seq_goods):
        positions_by_good.setdefault(g, []).append(pos)

    # Orden de labels = orden de primera aparición en la secuencia real
    labels = [g for g in dict.fromkeys(seq_goods).keys()]
    sizes_planner = [len(positions_by_good[g]) for g in labels]
    index_sets = [positions_by_good[g] for g in labels]
    K_planner = sum(sizes_planner)

    info_extra: Dict[str, Any] = {"K_planner": K_planner, "labels": labels, "sizes_planner": sizes_planner}

    # (Opcional) tamaños por grafo como referencia
    if mode == "graph" and primary_all:
        final_good = derive_final_good(production_graph)
        paths = count_paths_to_target_dag(production_graph, final_good)
        sizes_graph_full = {g: int(max(0, paths.get(g, 0))) for g in primary_all}
        info_extra["sizes_graph_full"] = sizes_graph_full
        # reordenar a nuestro orden 'labels' (primera aparición)
        sizes_graph_in_labels = [sizes_graph_full.get(g, 0) for g in labels]
        info_extra["sizes_graph_in_labels"] = sizes_graph_in_labels
        if sum(sizes_graph_in_labels) != K_planner:
            print("**Aviso**: suma de tamaños por grafo != K detectado por Planner. Se usará el layout del Planner.")

    return labels, sizes_planner, index_sets, tx_builder, L_min, info_extra


# =============================================================================
# Detección de alfabeto permitido en prefijo
# =============================================================================
def probe_allowed_indices_via_tx_builder(L_used: int,
                                         tx_builder: Callable[[np.ndarray], Any],
                                         max_index_probe: int = 16) -> List[int]:
    """
    Sonda simple: mete un genoma constante==idx y acepta los idx que no hagan
    explotar al Planner. (Ingenuo pero útil para tests.)
    """
    allowed = []
    for idx in range(max_index_probe):
        g = np.full(L_used, idx, dtype=int)
        try:
            _ = tx_builder(g)
            allowed.append(idx)
        except Exception:
            continue
    if not allowed:
        allowed = [0, 1]
    return sorted(set(allowed))


# =============================================================================
# Clases de equivalencia (multiconjuntos) por bien
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
# GA sobre el TAIL con prefijo FIJO (con logs opcionales)
# =============================================================================
def init_population_with_fixed_prefix(prefix: np.ndarray,
                                      L_used: int,
                                      population_size: int,
                                      fix_last_gene: bool,
                                      seed: Optional[int]) -> np.ndarray:
    rng = np.random.default_rng(seed)
    pop = rng.integers(0, 2, size=(population_size, L_used), dtype=int)
    K = len(prefix)
    pop[:, :K] = prefix.reshape(1, -1)
    if fix_last_gene:
        pop[:, -1] = 1
    return pop

def fitness_of_population(pop: np.ndarray,
                          production_graph,
                          pmatrix,
                          agents_information) -> np.ndarray:
    return np.array([
        Economy(production_graph=production_graph,
                pmatrix=pmatrix,
                agents_information=agents_information,
                genome=ind.tolist()).get_reports().get("utility", 0.0)
        for ind in pop
    ], dtype=float)

def select_parents(pop: np.ndarray, fit: np.ndarray, num_parents: int) -> np.ndarray:
    order = np.argsort(fit)[::-1]
    return pop[order[:num_parents], :].copy()

def crossover_tail_only(parents: np.ndarray,
                        offspring_size: Tuple[int, int],
                        K_lock: int) -> np.ndarray:
    """1p-crossover sólo en [K_lock .. L-1). Prefijo se copia intacto."""
    offspring = np.empty(offspring_size, dtype=int)
    L = offspring_size[1]
    cp = K_lock + max(1, (L - K_lock) // 2)
    for i in range(offspring_size[0]):
        p1 = parents[i % parents.shape[0], :]
        p2 = parents[(i + 1) % parents.shape[0], :]
        offspring[i, :K_lock] = p1[:K_lock]
        offspring[i, K_lock:cp] = p1[K_lock:cp]
        offspring[i, cp:] = p2[cp:]
    return offspring

def mutate_tail_bitflip(pop: np.ndarray,
                        K_lock: int,
                        mutation_rate: float,
                        fix_last_gene: bool) -> np.ndarray:
    rng = np.random.default_rng()
    mutated = pop.copy()
    last = mutated.shape[1] - 1 if fix_last_gene else mutated.shape[1]
    for i in range(mutated.shape[0]):
        for j in range(K_lock, last):
            if rng.random() < mutation_rate:
                mutated[i, j] = 1 - mutated[i, j]
    if fix_last_gene:
        mutated[:, -1] = 1
    return mutated

def run_ga_tail(prefix: np.ndarray,
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
                progress_label: Optional[str] = None
                ) -> Tuple[np.ndarray, float, List[float], List[float]]:
    """GA que optimiza sólo el tail manteniendo fijo el prefijo, con logs opcionales."""
    pop = init_population_with_fixed_prefix(prefix, L_used, population_size, fix_last_gene, seed)
    K = len(prefix)
    best_curve: List[float] = []
    mean_curve: List[float] = []

    for gen in range(1, num_generations + 1):
        fitness = fitness_of_population(pop, production_graph, pmatrix, agents_information)
        b = float(np.max(fitness)); m = float(np.mean(fitness))
        best_curve.append(b); mean_curve.append(m)

        if print_every and (gen == 1 or gen % print_every == 0 or gen == num_generations):
            tag = f"[{progress_label}] " if progress_label else ""
            print(f"{tag}Gen {gen:03d}: best={b:.6f} | mean={m:.6f}")

        parents = select_parents(pop, fitness, num_parents_mating)
        offspring = crossover_tail_only(parents, (pop.shape[0] - parents.shape[0], pop.shape[1]), K_lock=K)
        offspring = mutate_tail_bitflip(offspring, K_lock=K, mutation_rate=mutation_rate, fix_last_gene=fix_last_gene)

        # Reemplazo con elitismo simple
        pop[:parents.shape[0], :] = parents
        pop[parents.shape[0]:, :] = offspring
        if fix_last_gene:
            pop[:, -1] = 1
        pop[:, :K] = prefix.reshape(1, -1)

    fitness = fitness_of_population(pop, production_graph, pmatrix, agents_information)
    idx = int(np.argmax(fitness))
    return pop[idx, :].copy(), float(fitness[idx]), best_curve, mean_curve


# =============================================================================
# Orquestación (exhaustivo sobre combinaciones) respetando layout del Planner
# =============================================================================
def maximize_by_class_combinations(production_graph,
                                   pmatrix,
                                   agents_information,
                                   mode: str = "graph",          # mantiene 'graph'/'transactions' como etiqueta
                                   num_generations: int = 10,
                                   population_size: int = 20,
                                   num_parents_mating: int = 10,
                                   mutation_rate: float = 0.1,
                                   fix_last_gene: bool = True,
                                   seed: Optional[int] = 123,
                                   max_combos: Optional[int] = None,
                                   per_good_cap: Optional[int] = None,
                                   max_index_probe: int = 16,
                                   verbosity: int = 1,
                                   log_every: int = 1,
                                   no_plots: bool = False,
                                   top_combos_bars: int = 20
                                   ) -> Tuple[np.ndarray, float, Dict[str, Any]]:
    rng = np.random.default_rng(seed)

    # 1) Layout + tamaños por bien (usando secuencia real del Planner)
    labels, sizes, index_sets, tx_builder, L_min, info_extra = \
        detect_prefix_layout_and_sizes(production_graph, mode=mode)

    K = int(sum(sizes))
    L_used = max(L_min, K + 1)

    # 2) Alfabeto permitido (valores de genes en prefijo)
    alphabet = probe_allowed_indices_via_tx_builder(L_used, tx_builder, max_index_probe=max_index_probe)
    A = len(alphabet)

    # 3) Clases por bien → listas de vectores de conteos
    per_good_classes: List[List[np.ndarray]] = []
    per_good_totals: List[int] = []
    for k_g in sizes:
        total_g = num_equiv_classes(A, k_g)
        per_good_totals.append(total_g)
        if per_good_cap is not None and total_g > per_good_cap:
            classes_g = sample_count_vectors(A, k_g, per_good_cap, rng)
        else:
            classes_g = list(iter_count_vectors(A, k_g))
        per_good_classes.append(classes_g)

    # 4) Número total de combinaciones y política (exhaustivo o muestreo)
    combo_space = 1
    for total in per_good_totals:
        combo_space *= max(1, total)

    print("=== Detección de segmentos/clases (orden Planner) ===")
    print(f"Modo solicitado: {mode}")
    print(f"Bienes (orden de aparición): {labels}")
    print(f"k_g por bien (Planner): {sizes}  -> K={K}  | L_usado={L_used}")
    print(f"Alfabeto prefijo detectado: {alphabet}  (|A|={A})")
    if "sizes_graph_in_labels" in info_extra:
        print(f"Referencia grafo (k_g en orden labels): {info_extra['sizes_graph_in_labels']}")
    print("Clases por bien (totales):", per_good_totals)
    print(f"Espacio de combinaciones: {combo_space}")
    if max_combos is not None and combo_space > max_combos:
        print(f"**Aviso**: combo_space > max_combos={max_combos}. Se muestrearán combos.")

    def combo_iterator() -> Iterable[List[np.ndarray]]:
        if max_combos is None or combo_space <= max_combos:
            # EXHAUSTIVO
            for combo in itertools.product(*per_good_classes) if per_good_classes else [()]:
                yield list(combo)
        else:
            # MUESTREO
            for _ in range(max_combos):
                pick = []
                for classes_g in per_good_classes:
                    j = rng.integers(0, len(classes_g)) if classes_g else 0
                    pick.append(classes_g[j] if classes_g else np.zeros(0, dtype=int))
                yield pick

    # 5) GA por combinación, construyendo prefijos en **layout real**
    best_u = -float("inf")
    best_g = None
    best_combo_info: Dict[str, Any] = {}
    results_by_combo: List[Tuple[List[np.ndarray], float]] = []
    curves_by_combo: List[Tuple[List[np.ndarray], List[float], List[float]]] = []

    for it, counts_combo in enumerate(combo_iterator(), start=1):
        # Construir prefijo vacío de longitud K
        prefix = np.zeros(K, dtype=int)
        # Para cada bien g, construir sus valores canónicos y asignarlos
        # en las posiciones reales dadas por index_sets[g].
        for ig, counts_g in enumerate(counts_combo):
            vals_g = canonical_values_from_counts(alphabet, counts_g)
            pos_g = index_sets[ig]
            assert len(vals_g) == len(pos_g), "Counts no coinciden con k_g detectado."
            # asignación en el orden de aparición del Planner
            for v, j in zip(vals_g, pos_g):
                prefix[j] = int(v)

        combo_label = " | ".join(short_counts_str(c) for c in counts_combo) if counts_combo else "<vacío>"
        progress_label = f"Combo {it:04d}/{'?' if max_combos else combo_space}: {combo_label}"
        print_every = (log_every if verbosity >= 2 else None)

        g_star, u_star, best_curve, mean_curve = run_ga_tail(
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
            seed=(None if seed is None else seed + it),
            print_every=print_every,
            progress_label=progress_label
        )

        results_by_combo.append((counts_combo, u_star))
        curves_by_combo.append((counts_combo, best_curve, mean_curve))

        if verbosity == 1 and (u_star > best_u):
            print(f"[Nuevo BEST-SO-FAR] {progress_label}  -> best_u={u_star:.6f}")
            for gen, (b, m) in enumerate(zip(best_curve, mean_curve), start=1):
                if (gen == 1) or (gen % log_every == 0) or (gen == len(best_curve)):
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
                "mean_curve": mean_curve
            }

    # Curvas agregadas entre combinaciones
    if curves_by_combo:
        global_best_curve = np.maximum.reduce(
            [np.array(curve, dtype=float) for (_c, curve, _m) in curves_by_combo]
        ).tolist()
        global_mean_curve = np.mean(
            [np.array(mcurve, dtype=float) for (_c, _b, mcurve) in curves_by_combo],
            axis=0
        ).tolist()
    else:
        global_best_curve, global_mean_curve = [], []

    # Reporte textual
    print("\n=== Resumen ===")
    print(f"Combinaciones evaluadas: {len(results_by_combo)}")
    print(f"Mejor utilidad global: {best_u:.6f}")
    if best_combo_info:
        print(f"Mejor combinación: {best_combo_info['label']}")
        print(f"Prefijo (layout Planner): {best_combo_info['prefix']}")
        print("Evolución (mejor combinación):")
        for gen, (b, m) in enumerate(zip(best_combo_info['best_curve'],
                                         best_combo_info['mean_curve']), start=1):
            if (gen == 1) or (gen % log_every == 0) or (gen == len(best_combo_info['best_curve'])):
                print(f"  Gen {gen:03d}: best={b:.6f} | mean={m:.6f}")

    # Gráficas opcionales
    if not no_plots:
        if best_combo_info and best_combo_info["best_curve"]:
            plt.figure()
            plt.plot(best_combo_info["best_curve"], label="Best (mejor combinación)")
            if best_combo_info["mean_curve"]:
                plt.plot(best_combo_info["mean_curve"], label="Mean (mejor combinación)")
            plt.title("GA (Tail) — Mejor combinación")
            plt.xlabel("Generación"); plt.ylabel("Utilidad")
            plt.grid(True); plt.legend(); plt.tight_layout(); plt.show()

        if global_best_curve and global_mean_curve:
            plt.figure()
            plt.plot(global_best_curve, label="Global Best (max entre combos)")
            plt.plot(global_mean_curve, label="Global Mean (mean entre combos)")
            plt.title("GA (Tail) — Global (todas las combinaciones)")
            plt.xlabel("Generación"); plt.ylabel("Utilidad")
            plt.grid(True); plt.legend(); plt.tight_layout(); plt.show()

        if results_by_combo:
            ordered = sorted(results_by_combo, key=lambda x: x[1], reverse=True)
            top = ordered[:top_combos_bars]
            labels_bar = []
            values_bar = []
            for counts_combo, val in top:
                lbl = " | ".join(short_counts_str(c) for c in counts_combo) if counts_combo else "<vacío>"
                labels_bar.append(lbl); values_bar.append(val)
            plt.figure()
            x = np.arange(len(values_bar))
            plt.bar(x, values_bar)
            plt.xticks(x, labels_bar, rotation=90)
            plt.title(f"Mejor utilidad por combinación (Top {len(values_bar)})")
            plt.xlabel("Combinación (conteos por bien en orden de aparición)")
            plt.ylabel("Mejor utilidad")
            plt.tight_layout(); plt.show()

    traces = {
        "detection_mode": mode,
        "labels": labels,
        "sizes": sizes,
        "index_sets": index_sets,     # posiciones reales del prefijo por bien
        "alphabet": alphabet,
        "K": K,
        "L_used": L_used,
        "combo_space": combo_space,
        "per_good_totals": per_good_totals,
        "best_combo": best_combo_info,
        "global_best_curve": global_best_curve,
        "global_mean_curve": global_mean_curve,
        "results_by_combo": [
            (["(" + short_counts_str(c) + ")" for c in counts], float(u))
            for counts, u in results_by_combo
        ],
        "diagnostics": info_extra
    }
    return best_g, best_u, traces


# =============================================================================
# CLI / Demo
# =============================================================================
def parse_args():
    p = argparse.ArgumentParser(description="GA del tail por combinación de clases per-good (EXHAUSTIVO por defecto), respetando el orden del Planner.")
    p.add_argument("--mode", choices=["graph", "transactions"], default="graph",
                   help="Etiqueta de detección (los tamaños por grafo se reportan como referencia; el layout SIEMPRE se toma del Planner).")
    p.add_argument("--generations", type=int, default=10)
    p.add_argument("--popsize", type=int, default=20)
    p.add_argument("--parents", type=int, default=10)
    p.add_argument("--mutation", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=40)
    p.add_argument("--no-fix-last", action="store_true", help="No fijar el último gen a 1.")
    p.add_argument("--max-combos", type=int, default=None,
                   help="Máx. combinaciones (producto de clases) a evaluar. Si no se da, evalúa todas (EXHAUSTIVO).")
    p.add_argument("--per-good-cap", type=int, default=None,
                   help="Límite de clases por bien (si se excede, se muestrean 'per-good-cap').")
    p.add_argument("--max-index-probe", type=int, default=16,
                   help="Máximo valor entero a probar para detectar alfabeto del prefijo.")
    p.add_argument("--verbosity", type=int, choices=[0,1,2], default=1,
                   help="0: sin logs por generación; 1: logs por generación sólo si es nuevo best-so-far; 2: logs por generación en TODOS los combos.")
    p.add_argument("--log-every", type=int, default=1,
                   help="Frecuencia de impresión de generaciones (cada N).")
    p.add_argument("--no-plots", action="store_true", help="No mostrar gráficos.")
    p.add_argument("--top-combos-bars", type=int, default=20, help="Top-N en la barra por combinación.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # --------- EJEMPLO con tu grafo y matriz especificados ---------
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
]
, dtype=int)

    goods_list = sorted({n for u, v in Grafo for n in (u, v)})
    accounts_path = get_default_chart_of_accounts_path()
    agents_info = {
        "MKT": {"type": "MKT", "inventory_strategy": "FIFO", "firm_related_goods": goods_list,
                "income_statement_type": "standard", "accounts_yaml_path": accounts_path, "price_mapping": 0},
        "NCT": {"type": "NCT", "inventory_strategy": "FIFO", "firm_related_goods": goods_list,
                "income_statement_type": "standard", "accounts_yaml_path": accounts_path, "price_mapping": 1},
        "ZF":  {"type": "ZF",  "inventory_strategy": "FIFO", "firm_related_goods": goods_list,
                "income_statement_type": "standard", "accounts_yaml_path": accounts_path, "price_mapping": 2},
    }

    best_g, best_u, traces = maximize_by_class_combinations(
        production_graph=Grafo,
        pmatrix=Matriz,
        agents_information=agents_info,
        mode=args.mode,
        num_generations=args.generations,
        population_size=args.popsize,
        num_parents_mating=args.parents,
        mutation_rate=args.mutation,
        fix_last_gene=(not args.no_fix_last),
        seed=args.seed,
        max_combos=args.max_combos,
        per_good_cap=args.per_good_cap,
        max_index_probe=args.max_index_probe,
        verbosity=args.verbosity,
        log_every=max(1, args.log_every),
        no_plots=args.no_plots,
        top_combos_bars=args.top_combos_bars
    )

    print("\n=== GA tail por combinación de clases (per-good) — Resultado final ===")
    print(f"Bienes (orden Planner): {traces['labels']}")
    print(f"k_g (Planner): {traces['sizes']}  -> K={traces['K']}  L={traces['L_used']}")
    print(f"Index sets por bien (posiciones del prefijo): {traces['index_sets']}")
    print(f"Alfabeto prefijo detectado: {traces['alphabet']}")
    print(f"Espacio de combinaciones: {traces['combo_space']}")
    print(f"Combinaciones evaluadas: {len(traces['results_by_combo'])}")
    print(f"\nMejor utilidad: {best_u:.6f}")
    print(f"Mejor genoma: {format_genome(best_g)}")
    if traces.get("best_combo"):
        print("\nMejor combinación (conteos por bien, en orden de aparición):")
        for label, counts in zip(traces["labels"], traces["best_combo"]["counts_combo"]):
            print(f"  {label}: {counts}")
        print(f"Prefijo (layout Planner): {traces['best_combo']['prefix']}")

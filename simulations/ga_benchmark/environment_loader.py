# simulations/ga_benchmark/environment_loader.py
# =============================================================================
# Environment Loader for FTZ_EvoBench (selection policy + env meta)
# =============================================================================

from __future__ import annotations
import csv, json, random
from pathlib import Path
from typing import Tuple, Dict, List, Optional
import numpy as np

# --- rutas locales ---
BASE_DIR   = Path(__file__).resolve().parent           # .../simulations/ga_benchmark
DATA_DIR   = BASE_DIR / "data"
GRAPHS_DIR = DATA_DIR / "graphs"
PRICES_DIR = DATA_DIR / "prices"
ENV_INDEX  = DATA_DIR / "env_index.csv"

__all__ = [
    "generate_environment",
    "list_environments",
]

# -----------------------------------------------------------------------------
# helpers de índice
# -----------------------------------------------------------------------------
def _ensure_index_exists() -> None:
    """
    Si falta env_index.csv, lo reconstruye emparejando cada grafo con su archivo
    de precios '<graph_id>_prices.json'. No crea filas 'huérfanas'.
    """
    if ENV_INDEX.exists():
        return

    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
    PRICES_DIR.mkdir(parents=True, exist_ok=True)

    rows: List[List[str]] = []
    for gpath in sorted(GRAPHS_DIR.glob("*.json")):
        gid = gpath.stem  # p.ej. 'g6_low_0'
        price_file = f"{gid}_prices.json"
        ppath = PRICES_DIR / price_file
        if not ppath.exists():
            continue
        try:
            g = json.loads(gpath.read_text(encoding="utf-8"))
            N = g.get("N", 0)
            density = g.get("density", 0.0)
            max_depth = g.get("max_depth", 0)
        except Exception:
            continue
        rows.append([gid, N, density, max_depth, price_file])

    if rows:
        ENV_INDEX.parent.mkdir(parents=True, exist_ok=True)
        with open(ENV_INDEX, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["graph_id", "N", "density", "max_depth", "price_file"])
            w.writerows(rows)

def _read_index_rows() -> List[Dict[str, str]]:
    """Lee env_index.csv como lista de dicts; exige cabeceras estándar."""
    if not ENV_INDEX.exists():
        raise FileNotFoundError(
            f"env_index.csv not found at {ENV_INDEX}. "
            "Run `python -m simulations.ga_benchmark.generate_data` first."
        )
    rows: List[Dict[str, str]] = []
    with open(ENV_INDEX, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    if not rows:
        raise RuntimeError("env_index.csv is empty; no environments available.")
    return rows

def _unique_nodes_from_edges(edges: List[List[str] | tuple[str, str]]) -> List[str]:
    s = set()
    for u, v in edges:
        s.add(u); s.add(v)
    return sorted(s)

def _parse_level_from_graph_id(graph_id: str) -> Optional[str]:
    """
    Espera formato 'g{N}_{level}_{rep}'. Devuelve level en minúsculas, si existe.
    """
    try:
        parts = graph_id.split("_")
        if len(parts) >= 3 and parts[0].startswith("g"):
            return parts[1].lower()
    except Exception:
        pass
    return None

# -----------------------------------------------------------------------------
# API pública: listar entornos del índice con filtros
# -----------------------------------------------------------------------------
def list_environments(
    sizes: Optional[List[int]] = None,
    levels: Optional[List[str]] = None,
    max_envs: Optional[int] = None,
    shuffle: bool = False,
    shuffle_seed: Optional[int] = None,
) -> List[Dict[str, str]]:
    """
    Devuelve filas del env_index con posibles filtros:
      - sizes: lista de N (enteros) permitidos
      - levels: lista de 'low'|'medium'|'high' (según graph_id)
      - max_envs: recorta la lista final
      - shuffle(+seed): baraja el orden de evaluación
    Cada fila es un dict con al menos: graph_id, N, density, max_depth, price_file.
    """
    _ensure_index_exists()
    rows = _read_index_rows()

    # normalizar filtros
    levels_norm = [str(x).lower() for x in levels] if levels else None
    sizes_set = set(int(x) for x in sizes) if sizes else None

    filtered: List[Dict[str, str]] = []
    for r in rows:
        n_ok = True
        l_ok = True
        if sizes_set is not None:
            try:
                n_ok = int(r["N"]) in sizes_set
            except Exception:
                n_ok = False
        if levels_norm is not None:
            lvl = _parse_level_from_graph_id(str(r["graph_id"]))
            l_ok = (lvl in levels_norm) if lvl is not None else False
        if n_ok and l_ok:
            filtered.append(r)

    if shuffle:
        rng = random.Random(shuffle_seed)
        rng.shuffle(filtered)

    if max_envs is not None:
        try:
            k = int(max_envs)
            filtered = filtered[:max(0, k)]
        except Exception:
            pass

    return filtered

# -----------------------------------------------------------------------------
# API pública: cargar un entorno concreto (+ metadatos)
# -----------------------------------------------------------------------------
def generate_environment(
    seed: int = 0,
    selection: str = "by_seed",            # "by_seed" | "fixed_index" | "graph_id"
    fixed_index: Optional[int] = None,
    graph_id: Optional[str] = None,
):
    """
    Carga un par (grafo, precios) y devuelve:
      agents_information, production_graph, pmatrix, env_meta
    selection:
      - "by_seed": elige fila seed % #envs
      - "fixed_index": elige la fila fixed_index % #envs
      - "graph_id": elige la fila con ese graph_id exacto
    env_meta = {"graph_id": <id>, "price_file": <file>, "N": <int>, "density": <float>, "max_depth": <int>}
    """
    _ensure_index_exists()
    rows = _read_index_rows()

    # Resolver fila del índice
    if selection == "graph_id" and graph_id is not None:
        row = next((r for r in rows if r["graph_id"] == graph_id), None)
        if row is None:
            raise ValueError(f"graph_id '{graph_id}' not found in env_index.csv.")
    elif selection == "fixed_index" and fixed_index is not None:
        i = int(fixed_index) % len(rows)
        row = rows[i]
    else:  # by_seed
        i = seed % len(rows)
        row = rows[i]

    gid = row["graph_id"]
    price_file = row["price_file"]

    graph_path  = GRAPHS_DIR / f"{gid}.json"
    prices_path = PRICES_DIR / price_file
    if not graph_path.exists():
        raise FileNotFoundError(f"Graph file not found: {graph_path}")
    if not prices_path.exists():
        raise FileNotFoundError(f"Prices file not found: {prices_path}")

    # Cargar grafo
    g = json.loads(graph_path.read_text(encoding="utf-8"))
    edges_raw = g.get("edges", [])
    production_graph: List[tuple[str, str]] = [tuple(e) for e in edges_raw]
    goods: List[str] = _unique_nodes_from_edges(production_graph)
    n_goods = len(goods)
    if n_goods == 0:
        raise ValueError(f"Graph '{gid}' contains no edges/nodes.")

    # Cargar precios (tensor N×3×3)
    p = json.loads(prices_path.read_text(encoding="utf-8"))
    matrices = np.asarray(p["matrices"], dtype=float)
    if matrices.ndim != 3 or matrices.shape[1:] != (3, 3):
        raise ValueError(f"Invalid price tensor shape {matrices.shape}. Expected (N_goods, 3, 3).")
    if matrices.shape[0] < n_goods:
        raise ValueError(
            f"Price tensor has {matrices.shape[0]} blocks but {n_goods} goods were found in graph '{gid}'."
        )
    elif matrices.shape[0] > n_goods:
        # Defensa: truncar exceso (puede ocurrir si N en JSON > únicos en edges)
        matrices = matrices[:n_goods, :, :]

    # Localizar chart_of_accounts.yaml subiendo en el repo
    accounts_yaml = None
    for parent in BASE_DIR.parents:
        cand = parent / "chart_of_accounts.yaml"
        if cand.exists():
            accounts_yaml = cand
            break
    if accounts_yaml is None:
        raise FileNotFoundError("chart_of_accounts.yaml not found in repository root or any parent folder.")

    # Agentes compatibles con downstream
    agents_information: Dict[str, Dict] = {}
    for name, pmap in (("MKT", 0), ("NCT", 1), ("ZF", 2)):
        agents_information[name] = {
            "type": name,
            "inventory_strategy": "FIFO",
            "firm_related_goods": list(goods),
            "income_statement_type": "standard",
            "accounts_yaml_path": str(accounts_yaml),
            "price_mapping": pmap,
        }

    env_meta = {
        "graph_id": gid,
        "price_file": price_file,
        "N": int(row.get("N", len(goods))),
        "density": float(row.get("density", 0.0)),
        "max_depth": int(row.get("max_depth", 0)),
    }

    return agents_information, production_graph, matrices, env_meta

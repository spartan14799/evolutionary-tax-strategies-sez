# tests/smoke_test_ga.py
# -*- coding: utf-8 -*-
"""
Smoke test for the three GA variants using the *real* module names:
  1) algorithms.ga.flat.run_ga_flat
  2) algorithms.ga.equivclass_exhaustive.run_ga_equivclass_exhaustive
  3) algorithms.ga.equivclass_joint.run_ga_equivclass_joint

It:
- Builds a tiny Economy instance (graph + price matrix).
- Infers a viable genome length for the flat GA by probing the Planner.
- Runs each GA with a small budget.
- Checks the hard constraint last_gene == 1.
- Prints a compact report.
"""

from __future__ import annotations
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from src.fine_tuning.wrappers.pso_wrapper import run_pso_w1
import numpy as np

# ---------------------------------------------------------------------------
# Repo root on sys.path
# ---------------------------------------------------------------------------
THIS_FILE = Path(__file__).resolve()
cur = THIS_FILE.parent
REPO_ROOT = cur
while cur != cur.parent:
    if (cur / "classes").exists() or (cur / "algorithms").exists() or \
       (cur / "pyproject.toml").exists() or (cur / "setup.cfg").exists():
        REPO_ROOT = cur
        break
    cur = cur.parent

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ---------------------------------------------------------------------------
# Domain imports
# ---------------------------------------------------------------------------
from src.simulation.economy.economy import Economy
from src.simulation.economy.production_process.production_graph import ProductionGraph
from src.simulation.economy.production_process.production_process import ProductionProcess
from src.simulation.planner.planner import Planner

# Real GA modules (no shims)
from src.algorithms.flat import run_ga_flat
from src.algorithms.equivclass_exhaustive import run_ga_equivclass_exhaustive
from src.algorithms.equivclass_joint import run_ga_equivclass_joint
from src.algorithms import common as ga_common  # for probes

# ---------------------------------------------------------------------------
# Problem instance (edit freely)
# ---------------------------------------------------------------------------
GRAPH_LINKS: List[Tuple[str, str]] = [
    ("A", "B"),
    ("B", "C"),
    ("B", "D"),
    ("B", "E"),
    ("C", "E"),
    ("D", "E"),
    ("E", "F"),
]

PRICE_MATRIX = np.array(
    [
        [[10, 1, 1], [11, 1, 1], [15, 1, 1]],
        [[ 2, 2, 2], [ 2, 2, 2], [ 2, 2, 2]],
        [[ 3, 3, 3], [ 3, 3, 3], [ 3, 3, 3]],
        [[ 4, 4, 4], [ 4, 4, 4], [ 4, 4, 4]],
        [[ 5, 5, 5], [ 5, 5, 5], [ 5, 5, 5]],
        [[ 6, 6, 6], [ 6, 6, 6], [ 6, 6, 6]],
    ],
    dtype=int,
)

GOODS = sorted({n for u, v in GRAPH_LINKS for n in (u, v)})
ACCOUNTS_PATH = REPO_ROOT / "chart_of_accounts.yaml"

AGENTS: Dict[str, Dict[str, Any]] = {
    "MKT": {
        "type": "MKT",
        "inventory_strategy": "FIFO",
        "firm_related_goods": [],
        "income_statement_type": "standard",
        "accounts_yaml_path": ACCOUNTS_PATH,
        "price_mapping": 0,
    },
    "NCT": {
        "type": "NCT",
        "inventory_strategy": "FIFO",
        "firm_related_goods": GOODS,
        "income_statement_type": "standard",
        "accounts_yaml_path": ACCOUNTS_PATH,
        "price_mapping": 1,
    },
    "ZF": {
        "type": "ZF",
        "inventory_strategy": "FIFO",
        "firm_related_goods": GOODS,
        "income_statement_type": "standard",
        "accounts_yaml_path": ACCOUNTS_PATH,
        "price_mapping": 2,
    },
}

# ---------------------------------------------------------------------------
# Planner probe: infer minimal viable genome length for the flat GA
# ---------------------------------------------------------------------------
def infer_min_genome_length(graph_edges: Sequence[Tuple[str, str]]) -> int:
    primary = ga_common.derive_primary_goods(graph_edges)
    base_L = max(2, len(primary) + 1)
    builder = ga_common.make_transactions_builder(graph_edges)
    return ga_common.calibrate_min_len_via_builder(builder, base_L=base_L)

# ---------------------------------------------------------------------------
# Pretty print helpers
# ---------------------------------------------------------------------------
def _short(v: np.ndarray, n: int = 24) -> str:
    v = np.asarray(v, dtype=int).tolist()
    if len(v) <= n:
        return str(v)
    return f"{v[:n]} ... (+{len(v)-n} more)"

def print_report(results: List[Dict[str, Any]]) -> None:
    print("\n==================== SMOKE TEST REPORT ====================")
    for r in results:
        name = r["name"]
        fit = r["best_utility"]
        genome = r["best_genome"]
        last = r["last_gene"]
        extra = r.get("extra", {})
        ok_last = (last == 1)
        print(f"\n[{name}]")
        print(f"  best_utility : {fit:.6f}")
        print(f"  best_genome  : { _short(genome) }")
        print(f"  last_gene==1 : {'OK' if ok_last else 'FAIL'} (value={last})")
        if extra:
            for k, v in extra.items():
                print(f"  {k:<12}: {v}")
    print("\n===========================================================\n")

# ---------------------------------------------------------------------------
# Runners (each returns a standard dict for unified reporting)
# ---------------------------------------------------------------------------
def run_flat_smoke() -> Dict[str, Any]:
    L = infer_min_genome_length(GRAPH_LINKS)
    out = run_ga_flat(
        production_graph=GRAPH_LINKS,
        pmatrix=PRICE_MATRIX,
        agents_information=AGENTS,
        genome_shape=L,
        generations=6,
        popsize=24,
        parents=12,
        mutation_rate=0.05,
        fix_last_gene=True,
        seed=123,
    )
    best_genome = np.asarray(out["best_genome"], dtype=int)
    best_utility = float(out["best_utility"])
    return {
        "name": "flat",
        "best_utility": best_utility,
        "best_genome": best_genome,
        "last_gene": int(best_genome[-1]) if best_genome.size else None,
        "extra": {"genome_shape": L, "gens": 6, "pop": 24},
    }

def run_exhaustive_smoke() -> Dict[str, Any]:
    out = run_ga_equivclass_exhaustive(
        production_graph=GRAPH_LINKS,
        pmatrix=PRICE_MATRIX,
        agents_information=AGENTS,
        mode="graph",
        generations=1,
        popsize=1,
        parents=9,
        mutation_rate=0.10,
        fix_last_gene=True,
        seed=123,
        max_combos=1,       # cap para smoke test
        per_good_cap=None,
        max_index_probe=3,
        verbosity=0,
        log_every=1,
        no_plots=True,
        top_combos_bars=10,
    )
    best_genome = np.asarray(out["best_genome"], dtype=int)
    best_utility = float(out["best_utility"])
    meta = out.get("meta", {})
    return {
        "name": "exhaustive+tail",
        "best_utility": best_utility,
        "best_genome": best_genome,
        "last_gene": int(best_genome[-1]) if best_genome.size else None,
        "extra": {"K": meta.get("K"), "L_used": meta.get("L_used"), "combo_space": meta.get("combo_space")},
    }

def run_joint_smoke() -> Dict[str, Any]:
    out = run_ga_equivclass_joint(
        production_graph=GRAPH_LINKS,
        pmatrix=PRICE_MATRIX,
        agents_information=AGENTS,
        mode="graph",
        generations=8,
        popsize=24,
        parents=10,
        sel_mutation=0.20,
        tail_mutation=0.05,
        per_good_cap=None,
        max_index_probe=8,
        fix_last_gene=True,
        seed=44,
        verbosity=1,
        log_every=2,
    )
    best_genome = np.asarray(out["best_genome"], dtype=int)
    best_utility = float(out["best_utility"])
    meta = out.get("meta", {})
    return {
        "name": "joint(selectors+tail)",
        "best_utility": best_utility,
        "best_genome": best_genome,
        "last_gene": int(best_genome[-1]) if best_genome.size else None,
        "extra": {"K": meta.get("K"), "L_used": meta.get("L_used"), "pool_sizes": meta.get("pool_sizes")},
    }



# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    results: List[Dict[str, Any]] = []

    try:
        results.append(run_flat_smoke())
    except Exception as e:
        print(f"[flat] ERROR: {e}")

    try:
        results.append(run_exhaustive_smoke())
    except Exception as e:
        print(f"[exhaustive+tail] ERROR: {e}")

    try:
        results.append(run_joint_smoke())
    except Exception as e:
        print(f"[joint] ERROR: {e}")

    if results:
        print_report(results)
    else:
        print("No results to report. All runs failed — please check the error messages above.")

# ============================
# helpers_ablation.py
# ============================
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# ----------------------------
# Resolve repository root
# ----------------------------
_THIS = Path.cwd()
REPO_ROOT = _THIS
cur = _THIS
while cur != cur.parent:
    if (cur / "classes").exists() or (cur / "algorithms").exists() or (cur / "pyproject.toml").exists():
        REPO_ROOT = cur
        break
    cur = cur.parent

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ----------------------------
# Visualization utilities
# ----------------------------
def topo_vertical_layout(
    G: nx.DiGraph,
    layer_gap: float = 1.0,
    node_gap: float = 1.6,
    bottom_to_top: bool = False,
    stagger_singletons: bool = True,
    singleton_dx: float = 0.8,
):
    """Positions nodes top→bottom by topological order."""
    if not nx.is_directed_acyclic_graph(G):
        return nx.spring_layout(G, seed=42)

    layers = list(nx.topological_generations(G))
    pos = {}
    L = len(layers)

    for li, layer in enumerate(layers):
        y = li * layer_gap if not bottom_to_top else -li * layer_gap
        n = len(layer)
        if n == 1 and stagger_singletons:
            x = (li - (L - 1) / 2.0) * singleton_dx
            pos[layer[0]] = (x, y)
        else:
            width = (n - 1) * node_gap
            x0 = -width / 2.0
            for j, node in enumerate(layer):
                pos[node] = (x0 + j * node_gap, y)
    return pos


def draw_dag_topo(G: nx.DiGraph, ax=None, title: Optional[str] = None):
    """Draw DAG with top→bottom orientation."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))

    pos = topo_vertical_layout(G, layer_gap=1.0, node_gap=1.6)
    nx.draw(
        G, pos, with_labels=True, node_size=600,
        node_color="#cfe8ff", edgecolors="#7aa7d7", linewidths=2.0,
        arrows=True, arrowstyle='-|>', arrowsize=15, ax=ax
    )
    ax.set_axis_off()
    if title:
        ax.set_title(title)
    return ax


# ----------------------------
# Data & agents utilities
# ----------------------------
def attach_goods_to_agents(agents: Dict[str, Dict[str, Any]], goods: List[str]) -> Dict[str, Dict[str, Any]]:
    """Attach full goods list to each agent."""
    out = {}
    for k, v in agents.items():
        d = dict(v)
        d["firm_related_goods"] = list(goods)
        out[k] = d
    return out


def describe_price_matrix(P: np.ndarray) -> pd.DataFrame:
    """Summarize price matrix."""
    A = np.asarray(P, dtype=float)
    return pd.DataFrame({
        "shape": [A.shape],
        "min": [float(A.min())],
        "max": [float(A.max())],
        "mean": [float(A.mean())],
        "dtype": [str(A.dtype)],
    })


# ----------------------------
# Build environments
# ----------------------------
def build_environment(method: str = "macro_micro", case: str = "Base") -> Tuple[List[Tuple[str, str]], np.ndarray, Dict[str, Any], Dict[str, Any]]:
    """
    Return (graph_links, price_matrix, agents, hyperparams)
    for a specific scenario.
    """
    # Graph: linear 3-stage production
    G_links = [("Input", "Intermediate 1"), ("Intermediate 1", "Intermediate 2"), ("Intermediate 2", "Final Good")]
    goods = sorted({n for u, v in G_links for n in (u, v)})

    # Agents
    ACCOUNTS_PATH = REPO_ROOT / "chart_of_accounts.yaml"
    BASE_AGENTS = {
        "MKT": {"type": "MKT", "inventory_strategy": "FIFO", "firm_related_goods": [],
                "income_statement_type": "standard", "accounts_yaml_path": ACCOUNTS_PATH, "price_mapping": 0},
        "NCT": {"type": "NCT", "inventory_strategy": "FIFO", "firm_related_goods": [],
                "income_statement_type": "standard", "accounts_yaml_path": ACCOUNTS_PATH, "price_mapping": 1},
        "ZF":  {"type": "ZF",  "inventory_strategy": "FIFO", "firm_related_goods": [],
                "income_statement_type": "standard", "accounts_yaml_path": ACCOUNTS_PATH, "price_mapping": 2},
    }

    AGENTS = attach_goods_to_agents(BASE_AGENTS, goods)

    # ------------------------
    # Price Matrices
    # ------------------------
    P_base = np.array([
    [[100,100,100],[100,100,100],[100,100,100]],
    [[101,101,101],[101,101,101],[101,101,101]],
    [[102,102,102],[102,102,102],[102,102,102]],
    [[105,105,105],[103,103,103],[103,103,103]]
        ])


    # Tax rate differential → FTZ has lower effective cost structure
    P_tax_diff = np.array([
    [[100,100,100],[100,100,100],[100,100,100]],
    [[101,101,101],[101,101,101],[101,101,101]],
    [[102,102,102],[102,102,102],[102,102,102]],
    [[105,105,105],[103,103,103],[103,103,103]]
        ])


    # Price differentiation → FTZ achieves higher selling prices
    P_price_diff = np.array([
    [[100,100,100],[100,100,100],[100,99,100]],
    [[101,101,101],[101,101,101],[101,100,101]],
    [[102,102,102],[102,102,102],[102,101,102]],
    [[105,105,105],[103,103,103],[103,102,103]]
        ])


    matrices = {
        "Base": P_base,
        "TaxDifferential": P_tax_diff,
        "PriceDifferentiation": P_price_diff,
    }

    if case not in matrices:
        raise ValueError(f"Unknown case '{case}'. Choose one of: {list(matrices.keys())}")

    P = matrices[case]

    # ------------------------
    # Hyperparameters
    # ------------------------
    hp = {
        "generations": 10,
        "popsize": 70,
        "parents": 50,
        "elite_fraction": 0.2,
        "tourn_size": 3,
        "lambda_in": 0.3,
        "lambda_out": 0.3,
        "p_macro": 1.0,
        "p_micro": 1.0,
        "sel_mutation": 0.05,
        "tail_mutation": 0.02,
        "fix_last_gene": True,
        "seed": 123,
        "verbosity": 1,
        "log_every": 1,
    }

    return G_links, P, AGENTS, hp


# ----------------------------
# Run and display scenario
# ----------------------------
def run_case(case: str = "Base"):
    """
    Execute and display the full GA workflow for the chosen scenario.
    Prints all genomes that achieved the best utility (ties included).
    """
    from src.algorithms.ga.macro_micro import run_ga_macro_micro
    import numpy as np

    # 1️⃣ Build environment
    G_links, P, AGENTS, hp = build_environment(method="macro_micro", case=case)

    # 2️⃣ Run GA
    print("\n Running Macro–Micro GA...\n")
    res = run_ga_macro_micro(
        production_graph=G_links,
        pmatrix=P,
        agents_information=AGENTS,
        **hp
    )

    # 3️⃣ Extract results
    best_utility = float(res.get("best_utility", 0.0))
    all_best = res.get("all_best_genomes", [])
    n_equiv = len(all_best)

    # 4️⃣ Print summary
    print("✅ Optimization completed.\n")
    print(f"  Scenario            : {case}")
    print(f"  Best utility        : {best_utility:.4f}")
    print(f"  N° equivalent optima: {n_equiv}\n")

    if n_equiv > 0:
        print("  Equivalent best genomes:")
        for i, g in enumerate(all_best, start=1):
            print(f"   {i:02d}. {g}")
    else:
        print("  (No equivalent genomes found — unique optimum.)")

    # Optional metadata
    if "meta" in res:
        print(f"\n  Meta info: {res['meta'].get('genome_internal', {})}")
    print("\nDone.\n")

    return res

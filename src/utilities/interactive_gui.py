# -*- coding: utf-8 -*-
"""
ga_gui_min.py — GA Workbench with integrated domain visualization.

Overview
--------
This GUI orchestrates three Genetic Algorithm variants (Flat, Joint, Exhaustive)
against the Economy domain and renders:
  • Live logs while a run is executing.
  • Fitness/diagnostic charts with a larger display area.
  • A consolidated "Best Solution — Graph & Reports" tab:
      - Production graph (canvas-based).
      - Utility, production plan (orders), and compact agent reports (NCT/ZF).
  • A dedicated "Ledgers" tab:
      - Entry lines for NCT and ZF with a detail pane.

It also supports evaluating a single explicit genome ("Run Genome") using the
same inputs (graph + price matrix) and updates the visualization tabs accordingly.

Implementation notes
--------------------
• The application imports existing package code only; no GA or Economy logic is
  duplicated. All calls delegate to:
    - algorithms.ga.flat.run_ga_flat
    - algorithms.ga.equivclass_joint.run_ga_equivclass_joint
    - algorithms.ga.equivclass_exhaustive.run_ga_equivclass_exhaustive
    - classes.economy.economy.Economy
• A background thread executes long-running computations. Stdout is redirected
  to a thread-safe queue and rendered in the Log panel without blocking the UI.
• Inputs for graph edges and price matrix are required and parsed via
  ast.literal_eval to accept simple Python-like literals.

Dependencies
------------
tkinter, numpy, networkx, matplotlib, pandas (for table conveniences).
"""

from __future__ import annotations

import ast
import io
import sys
import threading
import queue
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from tkinter.scrolledtext import ScrolledText

import numpy as np
import pandas as pd
import networkx as nx
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# -----------------------------------------------------------------------------
# Repository root detection and package imports
# -----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
while not (ROOT / "classes" / "economy" / "economy.py").exists() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.simulation.economy.economy import Economy  # noqa: E402
from src.algorithms.ga import (  # noqa: E402
    run_ga_flat, run_ga_equivclass_joint, run_ga_equivclass_exhaustive
    )
from src.algorithms.ga.common import detect_prefix_layout_and_sizes  # noqa: E402


# -----------------------------------------------------------------------------
# Input parsing and small utilities
# -----------------------------------------------------------------------------
def safe_literal(text: str, what: str):
    """Parses a Python literal safely; raises a ValueError with context on failure."""
    try:
        return ast.literal_eval(text)
    except Exception as exc:
        raise ValueError(f"{what} is not a valid Python literal.\nDetails: {exc}") from exc


def build_graph_from_text(edges_text: str) -> nx.DiGraph:
    """Constructs a DiGraph from a literal list of (u, v) pairs."""
    edges = safe_literal(edges_text, "Production graph edges")
    G = nx.DiGraph()
    G.add_edges_from((str(u), str(v)) for (u, v) in edges)
    return G


def build_agents_for_goods(goods: List[str]) -> Dict[str, Dict[str, Any]]:
    """Builds agents_information for MKT/NCT/ZF using a shared chart of accounts."""
    accounts_path = ROOT / "chart_of_accounts.yaml"
    return {
        "MKT": {
            "type": "MKT",
            "inventory_strategy": "FIFO",
            "firm_related_goods": list(goods),
            "income_statement_type": "standard",
            "accounts_yaml_path": accounts_path,
            "price_mapping": 0,
        },
        "NCT": {
            "type": "NCT",
            "inventory_strategy": "FIFO",
            "firm_related_goods": list(goods),
            "income_statement_type": "standard",
            "accounts_yaml_path": accounts_path,
            "price_mapping": 1,
        },
        "ZF": {
            "type": "ZF",
            "inventory_strategy": "FIFO",
            "firm_related_goods": list(goods),
            "income_statement_type": "standard",
            "accounts_yaml_path": accounts_path,
            "price_mapping": 2,
        },
    }


class LiveStdout:
    """Streams line-buffered stdout to a queue for safe insertion into the Tk text widget."""
    def __init__(self, q: queue.Queue):
        self.q = q
        self._buf = io.StringIO()

    def write(self, s: str):
        self._buf.write(s)
        if s.endswith("\n"):
            self.q.put(self._buf.getvalue())
            self._buf = io.StringIO()

    def flush(self):
        pass


# -----------------------------------------------------------------------------
# Graph canvas (compact and legible)
# -----------------------------------------------------------------------------
def _topo_vertical_layout(
    G: nx.DiGraph,
    layer_gap: float = 1.0,
    node_gap: float = 1.6,
    bottom_to_top: bool = False,
    stagger_singletons: bool = True,
    singleton_dx: float = 0.8,
):
    """Computes a readable vertical topological layout for DAGs; falls back to spring for cyclic graphs."""
    import networkx as nx
    if not nx.is_directed_acyclic_graph(G):
        return nx.spring_layout(G, seed=42)

    layers = list(nx.topological_generations(G))
    pos: Dict[Any, Tuple[float, float]] = {}
    L = len(layers)
    for li, layer in enumerate(layers):
        y = -li * layer_gap if bottom_to_top else li * layer_gap
        n = max(1, len(layer))
        if n == 1 and stagger_singletons:
            x = (li - (L - 1) / 2.0) * singleton_dx
            pos[layer[0]] = (x, y)
        else:
            width = (n - 1) * node_gap
            x0 = -width / 2.0
            for j, node in enumerate(layer):
                pos[node] = (x0 + j * node_gap, y)
    return pos


class GraphView(ttk.Frame):
    """Owns a canvas and renders a directed production graph with curved arrows and labeled nodes."""

    def __init__(self, parent: tk.Widget):
        super().__init__(parent)
        self.rowconfigure(1, weight=1)
        self.columnconfigure(0, weight=1)

        bar = ttk.Frame(self)
        bar.grid(row=0, column=0, sticky="ew")
        ttk.Label(bar, text="Production Graph", font=("Arial", 12, "bold")).pack(side="left", padx=(6, 0))
        ttk.Button(bar, text="Fit", command=self.fit_to_view).pack(side="right", padx=3)
        ttk.Button(bar, text="+", width=3, command=lambda: self.zoom(1.25)).pack(side="right", padx=1)
        ttk.Button(bar, text="−", width=3, command=lambda: self.zoom(0.8)).pack(side="right", padx=1)

        self.canvas = tk.Canvas(self, bg="white", highlightthickness=1, highlightbackground="#bbb")
        self.canvas.grid(row=1, column=0, sticky="nsew")

        self.graph: Optional[nx.DiGraph] = None
        self.pos: Dict = {}
        self.scale: float = 1.0
        self.tx: float = 0.0
        self.ty: float = 0.0
        self.bbox_world: Optional[Tuple[float, float, float, float]] = None
        self.node_radius = 14

        self.canvas.bind("<Configure>", lambda e: self._on_resize())
        self.canvas.bind("<ButtonPress-1>", self._on_start_pan)
        self.canvas.bind("<B1-Motion>", self._on_pan)
        self.canvas.bind("<MouseWheel>", self._on_wheel)       # Windows/macOS
        self.canvas.bind("<Button-4>", lambda e: self.zoom(1.1))  # Linux
        self.canvas.bind("<Button-5>", lambda e: self.zoom(0.9))

        self._pan_last = None

    def set_graph(self, G: nx.DiGraph):
        self.graph = G
        self.pos = _topo_vertical_layout(G)
        self._compute_bbox_world()
        self.fit_to_view()

    def fit_to_view(self):
        if not self.graph or not self.bbox_world:
            self._clear()
            return
        w = max(self.canvas.winfo_width(), 1)
        h = max(self.canvas.winfo_height(), 1)
        x0, x1, y0, y1 = self.bbox_world
        world_w = max(x1 - x0, 1e-6)
        world_h = max(y1 - y0, 1e-6)
        padding = 0.08
        sx = (1 - 2 * padding) * w / world_w
        sy = (1 - 2 * padding) * h / world_h
        self.scale = min(sx, sy)

        cx_world = (x0 + x1) / 2
        cy_world = (y0 + y1) / 2
        cx_screen = w / 2
        cy_screen = h / 2

        self.tx = cx_screen - self.scale * cx_world
        self.ty = cy_screen - self.scale * cy_world
        self.render()

    def zoom(self, factor: float):
        self.scale *= factor
        self.render()

    def _compute_bbox_world(self):
        xs = [p[0] for p in self.pos.values()]
        ys = [p[1] for p in self.pos.values()]
        if not xs or not ys:
            self.bbox_world = None
            return
        x0, x1 = min(xs), max(xs)
        y0, y1 = min(ys), max(ys)
        pad_x = max((x1 - x0) * 0.12, 0.2)
        pad_y = max((y1 - y0) * 0.12, 0.2)
        self.bbox_world = (x0 - pad_x, x1 + pad_x, y0 - pad_y, y1 + pad_y)

    def _on_resize(self):
        if self.graph is None:
            self._clear()
        else:
            self.render()

    def _on_start_pan(self, event):
        self._pan_last = (event.x, event.y)

    def _on_pan(self, event):
        if self._pan_last is None:
            return
        dx = event.x - self._pan_last[0]
        dy = event.y - self._pan_last[1]
        self.tx += dx
        self.ty += dy
        self._pan_last = (event.x, event.y)
        self.render()

    def _on_wheel(self, event):
        factor = 1.1 if event.delta > 0 else 0.9
        px, py = event.x, event.y
        wx, wy = self._screen_to_world(px, py)
        self.scale *= factor
        sx, sy = self._world_to_screen(wx, wy)
        self.tx += (px - sx)
        self.ty += (py - sy)
        self.render()

    def _world_to_screen(self, x: float, y: float) -> Tuple[float, float]:
        return (self.scale * x + self.tx, self.scale * y + self.ty)

    def _screen_to_world(self, sx: float, sy: float) -> Tuple[float, float]:
        return ((sx - self.tx) / self.scale, (sy - self.ty) / self.scale)

    def _clear(self):
        self.canvas.delete("all")

    def render(self):
        self.canvas.delete("all")
        if not self.graph:
            return

        r = self.node_radius
        edge_sep_px = 16.0
        edge_color = "#555555"
        edge_width = 1.4
        arrowshape = (10, 12, 4)

        for u in self.graph.nodes():
            targets = list(self.graph.successors(u))
            k = len(targets)
            for idx, v in enumerate(targets):
                x0, y0 = self.pos[u]
                x1, y1 = self.pos[v]
                sx0, sy0 = self._world_to_screen(x0, y0)
                sx1, sy1 = self._world_to_screen(x1, y1)
                midx = (sx0 + sx1) / 2.0 + (idx - (k - 1) / 2.0) * edge_sep_px
                midy = (sy0 + sy1) / 2.0
                self.canvas.create_line(
                    sx0, sy0, midx, midy, sx1, sy1,
                    smooth=True, splinesteps=12,
                    arrow=tk.LAST, arrowshape=arrowshape,
                    width=edge_width, fill=edge_color
                )

        for n, (x, y) in self.pos.items():
            sx, sy = self._world_to_screen(x, y)
            self.canvas.create_oval(
                sx - r, sy - r, sx + r, sy + r, fill="#cfe8ff", outline="#7aa7d7", width=2
            )
            self.canvas.create_text(sx, sy, text=str(n), font=("Arial", 11, "bold"), fill="#274b6d")


# -----------------------------------------------------------------------------
# Table helpers (Treeview + DataFrame convenience)
# -----------------------------------------------------------------------------
def make_treeview(parent: tk.Widget, height: int = 16) -> ttk.Treeview:
    """Creates a Treeview with scrollbars and standard sizing inside a single grid cell."""
    frame = ttk.Frame(parent)
    frame.grid(row=0, column=0, sticky="nsew")
    parent.rowconfigure(0, weight=1)
    parent.columnconfigure(0, weight=1)

    tree = ttk.Treeview(frame, show="headings", height=height)
    s_x = ttk.Scrollbar(frame, orient="horizontal", command=tree.xview)
    s_y = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
    tree.configure(xscrollcommand=s_x.set, yscrollcommand=s_y.set)

    tree.grid(row=0, column=0, sticky="nsew")
    s_y.grid(row=0, column=1, sticky="ns")
    s_x.grid(row=1, column=0, sticky="ew")

    frame.rowconfigure(0, weight=1)
    frame.columnconfigure(0, weight=1)
    return tree


def fill_treeview(tree: ttk.Treeview, df: pd.DataFrame, column_prefs: Optional[Dict[str, Dict]] = None) -> None:
    """Clears and fills a Treeview from a DataFrame using per-column sizing prefs."""
    column_prefs = column_prefs or {}
    for col in tree["columns"] if "columns" in tree.configure() else []:
        tree.heading(col, text="")
    tree.delete(*tree.get_children())
    cols = list(df.columns)
    tree["columns"] = cols
    for col in cols:
        prefs = column_prefs.get(col, {})
        width = int(prefs.get("width", 160))
        anchor = prefs.get("anchor", "w")
        stretch = bool(prefs.get("stretch", True))
        tree.heading(col, text=col)
        tree.column(col, anchor=anchor, stretch=stretch, width=width)
    for _, row in df.iterrows():
        tree.insert("", "end", values=[str(v) for v in row.tolist()])


# -----------------------------------------------------------------------------
# Main Application
# -----------------------------------------------------------------------------
class GAWorkbench(tk.Tk):
    """Owns the complete UI, orchestrates runs, and renders analyses and domain outputs."""

    def __init__(self):
        super().__init__()
        self.title("GA Workbench — Flat | Joint | Exhaustive")
        self.geometry("1400x900")
        self.minsize(1200, 800)

        # Internal state
        self._runner_thread: Optional[threading.Thread] = None
        self._log_q: queue.Queue = queue.Queue()
        self._orig_stdout = sys.stdout
        self._last_method: Optional[str] = None
        self._last_result: Optional[Dict[str, Any]] = None
        self._last_domain: Optional[Dict[str, Any]] = None  # {"G":..., "prices":..., "agents":...}
        self._last_best_genome: Optional[np.ndarray] = None

        self._build_ui()
        self.after(120, self._drain_logs)

    # -- UI Layout ------------------------------------------------------------
    def _build_ui(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("Treeview", rowheight=22)
        style.configure("TLabelframe.Label", font=("Arial", 12, "bold"))
        style.configure("TLabel", font=("Arial", 10))
        style.configure("TNotebook.Tab", padding=(12, 8))

        # Top: Inputs (Graph, Prices, Genome)
        top = ttk.Frame(self)
        top.pack(side="top", fill="x", padx=10, pady=8)

        ttk.Label(top, text="Production Graph (edges):").grid(row=0, column=0, sticky="w")
        self.txt_edges = ScrolledText(top, height=4, wrap="none")
        self.txt_edges.grid(row=1, column=0, sticky="nsew", padx=(0, 8))

        ttk.Button(top, text="Load YAML…", command=self._load_edges_from_yaml).grid(row=1, column=1, sticky="nsw", padx=(0, 6))

        ttk.Label(top, text="Price Matrix:").grid(row=0, column=2, sticky="w")
        self.txt_prices = ScrolledText(top, height=4, wrap="none")
        self.txt_prices.grid(row=1, column=2, sticky="nsew", padx=(0, 8))

        ttk.Label(top, text="Genome (optional, to evaluate directly):").grid(row=0, column=3, sticky="w")
        self.txt_genome = ScrolledText(top, height=4, wrap="none")
        self.txt_genome.grid(row=1, column=3, sticky="nsew", padx=(0, 8))
        ttk.Button(top, text="Run Genome", command=self._run_genome_only).grid(row=1, column=4, sticky="nsw")

        for c in (0, 2, 3):
            top.columnconfigure(c, weight=1)

        # Notebook for GA params and analytics
        self.nb = ttk.Notebook(self)
        self.nb.pack(side="top", fill="x", padx=10)

        self._build_tab_flat()
        self._build_tab_joint()
        self._build_tab_exhaustive()

        # Middle: Charts (wider) + Log + Maximizers
        center = ttk.Panedwindow(self, orient=tk.HORIZONTAL)
        center.pack(fill="both", expand=True, padx=10, pady=8)

        charts = ttk.Labelframe(center, text="Charts")
        center.add(charts, weight=4)  # more space for charts

        sel_frame = ttk.Frame(charts)
        sel_frame.pack(fill="x", padx=6, pady=(6, 0))
        ttk.Label(sel_frame, text="Chart:").pack(side="left")
        self.chart_var = tk.StringVar(value="Fitness curves")
        self.chart_combo = ttk.Combobox(
            sel_frame, textvariable=self.chart_var, state="readonly",
            values=[
                "Fitness curves",
                "Best improvement (Δbest)",
                "Best genome (bits)",
                "Best-by-selectors (Joint)",
                "Global curves (Exhaustive)",
            ]
        )
        self.chart_combo.pack(side="left", padx=6)
        ttk.Button(sel_frame, text="Render", command=self._render_chart).pack(side="left")

        self.fig = Figure(figsize=(7.2, 4.6), dpi=100)  # larger canvas
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlabel("Generation")
        self.ax.set_ylabel("Utility")
        self.canvas = FigureCanvasTkAgg(self.fig, master=charts)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=6, pady=6)

        right = ttk.Panedwindow(center, orient=tk.VERTICAL)
        center.add(right, weight=2)

        log_frame = ttk.Labelframe(right, text="Log")
        right.add(log_frame, weight=2)
        self.txt_log = ScrolledText(log_frame, height=10)
        self.txt_log.pack(fill="both", expand=True, padx=6, pady=6)

        # ---- Maximizers panel (with best utility header + utility column)
        best_frame = ttk.Labelframe(right, text="Maximizers (ties)")
        right.add(best_frame, weight=3)

        header = ttk.Frame(best_frame)
        header.pack(fill="x", padx=6, pady=(6, 0))
        ttk.Label(header, text="Best utility (this run):").pack(side="left")
        self.best_u_var = tk.StringVar(value="—")
        ttk.Label(header, textvariable=self.best_u_var, font=("Arial", 11, "bold")).pack(side="left", padx=(6, 0))

        self.best_tree = ttk.Treeview(best_frame, show="headings", height=10)
        self.best_tree["columns"] = ("#", "len", "utility", "genome")
        self.best_tree.heading("#", text="#");              self.best_tree.column("#", width=60, anchor="e", stretch=False)
        self.best_tree.heading("len", text="LEN");          self.best_tree.column("len", width=60, anchor="e", stretch=False)
        self.best_tree.heading("utility", text="UTILITY");  self.best_tree.column("utility", width=120, anchor="e", stretch=False)
        self.best_tree.heading("genome", text="GENOME");    self.best_tree.column("genome", width=800, anchor="w", stretch=True)
        self.best_tree.pack(fill="both", expand=True, padx=6, pady=6)

        # Small toolbar to copy genomes from the maximizers table
        btns = ttk.Frame(best_frame)
        btns.pack(fill="x", padx=6, pady=(0, 6))

        # Copy only the currently selected genome row
        ttk.Button(btns, text="Copy selected", width=14,
                   command=self._copy_selected_maximizer).pack(side="left")

        # Copy all maximizing genomes as newline-separated lists
        ttk.Button(btns, text="Copy all", width=10,
                   command=self._copy_all_maximizers).pack(side="left", padx=(6, 0))

        # Optional: keyboard shortcuts for convenience (Ctrl/Cmd + C)
        self.best_tree.bind("<Control-c>", lambda e: self._copy_selected_maximizer())
        self.best_tree.bind("<Command-c>", lambda e: self._copy_selected_maximizer())  # macOS

        # Bottom: Meta
        meta = ttk.Labelframe(self, text="Meta")
        meta.pack(fill="x", padx=10, pady=(0, 10))
        self.meta_var = tk.StringVar(value="—")
        ttk.Label(meta, textvariable=self.meta_var, anchor="w").pack(fill="x", padx=6, pady=6)

        # Domain Visualization tabs
        self._build_results_tabs()

    # -- GA Parameter Tabs ----------------------------------------------------
    def _build_tab_flat(self):
        tab = ttk.Frame(self.nb)
        self.nb.add(tab, text="Flat")
        items = [
            ("generations", tk.IntVar(value=15)),
            ("popsize", tk.IntVar(value=30)),
            ("parents", tk.IntVar(value=20)),
            ("mutation_rate", tk.DoubleVar(value=0.05)),
            ("fix_last_gene", tk.BooleanVar(value=True)),
            ("seed", tk.IntVar(value=42)),
        ]
        self.flat_params = {k: v for k, v in items}
        for i, (k, var) in enumerate(items):
            ttk.Label(tab, text=k).grid(row=i, column=0, sticky="w", padx=6, pady=2)
            if isinstance(var, tk.BooleanVar):
                ttk.Checkbutton(tab, variable=var).grid(row=i, column=1, sticky="w")
            else:
                ttk.Entry(tab, textvariable=var, width=12).grid(row=i, column=1, sticky="w")
        ttk.Button(tab, text="Run Flat", command=self._run_flat).grid(row=len(items), column=0, columnspan=2, pady=6)

    def _build_tab_joint(self):
        tab = ttk.Frame(self.nb)
        self.nb.add(tab, text="Joint")
        items = [
            ("generations", tk.IntVar(value=20)),
            ("popsize", tk.IntVar(value=30)),
            ("parents", tk.IntVar(value=15)),
            ("sel_mutation", tk.DoubleVar(value=0.25)),
            ("tail_mutation", tk.DoubleVar(value=0.05)),
            ("per_good_cap", tk.StringVar(value="")),
            ("max_index_probe", tk.IntVar(value=3)),
            ("fix_last_gene", tk.BooleanVar(value=True)),
            ("seed", tk.IntVar(value=44)),
            ("verbosity", tk.IntVar(value=1)),
            ("log_every", tk.IntVar(value=1)),
        ]
        self.joint_params = {k: v for k, v in items}
        for i, (k, var) in enumerate(items):
            ttk.Label(tab, text=k).grid(row=i, column=0, sticky="w", padx=6, pady=2)
            if isinstance(var, tk.BooleanVar):
                ttk.Checkbutton(tab, variable=var).grid(row=i, column=1, sticky="w")
            else:
                ttk.Entry(tab, textvariable=var, width=12).grid(row=i, column=1, sticky="w")
        ttk.Button(tab, text="Run Joint", command=self._run_joint).grid(row=len(items), column=0, columnspan=2, pady=6)

    def _build_tab_exhaustive(self):
        tab = ttk.Frame(self.nb)
        self.nb.add(tab, text="Exhaustive-by-class")
        items = [
            ("generations", tk.IntVar(value=10)),
            ("popsize", tk.IntVar(value=20)),
            ("parents", tk.IntVar(value=10)),
            ("mutation_rate", tk.DoubleVar(value=0.10)),
            ("fix_last_gene", tk.BooleanVar(value=True)),
            ("seed", tk.IntVar(value=40)),
            ("max_combos", tk.StringVar(value="")),
            ("per_good_cap", tk.StringVar(value="")),
            ("max_index_probe", tk.IntVar(value=3)),
            ("verbosity", tk.IntVar(value=1)),
            ("log_every", tk.IntVar(value=1)),
            ("max_best_returned", tk.StringVar(value="")),
        ]
        self.ex_params = {k: v for k, v in items}
        for i, (k, var) in enumerate(items):
            ttk.Label(tab, text=k).grid(row=i, column=0, sticky="w", padx=6, pady=2)
            if isinstance(var, tk.BooleanVar):
                ttk.Checkbutton(tab, variable=var).grid(row=i, column=1, sticky="w")
            else:
                ttk.Entry(tab, textvariable=var, width=12).grid(row=i, column=1, sticky="w")
        ttk.Button(tab, text="Run Exhaustive", command=self._run_exhaustive).grid(row=len(items), column=0, columnspan=2, pady=6)

    # -- Domain Visualization Tabs -------------------------------------------
    def _build_results_tabs(self):
        self.results_nb = ttk.Notebook(self)
        self.results_nb.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        # Tab 1: Best Solution — Graph & Reports
        self.tab_best = ttk.Frame(self.results_nb)
        self.results_nb.add(self.tab_best, text="Best Solution — Graph & Reports")

        self.tab_best.rowconfigure(0, weight=1)
        self.tab_best.columnconfigure(0, weight=1)

        best_split = ttk.Panedwindow(self.tab_best, orient=tk.HORIZONTAL)
        best_split.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)

        # Left: Graph
        graph_frame = ttk.Frame(best_split)
        graph_frame.rowconfigure(0, weight=1)
        graph_frame.columnconfigure(0, weight=1)
        self.graph_view = GraphView(graph_frame)
        self.graph_view.grid(row=0, column=0, sticky="nsew")
        best_split.add(graph_frame, weight=1)

        # Right: Reports + Plan + Utility
        right_v = ttk.Panedwindow(best_split, orient=tk.VERTICAL)
        best_split.add(right_v, weight=2)

        reports_split = ttk.Panedwindow(right_v, orient=tk.HORIZONTAL)

        nct_lf = ttk.Labelframe(reports_split, text="NCT Report")
        nct_lf.rowconfigure(0, weight=1)
        nct_lf.columnconfigure(0, weight=1)
        self.nct_tree = make_treeview(nct_lf)
        reports_split.add(nct_lf, weight=1)

        zf_lf = ttk.Labelframe(reports_split, text="ZF Report")
        zf_lf.rowconfigure(0, weight=1)
        zf_lf.columnconfigure(0, weight=1)
        self.zf_tree = make_treeview(zf_lf)
        reports_split.add(zf_lf, weight=1)

        right_v.add(reports_split, weight=3)

        plan_utility_frame = ttk.Frame(right_v)
        plan_utility_frame.rowconfigure(0, weight=1)
        plan_utility_frame.columnconfigure(0, weight=4)
        plan_utility_frame.columnconfigure(1, weight=1)

        plan_lf = ttk.Labelframe(plan_utility_frame, text="Production Plan")
        plan_lf.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        plan_lf.rowconfigure(0, weight=1)
        plan_lf.columnconfigure(0, weight=1)
        self.plan_tree = make_treeview(plan_lf, height=12)

        utility_lf = ttk.Labelframe(plan_utility_frame, text="Utility")
        utility_lf.grid(row=0, column=1, sticky="nsew")
        self.utility_var = tk.StringVar(value="—")
        utility_label = ttk.Label(utility_lf, textvariable=self.utility_var, anchor="center", font=("Arial", 16, "bold"))
        utility_label.pack(expand=True, fill="both", padx=10, pady=10)

        right_v.add(plan_utility_frame, weight=2)

        # Tab 2: Ledgers
        self.tab_ledgers = ttk.Frame(self.results_nb)
        self.results_nb.add(self.tab_ledgers, text="Ledgers")
        self.tab_ledgers.rowconfigure(0, weight=1)
        self.tab_ledgers.columnconfigure(0, weight=1)

        ledger_nb = ttk.Notebook(self.tab_ledgers)
        ledger_nb.grid(row=0, column=0, sticky="nsew", padx=6, pady=6)

        # NCT
        nct_tab = ttk.Frame(ledger_nb)
        ledger_nb.add(nct_tab, text="NCT")
        nct_tab.rowconfigure(0, weight=1)
        nct_tab.columnconfigure(0, weight=1)

        nct_split = ttk.Panedwindow(nct_tab, orient=tk.HORIZONTAL)
        nct_split.grid(row=0, column=0, sticky="nsew")

        nct_table_lf = ttk.Labelframe(nct_split, text="NCT Ledger Entries")
        nct_table_lf.rowconfigure(0, weight=1)
        nct_table_lf.columnconfigure(0, weight=1)
        self.nct_ledger_tree = make_treeview(nct_table_lf, height=25)
        nct_split.add(nct_table_lf, weight=2)

        nct_detail_lf = ttk.Labelframe(nct_split, text="Detail")
        nct_detail_lf.rowconfigure(0, weight=1)
        nct_detail_lf.columnconfigure(0, weight=1)
        self.nct_detail_text = ScrolledText(nct_detail_lf, wrap="word")
        self.nct_detail_text.grid(row=0, column=0, sticky="nsew", padx=6, pady=6)
        self.nct_detail_text.config(state="disabled")
        nct_copy = ttk.Button(nct_detail_lf, text="Copy", command=lambda: self._copy_detail(self.nct_detail_text))
        nct_copy.grid(row=1, column=0, sticky="e", padx=6, pady=(0, 6))
        nct_split.add(nct_detail_lf, weight=3)

        # ZF
        zf_tab = ttk.Frame(ledger_nb)
        ledger_nb.add(zf_tab, text="ZF")
        zf_tab.rowconfigure(0, weight=1)
        zf_tab.columnconfigure(0, weight=1)

        zf_split = ttk.Panedwindow(zf_tab, orient=tk.HORIZONTAL)
        zf_split.grid(row=0, column=0, sticky="nsew")

        zf_table_lf = ttk.Labelframe(zf_split, text="ZF Ledger Entries")
        zf_table_lf.rowconfigure(0, weight=1)
        zf_table_lf.columnconfigure(0, weight=1)
        self.zf_ledger_tree = make_treeview(zf_table_lf, height=25)
        zf_split.add(zf_table_lf, weight=2)

        zf_detail_lf = ttk.Labelframe(zf_split, text="Detail")
        zf_detail_lf.rowconfigure(0, weight=1)
        zf_detail_lf.columnconfigure(0, weight=1)
        self.zf_detail_text = ScrolledText(zf_detail_lf, wrap="word")
        self.zf_detail_text.grid(row=0, column=0, sticky="nsew", padx=6, pady=6)
        self.zf_detail_text.config(state="disabled")
        zf_copy = ttk.Button(zf_detail_lf, text="Copy", command=lambda: self._copy_detail(self.zf_detail_text))
        zf_copy.grid(row=1, column=0, sticky="e", padx=6, pady=(0, 6))
        zf_split.add(zf_detail_lf, weight=3)

        # Selection handlers
        self.nct_ledger_tree.bind("<<TreeviewSelect>>", lambda e: self._on_ledger_select(e, self.nct_ledger_tree, self.nct_detail_text))
        self.zf_ledger_tree.bind("<<TreeviewSelect>>", lambda e: self._on_ledger_select(e, self.zf_ledger_tree, self.zf_detail_text))

    # -- Input helpers --------------------------------------------------------
    def _load_edges_from_yaml(self):
        path = filedialog.askopenfilename(filetypes=[("YAML", "*.yml *.yaml"), ("All files", "*.*")])
        if not path:
            return
        import yaml
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        edges = data.get("edges", [])
        self.txt_edges.delete("1.0", tk.END)
        self.txt_edges.insert("1.0", str(edges))

    def _prepare_domain_inputs(self) -> Tuple[nx.DiGraph, np.ndarray, Dict[str, Any]]:
        """Assembles graph, price matrix, and agents_information from UI controls; raises on invalid inputs."""
        G = build_graph_from_text(self.txt_edges.get("1.0", tk.END))
        prices = np.array(safe_literal(self.txt_prices.get("1.0", tk.END), "Price matrix"), dtype=float)
        goods = list(G.nodes())
        agents = build_agents_for_goods(goods)
        return G, prices, agents

    def _lock_inputs(self, locked: bool):
        state = "disabled" if locked else "normal"
        for w in (self.nb, self.txt_edges, self.txt_prices, self.txt_genome):
            try:
                w.configure(state=state)
            except tk.TclError:
                pass

    def _start_background(self, target, *args, **kwargs):
        """Starts a background worker thread, wires stdout to the live log, and clears presentation state."""
        if self._runner_thread and self._runner_thread.is_alive():
            messagebox.showwarning("Running", "A run is already in progress.")
            return

        # Clear chart, meta, logs, and best list
        self.ax.clear()
        self.ax.set_xlabel("Generation")
        self.ax.set_ylabel("Utility")
        self.canvas.draw()
        for item in self.best_tree.get_children():
            self.best_tree.delete(item)
        self.meta_var.set("—")
        self.txt_log.delete("1.0", tk.END)

        sys.stdout = LiveStdout(self._log_q)
        self._lock_inputs(True)

        t = threading.Thread(target=self._run_wrapper, args=(target, args, kwargs), daemon=True)
        self._runner_thread = t
        t.start()

    def _run_wrapper(self, target, args, kwargs):
        try:
            method, result, domain = target(*args, **kwargs)
            self._last_method = method
            self._last_result = result
            self._last_domain = domain
            self._last_best_genome = np.array(result.get("best_genome", []), dtype=int) if method != "genome" else np.array(domain["genome"], dtype=int)
            self.after(0, lambda: self._render_after_run(method, result, domain))
        except Exception as exc:
            self.after(0, lambda e=exc: messagebox.showerror("Execution error", f"{type(e).__name__}: {e}"))
        finally:
            sys.stdout = self._orig_stdout
            self.after(0, lambda: self._lock_inputs(False))


    def _drain_logs(self):
        """Flushes queued stdout chunks to the log area periodically."""
        try:
            while True:
                chunk = self._log_q.get_nowait()
                self.txt_log.insert(tk.END, chunk)
                self.txt_log.see(tk.END)
        except queue.Empty:
            pass
        self.after(120, self._drain_logs)

    # -- Ledger helpers -------------------------------------------------------
    def _on_ledger_select(self, _event, tree: ttk.Treeview, detail_text: ScrolledText):
        sel = tree.selection()
        if not sel:
            return
        values = tree.item(sel[0], "values")
        full_entry = values[1] if len(values) > 1 else ""
        detail_text.config(state="normal")
        detail_text.delete("1.0", tk.END)
        detail_text.insert("1.0", full_entry)
        detail_text.config(state="disabled")

    def _copy_detail(self, detail_text: ScrolledText):
        self.clipboard_clear()
        self.clipboard_append(detail_text.get("1.0", tk.END))
        self.update()

    def _copy_selected_maximizer(self):
        """
        Copies the genome text of the currently selected row in the maximizers table.
        This avoids text selection inside cells (Treeview does not support it).
        """
        sel = self.best_tree.selection()
        if not sel:
            return
        vals = self.best_tree.item(sel[0], "values")
        genome_txt = vals[3] if len(vals) > 3 else ""
        self.clipboard_clear()
        self.clipboard_append(str(genome_txt).strip())
        self.update()

    def _copy_all_maximizers(self):
        """
        Copies all maximizing genomes (ties) to the clipboard, one per line.
        If the GA did not provide 'all_best_genomes', it falls back to 'best_genome'.
        """
        res = self._last_result or {}
        all_best = res.get("all_best_genomes") or []
        if not all_best:
            best = res.get("best_genome")
            if best is not None:
                all_best = [best]

        # Normalize to simple Python lists of ints and render as newline-separated rows.
        out_lines = []
        for g in all_best:
            try:
                arr = np.asarray(g, dtype=int).tolist()
            except Exception:
                arr = g
            out_lines.append(str(arr))

        txt = "\n".join(out_lines) if out_lines else ""
        self.clipboard_clear()
        self.clipboard_append(txt)
        self.update()


    # -- GA handlers ----------------------------------------------------------
    def _run_flat(self):
        try:
            G, prices, agents = self._prepare_domain_inputs()
            # Determine a viable genome length via Planner detection (used only for flat GA shape).
            labels, sizes, _idx, _txb, L_min, _info = detect_prefix_layout_and_sizes(G, mode="graph")
            K = int(sum(sizes))
            L_used = max(int(L_min), K + 1)
        except Exception as exc:
            messagebox.showerror("Invalid inputs", str(exc))
            return

        def job():
            res = run_ga_flat(
                production_graph=G,
                pmatrix=prices,
                agents_information=agents,
                genome_shape=L_used,
                generations=int(self.flat_params["generations"].get()),
                popsize=int(self.flat_params["popsize"].get()),
                parents=int(self.flat_params["parents"].get()),
                mutation_rate=float(self.flat_params["mutation_rate"].get()),
                fix_last_gene=bool(self.flat_params["fix_last_gene"].get()),
                seed=int(self.flat_params["seed"].get()),
            )
            domain = {"G": G, "prices": prices, "agents": agents}
            return ("flat", res, domain)

        self._start_background(job)

    def _run_joint(self):
        try:
            G, prices, agents = self._prepare_domain_inputs()
        except Exception as exc:
            messagebox.showerror("Invalid inputs", str(exc))
            return

        def to_int_or_none(s: str) -> Optional[int]:
            s = (s or "").strip()
            return int(s) if s.isdigit() else None

        def job():
            res = run_ga_equivclass_joint(
                production_graph=G,
                pmatrix=prices,
                agents_information=agents,
                generations=int(self.joint_params["generations"].get()),
                popsize=int(self.joint_params["popsize"].get()),
                parents=int(self.joint_params["parents"].get()),
                sel_mutation=float(self.joint_params["sel_mutation"].get()),
                tail_mutation=float(self.joint_params["tail_mutation"].get()),
                per_good_cap=to_int_or_none(self.joint_params["per_good_cap"].get()),
                max_index_probe=int(self.joint_params["max_index_probe"].get()),
                fix_last_gene=bool(self.joint_params["fix_last_gene"].get()),
                seed=int(self.joint_params["seed"].get()),
                verbosity=int(self.joint_params["verbosity"].get()),
                log_every=int(self.joint_params["log_every"].get()),
            )
            domain = {"G": G, "prices": prices, "agents": agents}
            return ("joint", res, domain)

        self._start_background(job)

    def _run_exhaustive(self):
        try:
            G, prices, agents = self._prepare_domain_inputs()
        except Exception as exc:
            messagebox.showerror("Invalid inputs", str(exc))
            return

        def to_int_or_none(s: str) -> Optional[int]:
            s = (s or "").strip()
            return int(s) if s.isdigit() else None

        def job():
            res = run_ga_equivclass_exhaustive(
                production_graph=G,
                pmatrix=prices,
                agents_information=agents,
                generations=int(self.ex_params["generations"].get()),
                popsize=int(self.ex_params["popsize"].get()),
                parents=int(self.ex_params["parents"].get()),
                mutation_rate=float(self.ex_params["mutation_rate"].get()),
                fix_last_gene=bool(self.ex_params["fix_last_gene"].get()),
                seed=int(self.ex_params["seed"].get()),
                max_combos=to_int_or_none(self.ex_params["max_combos"].get()),
                per_good_cap=to_int_or_none(self.ex_params["per_good_cap"].get()),
                max_index_probe=int(self.ex_params["max_index_probe"].get()),
                verbosity=int(self.ex_params["verbosity"].get()),
                log_every=int(self.ex_params["log_every"].get()),
                max_best_returned=to_int_or_none(self.ex_params["max_best_returned"].get()),
            )
            domain = {"G": G, "prices": prices, "agents": agents}
            return ("exhaustive", res, domain)

        self._start_background(job)

    # -- Single-genome evaluation --------------------------------------------
    def _run_genome_only(self):
        """Evaluates a single explicit genome and updates the visualization tabs."""
        try:
            G, prices, agents = self._prepare_domain_inputs()
            genome = np.array(safe_literal(self.txt_genome.get("1.0", tk.END), "Genome"), dtype=int)
        except Exception as exc:
            messagebox.showerror("Invalid inputs", str(exc))
            return

        def job():
            # Compute its utility so the run shows a consistent best utility.
            econ = Economy(G.edges(), prices, agents, genome)
            rep = econ.get_reports()
            u = float(rep.get("utility", 0.0))
            print(f"[GENOME] Utility = {u:.6f}")
            domain = {"G": G, "prices": prices, "agents": agents, "genome": genome.tolist()}
            pseudo_result = {
                "best_genome": genome.tolist(),
                "best_utility": u,
                "all_best_genomes": [genome.tolist()],
                "curves": {},
                "meta": {},
            }
            return ("genome", pseudo_result, domain)

        self._start_background(job)

    def _is_empty_seq(self, x):
        import numpy as np
        return (x is None) or (isinstance(x, (list, tuple, np.ndarray)) and len(x) == 0)

    def _pick_best_genome(self, res, domain, method):
        """Returns np.ndarray or None, the 'best' genome with safe fallbacks."""
        import numpy as np
        g = res.get("best_genome", None)
        if self._is_empty_seq(g):
            ab = res.get("all_best_genomes") or []
            g = ab[0] if ab else None
        if self._is_empty_seq(g) and method == "genome":
            g = (domain or {}).get("genome")
        return (np.array(g, dtype=int) if g is not None else None)


    # -- Post-run rendering ---------------------------------------------------
    def _render_after_run(self, method: str, result: Dict[str, Any], domain: Dict[str, Any]):
        """Populates best list, summary meta, charts, and domain tabs based on the run type."""
        # ---- Best list with per-genome utility
        for item in self.best_tree.get_children():
            self.best_tree.delete(item)

        all_best = result.get("all_best_genomes", []) or []
        best_u = result.get("best_utility", None)

        def _utility_of(gen):
            try:
                econ_i = Economy(domain["G"].edges(), domain["prices"], domain["agents"], np.array(gen, dtype=int))
                return float(econ_i.get_reports().get("utility", 0.0))
            except Exception:
                return None

        utils = []
        if all_best:
            for i, g in enumerate(all_best, start=1):
                u = _utility_of(g)
                utils.append(u)
                u_txt = ("—" if u is None else f"{u:.6f}")
                self.best_tree.insert("", "end", values=(i, len(g), u_txt, str(g)))
        else:
            g = result.get("best_genome", [])
            if g:
                u = best_u if best_u is not None else _utility_of(g)
                utils.append(u)
                u_txt = ("—" if u is None else f"{u:.6f}")
                self.best_tree.insert("", "end", values=(1, len(g), u_txt, str(g)))

        if best_u is None:
            best_u = max([x for x in utils if x is not None], default=None)
        self.best_u_var.set("—" if best_u is None else f"{best_u:.6f}")

        # ---- Meta summary
        meta = result.get("meta", {}) or {}
        best_g = result.get("best_genome", [])
        if method == "flat":
            meta_txt = f"[FLAT] best_u={best_u if best_u is None else f'{best_u:.6f}'} | genome_len={len(best_g)}"
        elif method == "joint":
            meta_txt = (
                f"[JOINT] best_u={best_u:.6f} | labels={meta.get('labels')} | sizes={meta.get('sizes')} "
                f"| K={meta.get('K')} L_used={meta.get('L_used')} | alphabet={meta.get('alphabet')} "
                f"| pool_sizes={meta.get('pool_sizes')} | num_best={meta.get('num_best_genomes')}"
            )
        elif method == "exhaustive":
            meta_txt = (
                f"[EXHAUSTIVE] best_u={best_u:.6f} | K={meta.get('K')} L_used={meta.get('L_used')} "
                f"| labels={meta.get('labels')} | sizes={meta.get('sizes')} "
                f"| num_best_returned={meta.get('num_best_genomes')} (found={meta.get('num_best_found')})"
            )
        else:  # genome
            meta_txt = "[GENOME] evaluated a single explicit genome (no GA run)."
        self.meta_var.set(meta_txt)

        # Default chart choice and draw
        self._last_method = method
        self._last_result = result
        if method in ("flat", "joint"):
            self.chart_var.set("Fitness curves")
        elif method == "exhaustive":
            self.chart_var.set("Global curves (Exhaustive)")
        else:
            self.chart_var.set("Best genome (bits)")
        
        g0 = result.get("best_genome", None)
        try:
            is_empty = (g0 is None) or (len(g0) == 0)
        except Exception:
            is_empty = False  # si no es “sizable”, lo dejamos como está

        if is_empty:
            if all_best:
                result["best_genome"] = all_best[0]
            elif method == "genome" and "genome" in domain:
                result["best_genome"] = domain["genome"]


        self._render_chart()

        # Domain tabs: evaluate with best genome (or explicit genome) and render
        if method == "genome":
            best_genome = np.array(domain["genome"], dtype=int)
        else:
            best_genome = np.array(result.get("best_genome", []), dtype=int)
        self._render_domain_with_genome(domain["G"], domain["prices"], domain["agents"], best_genome)

    def _render_chart(self):
        """Renders the selected chart using the last available result."""
        self.ax.clear()
        self.ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.6)

        method = self._last_method
        res = self._last_result or {}
        curves = res.get("curves", {}) or {}
        choice = self.chart_var.get()

        # Helpers locales (no dependen de nada externo)
        def _has_data(series):
            if series is None:
                return False
            try:
                return len(series) > 0
            except Exception:
                return False

        def _first_nonempty(*candidates):
            for cand in candidates:
                if cand is None:
                    continue
                try:
                    if len(cand) == 0:
                        continue
                except Exception:
                    # Si no tiene len (escalares), lo aceptamos
                    pass
                return cand
            return None

        if choice == "Fitness curves":
            for key in ("best", "mean", "median"):
                s = curves.get(key, None)
                if _has_data(s):
                    self.ax.plot(s, label=key)
            for key in ("best_combo_best", "best_combo_mean"):
                s = curves.get(key, None)
                if _has_data(s):
                    self.ax.plot(s, label=key)
            self.ax.set_xlabel("Generation")
            self.ax.set_ylabel("Utility")
            if self.ax.has_data():
                self.ax.legend(loc="best")
            else:
                self.ax.text(0.5, 0.5, "No curves available", ha="center", va="center")

        elif choice == "Best improvement (Δbest)":
            c = None
            for key in ("best", "global_best"):
                s = curves.get(key, None)
                if _has_data(s):
                    c = np.array(s, dtype=float)
                    break
            if c is None or c.size == 0:
                self.ax.text(0.5, 0.5, "No best curve available", ha="center", va="center")
            else:
                delta = np.diff(c, prepend=c[0])
                x = np.arange(len(delta))
                self.ax.bar(x, delta)
                self.ax.set_xlabel("Generation")
                self.ax.set_ylabel("Δbest")
                self.ax.set_title("Best improvement per generation")

        elif choice == "Best genome (bits)":
            # Elegimos el mejor genoma con fallbacks seguros y sin 'if ndarray'
            cand1 = res.get("best_genome", None)
            ab = res.get("all_best_genomes")
            cand2 = ab[0] if isinstance(ab, (list, tuple)) and len(ab) > 0 else None
            cand3 = (self._last_domain or {}).get("genome") if method == "genome" else None
            g = _first_nonempty(cand1, cand2, cand3)

            if g is None or (hasattr(g, "__len__") and len(g) == 0):
                self.ax.text(0.5, 0.5, "No best genome available", ha="center", va="center")
            else:
                g = np.array(g, dtype=int)
                x = np.arange(len(g))
                self.ax.bar(x, g)
                self.ax.set_xlabel("Gene index")
                self.ax.set_ylabel("Value")
                self.ax.set_title("Best genome (phenotype)")

        elif choice == "Best-by-selectors (Joint)":
            if method != "joint":
                self.ax.text(0.5, 0.5, "Only available for Joint GA", ha="center", va="center")
            else:
                bsel = res.get("best_by_selectors", []) or []
                top = bsel[:12]
                if not top:
                    self.ax.text(0.5, 0.5, "No selector diagnostics available", ha="center", va="center")
                else:
                    labels = [k for (k, _v) in top]
                    vals = [float(v) for (_k, v) in top]
                    x = np.arange(len(vals))
                    self.ax.bar(x, vals)
                    self.ax.set_xticks(x, labels, rotation=45, ha="right")
                    self.ax.set_ylabel("Best utility per selector tuple")
                    self.ax.set_title("Top selector combinations")

        elif choice == "Global curves (Exhaustive)":
            any_plot = False
            for key, lbl in (("global_best", "global_best"), ("global_mean", "global_mean")):
                s = curves.get(key, None)
                if _has_data(s):
                    self.ax.plot(s, label=lbl)
                    any_plot = True
            self.ax.set_xlabel("Combination progress proxy")
            self.ax.set_ylabel("Utility")
            if any_plot:
                self.ax.legend(loc="best")
            else:
                self.ax.text(0.5, 0.5, "No global curves available", ha="center", va="center")

        self.canvas.draw()


    # -- Domain rendering using Economy --------------------------------------
    def _render_domain_with_genome(self, G: nx.DiGraph, prices: np.ndarray, agents: Dict[str, Any], genome: np.ndarray):
        """Runs Economy with the provided genome and renders graph, orders, reports, and ledgers."""
        try:
            econ = Economy(G.edges(), prices, agents, genome)
            plan = econ.orders
            results = econ.get_reports()
            self.graph_view.set_graph(G)
        except Exception as exc:
            messagebox.showerror("Economy error", str(exc))
            return

        # Utility
        self.utility_var.set(str(results.get("utility", "—")))

        # Plan table
        plan_df = pd.DataFrame([(i, str(t)) for i, t in enumerate(plan, start=1)], columns=["Step", "Transaction"])
        plan_prefs = {"Step": {"width": 60, "anchor": "e", "stretch": False}, "Transaction": {"width": 900, "anchor": "w", "stretch": True}}
        fill_treeview(self.plan_tree, plan_df, plan_prefs)

        # Reports (NCT, ZF)
        rep = results.get("reports", {})
        for key, tree in (("NCT", self.nct_tree), ("ZF", self.zf_tree)):
            data = rep.get(key, {})
            df = pd.DataFrame(list(data.items()), columns=["Account", "Balance"])
            if not df.empty:
                def _fmt(x):
                    try:
                        return f"{float(x):,.2f}"
                    except Exception:
                        return str(x)
                df["Balance"] = df["Balance"].map(_fmt)
            prefs = {"Account": {"width": 320, "anchor": "w", "stretch": True},
                     "Balance": {"width": 160, "anchor": "e", "stretch": False}}
            fill_treeview(tree, df, prefs)

        # Ledgers (NCT, ZF) with detail
        try:
            nct_entries = econ.agents["NCT"].accountant.ledger.get_all_entries()
        except Exception:
            nct_entries = []
        try:
            zf_entries = econ.agents["ZF"].accountant.ledger.get_all_entries()
        except Exception:
            zf_entries = []

        nct_df = pd.DataFrame([(i, str(t)) for i, t in enumerate(nct_entries, start=1)], columns=["#","Entry"])
        zf_df  = pd.DataFrame([(i, str(t)) for i, t in enumerate(zf_entries , start=1)], columns=["#","Entry"])

        ledger_prefs = {"#": {"width": 60, "anchor": "e", "stretch": False}, "Entry": {"width": 900, "anchor": "w", "stretch": True}}
        fill_treeview(self.nct_ledger_tree, nct_df, ledger_prefs)
        fill_treeview(self.zf_ledger_tree, zf_df, ledger_prefs)

        # Auto-select first entry to populate detail panes
        for tree, detail in [(self.nct_ledger_tree, self.nct_detail_text), (self.zf_ledger_tree, self.zf_detail_text)]:
            children = tree.get_children()
            detail.config(state="normal")
            detail.delete("1.0", tk.END)
            if children:
                tree.selection_set(children[0])
                vals = tree.item(children[0], "values")
                detail.insert("1.0", vals[1] if len(vals) > 1 else "")
            detail.config(state="disabled")


# -----------------------------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    app = GAWorkbench()
    app.mainloop()

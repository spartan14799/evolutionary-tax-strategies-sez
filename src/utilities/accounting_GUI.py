# -*- coding: utf-8 -*-
"""
Economy Simulation Interface (Canvas-based graph view, reports, plan, ledgers).
"""

from __future__ import annotations

import ast
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import tkinter as tk
from tkinter import messagebox, ttk
from tkinter.scrolledtext import ScrolledText

import networkx as nx
import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------- #
# Paths / imports
# ----------------------------------------------------------------------------- #

ROOT_DIR = Path(__file__).resolve().parent
while not (ROOT_DIR / "src" / "simulation" / "economy" / "economy.py").exists() and ROOT_DIR != ROOT_DIR.parent:
    ROOT_DIR = ROOT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config_paths import get_default_chart_of_accounts_path  # noqa: E402
from src.simulation.economy.economy import Economy  # noqa: E402

accounts_path = get_default_chart_of_accounts_path()

# ----------------------------------------------------------------------------- #
# Agents config
# ----------------------------------------------------------------------------- #

base_agents_info: Dict[str, Dict] = {
    "MKT": {
        "type": "MKT",
        "inventory_strategy": "FIFO",
        "firm_related_goods": [],
        "income_statement_type": "standard",
        "accounts_yaml_path": accounts_path,
        "price_mapping": 0,
    },
    "NCT": {
        "type": "NCT",
        "inventory_strategy": "FIFO",
        "firm_related_goods": [],
        "income_statement_type": "standard",
        "accounts_yaml_path": accounts_path,
        "price_mapping": 1,
    },
    "ZF": {
        "type": "ZF",
        "inventory_strategy": "FIFO",
        "firm_related_goods": [],
        "income_statement_type": "standard",
        "accounts_yaml_path": accounts_path,
        "price_mapping": 2,
    },
}


def populate_goods_agents_info(
    agents_info_dict: Dict[str, Dict], goods_list: List[str]
) -> Dict[str, Dict]:
    updated: Dict[str, Dict] = {}
    for agent, info in agents_info_dict.items():
        info_copy = dict(info)
        info_copy["firm_related_goods"] = list(goods_list)
        updated[agent] = info_copy
    return updated


# ----------------------------------------------------------------------------- #
# Graph view (Canvas)
# ----------------------------------------------------------------------------- #

def topo_vertical_layout(
    G,
    layer_gap: float = 1.0,
    node_gap: float = 1.6,
    bottom_to_top: bool = False,  
    stagger_singletons: bool = True,
    singleton_dx: float = 0.8,
):
    import networkx as nx
    if not nx.is_directed_acyclic_graph(G):
        return nx.spring_layout(G, seed=42)

    layers = list(nx.topological_generations(G))  # sources first, sinks last
    pos = {}
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
    def __init__(self, parent: tk.Widget):
        super().__init__(parent)
        self.rowconfigure(1, weight=1)
        self.columnconfigure(0, weight=1)

        bar = ttk.Frame(self)
        bar.grid(row=0, column=0, sticky="ew")
        ttk.Label(bar, text="Production Graph", font=("Arial", 12, "bold")).pack(
            side="left", padx=(6, 0)
        )
        ttk.Button(bar, text="Fit", command=self.fit_to_view).pack(side="right", padx=3)
        ttk.Button(bar, text="+", width=3, command=lambda: self.zoom(1.25)).pack(
            side="right", padx=1
        )
        ttk.Button(bar, text="−", width=3, command=lambda: self.zoom(0.8)).pack(
            side="right", padx=1
        )

        self.canvas = tk.Canvas(self, bg="white", highlightthickness=1, highlightbackground="#bbb")
        self.canvas.grid(row=1, column=0, sticky="nsew")

        self.graph: nx.DiGraph | None = None
        self.pos: Dict = {}
        self.scale: float = 1.0
        self.tx: float = 0.0
        self.ty: float = 0.0
        self.bbox_world: Tuple[float, float, float, float] | None = None
        self.node_radius = 14

        self.canvas.bind("<Configure>", lambda e: self._on_resize())
        self.canvas.bind("<ButtonPress-1>", self._on_start_pan)
        self.canvas.bind("<B1-Motion>", self._on_pan)
        self.canvas.bind("<MouseWheel>", self._on_wheel)      # Windows / macOS
        self.canvas.bind("<Button-4>", lambda e: self.zoom(1.1))  # Linux
        self.canvas.bind("<Button-5>", lambda e: self.zoom(0.9))

        self._pan_last = None

    def set_graph(self, G: nx.DiGraph):
        self.graph = G
        self.pos = topo_vertical_layout(
            G, layer_gap=1.0, node_gap=1.6, bottom_to_top=False,
            stagger_singletons=True, singleton_dx=0.8
        )
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

        # --- EDGES: curved and with arrows
        edge_sep_px = 16.0    # horizontal separation between parallel edges
        edge_color  = "#555555"
        edge_width  = 1.4
        arrowshape  = (10, 12, 4)  # (length, width, wing); clear and compact arrow
        for u in self.graph.nodes():
            targets = list(self.graph.successors(u))
            k = len(targets)
            for idx, v in enumerate(targets):
                x0, y0 = self.pos[u]
                x1, y1 = self.pos[v]
                sx0, sy0 = self._world_to_screen(x0, y0)
                sx1, sy1 = self._world_to_screen(x1, y1)

                # Control point for the curve: midpoint + horizontal offset
                midx = (sx0 + sx1) / 2.0 + (idx - (k - 1) / 2.0) * edge_sep_px
                midy = (sy0 + sy1) / 2.0

                self.canvas.create_line(
                    sx0, sy0, midx, midy, sx1, sy1,
                    smooth=True, splinesteps=12,
                    arrow=tk.LAST, arrowshape=arrowshape,
                    width=edge_width, fill=edge_color
                )

        # --- NODES: Circle + label
        for n, (x, y) in self.pos.items():
            sx, sy = self._world_to_screen(x, y)

            self.canvas.create_oval(
                sx - r, sy - r, sx + r, sy + r,
                fill="#cfe8ff", outline="#7aa7d7", width=2
            )
            self.canvas.create_text(
                sx, sy, text=str(n), font=("Arial", 11, "bold"), fill="#274b6d"
            )



# ----------------------------------------------------------------------------- #
# UI helpers
# ----------------------------------------------------------------------------- #

def make_treeview(parent: tk.Widget, height: int = 18) -> ttk.Treeview:
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


def fill_treeview(
    tree: ttk.Treeview,
    df: pd.DataFrame,
    column_prefs: Dict[str, Dict] | None = None,
) -> None:
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


def safe_eval_list(txt: str, what: str):
    try:
        return ast.literal_eval(txt)
    except Exception as exc:
        raise ValueError(
            f"The content of {what} is not a valid list/array.\nDetails: {exc}"
        ) from exc


# ----------------------------------------------------------------------------- #
# UI logic
# ----------------------------------------------------------------------------- #

def run_model():
    try:
        graph_links = safe_eval_list(links_input.get("1.0", tk.END), "Graph links")
        price_matrix = np.array(
            safe_eval_list(price_matrix_input.get("1.0", tk.END), "Price matrix")
        )
        economy_genome = np.array(
            safe_eval_list(genome_input.get("1.0", tk.END), "Plan genome")
        )
    except ValueError as err:
        messagebox.showerror("Invalid input", str(err))
        return

    economy_graph = nx.DiGraph()
    economy_graph.add_edges_from(graph_links)
    related_goods = list(economy_graph.nodes())
    agents_dict = populate_goods_agents_info(base_agents_info, related_goods)

    graph_view.set_graph(economy_graph)

    economy = Economy(graph_links, price_matrix, agents_dict, economy_genome)
    production_plan = economy.orders
    results = economy.get_reports()

    report_nct = results["reports"]["NCT"]
    report_zf = results["reports"]["ZF"]

    nct_df = pd.DataFrame(list(report_nct.items()), columns=["Account", "Balance"])
    zf_df = pd.DataFrame(list(report_zf.items()), columns=["Account", "Balance"])

    for df in (nct_df, zf_df):
        df["Balance"] = df["Balance"].map(lambda x: f"{float(x):,.2f}")

    plan_df = pd.DataFrame(
        [(i, str(t)) for i, t in enumerate(production_plan, start=1)],
        columns=["Step", "Transaction"],
    )

    utility_var.set(str(results["utility"]))

    report_prefs = {
        "Account": {"width": 320, "anchor": "w", "stretch": True},
        "Balance": {"width": 120, "anchor": "e", "stretch": False},
    }
    plan_prefs = {
        "Step": {"width": 60, "anchor": "e", "stretch": False},
        "Transaction": {"width": 900, "anchor": "w", "stretch": True},
    }

    fill_treeview(nct_tree, nct_df, report_prefs)
    fill_treeview(zf_tree, zf_df, report_prefs)
    fill_treeview(plan_tree, plan_df, plan_prefs)

    nct_ledger_entries = economy.agents["NCT"].accountant.ledger.get_all_entries()
    zf_ledger_entries = economy.agents["ZF"].accountant.ledger.get_all_entries()

    nct_entries_df = pd.DataFrame(
        [(i, str(t)) for i, t in enumerate(nct_ledger_entries, start=1)],
        columns=["Step", "Entry"],
    )
    zf_entries_df = pd.DataFrame(
        [(i, str(t)) for i, t in enumerate(zf_ledger_entries, start=1)],
        columns=["Step", "Entry"],
    )

    ledger_prefs = {
        "Step": {"width": 60, "anchor": "e", "stretch": False},
        "Entry": {"width": 900, "anchor": "w", "stretch": True},
    }

    fill_treeview(nct_ledger_tree, nct_entries_df, ledger_prefs)
    fill_treeview(zf_ledger_tree, zf_entries_df, ledger_prefs)

    for tree, detail in [
        (nct_ledger_tree, nct_detail_text),
        (zf_ledger_tree, zf_detail_text),
    ]:
        children = tree.get_children()
        if children:
            tree.selection_set(children[0])
            values = tree.item(children[0], "values")
            detail.config(state="normal")
            detail.delete("1.0", tk.END)
            detail.insert("1.0", values[1] if len(values) > 1 else "")
            detail.config(state="disabled")


def on_ledger_select(event, tree: ttk.Treeview, detail_text: ScrolledText):
    sel = tree.selection()
    if not sel:
        return
    values = tree.item(sel[0], "values")
    full_entry = values[1] if len(values) > 1 else ""
    detail_text.config(state="normal")
    detail_text.delete("1.0", tk.END)
    detail_text.insert("1.0", full_entry)
    detail_text.config(state="disabled")


def on_copy_detail(detail_text: ScrolledText):
    main_window.clipboard_clear()
    main_window.clipboard_append(detail_text.get("1.0", tk.END))
    main_window.update()


# ----------------------------------------------------------------------------- #
# UI construction
# ----------------------------------------------------------------------------- #

main_window = tk.Tk()
main_window.title("Economy Simulation Interface")
main_window.geometry("1500x950")
main_window.minsize(1200, 750)

style = ttk.Style()
try:
    style.theme_use("clam")
except tk.TclError:
    pass

style.configure("Treeview", font=("Consolas", 10), rowheight=22)
style.configure("TLabel", font=("Arial", 11))
style.configure("TLabelframe.Label", font=("Arial", 12, "bold"))
style.configure("TNotebook.Tab", padding=(12, 8))

notebook = ttk.Notebook(main_window)
notebook.grid(row=0, column=0, sticky="nsew")
main_window.rowconfigure(0, weight=1)
main_window.columnconfigure(0, weight=1)

# Setup tab
setup_tab = ttk.Frame(notebook)
notebook.add(setup_tab, text="Setup")

for c in range(3):
    setup_tab.columnconfigure(c, weight=1, uniform="setup")
setup_tab.rowconfigure(1, weight=1)

links_lf = ttk.LabelFrame(setup_tab, text="Graph edges (list of tuples)")
links_lf.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
links_input = ScrolledText(links_lf, wrap="none")
links_input.pack(fill="both", expand=True, padx=6, pady=6)

prices_lf = ttk.LabelFrame(setup_tab, text="Price matrix (NumPy-like)")
prices_lf.grid(row=1, column=1, sticky="nsew", padx=10, pady=10)
price_matrix_input = ScrolledText(prices_lf, wrap="none")
price_matrix_input.pack(fill="both", expand=True, padx=6, pady=6)

genome_lf = ttk.LabelFrame(setup_tab, text="Plan genome (NumPy-like)")
genome_lf.grid(row=1, column=2, sticky="nsew", padx=10, pady=10)
genome_input = ScrolledText(genome_lf, wrap="none")
genome_input.pack(fill="both", expand=True, padx=6, pady=6)

toolbar_setup = ttk.Frame(setup_tab)
toolbar_setup.grid(row=2, column=0, columnspan=3, sticky="ew", padx=10, pady=(0, 10))
toolbar_setup.columnconfigure(0, weight=1)
run_button = ttk.Button(toolbar_setup, text="Run", command=run_model)
run_button.grid(row=0, column=0, sticky="e")

# Results tab
results_tab = ttk.Frame(notebook)
notebook.add(results_tab, text="Results")

results_tab.rowconfigure(0, weight=1)
results_tab.columnconfigure(0, weight=1)

left_right_pane = ttk.Panedwindow(results_tab, orient=tk.HORIZONTAL)
left_right_pane.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

graph_frame = ttk.Frame(left_right_pane)
graph_frame.rowconfigure(0, weight=1)
graph_frame.columnconfigure(0, weight=1)
graph_view = GraphView(graph_frame)
graph_view.grid(row=0, column=0, sticky="nsew")
left_right_pane.add(graph_frame, weight=1)

right_vertical_pane = ttk.Panedwindow(left_right_pane, orient=tk.VERTICAL)
left_right_pane.add(right_vertical_pane, weight=2)

reports_split = ttk.Panedwindow(right_vertical_pane, orient=tk.HORIZONTAL)

nct_lf = ttk.Labelframe(reports_split, text="NCT Report")
nct_lf.rowconfigure(0, weight=1)
nct_lf.columnconfigure(0, weight=1)
nct_tree = make_treeview(nct_lf)
reports_split.add(nct_lf, weight=1)

zf_lf = ttk.Labelframe(reports_split, text="ZF Report")
zf_lf.rowconfigure(0, weight=1)
zf_lf.columnconfigure(0, weight=1)
zf_tree = make_treeview(zf_lf)
reports_split.add(zf_lf, weight=1)

right_vertical_pane.add(reports_split, weight=3)

plan_utility_frame = ttk.Frame(right_vertical_pane)
plan_utility_frame.rowconfigure(0, weight=1)
plan_utility_frame.columnconfigure(0, weight=4)
plan_utility_frame.columnconfigure(1, weight=1)

plan_lf = ttk.Labelframe(plan_utility_frame, text="Production Plan")
plan_lf.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
plan_lf.rowconfigure(0, weight=1)
plan_lf.columnconfigure(0, weight=1)
plan_tree = make_treeview(plan_lf, height=12)

utility_lf = ttk.Labelframe(plan_utility_frame, text="Utility")
utility_lf.grid(row=0, column=1, sticky="nsew")
utility_var = tk.StringVar(value="—")
utility_label = ttk.Label(utility_lf, textvariable=utility_var, anchor="center", font=("Arial", 16, "bold"))
utility_label.pack(expand=True, fill="both", padx=10, pady=10)

right_vertical_pane.add(plan_utility_frame, weight=2)

# Ledgers tab
ledgers_tab = ttk.Frame(notebook)
notebook.add(ledgers_tab, text="Ledgers")

ledgers_tab.rowconfigure(0, weight=1)
ledgers_tab.columnconfigure(0, weight=1)

ledger_nb = ttk.Notebook(ledgers_tab)
ledger_nb.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

nct_tab = ttk.Frame(ledger_nb)
ledger_nb.add(nct_tab, text="NCT")
nct_tab.rowconfigure(0, weight=1)
nct_tab.columnconfigure(0, weight=1)

nct_split = ttk.Panedwindow(nct_tab, orient=tk.HORIZONTAL)
nct_split.grid(row=0, column=0, sticky="nsew")

nct_table_lf = ttk.Labelframe(nct_split, text="NCT Ledger Entries")
nct_table_lf.rowconfigure(0, weight=1)
nct_table_lf.columnconfigure(0, weight=1)
nct_ledger_tree = make_treeview(nct_table_lf, height=25)
nct_split.add(nct_table_lf, weight=2)

nct_detail_lf = ttk.Labelframe(nct_split, text="Detail")
nct_detail_lf.rowconfigure(0, weight=1)
nct_detail_lf.columnconfigure(0, weight=1)
nct_detail_text = ScrolledText(nct_detail_lf, wrap="word")
nct_detail_text.grid(row=0, column=0, sticky="nsew", padx=6, pady=6)
nct_detail_text.config(state="disabled")
nct_copy = ttk.Button(nct_detail_lf, text="Copy", command=lambda: on_copy_detail(nct_detail_text))
nct_copy.grid(row=1, column=0, sticky="e", padx=6, pady=(0, 6))
nct_split.add(nct_detail_lf, weight=3)

zf_tab = ttk.Frame(ledger_nb)
ledger_nb.add(zf_tab, text="ZF")
zf_tab.rowconfigure(0, weight=1)
zf_tab.columnconfigure(0, weight=1)

zf_split = ttk.Panedwindow(zf_tab, orient=tk.HORIZONTAL)
zf_split.grid(row=0, column=0, sticky="nsew")

zf_table_lf = ttk.Labelframe(zf_split, text="ZF Ledger Entries")
zf_table_lf.rowconfigure(0, weight=1)
zf_table_lf.columnconfigure(0, weight=1)
zf_ledger_tree = make_treeview(zf_table_lf, height=25)
zf_split.add(zf_table_lf, weight=2)

zf_detail_lf = ttk.Labelframe(zf_split, text="Detail")
zf_detail_lf.rowconfigure(0, weight=1)
zf_detail_lf.columnconfigure(0, weight=1)
zf_detail_text = ScrolledText(zf_detail_lf, wrap="word")
zf_detail_text.grid(row=0, column=0, sticky="nsew", padx=6, pady=6)
zf_detail_text.config(state="disabled")
zf_copy = ttk.Button(zf_detail_lf, text="Copy", command=lambda: on_copy_detail(zf_detail_text))
zf_copy.grid(row=1, column=0, sticky="e", padx=6, pady=(0, 6))
zf_split.add(zf_detail_lf, weight=3)

nct_ledger_tree.bind("<<TreeviewSelect>>", lambda e: on_ledger_select(e, nct_ledger_tree, nct_detail_text))
zf_ledger_tree.bind("<<TreeviewSelect>>", lambda e: on_ledger_select(e, zf_ledger_tree, zf_detail_text))

# ----------------------------------------------------------------------------- #
# Prefill from environment + optional auto-run
# ----------------------------------------------------------------------------- #

def _prefill_from_env_and_maybe_run():
    L = os.environ.get("ECON_LINKS")
    M = os.environ.get("ECON_PRICE_MATRIX")
    G = os.environ.get("ECON_GENOME")

    if L:
        links_input.delete("1.0", tk.END)
        links_input.insert("1.0", L.strip())
    if M:
        price_matrix_input.delete("1.0", tk.END)
        price_matrix_input.insert("1.0", M.strip())
    if G:
        genome_input.delete("1.0", tk.END)
        genome_input.insert("1.0", G.strip())

    if os.environ.get("ECON_AUTO_RUN", "") == "1":
        try:
            run_model()
            for i in range(notebook.index("end")):
                if notebook.tab(i, "text").lower().startswith("result"):
                    notebook.select(i)
                    break
        except Exception as e:
            print("Auto-run failed:", e)


# ----------------------------------------------------------------------------- #
# Mainloop
# ----------------------------------------------------------------------------- #

if __name__ == "__main__":
    main_window.after(100, _prefill_from_env_and_maybe_run)
    main_window.mainloop()

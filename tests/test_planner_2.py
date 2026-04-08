import os 
import sys 

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))




import networkx as nx
import importlib
from collections import deque
import matplotlib.pyplot as plt
from src.simulation.economy.production_process.production_graph import ProductionGraph
from src.simulation.economy.production_process.production_process import ProductionProcess
from src.simulation.planner.planner import Planner

from networkx.drawing.nx_pydot import graphviz_layout



try:
    from networkx.drawing.nx_agraph import graphviz_layout
except ImportError:
    from networkx.drawing.nx_pydot import graphviz_layout




G = nx.DiGraph()

G.add_edges_from([
        ("Input", "Intermediate"),
        ("Intermediate", "Final"),
    ])



plan = [0,1,1,0,1,0,1,0,1,1,1,0,1,1,1,0,0,1]

print(list(nx.topological_sort(G)))


test_process = ProductionProcess(ProductionGraph(G))


base_plan = test_process.create_production_plan(plan)
base_plan



test_planner = Planner(test_process)

base_plan = test_planner.create_production_plan(plan)
base_plan

test = test_planner.execute_plan(plan)
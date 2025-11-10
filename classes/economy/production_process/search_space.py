from typing import List
import networkx as nx
import numpy as np
from classes.economy.production_process.production_process import ProductionProcess
from classes.economy.production_process.production_graph import ProductionGraph

import math


class SearchSpace:
    def __init__(self, link_list: List[tuple]):
        digraph = nx.DiGraph()
        digraph.add_edges_from(link_list)

        prod_graph = ProductionGraph(digraph)
        self.prod_process = ProductionProcess(prod_graph)

    def calculate_search_space(self , fix_last_gene=True , verbosity = False) -> int:
        classification = self.prod_process.get_goods_classification()
        required_quantities = self.prod_process.get_required_quantities()

        primary_goods = [g for g, cls in classification.items() if cls == "primary"]

        # Total required quantities (all goods)
        n = sum(required_quantities.values())+1

        # Required quantities for primary goods only
        k = sum(required_quantities[g] for g in primary_goods)

        # Product over primary goods of (q_g + 1)
        primary_combinations = np.prod([(required_quantities[g] + 1) for g in primary_goods])

        # Binary decisions for each required unit of non-primary goods
        if fix_last_gene:

            reduced_search_space =math.ceil( primary_combinations * (2 ** (n - k-1)))
        else:
            reduced_search_space = math.ceil(primary_combinations * (2 ** (n-k)))

        search_space = 2**n

        if verbosity:
            print(f"n = {n}, k = {k}, primary_combinations = {primary_combinations}")
        return {'search_space': int(search_space), 'reduced_search_space': int(reduced_search_space)}


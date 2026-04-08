import numpy as np
import networkx as nx
from typing import List, Tuple
from src.simulation.economy.production_process.production_process import ProductionProcess
from src.simulation.economy.production_process.production_graph import ProductionGraph

import math

class PriceMarkupGenerator:
    def __init__(self, list_links: List[Tuple[str, str]]):
        """
        Builds the production process and prepares topological order.
        """
        digraph = nx.DiGraph()
        digraph.add_edges_from(list_links)

        prod_graph = ProductionGraph(digraph)
        self.prod_process = ProductionProcess(prod_graph)

        self.topo_goods = list(nx.topological_sort(digraph))
        self.good_indices = {g: i for i, g in enumerate(self.topo_goods)}

        self.good_prices: dict[str, np.ndarray] = {}
        self.perturbed_prices: dict[str, np.ndarray] = {}

    # -------------------------------------------------------------------------
    def get_good_topo_indices(self):
        return self.good_indices

    # -------------------------------------------------------------------------
    def generate_price_tensor(
        self, good: str, base: float, m1: float, m2: float, m3: float, n_agents: int
    ) -> np.ndarray:
        """
        Recursively computes the price matrix for each good according to:
          • Primary goods: fixed base price.
          • Intermediate goods: (1 + m1) × sum of input prices.
          • Final goods: (1 + m2) × sum of input prices.
          • Additionally, for final goods, the first row receives +m3 (row-wise markup).
        """
        # Return cached matrix if already computed
        if good in self.good_prices:
            return self.good_prices[good]

        goods_class = self.prod_process.get_goods_classification()
        classification = goods_class.get(good)

        # ---- Base case: primary goods ----
        if classification == "primary":
            price_matrix = np.full((n_agents, n_agents), base)

        # ---- Recursive case ----
        else:
            inputs = self.prod_process.get_inputs(good)
            if not inputs:
                raise ValueError(f"No inputs found for non-primary good '{good}'")

            input_prices = [
                self.generate_price_tensor(inp, base, m1, m2, m3, n_agents)[0, 0]
                for inp in inputs
            ]
            summed_inputs = np.sum(input_prices)

            if classification == "intermediate":
                final_price = int(summed_inputs * (1 + m1))
            elif classification == "final":
                final_price = int(summed_inputs * (1 + m2))
            else:
                raise ValueError(f"Unrecognized good classification: {classification}")

            price_matrix = np.full((n_agents, n_agents), final_price)

            # ---- Apply m3 to first row (buyers of agent 0) ----
            if classification == "final":
                price_matrix[0, :] = (price_matrix[0, :] * (1 + m3)).astype(int)

        # Cache and return
        self.good_prices[good] = price_matrix
        return price_matrix

    # -------------------------------------------------------------------------
    def generate_all_price_tensors(
        self, base: float, m1: float, m2: float, m3: float, n_agents: int
    ) -> dict[str, np.ndarray]:
        """
        Computes and stores all goods' price matrices in topological order.
        """
        for good in self.topo_goods:
            self.generate_price_tensor(good, base, m1, m2, m3, n_agents)
        return self.good_prices

    # -------------------------------------------------------------------------
    def generate_perturbations(
        self,
        seed: int,
        sigma: float = 0.05,
        ignored_indices: List[int] = None
    ) -> dict[str, np.ndarray]:
        """
        Generates a perturbed version of each good's price matrix using lognormal noise.

        Each perturbed price follows:
            P_ij' = ceil(P_ij * (1 + r_ij)),
        where r_ij ~ LogNormal(mean=0, sigma) - 1.

        Only *off-diagonal* entries are perturbed, and rows/columns in `ignored_indices`
        remain fixed.

        Args:
            seed (int): Random seed for reproducibility.
            sigma (float): Std. deviation of lognormal noise (controls volatility).
            ignored_indices (List[int]): Rows/columns to ignore in perturbations.

        Returns:
            dict[str, np.ndarray]: Mapping of good → perturbed price matrix.
        """
        if not self.good_prices:
            raise ValueError("Generate base price matrices first using generate_all_price_tensors().")

        if ignored_indices is None:
            ignored_indices = []

        np.random.seed(seed)
        perturbed_dict = {}

        for good, matrix in self.good_prices.items():
            perturbed = matrix.copy()
            n = perturbed.shape[0]

            for i in range(n):
                for j in range(n):
                    # Skip diagonal and ignored rows/cols
                    if i == j or i in ignored_indices or j in ignored_indices:
                        continue

                    # Lognormal shock centered around 0
                    r = np.random.lognormal(mean=0, sigma=sigma) - 1

                    # Apply perturbation and round up (ceiling)
                    new_value = np.ceil(matrix[i, j] * (1 + r))
                    perturbed[i, j] = new_value

            perturbed_dict[good] = perturbed

        self.perturbed_prices = perturbed_dict
        return perturbed_dict

                
       

        

        




        
        
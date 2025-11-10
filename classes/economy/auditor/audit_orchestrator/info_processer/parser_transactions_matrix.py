
### Author: ANDRES LEGUIZAMON / CARLOS SANCHEZ / GERMAN RODRIGUEZ / SOFIA OCAMPO 
### 2025
### TODO : LICENSE / COPYRIGHT 



### DESCRIPTION: Class that takes a list of transactions and constructs the quantity flow matrix between agents for each good 


# Standard Format of transactions BuyOrder(buyer='ZF', seller='MKT', good='Input', transaction_type='Buy', price=24)

from classes.economy.order_book.order.orders import BuyOrder,ProductionOrder

import numpy as np 

from typing import List , Tuple , Union , Dict


import numpy as np
from typing import List, Union

class TransactionsMatrixParser:
    def __init__(self, mix: float):
        """
        Initialize a TransactionsMatrixParser instance.

        Parameters
        ----------
        mix : float
            Weighting parameter between 0 and 1 (0=quantity only, 1=money only),
            used in the mixing of monetary and quantity flow weights.

        Attributes
        ----------
        base_agent_map : dict
            Map of agents to indices
        raw_quantity_matrices : dict
            Stores raw quantity flow matrices per good (counts of transactions)
        normalized_quantity_matrices : dict
            Stores row-normalized flow matrices per good
        raw_monetary_matrices : dict
            Stores accumulated monetary volumes per good
        quantity_flow_volume : dict
            Stores total flow volumes per good (sum of flows)
        mixed_weights : dict
            Stores the mixed monetary–flow weights
        mix : float
            Weighting parameter between 0 and 1 (0=quantity only, 1=money only)
        """
        self.base_agent_map = {"MKT": 0, "NCT": 1, "ZF": 2}

        # renamed attributes
        self.raw_quantity_matrices: Dict[str, np.ndarray] = {}
        self.normalized_quantity_matrices: Dict[str, np.ndarray] = {}
        self.raw_monetary_matrices: Dict[str, float] = {}
        self.quantity_flow_volume: Dict[str, float] = {}
        self.mixed_weights: Dict[str, float] = {}

        self.mix = mix

    # ----------------------------------------------------------------------
    def filter_buy_orders(self, order_list: List[Union["ProductionOrder", "BuyOrder"]]):
        """Filters a list of transactions to only include BuyOrders."""
        return [orden for orden in order_list if orden.transaction_type == "Buy"]

    # ----------------------------------------------------------------------
    def _ensure_matrix_size(self, good: str, n: int):
        """Ensure the raw quantity matrix for a good exists and has size n × n."""
        if good not in self.raw_quantity_matrices:
            self.raw_quantity_matrices[good] = np.zeros((n, n), dtype=float)
        else:
            matrix = self.raw_quantity_matrices[good]
            if matrix.shape[0] < n:
                new_matrix = np.zeros((n, n), dtype=float)
                new_matrix[:matrix.shape[0], :matrix.shape[1]] = matrix
                self.raw_quantity_matrices[good] = new_matrix

    # ----------------------------------------------------------------------
    def process_order_list(self, order_list: List[Union["ProductionOrder", "BuyOrder"]]):
        """
        Process a list of BuyOrders and update quantity and monetary matrices,
        volumes, and normalized versions.
        """
        filtered_orders = self.filter_buy_orders(order_list)

        for order in filtered_orders:
            buyer, seller, price, good = order.buyer, order.seller, order.price, order.good

            # Ensure buyer and seller are in agent map
            for agent in (buyer, seller):
                if agent not in self.base_agent_map:
                    last_index = max(self.base_agent_map.values(), default=-1)
                    self.base_agent_map[agent] = last_index + 1

            i = self.base_agent_map[buyer]   # row index
            j = self.base_agent_map[seller]  # col index

            # Ensure matrix is initialized/expanded
            n = len(self.base_agent_map)
            self._ensure_matrix_size(good, n)
            matrix = self.raw_quantity_matrices[good]

            # Register flow (+1 transaction per order)
            matrix[i, j] += 1

            # Update volumes
            self.quantity_flow_volume[good] = self.quantity_flow_volume.get(good, 0) + 1
            self.raw_monetary_matrices[good] = self.raw_monetary_matrices.get(good, 0) + price

        # After all transactions → compute normalized matrices
        self.generate_normalized_matrices()

    # ----------------------------------------------------------------------
    def generate_normalized_matrices(self):
        """Creates row-normalized versions of raw quantity matrices."""
        for good, matrix in self.raw_quantity_matrices.items():
            row_sums = matrix.sum(axis=1, keepdims=True)
            normalized = np.divide(
                matrix,
                row_sums,
                out=np.zeros_like(matrix, dtype=float),
                where=row_sums != 0
            )
            self.normalized_quantity_matrices[good] = normalized

    # ----------------------------------------------------------------------
    def calculate_mixed_weights(self) -> Dict[str, float]:
        """
        Calculate normalized weights for goods based on monetary and quantity volumes.
        """
        mix = self.mix

        if not (0 <= mix <= 1):
            raise ValueError("mix must be between 0 and 1")

        if not self.raw_monetary_matrices or not self.quantity_flow_volume:
            return {}

        # Get common goods
        common_goods = set(self.raw_monetary_matrices.keys()) & set(self.quantity_flow_volume.keys())
        if not common_goods:
            return {}

        # Totals
        total_quantity = sum(self.quantity_flow_volume[g] for g in common_goods)
        total_money = sum(self.raw_monetary_matrices[g] for g in common_goods)

        # Normalized component weights
        quantity_weights = (
            {g: self.quantity_flow_volume[g] / total_quantity for g in common_goods}
            if total_quantity > 0 else {}
        )
        money_weights = (
            {g: self.raw_monetary_matrices[g] / total_money for g in common_goods}
            if total_money > 0 else {}
        )

        # Combine with mix
        weights = {}
        for g in common_goods:
            q_weight = quantity_weights.get(g, 0)
            m_weight = money_weights.get(g, 0)
            weights[g] = mix * m_weight + (1 - mix) * q_weight

        # Normalize final weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {g: w / total_weight for g, w in weights.items()}

        self.mixed_weights = weights
        return weights
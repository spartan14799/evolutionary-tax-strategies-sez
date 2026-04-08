from src.simulation.economy.order_book.order.orders import BaseBuyOrder, BaseProductionOrder , BuyOrder, ProductionOrder
from src.simulation.economy.production_process.production_graph import ProductionGraph
import networkx as nx

from typing import List, Tuple, Union

import numpy as np 


### Author: ANDRES LEGUIZAMON / CARLOS SANCHEZ / GERMAN RODRIGUEZ / SOFIA OCAMPO 
### 2025
### TODO : LICENSE / COPYRIGHT 

### DESCRIPTION: Class that  looks the apropaited price for Buy order given the price matrix enunciated in the model description
# Price matrix is an n-dimensional numpy array where the first dimension is the good index, the second is the buyer location, and the third is the seller location.

class PriceLooker: 
    def __init__(self , pmatrix: np.ndarray, agent_map: dict , pgraph: ProductionGraph ) -> None: 
        
        """
        Initialize a PriceLooker instance.
        
        Parameters
        ----------
        pmatrix : np.ndarray
            3-dimensional numpy array representing the price matrix.
        agent_map : dict
            Dictionary mapping agent IDs to locations.
        pgraph : ProductionGraph
            A ProductionGraph instance representing the production graph.
        
        Raises
        ------
        ValueError
            If the production graph is not directed.
            If the price matrix is not a 3-dimensional numpy array.
            If the price matrix dimensions do not match the number of goods and locations.
        """
        
        self.pmatrix = pmatrix
        self.agent_map = agent_map
        self.pgraph = pgraph

        digraph = self.pgraph.get_production_graph()

        if not nx.is_directed(digraph):
            raise ValueError("Production graph must be directed")
        
        n_agents = len(set(agent_map.values())) # Unique locations
        n_goods = len(digraph.nodes)

        if pmatrix.ndim != 3:
            raise ValueError("Price matrix must be a 3-dimensional numpy array")
        
        m , n0 , n1 = pmatrix.shape
        if m != n_goods or n0 != n_agents or n1 != n_agents:
            raise ValueError(f"Price matrix dimensions must be ({n_goods}, {n_agents}, {n_agents})")
        
        good_indices = list(nx.topological_sort(digraph))
        self.good_indices = {good: index for index, good in enumerate(good_indices)}
        

    
    def determine_price(self, order: BaseBuyOrder) -> float:
        """
        Determine the price for a given order based on the agent's location and the good involved.
        """
        buyer = order.buyer
        seller = order.seller 
        good = order.good

        if not isinstance(order, BaseBuyOrder):
            raise ValueError("Order must be a BaseBuyOrder to determine price")
        
        buyer_location = self.agent_map[buyer]
        seller_location = self.agent_map[seller]
        good_index = self.good_indices[good]

        price = self.pmatrix[good_index, buyer_location, seller_location]
        return price
import networkx as nx
from typing import List, Dict

from dataclasses import dataclass

from .production_graph import ProductionGraph

### Author: ANDRES LEGUIZAMON / CARLOS SANCHEZ / GERMAN RODRIGUEZ / SOFIA OCAMPO 
### 2025
### TODO : LICENSE / COPYRIGHT 

### DESCRIPTION: Class that represents the local production graph of the firm, its derived from a parent production graph

class ConstrainedProductionGraph(ProductionGraph):
    def __init__(self, constrained_nodes: List, parent_production_graph: ProductionGraph) -> None:

        """
        Creates a constrained production graph from a parent production graph given a list of nodes.

        Parameters
        ----------
        constrained_nodes : List
            A list of nodes to be included in the constrained production graph.
        parent_production_graph : ProductionGraph
            The original production graph from which the constrained production graph is derived.
        """
        # Create the subgraph from the parent
        constrained_subgraph = parent_production_graph.get_production_graph().subgraph(constrained_nodes).copy()


        
        # Call parent class constructor with the new graph
        super().__init__(production_graph=constrained_subgraph)
        
        # Store references if needed later
        self.parent_production_graph = parent_production_graph
        self.constrained_nodes = constrained_nodes
        self.global_nodes = parent_production_graph.get_nodes()
       
    def classify_goods(self):
        """
        Classifies goods in the constrained production graph, extending the base classification
        from the parent production graph. Goods are classified as primary, intermediate, final,
        non-produced, or non-related.

        Non-produced goods are nodes in the constrained graph that are classified as non-related
        in the parent graph. Non-related goods are nodes present in the global production graph
        but not in the constrained production graph.

        Returns
        -------
        Dict[str, str]
            A dictionary where each key is a node in the global production graph, and its value is
            a string indicating whether it is a primary, intermediate, final, non-produced, or
            non-related good.
        """

        base_classification = super().classify_goods()

        extended_classification = {
            good: ("non produced" if classification == "non related" else classification)
            for good, classification in base_classification.items()
        }
    
        for node in self.global_nodes:
            if node not in self.constrained_nodes:
                extended_classification[node] = "non related"

        return extended_classification
    
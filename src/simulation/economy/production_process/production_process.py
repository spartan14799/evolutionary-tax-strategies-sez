import networkx as nx
from typing import List, Dict

from dataclasses import dataclass

from .production_graph import ProductionGraph


### Author: ANDRES LEGUIZAMON / CARLOS SANCHEZ / GERMAN RODRIGUEZ / SOFIA OCAMPO 
### 2025
### TODO : LICENSE / COPYRIGHT 

### DESCRIPTION: Class that gives information abouyt the production process as an interface to the planner
# This class implement the production process information that will be further used by the planner

class ProductionProcess:
    def __init__(self, pgraph: ProductionGraph):
        """
    Initializes a `ProductionProcess` object with a production graph.

    Args:
        pgraph (nx.DiGraph): Production graph.

    Attributes:
        _direct_inputs (Dict[str, List[str]]): Dictionary that maps each node to a list of nodes
            to which it is directly connected in a directed graph (i.e., its direct inputs).
        _direct_outputs (Dict[str, List[str]]): Dictionary that maps each node to a list of nodes
            that are directly connected to it in a directed graph (i.e., its direct outputs).
        _goods_classification (Dict[str, str]): Dictionary that maps each node to its classification as
            "primary_good", "intermediate_good", or "final_good".
    """
        self._pgraph = pgraph

        self._direct_inputs = pgraph.generate_direct_inputs()  

        self._direct_outputs = pgraph.generate_direct_outputs()

        self._goods_classification = pgraph.classify_goods()  # diccionario

        self._required_quantities = pgraph.generate_required_good_quantities()

    def get_inputs(self, good):
        return self._direct_inputs[good]
    
    def get_outputs(self, good):
        return self._direct_outputs[good]

    def get_goods_classification(self):
        return self._goods_classification

    def get_required_quantities(self):
        return self._required_quantities

    def get_graph(self):
        """
        Returns the production graph used in the production process. 

        Returns
        -------
         ProductionGraph
            The production graph used in the production process.
        """
        return self._pgraph

    def create_production_plan(self, plan: List[int]):
        action_map = {
            "primary": "Buy",
            "intermediate": "Produce",  # Maps actions depending on the type of good 
            "final": "Produce",
        }

        agent_map = {1: "NCT", 0: "ZF"}  # Mapping to the agent types 

        #### calculate the total required steps in the production process 

        required_steps = sum(self._required_quantities.values())+1

        # Validate plan length 
        # Ensure the plan has enough steps to cover all required goods
        if len(plan) < required_steps:
            raise ValueError(
                "Plan must have at least {} steps(indications).".format(required_steps)
            )
        production_plan = []  # Stores the list of transactions involving the production plan 

        position = 0

        # Assing Buy action to primary goods 

        for good, quantity in self._required_quantities.items():
            for _ in range(quantity):
                production_plan.append(
                    (
                        good,
                        action_map[self._goods_classification[good]],
                        agent_map[plan[position]],
                    )
                )
                position += 1

        # Asignation of production for final and intermediate goods 

        # Append the sale to the market as the last action in the production plan 

        final_good = [
            good
            for good, classification in self._goods_classification.items()
            if classification == "final"
        ][0]
        
        penultimate_agent = agent_map[plan[required_steps-2]]

        last_agent = agent_map[plan[required_steps-1]]

        if penultimate_agent != last_agent:

            production_plan.append((final_good, 'Buy', last_agent))
            

        return production_plan
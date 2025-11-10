
from collections import defaultdict
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

from classes.economy.production_process.production_graph import ProductionGraph
from classes.economy.production_process.constrained_production_graph import ConstrainedProductionGraph

#! TODO añadir método que identifica insumos y productos 
#! TODO: Añadir un método o atributo dentro del context quie me diga la ubicación del agente 

### Author: ANDRES LEGUIZAMON / CARLOS SANCHEZ / GERMAN RODRIGUEZ / SOFIA OCAMPO 
### 2025
### TODO : LICENSE / COPYRIGHT 

### DESCRIPTION: Class that represents the economic context of the agent , the economic context will be used in the accounting process 


@dataclass
class AgentContext(ABC):
    pass 

class EconomicContext(AgentContext):
    """ Class to store production information , account information, and any other relevant information of the agent """
    def __init__(self,global_production_graph: ProductionGraph , firm_related_goods: List[str], income_statement_type: str = "standard"):
        """
        Initialize the EconomicContext with a global production graph and a list of firm-related goods.

        Parameters
        ----------
        global_production_graph : ProductionGraph
            The global production graph of the economy.
        firm_related_goods : List[str]
            A list of goods related to the firm, used to create the local production graph.
        """

        self.global_production_graph = global_production_graph
        self.local_production_graph = ConstrainedProductionGraph(firm_related_goods, global_production_graph)
        self.income_statement_type = income_statement_type
        
    def get_global_production_graph(self) -> ProductionGraph:
        """
        Get the global production graph of the economy.

        Returns
        -------
        ProductionGraph
            The global production graph of the economy.
        """
        return self.global_production_graph
    
    def get_local_production_graph(self) -> ConstrainedProductionGraph:
        """
        Get the local production graph of the agent, which is a subgraph of the global production graph
        containing only the nodes related to the agent's goods.

        Returns
        -------
        ConstrainedProductionGraph
            The local production graph of the agent.
        """
        return self.local_production_graph
    
    @property
    def local_classification(self):
        """
        Get the classification of the goods in the local production graph.

        Returns
        -------
        Dict[str, str]
            A dictionary where each key is a node in the local production graph and its value is
            a string indicating whether it is a primary, intermediate, final or non related
            good.
        """
        return self.local_production_graph.classify_goods()
        



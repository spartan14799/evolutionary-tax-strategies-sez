import networkx as nx
from typing import List, Dict

from dataclasses import dataclass

### Author: ANDRES LEGUIZAMON / CARLOS SANCHEZ / GERMAN RODRIGUEZ / SOFIA OCAMPO 
### 2025
### TODO : LICENSE / COPYRIGHT 

### DESCRIPTION: Class that represents the global production graph of the economy 

class ProductionGraph:
    def __init__(self, production_graph: nx.DiGraph):
        if not nx.is_directed_acyclic_graph(production_graph):
            raise ValueError("The production graph must be a directed acyclic graph.")
        self.production_graph = production_graph

    def get_production_graph(self) -> nx.DiGraph:
        return self.production_graph
    
    def get_nodes(self) -> List[str]:
        """
        Gets the nodes in th directed graph organized in topological sort, 
        this preserves in some sense the "order" given by the production binary relation defined in the graph

        Returns:
            List[str]: List of nodes in the graph in topological sort
        """
        return list(nx.topological_sort(self.production_graph))

    def verify_direct_conection(self, node1: str, node2: str) -> bool:
        
        """
        Verifies if there is a direct connection between two nodes in the directed graph.
        i.e. if there is an edge from node1 to node2.

        Args:
            node1 (str): First node to check.
            node2 (str): Second node to check.

        Returns:
            bool: True if there is a direct connection, False otherwise.
        """
        return self.production_graph.has_edge(node1, node2)
    
    def generate_direct_inputs(self) -> Dict[str, List[str]]:
       
        """
        Generates a dictionary that maps each node to a list of nodes
        that are directly connected to it in the directed graph.
        (i.e nodes with directed connections to the node)


        Args:
            Uses the internal directed graph

        Returns:
            Dict[str, List[str]]: A dictionary where each key is a node in the graph
            and its value is a list of nodes to which it is directly connected.
        """
        nodes = self.get_nodes()
        direct_inputs = {}

        for node in nodes:
            direct_inputs[node] = []

        for node1, node2 in self.production_graph.edges():
            direct_inputs[node2].append(node1)

        return direct_inputs 
    


    def generate_direct_outputs(self) -> Dict[str, List[str]]:
     
        """
        Generates a dictionary that maps each node to a list of nodes
        that are directly connected from it in the directed graph.
        (i.e., nodes with directed connections originating from the node)

        Returns:
            Dict[str, List[str]]: A dictionary where each key is a node in the graph
            and its value is a list of nodes to which it is directly connected from.
        """

        nodes = self.get_nodes()
        direct_outputs = {}

        for node in nodes:
            direct_outputs[node] = []
        for node1, node2 in self.production_graph.edges():
            direct_outputs[node1].append(node2)
        return direct_outputs



    def classify_goods(self) -> Dict[str, List[str]]:
   

        """
        Classifies goods in the graph into primary, intermediate , final and non related goods.

        Primary goods are goods that have no inputs, i.e. they are not used to
        produce any other goods (empty inputs). 
        
        Intermediate goods: goods that have inputs
        and are used to produce other goods(non empty inputs and outputs).
         
        Final goods : goods that have
        inputs but are not used to produce any other goods(empty outputs).

        Non Related Goods: Goods that have neither inputs nor outputs (empty inputs and outputs).

        Parameters
        ----------
            Internal Graph of production in the class.

        Returns
        -------
        Dict[str, str]
            A dictionary where each key is a node in the graph and its value is
            a string indicating whether it is a primary, intermediate final or non related
            good.
        """
       
        inputs = self.generate_direct_inputs()
        outputs = self.generate_direct_outputs()
        good_classification = {}

        for node in inputs:
            if not inputs[node] and not outputs[node]:
                good_classification[node] = "non related"
            elif not inputs[node]:
                good_classification[node] = "primary"
            elif not outputs[node]:
                good_classification[node] = "final"
            else:
                good_classification[node] = "intermediate"

        return good_classification


    
    def get_primary_goods(self) -> List[str]:
    

        """
        Gets a list of all primary goods in the graph.

        Parameters
        ----------
        None: Uses internal graph of production of the class 

        Returns
        -------
        List[str]
            A list of all primary goods in the graph.
        """
        good_classification = self.classify_goods()
        primary_goods = [
            good
            for good, classification in good_classification.items()
            if classification == "primary"
        ]
        return primary_goods


    def get_intermediate_goods(self) -> List[str]:
     
        """
        Gets a list of all intermediate goods in the graph.

        Parameters
        ----------
        None: Uses internal graph of production of the class 

        Returns
        -------
        List[str]
            A list of all intermediate goods in the graph.
        """
        good_classification = self.classify_goods()
        intermediate_goods = [
            good
            for good, classification in good_classification.items()
            if classification == "intermediate"
        ]
        return intermediate_goods

    def get_final_goods(self) -> List[str]:
       
        """
        Gets a list of all final goods in the graph.

        Parameters
        ----------
        None: Uses internal graph of production of the class 

        Returns
        -------
        List[str]
            A list of all final goods in the graph.
        """
        good_classification = self.classify_goods()
        final_goods = [
            good
            for good, classification in good_classification.items()
            if classification == "final"
        ]
        return final_goods

    def get_non_related_goods(self) -> List[str]:
       
        """
        Gets a list of all non related goods in the graph.

        Parameters
        ----------
        None: Uses internal graph of production of the class 

        Returns
        -------
        List[str]
            A list of all non related goods in the graph.
        """
        good_classification = self.classify_goods()
        non_related_goods = [
            good
            for good, classification in good_classification.items()
            if classification == "non related"
        ]
        return non_related_goods

    def count_paths_x_to_y(self, x:str, y:str) -> int:
     
        """
        Counts the number of paths from node x to node y in the graph.

        Parameters
        ----------
        x : str
            Initial node.
        y : str
            Final node.

        Returns
        -------
        int
            The number of paths from node x to node y in the graph.
        """
        
        paths = nx.all_simple_paths(self.production_graph, source=x, target=y)
        return len(list(paths))
    
    def generate_required_good_quantities(self) -> Dict[str, int]:
  
        """
        Generates the quantity of each good required to produce the final good. 
        This assumes that economy global production process has a global Leontieff with unitary coefficents
        production process

        Parameters
        ----------
        None: Uses internal graph of production of the class 

        Returns
        -------
        Dict[str, int]
            A dictionary where each key is a node in the graph and its value is
            the quantity of the good required to produce the final good.
        """
        #! TODO: Current implementation assumes a monoproduct graph 
        required_good_quantities = {}
        goods = self.get_primary_goods()
        goods = goods + self.get_intermediate_goods()
        final_good = self.get_final_goods()[0] #! Modifying this line allows for more than one final good

        for good in goods:
            required_good_quantities[good] = self.count_paths_x_to_y(good, final_good)
        # Agregar el bien final con cantidad 1

        for final_good in self.get_final_goods():
            required_good_quantities[final_good] = 1
        return required_good_quantities





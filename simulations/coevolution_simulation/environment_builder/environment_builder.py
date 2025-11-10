
### Author: ANDRES LEGUIZAMON / CARLOS SANCHEZ / GERMAN RODRIGUEZ / SOFIA OCAMPO 
### 2025
### TODO : LICENSE / COPYRIGHT 

### DESCRIPTION: Classs That parametrically generates environments for simulation from yaml environemnt files 

import yaml

from pathlib import Path


from typing import List, Tuple , Dict , Any 

import networkx as nx

import numpy as np 




class EnvironmentBuilder:
    def __init__(self):
        """
        Initializes the EnvironmentBuilder.

        This class will be used to generate environments from yaml configuration files.
        """
        pass

    def read_graph_yaml(self, graph_yaml_path: Path) -> List[Tuple[str, str]]:
        """
        Reads a graph configuration from a YAML file.

        Args:
            graph_yaml_path (Path): Path to the YAML file containing the graph configuration (list of links)

        Returns:
            List[Tuple[str, str]]: List of links representing the graph structure ready for NetworkX
        """
        with open(graph_yaml_path, "r") as file:

            graph_config = yaml.safe_load(file)

        return graph_config["link_list"]
    
    def read_price_matrix_yaml(self, price_matrix_yaml_path: Path) -> np.ndarray:
        """
        Reads a price matrix (tensor) from a YAML file and converts it into a NumPy array.

        Args:
            price_matrix_yaml_path (Path): Path to the YAML file containing the price tensor.

        Returns:
            np.ndarray: A 3D NumPy array of integers representing the price matrix.
        """
        with open(price_matrix_yaml_path, "r") as file:
            price_data = yaml.safe_load(file)

        tensor = np.array(price_data["tensor"], dtype=int)
        return tensor

    def read_agents_info_yaml(self, agents_yaml_path: Path) -> Dict[str, Any]:
        """
        Reads the agents information YAML file.

        Args:
            agents_yaml_path (Path): Path to the YAML file containing agent configurations.

        Returns:
            dict: Dictionary mapping agent names to their configuration dictionaries.
        """
        with open(agents_yaml_path, "r", encoding="utf-8") as file:

            agents_info = yaml.safe_load(file)

        return agents_info["agents_info"]
    

    
    def build_environment(self, environment_yaml_path: Path) -> Dict[str, Any]:
        """
        Builds the full environment by reading all referenced YAML files.

        Args:
            environment_yaml_path (Path): Path to the YAML file containing the environment configuration.

        Returns:
            dict: Dictionary with fully loaded environment data:
                {
                    "graph_info": list of (node_from, node_to),
                    "price_matrix": np.ndarray,
                    "agents_info": dict,
                    "chart_of_accounts": Path
                }
        """
        with open(environment_yaml_path, "r") as file:
            environment_config = yaml.safe_load(file)

        base_dir = environment_yaml_path.parent
        yaml_paths = environment_config["environment"]["yaml_paths"]

        #--------------------------------------

        #1. Parse Chart of Accounts Path
        chart_path = base_dir / yaml_paths["chart_of_accounts"]

        #--------------------------------------

        #2. Parse Graph Path
        graph_path = base_dir / yaml_paths["graph_info"]

        #2.1 read Graph and store link list
        graph_info =  self.read_graph_yaml(graph_path)

        graph = nx.DiGraph()
        graph.add_edges_from(graph_info)

        goods_list = list(graph.nodes)

        graph_len = len(goods_list)

        #--------------------------------------

        #3. Parse Price Matrix Path
        price_path = base_dir / yaml_paths["price_matrix"]

        #3.1 read Price Matrix and store as numpy array
        price_matrix = self.read_price_matrix_yaml(price_path)

        assert price_matrix.shape == (graph_len, 3, 3), f"Price matrix shape {price_matrix.shape} does not match expected shape {(graph_len, 3, 3)} based on graph nodes {goods_list}"

        #--------------------------------------

        #4. Parse Agents Info Path
        agents_path = base_dir / yaml_paths["agents_info"]
        
        #4.1 read Agents Info and store as dictionary

        agents_info = self.read_agents_info_yaml(agents_path)

        for agent_info in agents_info.values():

            agent_info['accounts_yaml_path'] = chart_path
            agent_info['firm_related_goods'] = goods_list



        env = {}

        env['graph_info'] = graph_info

        env['price_matrix'] = price_matrix

        env['agents_info'] = agents_info

        env['chart_of_accounts_path'] = chart_path

        env["nx_graph"] = graph

        return env
   
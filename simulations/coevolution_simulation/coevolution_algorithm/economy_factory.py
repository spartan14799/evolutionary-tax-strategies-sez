
### Author: ANDRES LEGUIZAMON / CARLOS SANCHEZ / GERMAN RODRIGUEZ / SOFIA OCAMPO 
### 2025
### TODO : LICENSE / COPYRIGHT 

### DESCRIPTION: Classs That parametrically generates economy classes for simulation from yaml environemnt files 

import yaml

from pathlib import Path


from typing import List, Tuple , Dict , Any 

import networkx as nx

import numpy as np

from simulations.coevolution_simulation.environment_builder.environment_builder import EnvironmentBuilder

from classes.economy.economy import Economy




class EconomyFactory:
    def __init__(self , env:Dict) -> None:
                    
            """
            Class representing a simulation environment built from preprocessed YAML files.

            Parameters
            ----------
            env : dict
                Dictionary containing the environment configuration, as returned by
                `EnvironmentBuilder.build_environment()`.

                Structure:
                ----------
                graph_info : list[tuple[str, str]]
                    List of directed edges (origin_good, destination_good) defining
                    the structure of the production graph.

                price_matrix : np.ndarray
                    3D NumPy tensor (n_goods × n_goods × n_goods) representing
                    relative prices or transformation costs among goods.

                agents_info : dict[str, dict]
                    Mapping of agent identifiers to their configuration dictionaries.
                    Each entry defines:
                        - type : str
                        - inventory_strategy : str
                        - firm_related_goods : list[str]
                        - income_statement_type : str
                        - accounts_yaml_path : pathlib.Path
                        - price_mapping : int

                chart_of_accounts_path : pathlib.Path
                    Absolute path to the YAML file defining the chart of accounts used
                    across all agents.

                nx_graph : networkx.DiGraph
                    Directed graph built from `graph_info`, used for validating and
                    traversing production dependencies.

            Notes
            -----
            This dictionary centralizes all simulation components—graph topology,
            price matrix, agents, and accounting scheme—and is intended to be passed
            directly to higher-level classes such as `Economy`, `Simulation`, or
            `Planner` for initialization and execution.
            """

            self.env = env 
            pass
    
    def create_economy(self, genome_evader , genome_auditor) -> Economy:
        
        return Economy(
            self.env["graph_info"],
            self.env["price_matrix"],
            self.env["agents_info"],
            genome_evader,
            genome_auditor
        )
import sys
sys.path.append(r"C:\Users\andre\Repositories\FTZ_model_2.0")

from pathlib import Path

from classes.economy.economy import Economy

import numpy as np 
from classes.economy.agent.reporting.income_statement import generate_income_statement
import networkx as nx

graph_info =  [("Input", "Intermediate"),
        ("Intermediate", "Final")]


digraph_test = nx.DiGraph(graph_info)


price_matrix = np.array([[[102, 132, 129],
                          [ 65,  40,  85],
                          [ 5000,  17, 113]],

                         [[ 25,  42,  82],
                          [ 40,  56,  15],
                          [ 89,  55,  94]],

                         [[144, 125,  91],
                          [137, 107,  41],
                          [ 97,  88,  48]]])

print(price_matrix)


goods_list = list(digraph_test.nodes())
goods_list



ROOT_DIR = Path(r"C:\Users\andre\Repositories\FTZ_model_2.0")
accounts_path = ROOT_DIR / "chart_of_accounts.yaml"


agents_info = { "MKT":
               {"type":"MKT",
                 "inventory_strategy": "FIFO", 
                 "firm_related_goods":goods_list, 
                 "income_statement_type": "standard" ,
                "accounts_yaml_path": accounts_path, 
                "price_mapping":0} , 

                "NCT":
               {"type":"NCT",
                 "inventory_strategy": "FIFO", 
                 "firm_related_goods":goods_list, 
                 "income_statement_type": "standard" ,
                "accounts_yaml_path": accounts_path, 
                "price_mapping":1} , 

                "ZF":
               {"type":"ZF",
                 "inventory_strategy": "FIFO", 
                 "firm_related_goods":goods_list, 
                 "income_statement_type": "standard" ,
                "accounts_yaml_path": accounts_path, 
                "price_mapping":2}
               }


test_economy = Economy(graph_info, price_matrix, agents_info , genome = [1,1,0,1,1,0,0,0,1,1,1])


for order in test_economy.orders:
    print(order)




a = test_economy.get_reports()

print(a)
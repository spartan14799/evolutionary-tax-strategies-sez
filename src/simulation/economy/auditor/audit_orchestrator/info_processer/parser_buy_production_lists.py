
### DESCRIPTION: Class that takes a list of transactions and constructs the quantity flow matrix between agents for each good 


# Standard Format of transactions BuyOrder(buyer='ZF', seller='MKT', good='Input', transaction_type='Buy', price=24)

from src.simulation.economy.order_book.order.orders import BuyOrder,ProductionOrder

import numpy as np 

from typing import List , Tuple , Union, Dict


import numpy as np
from typing import List, Union , Dict
from collections import defaultdict


class BuyProductionListsParser:
    def __init__(self):
        """
        Parser utility to separate Buy and Production orders into a nested dictionary structure.

        This class provides methods to process individual orders or lists of orders,
        grouping them by agent and by good. It is intended to create a data structure of the form:

            {
                "AgentA": {
                    "good1": [order1, order2, ...],
                    "good2": [order3, ...]
                },
                "AgentB": {
                    "good3": [order4, ...]
                }
            }

            Where the orders are Buy Orders where agents was the buyer of the key good and Production Orders where the agent was the producer of the key good.
            
        """
        pass

    def parse_order(
        self,
        order: Union["BuyOrder", "ProductionOrder"],
        orders_dict: Dict[str, Dict[str, List[Union["BuyOrder", "ProductionOrder"]]]],
    ) -> None:
        """
        Parse a single Buy or Production order and insert it into the provided dictionary.

        Parameters
        ----------
        order : Union["BuyOrder", "ProductionOrder"]
            The transaction to be parsed. Must have attributes:
                - transaction_type : str, either "Buy" or "Production"
                - good : str, the good being transacted or produced
                - buyer / producer : str, the agent involved (depends on type)
        orders_dict : Dict[str, Dict[str, List[Union["BuyOrder", "ProductionOrder"]]]]
            The dictionary that stores parsed orders. It will be updated in place.
            Structure:
                {agent: {good: [list_of_orders]}}

        Raises
        ------
        ValueError
            If the order's transaction type is not "Buy" or "Production".

        Example
        -------
        >>> parser = BuyProductionListsParser()
        >>> orders_dict = {}
        >>> order = BuyOrder(buyer="NCT", good="good1", ...)
        >>> parser.parse_order(order, orders_dict)
        >>> print(orders_dict)
        {"NCT": {"good1": [<BuyOrder ...>]}}
        """
        txn_type = order.transaction_type

        if txn_type == "Buy":
            agent = order.buyer
            good = order.good

        elif txn_type == "Production":
            agent = order.producer
            good =order.produced_good
        else:
            raise ValueError(f"Not recognized transaction type: {txn_type}")

        # Ensure nested structure exists
        if agent not in orders_dict:
            orders_dict[agent] = defaultdict(list)
        orders_dict[agent][good].append(order)

    def parse_order_list(
        self,
        order_list: List[Union["BuyOrder", "ProductionOrder"]]
    ) -> Dict[str, Dict[str, List[Union["BuyOrder", "ProductionOrder"]]]]:
        """
        Parse a list of Buy and Production orders into the provided dictionary.

        Parameters
        ----------
        order_list : List[Union["BuyOrder", "ProductionOrder"]]
            List of orders to parse.
        orders_dict : Dict[str, Dict[str, List[Union["BuyOrder", "ProductionOrder"]]]]
            Dictionary that will be updated with parsed orders.

        Notes
        -----
        - This method calls `parse_order` internally for each order.
        - The dictionary is updated in place and not returned.

        Example
        -------
        >>> parser = BuyProductionListsParser()
        >>> orders_dict = {}
        >>> order1 = BuyOrder(buyer="NCT", good="good1", ...)
        >>> order2 = ProductionOrder(producer="ZF", good="good2", ...)
        >>> parser.parse_order_list([order1, order2], orders_dict)
        >>> print(orders_dict)
        {
            "NCT": {"good1": [<BuyOrder ...>]},
            "ZF": {"good2": [<ProductionOrder ...>]}
        }
        """
        orders_dict = {}
        for order in order_list:
            self.parse_order(order, orders_dict)
        return orders_dict




    

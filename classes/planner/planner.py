
import networkx as nx
import numpy as np
from typing import List, Dict, Tuple, Any 
from classes.economy.production_process.production_graph import ProductionGraph
from classes.economy.production_process.production_process import ProductionProcess


class DummyAgent: 
    id = 0
    def __init__(self, name , inventory: Dict[str, List[str]]):
        self.name = name
        self.inventory = inventory

        self.id = DummyAgent.id
        DummyAgent.id += 1

    def __getattribute__(self, name):
        try: 
            return object.__getattribute__(self, name)
        except AttributeError:
            return f"Agent {self.id} does not have attribute {name}"
        
    def add_to_inventory(self, good, asset): 
        if good in self.inventory:
            self.inventory[good].append(asset)
        else:
            self.inventory[good] = [asset]

    def remove_from_inventory(self, good, asset):
        if good in self.inventory and asset in self.inventory[good]:
            self.inventory[good].remove(asset)
            if len(self.inventory[good]) == 0:
                del self.inventory[good]
        else:
            raise ValueError(f"Asset {asset} not found in inventory for good {good}")
        




def cut_list(
    list: List[Tuple[str, str, str]], order: str, first_part: bool
) -> List[Tuple[str, str, str]]:
    """
Splits the list at the last occurrence of the specified action.

Args:
    lista (List[Tuple[str, str, str]]): List of orders.
    orden (str): Action at which the list should be split.
    primera_parte (bool): If True, returns the first part up to the last occurrence of 'orden'.
                        If False, returns the second part after the last occurrence of 'orden'.

Returns:
    List[Tuple[str, str, str]]: List split according to the selected option.
"""
    for i in range(len(list) - 1, -1, -1):
        if list[i][1] == order:
            return list[: i + 1] if first_part else list[i + 1 :]

    print("Order not found in list.")
    return list  # if order not found, return the entire list



class Planner:
    def __init__(self, production_process : ProductionProcess):

        """
        Initializes a Planner object with a production process.

        Args:
            production_process (ProductionProcess): Object with the production process information.

        Attributes:
            _production_process (ProductionProcess): Object with the production process information.
            required_quantities (Dict[str, int]): Dictionary with the required quantities of each good.
            good_classification (Dict[str, str]): Dictionary with the classification of each good.
            _market (DummyAgent): Dummy agent representing the market.
            _agents (Dict[str, DummyAgent]): Dictionary with the dummy agents.
            transactions (List[Tuple[str, str, str]]): List of transactions in the simulation.
        """
        self._production_process = production_process


        self.required_quantities = (
            self._production_process.get_required_quantities()  # Dictionary with required quantities
        )



        self.good_classification = self._production_process.get_goods_classification()  # Dictionary with good classification



        # Here it starts a Dummy paseudo simulation to make the mapping of the production process to the agents and the market

        # Inicializar al mercado con los bienes primarios

        self._market = DummyAgent("MKT", {})
        for good, quantity in self.required_quantities.items():

            if self.good_classification[good] == "primary":
                for i in range(1, quantity + 1):
                    self._market.add_to_inventory(good, f"unit_{i}")

        # Initialize dummy simulation agents 

        self._agents = {
            "NCT": DummyAgent("NCT", {}),
            "ZF": DummyAgent("ZF", {}),
            "MKT": self._market,
        }

        # Initialize transaction sequence 
        self.transactions  = []

    def create_production_plan(self, plan: List[int]):
        return self._production_process.create_production_plan(plan)

    def reset(self):
        """
        Resets the planner to its initial state of the simulation, reinitializing the production process.

        """
        self.__init__(self._production_process)

    def process_buy_order(
        self, order: Tuple[str, str, str], initial: bool = False, step: int = 0
    ):  # <-- Cambio
        """
        Processes  a Buy order in the form of (Good, Buy , Agent) 
        """
        good, action, agent = order

        if action != "Buy":
            raise ValueError(" Order must be of type 'Buy'.")

        # Filtrar los agentes (vendedores) según sea compra inicial o no
        if initial:

            potential_sellers = {k: v for k, v in self._agents.items() if k == "MKT"}

        else:
            potential_sellers = {
                k: v for k, v in self._agents.items() if k != agent
            }

        # Search for a seller that has the good available

        found_seller = None

        for agent_name, agent_instance in potential_sellers.items():
            if good in agent_instance.inventory and agent_instance.inventory[good]:
                found_seller = agent_name
                break

        if found_seller:
            # Transfer firs unit available from the seller to the buyer

            unit = self._agents[found_seller].inventory[good][0]
            self._agents[found_seller].remove_from_inventory(good, unit)
            self._agents[agent].add_to_inventory(good, unit)

            # Register the transacción: (step, (Buyer, Seller), "Buy", Good)  # <-- Cambio
            self.transactions.append(
                (step, (agent, found_seller), "Buy", good)
            )
        else:
            raise ValueError(f" No agent has good '{good}' available for sale.")
        


    def process_production_order(
            
        self, order: Tuple[str, str, str], step: int = 0
    ):  # <-- Cambio
        """

        Processes a production order in the form (Good, "produce", Agent).
        """
        good, action, agent = order

        if action != "Produce":
            raise ValueError(" Order must be of type 'Produce'.")
        

        # Obtain and validate inputs for production

        required_inputs  = self._production_process._direct_inputs[good]

        #  Validate that the agent has all required inputs in its inventory otherwise buy 
        for input in required_inputs:
            if (
                input not in self._agents[agent].inventory
                or not self._agents[agent].inventory[input]
            ):
                # Recursively try to buy the input

                self.process_order((input, "Buy", agent), step = step)

        # Add produced good to the agent's inventory

        self._agents[agent].add_to_inventory(good, f"unit_{good}")
      
        #  Consume the required inputs from the agent's inventory
        
        for input in required_inputs:
            unit_input = self._agents[agent].inventory[input][0]
            self._agents[agent].remove_from_inventory(input, unit_input)



        # register the transaction: (step, (agent,), "Produce", good)

        self.transactions.append(
            (step, (agent,), "Produce", good)
        )

    def process_order(
        self, order: Tuple[str, str, str], initial: bool = False, step: int = 0
    ): 
        """
    Given an order in the form (good, action, agent), delegates the processing
    to either 'process_purchase' or 'process_production'.
    """
        good, action, agent = order
        if action == "Buy":
  
            self.process_buy_order(order, initial=initial, step=step)
        elif action == "Produce":
            self.process_production_order(order, step=step)
        else:
            raise ValueError(f"Not recognized action: {action}")

    def execute_plan(self, plan: List[Tuple[str, str, str]]):
        """
Executes a production plan by processing each order.
Returns the list of transactions along with the step at which they occurred.
"""
        self.reset()

        # create complete plan 

        production_plan  = self.create_production_plan(plan)

        sequence_3 = [production_plan[-1]]

        filtered_plan = production_plan[:-1]

        # Separar secuencias de compra inicial vs. resto
        sequence_1 = cut_list(filtered_plan, "Buy", 1)
        sequence_2 = cut_list(filtered_plan, "Buy", 0)

        # Step counter 
        step = 1

        # Process the initial part of the sequence (inital buy orders from the market (agent buying primary goods))
        for order in sequence_1:
            self.process_order(order, initial=True, step=step)
            step += 1

        # Process Resting part of the sequence (production orders and buy orders from agents)
        for order in sequence_2:
            self.process_order(order, initial=False, step=step)
            step += 1
        
        for order in sequence_3:
            self.process_order(order, initial=False, step=step)
            step += 1

        # Add final transactions (sell to the market)
    
        good_classification = self._production_process.get_goods_classification()

        final_goods = [
            good for good, classification in good_classification.items() if classification == "final"
        ]

        final_good = final_goods[0] if final_goods else None

        last_producer = self.transactions[-1][1][0] 

        self.transactions.append(
            (step, ("MKT",last_producer), "Buy", final_good)
        )



        return self.transactions

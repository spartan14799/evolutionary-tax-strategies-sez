
### Author: ANDRES LEGUIZAMON / CARLOS SANCHEZ / GERMAN RODRIGUEZ / SOFIA OCAMPO 
### 2025
### TODO : LICENSE / COPYRIGHT 

### DESCRIPTION: Draft of the agent Class, as of now, only inventory movements are implemented. 

from typing import Dict, List, Tuple 


from ..assets.good import Good



from classes.economy.production_process.production_graph import ProductionGraph

from classes.economy.agent.context.agent_context import EconomicContext

from classes.economy.agent.inventory.inventory import Inventory

from  classes.economy.agent.inventory.inventory_strategy import InventoryStrategy

from classes.economy.agent.acc_agent import AccAgent





class Agent(): 
    id = 0 
    def __init__(self , type:str , global_production_graph: ProductionGraph, inventory_strategy: InventoryStrategy, firm_related_goods: List[str] 
                 , income_statement_type: str = "standard" , accounts_yaml_path: str = None):
        
        self.id = Agent.id
        
        Agent.id += 1

        self.type = type

        self._economic_context = EconomicContext(global_production_graph, firm_related_goods, income_statement_type)

        self._inventory = Inventory(inventory_strategy)

        self._accounting_agent = AccAgent(accounts_yaml_path)

        



    def get_inventory(self) -> Inventory:
        """
        Returns the inventory of the agent.
        
        Returns:
            Inventory: The inventory object containing goods.
        """
        return self._inventory
    

    def get_economic_context(self) -> EconomicContext:
        """
        Returns the economic context of the agent.
        
        Returns:
            EconomicContext: The economic context object containing production and accounting information.
        """
        return self._economic_context
    
    def get_id(self) -> int:
        """
        Returns the unique identifier of the agent.
        
        Returns:
            int: The unique identifier of the agent.
        """
        return self.id
    
    def add_good_to_inventory(self, good: Good):
        """
        Adds a good to the agent's inventory.
        
        Args:
            good (Good): The good to be added to the inventory.
        """
        self._inventory.add_good(good)

    def remove_good_from_inventory(self, good: Good):
        """
        Removes a good from the agent's inventory based on its type.
        
        Args:
            good_type (str): The type of good to be removed from the inventory.
        
        Raises:
            ValueError: If there are no available units of the specified good type.
        """
        self._inventory.inventory[good.type].remove(good)

    
    def generate_produce_menu(self, good:str) -> Dict[str, Good]:

        if good not in self._economic_context.get_local_production_graph().get_nodes():
            raise ValueError(f"Good '{good}' is not part of the local production graph.")
        
        if self._economic_context.local_classification[good] == "non-related" or self._economic_context.local_classification[good] == "non-produced":
            raise ValueError(f"Good '{good}' Cannot be produced: {good} is classified as non-related or non-produced in the local production graph.")
        

        # Check if the agent has enough inventory to produce the good 

        required_inputs = self._economic_context.get_local_production_graph().generate_direct_inputs()[good]

        for input in required_inputs:
            if self.get_inventory().get_stock_quantity(input) <= 0:
                raise ValueError(f"Good '{good}' Cannot be produced: there is not enough '{input}' quantity to produce , current quantity {self.get_inventory().get_stock_quantity(input)}" )

        production_menu = {}

        for input in required_inputs:

            unit = self.get_inventory().get_unit(input)

            production_menu[input] = unit

        return production_menu


        


    def produce_good(self, good:str): 

        """ 
        
        Validates the instruction to produce a good, valdiates production constarints and uses necessary inputs from 

        inventory to produce the good and uodate its cost 
        
        """

        # Good validation 

        if good not in self._economic_context.get_local_production_graph().get_nodes():
            raise ValueError(f"Good '{good}' is not part of the local production graph.")
        
        if self._economic_context.local_classification[good] == "non-related" or self._economic_context.local_classification[good] == "non-produced":
            raise ValueError(f"Good '{good}' Cannot be produced: {good} is classified as non-related or non-produced in the local production graph.")
        
        # Check if the agent has enough inventory to produce the good 

        required_inputs = self._economic_context.get_local_production_graph().generate_direct_inputs()[good]

        for input in required_inputs:
            if self.get_inventory().get_stock_quantity(input) <= 0:
                raise ValueError(f"Good '{good}' Cannot be produced: there is not enough '{input}' quantity to produce , current quantity {self.get_inventory().get_stock_quantity(input)}" )
    
       # Si hay suficiente de cada bien , llamar a una instancia de cada bien del inventario

        production_cost = 0
        
        production_menu = self.generate_produce_menu(good)

        for input, unit in production_menu.items():
            production_cost += unit.cost
            self.get_inventory().remove_instance(unit) # Remove the unit from inventory after using it
            
        # Create a new Good instance for the produced good
        produced_good = Good(type=good, price=  production_cost) 
        produced_good.cost = production_cost
        produced_good.last_cost = production_cost # Assuming price is set later

        # Produce the good and update the inventory

        self.get_inventory().add_good(produced_good)
    
    # ---------------- Accounting façade ----------------
    @property
    def accountant(self):
        """Provides the public accountant handle expected by accounting modules.
        It forwards to the internal AccAgent instance."""
        return self._accounting_agent.accountant

    @property
    def chart_of_accounts(self):
        """Exposes the agent's Chart of Accounts as a public attribute,
        forwarding to the internal AccAgent."""
        return self._accounting_agent.accountant.chart_of_accounts



    # ---------------- Production façade ----------------
    @property
    def local_production_graph(self):
        """Exposes the local production graph as a direct property, which is
        what external logic expects when classifying goods and building entries."""
        return self._economic_context.get_local_production_graph()

    # ---------------- Inventory façade -----------------
    @property
    def inventory(self):
        """Exposes the inventory through a direct property. Forwarding is used
        to keep compatibility with code that references `agent.inventory`."""
        return self.get_inventory()

    # ---------------- Optional: location ----------------
    @property
    def location(self):
        """Exposes a location attribute for classifiers that rely on it.
        If the model encodes location in `type`, it forwards that value."""
        return getattr(self, "type", None)
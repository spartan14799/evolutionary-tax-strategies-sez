
from ...assets.good import Good
from collections import defaultdict
from abc import ABC, abstractmethod
from .inventory_strategy import InventoryStrategy

from .inventory_strategy import FIFOInventoryStrategy, LIFOInventoryStrategy 

from typing import List, Optional

### Author: ANDRES LEGUIZAMON / CARLOS SANCHEZ / GERMAN RODRIGUEZ / SOFIA OCAMPO 
### 2025
### TODO : LICENSE / COPYRIGHT 

### DESCRIPTION: Class that represents the inventory of goods that an agent posseses

# --- -----------------------------------------------------------------------------Inventory Class -------------------------------------------------------- ---
class Inventory:
    def __init__(self, inventory_strategy: str):
        # Diccionario: clave = tipo de bien (str), valor = lista de Good
        """
        Initializes an Inventory object with a given inventory strategy.

        Args:
            inventory_strategy (InventoryStrategy): The strategy to use when retrieving goods from the inventory.
        """
     

        self.inventory = defaultdict(list)

        if inventory_strategy is None:
            raise ValueError("Inventory strategy cannot be None")

        if inventory_strategy not in ["FIFO", "LIFO"]:
            raise ValueError("Inventory strategy must be either 'FIFO' or 'LIFO'")
        
        if inventory_strategy == "FIFO":
            self.inventory_strategy = FIFOInventoryStrategy()

        if inventory_strategy == "LIFO":
            self.inventory_strategy = LIFOInventoryStrategy()




    def add_good(self, unit: Good):
        
        """
        Adds a unit to the inventory.
        
        Args:
            unit (Good): The unit to be added.
        """
        self.inventory[unit.type].append(unit)



    def get_unit(self, good_type: str) -> Optional[Good]:

        """
        Returns a unit from the inventory based on the specified type and strategy.
        
        Args:
            good_type (str): The type of good to be retrieved.
        
        Returns:
            Good: The selected unit or None if not enough units of the given type are available.
        """

        goods = self.inventory[good_type]
        selected_good = self.inventory_strategy.select_good(goods)
        if selected_good is not None:
            return selected_good
        else:
            raise ValueError("Not Enough Units of good Type"+" "+good_type)
        

    def remove_unit(self, good_type: str):
        """
        Removes a unit from the inventory based on the specified type and strategy.
        
        Args:
            good_type (str): The type of good to be removed.
        
        Raises:
            ValueError: If there are no available units of the specified good type.
        """

        unit = self.get_unit(good_type)
        if unit is not None:
            self.inventory[good_type].remove(unit)
        else:
            raise ValueError("Not Enough Units of good Type"+" "+good_type)    

    def get_stock_quantity(self, good_type: str) -> List[Good]:
      
        """
        Returns the quantity of a given good type in the inventory.
        
        Args:
            good_type (str): The type of good to be retrieved.
        
        Returns:
            int: The number of units of the specified good type in the inventory.
        """

        return len(self.inventory[good_type])
    
    def get_stock(self, good_type: str) -> List[Good]:
        return self.inventory[good_type]
    
    def __repr__(self):
        return f"Inventory({dict(self.inventory)})"
    
    def remove_instance(self, good: Good):
        # 1. Verificar que el tipo existe en el inventario
        """
        Removes a specific instance of a good from the inventory.

        Args:
            good (Good): The specific good instance to be removed.

        Raises:
            KeyError: If the good type is not found in the inventory.
            ValueError: If the specified good instance is not in the inventory.
        """

        if good.type not in self.inventory:
            raise KeyError(f"No inventory found for good type '{good.type}'")

        goods_list = self.inventory[good.type]

        # 2. Verificar que la instancia está en la lista
        if good not in goods_list:
            raise ValueError(f"The specified good instance is not in the inventory of type '{good.type}'")

        # 3. Eliminar la instancia
        goods_list.remove(good)
from abc import ABC, abstractmethod
from ...assets.good import Good

from typing import List, Optional

### Author: ANDRES LEGUIZAMON / CARLOS SANCHEZ / GERMAN RODRIGUEZ / SOFIA OCAMPO 
### 2025
### TODO : LICENSE / COPYRIGHT 

### DESCRIPTION: Class that represents a the strategy to select goods to remove from the inventory 

class InventoryStrategy(ABC):
    
    @abstractmethod
    def select_good(self, goods: List[Good]) -> Optional[Good]:
        pass

class FIFOInventoryStrategy(InventoryStrategy):
    
    def select_good(self, goods: List[Good]) -> Optional[Good]:
        if goods:
            return goods[0]  # El primero que llegó
        return None

class LIFOInventoryStrategy(InventoryStrategy):
    """Implementa la estrategia LIFO: Last-In, First-Out."""
    def select_good(self, goods: List[Good]) -> Optional[Good]:
        if goods:
            return goods[-1]  # El último que llegó
        return None

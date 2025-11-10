from abc import ABC, abstractmethod
from assets.good import Good

### Author: ANDRES LEGUIZAMON / CARLOS SANCHEZ / GERMAN RODRIGUEZ / SOFIA OCAMPO 
### 2025
### TODO : LICENSE / COPYRIGHT 

### DESCRIPTION: Class that represents a batch of homogeneus or mixed goods , custom methods to calculates prices and costs yet to be implemented 

class Batch(ABC):  # <- Hereda de ABC
    def __init__(self, good_list):

        self._good_list = good_list

    def add_good(self, good):
        self._good_list.append(good)

    def remove_good(self, good):  
        self._good_list.remove(good)
    
    @abstractmethod
    def calculate_cost(self):
        pass  


from dataclasses import dataclass , field 
from abc import ABC, abstractmethod
from typing import List


@dataclass
class BaseBuyOrder:
    buyer: str
    seller: str
    good: str
    transaction_type: str = field(init=False, default="Buy")
    
@dataclass
class BaseProductionOrder:
    producer: str
    produced_good: str
    transaction_type: str =   field(init=False, default="Production")
    

@dataclass
class BuyOrder(BaseBuyOrder):
  price: int


@dataclass
class ProductionOrder(BaseProductionOrder):
   pass 

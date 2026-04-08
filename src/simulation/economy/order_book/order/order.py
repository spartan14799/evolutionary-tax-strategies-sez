
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import List
from src.simulation.economy.agent.agent import Agent

from .order_types import OrderType




class Order:
    def __init__(self, transaction_type:OrderType, involved_agents: List[str], involved_goods: List[str]):
        self.transaction_type = transaction_type
        self.involved_agents = involved_agents
        self.involved_goods = involved_goods

    @property
    def transaction_type(self):
        return self._transaction_type
    
    @property
    def involved_agents(self):
        return self._involved_agents
    
    @property
    def involved_goods(self):
        return self.involved_goods
    


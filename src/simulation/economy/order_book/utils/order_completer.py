
### Author: ANDRES LEGUIZAMON / CARLOS SANCHEZ / GERMAN RODRIGUEZ / SOFIA OCAMPO 
### 2025
### TODO : LICENSE / COPYRIGHT 

### DESCRIPTION: Class that completes orders.

from src.simulation.economy.order_book.order.orders import BaseBuyOrder, BaseProductionOrder, BuyOrder, ProductionOrder
from src.simulation.economy.order_book.utils.price_looker import PriceLooker

from src.simulation.economy.production_process.production_graph import ProductionGraph

from typing import List, Tuple, Union


class OrderCompleter:
    def __init__(self,price_looker: PriceLooker):
        """
        Initializes the OrderCompleter with a PriceLooker instance.
        
        Args:
            price_looker (PriceLooker): An instance of PriceLooker used to determine prices for orders.
        """

        self.price_looker = price_looker
    
    def complete_order(self, order: Union[BaseBuyOrder, BaseProductionOrder]) -> Union[BuyOrder, ProductionOrder]:
        """
        Completes a buy order by determining the price based on the agent's location and the good involved.
        """
        if isinstance(order, BaseBuyOrder): 
            price = self.price_looker.determine_price(order)
            
            final_order = BuyOrder(
                buyer=order.buyer,
                seller=order.seller,
                good=order.good,
                price=price
            )
        if isinstance(order, BaseProductionOrder):
            # For production orders, we might not need to determine a price, but we can still return a ProductionOrder object.
            final_order = ProductionOrder(
                producer=order.producer,
                produced_good=order.produced_good
            )
        return final_order
    
    def complete_orders(self, orders: List[Union[BaseBuyOrder, BaseProductionOrder]]) -> List[Union[BuyOrder, ProductionOrder]]:
        return [self.complete_order(order) for order in orders]
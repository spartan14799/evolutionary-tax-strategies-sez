
### Author: ANDRES LEGUIZAMON / CARLOS SANCHEZ / GERMAN RODRIGUEZ / SOFIA OCAMPO 
### 2025
### TODO : LICENSE / COPYRIGHT 

### DESCRIPTION: Instance of good represent a real unit of a good, includes a price to quote a buy order, the last cost paid by an agent to acquire 
### the good and the cost to quote a sell order or a accounting registration of the good

class Good: 
    def __init__(self, type, price = 0):
        """
        Initializes a Good instance.
        
        Args:
            type (str): The type of the good.
            price (int, optional): The price of the good. Defaults to 0.
        """
        self.type = type
        self._price = price
        self._cost = 0
        self._last_cost = 0

    @property
    def type(self):
        return self._type

    @property
    def price(self):
        return self._price
    
    @price.setter
    def price(self, value):
        self._price = value

    @property
    def cost(self):
        return self._cost
    
    @cost.setter
    def cost(self, value):
        self._cost = value

    @property
    def last_cost(self):
        return self._last_cost
    
    @last_cost.setter
    def last_cost(self, value):
        self._last_cost = value

    @type.setter
    def type(self, value):
        self._type = value




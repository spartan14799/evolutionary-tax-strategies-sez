### Author: ANDRES LEGUIZAMON / CARLOS SANCHEZ / GERMAN RODRIGUEZ / SOFIA OCAMPO 
### 2025
### TODO : LICENSE / COPYRIGHT 

### DESCRIPTION: This class encapsulates the logic of an auditor that recieves an interface of informatio 
### of the list of good matrices, normalized matrices, good flow and monetary flow and calculates measures 


from decimal import Decimal

class StatementMethods():

    def __init__(self):
        pass

    def calculate_asymmetry(self, value1, value2):
        """
        Calculates the asymmetry between two values.

        Parameters
        ----------
        value1 : float
            First value to calculate the asymmetry.
        value2 : float
            Second value to calculate the asymmetry.

        Returns
        -------
        float
            Asymmetry metric, a float representing the difference between the two values normalized by their sum plus a small epsilon value to avoid division by zero.
        """
        eps = Decimal('1e-9')
        return abs(value1 - value2) / (abs(value1) + abs(value2) + eps)
    

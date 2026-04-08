
### Author: ANDRES LEGUIZAMON / CARLOS SANCHEZ / GERMAN RODRIGUEZ / SOFIA OCAMPO 
### 2025
### TODO : LICENSE / COPYRIGHT 

### DESCRIPTION: This class encapsulates the logic of an auditor that recieves an interface of informatio 
### of the list of good matrices, normalized matrices, good flow and monetary flow and calculates measures 


from src.simulation.economy.auditor.audit_orchestrator.audit_patterns.pattern_measures.pattern_methods import PatternMethods

from src.simulation.economy.auditor.audit_orchestrator.info_processer.parser_buy_production_lists import BuyProductionListsParser



class PatternAuditor:
    def __init__(self):
        """
        Initializes a PatternAuditor object.

        This object encapsulates the logic of an auditor that recieves an interface of information
        of the list of good matrices, normalized matrices, good flow and monetary flow and calculates measures

        Attributes:
            pattern_methods (PatternMethods): Object containing methods to calculate pattern metrics
            buy_production_lists_parser (BuyProductionListsParser): Object containing methods to parse buy and production lists
        """
        self.pattern_methods = PatternMethods()
        self.buy_production_lists_parser = BuyProductionListsParser()
        pass
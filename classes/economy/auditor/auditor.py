


### Author: ANDRES LEGUIZAMON / CARLOS SANCHEZ / GERMAN RODRIGUEZ / SOFIA OCAMPO 
### 2025
### TODO : LICENSE / COPYRIGHT 

### DESCRIPTION: Draft of the agent Class, as of now, only inventory movements are implemented. 

from classes.economy.auditor.audit_orchestrator.audit_orchestrator import AuditOrchestrator

from classes.economy.auditor.audit_score_calculator.audit_score_calculator import AuditScoreCalculator

from typing import Dict , List


class Auditor:

    def __init__(self,audit_genome:List):

        """
        Initializes an Auditor object with the given audit genome.

        Parameters
        ----------
        audit_genome : List
            List of parameters for the audit, in the following order:
            1. Quantity weight (float)
            2. Monetary weight (float)
            3. List of weights for each metric in the audit scheme (List of floats)

        Returns
        -------
        None
        """
        self.mix = audit_genome[0]/(audit_genome[0]+audit_genome[1])

        self.audit_scheme = audit_genome[2:]

        self.audit_orchestrator = AuditOrchestrator(self.mix)

        self.audit_score_calculator = AuditScoreCalculator()

        pass




    def run_audit(self,price_matrix,transaction_list,production_graph , agent_reports):

        calculated_metrics_dict = self.audit_orchestrator.run_audit(transaction_list,agent_reports)

        genome_weights  = self.audit_scheme

        audit_score = self.audit_score_calculator.compute_score(calculated_metrics_dict,genome_weights)

        return audit_score 
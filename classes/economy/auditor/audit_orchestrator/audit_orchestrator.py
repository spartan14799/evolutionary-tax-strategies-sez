### DESCRIPTION: Class that takes a list of transactions and constructs the quantity flow matrix between agents for each good 


# Standard Format of transactions BuyOrder(buyer='ZF', seller='MKT', good='Input', transaction_type='Buy', price=24)

from classes.economy.order_book.order.orders import BuyOrder,ProductionOrder

import numpy as np 

from typing import List , Tuple , Union , Dict









from classes.economy.auditor.audit_orchestrator.audit_patterns.spectral_measures.spectral_auditor import SpectralAuditor


from classes.economy.auditor.audit_orchestrator.audit_patterns.pattern_measures.pattern_auditor import PatternAuditor

from classes.economy.auditor.audit_orchestrator.audit_patterns.statement_measures.statement_auditor import StatementAuditor



from classes.economy.auditor.audit_orchestrator.info_processer.parser_transactions_matrix import TransactionsMatrixParser
class AuditOrchestrator:

    def __init__(self, mix: float):
        """
        Initialize the AuditOrchestrator with specified mix and weights.

        Parameters
        ----------
        mix : float
            Weighting parameter between 0 and 1 (0=quantity only, 1=money only),
            used in the mixing of monetary and quantity flow weights.

            Mix should be calcualted by the one level up auditor from a mapping of auditing scheme genome 

        """
        self.pattern_auditor = PatternAuditor()
        self.statement_auditor = StatementAuditor()

        self.spectral_auditor = SpectralAuditor()

        self.matrix_parser = TransactionsMatrixParser(mix)

        self.good_weights = {}

        self.metrics = {}

    def update_matrix_parser(self, order_list):

        self.matrix_parser.process_order_list(order_list)

        self.good_weights = self.matrix_parser.calculate_mixed_weights()

    def run_spectral_audit(self, weights):
        """
        Runs the spectral audit and returns all computed metrics as a dictionary.

        Parameters
        ----------
        weights : dict or array-like
            Weights used for aggregating the spectral indices.

        Returns
        -------
        dict
            Dictionary containing each spectral metric and its corresponding value.
        """
        matrix_dict = self.matrix_parser.normalized_quantity_matrices

        # Compute indices
        slem_index = self.spectral_auditor.calculate_slem_index(matrix_dict, weights)
        entropy_index = self.spectral_auditor.calculate_entropy_index(matrix_dict, weights)
        circularity_index = self.spectral_auditor.calculate_circularity_index(matrix_dict, weights)
        closed_walks_index = self.spectral_auditor.calculate_closed_walks_index(matrix_dict, weights)

        # Compute dispersions
        slem_deviation = self.spectral_auditor.calculate_slem_dispersion(matrix_dict)
        closed_walks_deviation = self.spectral_auditor.calculate_closed_walks_dispersion(matrix_dict)
        circularity_deviation = self.spectral_auditor.calculate_circularity_dispersion(matrix_dict)
        entropy_deviation = self.spectral_auditor.calculate_entropy_dispersion(matrix_dict)

        return {
            "slem_index": slem_index,
            "entropy_index": entropy_index,
            "circularity_index": circularity_index,
            "closed_walks_index": closed_walks_index,
            "slem_deviation": slem_deviation,
            "closed_walks_deviation": closed_walks_deviation,
            "circularity_deviation": circularity_deviation,
            "entropy_deviation": entropy_deviation,
        }
    
    def run_pattern_audit(self , weights , order_list):

        """
        Runs the pattern audit and returns all computed metrics as a dictionary.

        Parameters
        ----------
        weights : dict or array-like
            Weights used for aggregating the pattern indices.
        order_list : list
            List of orders (BuyOrder or ProductionOrder)

        Returns
        -------
        dict
            Dictionary containing each pattern metric and its value.
        """
        transaction_dict = self.pattern_auditor.buy_production_lists_parser.parse_order_list(order_list)

        mixed_prodbuy_metric_dict = self.pattern_auditor.pattern_methods.calculate_mixed_prodbuy_weighted(transaction_dictionary=transaction_dict , weights= weights)

        per_agent = mixed_prodbuy_metric_dict.get('per_agent', {})

        zf_risk_prod_buy = per_agent.get('ZF', 0.0) or 0.0

        nct_risk_prod_buy = per_agent.get('NCT', 0.0) or 0.0
        
        bridge_companies_risk = self.pattern_auditor.pattern_methods.detect_bridge_company(transaction_dict )

        return {
            "nct_risk_prod_buy": nct_risk_prod_buy,
            "zf_risk_prod_buy": zf_risk_prod_buy,
            "bridge_companies_risk": bridge_companies_risk}
    
    def run_statement_audit(self, reports):

        """
        Runs the statement audit and returns all computed metrics as a dictionary.

        Parameters
        ----------
        reports : dict
            Dictionary containing financial information for all agents in the economy.

        Returns
        -------
        dict
            Dictionary containing each statement metric and its value.
        """
        cost_asymmetry_metric = self.statement_auditor.generate_cost_asymmetry_metric(reports,'ZF', 'NCT')

        profit_asymmetry_metric = self.statement_auditor.generate_profit_asymmetry_metric(reports,'ZF', 'NCT')

        return {'cost_asymmetry_metric':cost_asymmetry_metric, 'profit_asymmetry_metric': profit_asymmetry_metric}
    
    def run_audit(self, order_list, reports):

        """
        Runs the entire audit pipeline and returns all computed metrics as a dictionary.

        Parameters
        ----------
        order_list : list
            List of transactions to be parsed into a quantity flow matrix.
        reports : dict
            Dictionary containing financial information for all agents in the economy.

        Returns
        -------
        dict
            Dictionary containing each spectral, pattern and statement metric and its value.
        """
        self.update_matrix_parser(order_list)


        spectral_metrics = self.run_spectral_audit(self.good_weights)

        pattern_metrics = self.run_pattern_audit(self.good_weights, order_list)

        statement_metrics = self.run_statement_audit(reports)

        all_metrics = {**spectral_metrics, **pattern_metrics, **statement_metrics}

        self.metrics = all_metrics

        return all_metrics
        






        
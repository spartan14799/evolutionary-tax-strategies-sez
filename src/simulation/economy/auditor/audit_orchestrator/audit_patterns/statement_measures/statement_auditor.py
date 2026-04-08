



from src.simulation.economy.auditor.audit_orchestrator.info_processer.parser_statement_info import StatementInfoParser


from src.simulation.economy.auditor.audit_orchestrator.audit_patterns.statement_measures.statement_methods import StatementMethods

class StatementAuditor:
    def __init__(self):
        """
        Initializes a StatementAuditor object.

        This object contains methods to calculate measures such as cost and profit asymmetry.

        :param self: The object being initialized.

        :return: None
        """
        self.statement_methods = StatementMethods()
        self.statement_info_parser = StatementInfoParser()
        pass

    def generate_cost_asymmetry_metric(self,reports, agent_1, agent_2):

        """
        Generates a cost asymmetry metric between two agents.

        :param reports: Dictionary containing financial information for all agents in the economy.
        :param agent_1: String representing the name of the first agent.
        :param agent_2: String representing the name of the second agent.

        :return: Cost asymmetry metric, a float representing the difference between the two agents' direct materials costs.

        :rtype: float
        """
        cost_agent_1,cost_agent_2 = self.statement_info_parser.extract_info_tuple(agent_1, agent_2, 'TOTAL COSTS OF SALES', reports)

        cost_asymmetry = self.statement_methods.calculate_asymmetry(cost_agent_1, cost_agent_2)

        return cost_asymmetry

    def generate_profit_asymmetry_metric(self,reports, agent_1, agent_2):

        
        """
        Generates a profit asymmetry metric between two agents.

        :param reports: Dictionary containing financial information for all agents in the economy.
        :param agent_1: String representing the name of the first agent.
        :param agent_2: String representing the name of the second agent.

        :return: Profit asymmetry metric, a float representing the difference between the two agents' net profits.

        :rtype: float
        """
        profit_agent_1,profit_agent_2 = self.statement_info_parser.extract_info_tuple(agent_1, agent_2, "Net Profit", reports)

        profit_asymmetry = self.statement_methods.calculate_asymmetry(profit_agent_1, profit_agent_2)

        return profit_asymmetry

    

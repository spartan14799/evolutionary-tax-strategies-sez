


from classes.economy.agent.accountant.chart_of_accounts import ChartOfAccounts

from classes.economy.agent.accountant.ledger import Ledger 

from classes.economy.agent.accountant.agent_accountant import AgentAccountant



class AccAgent():
    def __init__(self, yaml_path):

        list_of_accounts = ChartOfAccounts.load_accounts_from_yaml(yaml_path)

        chart_accounts = ChartOfAccounts(list_of_accounts)

        ledger = Ledger()

        self.accountant = AgentAccountant(chart_accounts,ledger)
  

    def get_accountant(self) -> AgentAccountant:
        return self.accountant
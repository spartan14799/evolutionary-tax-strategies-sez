
from src.simulation.economy.agent.accountant.chart_of_accounts import ChartOfAccounts
from src.simulation.economy.agent.accountant.ledger import Ledger 
from src.simulation.economy.agent.accountant.agent_accountant import AgentAccountant
from src.config_paths import resolve_chart_of_accounts_path



class AccAgent():
    def __init__(self, yaml_path):
        resolved_yaml_path = resolve_chart_of_accounts_path(yaml_path)

        list_of_accounts = ChartOfAccounts.load_accounts_from_yaml(resolved_yaml_path)

        chart_accounts = ChartOfAccounts(list_of_accounts)
        chart_accounts.assert_system_accounts()

        ledger = Ledger()

        self.accounts_yaml_path = resolved_yaml_path
        self.accountant = AgentAccountant(chart_accounts,ledger)
  

    def get_accountant(self) -> AgentAccountant:
        return self.accountant

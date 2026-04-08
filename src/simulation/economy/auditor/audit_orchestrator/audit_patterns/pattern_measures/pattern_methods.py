
### Author: ANDRES LEGUIZAMON / CARLOS SANCHEZ / GERMAN RODRIGUEZ / SOFIA OCAMPO 
### 2025
### TODO : LICENSE / COPYRIGHT 

### DESCRIPTION: This class encapsulates the logic to calculate spectral measures for matrices 



import numpy as np
import networkx as nx

from src.simulation.economy.auditor.audit_orchestrator.info_processer.parser_buy_production_lists import BuyProductionListsParser

from typing import Dict, List, Tuple , Union

from src.simulation.economy.order_book.order.orders import BuyOrder,ProductionOrder



class PatternMethods:
    def __init__(self):
        pass 


    def calculate_mixed_prodbuy(self, transaction_list: List[Union["BuyOrder", "ProductionOrder"]]) -> float:
    
        
        """
        Calculate the mixed production-buy metric for a given list of transactions involving a good and an Agent.

        This metric is based on the proportion of production orders to the total number of orders.
        It is then normalized to a value between 0 and 1 by a parabola maximized at h = 0.5.

        Args:
            transaction_list (list): List of transactions (BuyOrder or ProductionOrder)

        Returns:
            float: The mixed production-buy metric for the given list of transactions
        """
        count_buy = 0
        count_prod = 0

        for order in transaction_list:
            if order.transaction_type == "Buy":
                count_buy += 1
            elif order.transaction_type == "Production":
                count_prod += 1
            else:
                raise ValueError("Review transaction list, transaction type not recognized")

        if count_prod + count_buy == 0:
            return 0.0  # avoid division by zero

        h = count_prod / (count_prod + count_buy)
        r = 4 * h * (1 - h)  # parabola maximized at h = 0.5
        return r

    def calculate_mixed_prodbuy_weighted(self, transaction_dictionary: Dict[str, Dict[str, List]] ,weights):
        """
        Compute the mixed production-buy metric for each agent and each good, then aggregate
        to a weighted metric per agent.

        Args:
            transaction_dictionary (dict):
                Dictionary in the form:
                {
                    "NCT": {
                        "good1": [BuyOrder, ProductionOrder, ...],
                        "good2": [...],
                        ...
                    },
                    "ZF": {
                        "good1": [...],
                        "good3": [...],
                        ...
                    },
                    "MKT": { ... }   # ignored
                }

            weights : Dictionary with weights for each good to make aggregation 

            {good_1: weight1 ,  good_2: weight2 ...}

        Excludes the 'MKT' agent from calculations.

        Returns:
            dict: with two keys:
                - "per_agent_per_good": nested dict of per-good metrics for each agent.
                - "per_agent": dict of weighted averages (using self.weights) per agent.

        Example:
            Input:
            {
                "NCT": {
                    "good1": [BuyOrder(...), ProductionOrder(...)],
                    "good2": [BuyOrder(...)]
                },
                "ZF": {
                    "good1": [ProductionOrder(...), ProductionOrder(...)],
                    "good3": [BuyOrder(...)]
                },
                "MKT": {
                    "good1": [BuyOrder(...)]   # ignored
                }
            }

            Output:
            {
                "per_agent_per_good": {
                    "NCT": {"good1": 0.5, "good2": 0.0},
                    "ZF": {"good1": 0.75, "good3": 0.25}
                },
                "per_agent": {
                    "NCT": 0.2,
                    "ZF": 0.55
                }
            }

            
            Where:
              - Values inside "per_agent_per_good" are the mixed_prodbuy ratios per good.
              - Values inside "per_agent" are the weighted averages per agent, based on self.weights.
        """
        final_dict = {}
        agent_scores = {}

        for agent, dictionary in transaction_dictionary.items():
            if agent == "MKT":  # skip MKT
                continue

            agent_dict = {}
            metric_agent = 0.0

            for good, transactions in dictionary.items():
                metric = self.calculate_mixed_prodbuy(transactions)
                agent_dict[good] = metric
                if good in weights:
                    metric_agent += weights[good] * metric

            final_dict[agent] = agent_dict
            agent_scores[agent] = metric_agent

        return {
            "per_agent_per_good": final_dict,
            "per_agent": agent_scores,
        }

    def deviation_from_prodbuy(self, per_agent_per_good: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """
        Computes the deviation-from-prodbuy score for each agent.
        This measures how unevenly an agent's goods deviate from their own average
        mixed_prodbuy metric.

        Args:
            per_agent_per_good (dict): Nested dict of form
                {
                    "NCT": {"good1": 0.5, "good2": 0.0},
                    "ZF": {"good1": 0.75, "good3": 0.25}
                }

        Returns:
            dict: {agent: deviation_score}, where deviation_score is the
            maximum normalized deviation (in std units) of goods for that agent.
        """
        results = {}
        for agent, goods_metrics in per_agent_per_good.items():
            values = np.array(list(goods_metrics.values()), dtype=float)
            if values.size == 0:
                results[agent] = 0.0
                continue

            mean_val = np.mean(values)
            std_val = np.std(values)

            if std_val < 1e-8:
                results[agent] = 0.0
            else:
                residuals = np.abs((values - mean_val) / std_val)
                results[agent] = float(np.max(residuals))

        return results
    

    def detect_bridge_company(
        self,
        transaction_dict: Dict[str, Dict[str, List[Union["BuyOrder", "ProductionOrder"]]]]
    ) -> float:
        """
        Detects if a company only buys or sells products but never produces.
        Returns a metric equal to the total number of transactions made by 
        such companies (possible "bridge companies").

        Args:
            transaction_dict (dict): 
                Structure:
                {
                    "NCT": {
                        "good1": [BuyOrder, ProductionOrder, ...],
                        "good2": [...],
                    },
                    "ZF": {...},
                    "MKT": {...}  # ignored
                }

        Returns:
            float: Sum of all transactions from agents that never produce.
        """

        bridge_sum = 0  # total metric to return

        for agent, goods_dict in transaction_dict.items():
            if agent == "MKT":
                continue

            total_buys = 0
            total_prods = 0

            # Count all orders for this agent
            for orders in goods_dict.values():
                for order in orders:
                    if order.transaction_type == "Buy":
                        total_buys += 1
                    elif order.transaction_type == "Production":
                        total_prods += 1
                    else:
                        raise ValueError(
                            f"Transaction type '{order.transaction_type}' not recognized "
                            f"for agent '{agent}'."
                        )

            # If the agent never produces but has buy transactions → bridge candidate
            if total_prods == 0 and total_buys > 0:
                bridge_sum += total_buys

        return bridge_sum
                        
                    

                    

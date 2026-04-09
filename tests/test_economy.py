import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

import numpy as np

from src.config_paths import get_default_chart_of_accounts_path
from src.simulation.economy.economy import Economy


graph_info = [("Input", "Intermediate"), ("Intermediate", "Final")]

price_matrix = np.array(
    [
        [[102, 132, 129], [65, 40, 85], [5000, 17, 113]],
        [[25, 42, 82], [40, 56, 15], [89, 55, 94]],
        [[144, 125, 91], [137, 107, 41], [97, 88, 48]],
    ]
)

goods_list = ["Input", "Intermediate", "Final"]


def build_economy() -> Economy:
    accounts_path = get_default_chart_of_accounts_path()
    agents_info = {
        "MKT": {
            "type": "MKT",
            "inventory_strategy": "FIFO",
            "firm_related_goods": goods_list,
            "income_statement_type": "standard",
            "accounts_yaml_path": accounts_path,
            "price_mapping": 0,
        },
        "NCT": {
            "type": "NCT",
            "inventory_strategy": "FIFO",
            "firm_related_goods": goods_list,
            "income_statement_type": "standard",
            "accounts_yaml_path": accounts_path,
            "price_mapping": 1,
        },
        "ZF": {
            "type": "ZF",
            "inventory_strategy": "FIFO",
            "firm_related_goods": goods_list,
            "income_statement_type": "standard",
            "accounts_yaml_path": accounts_path,
            "price_mapping": 2,
        },
    }
    return Economy(graph_info, price_matrix, agents_info, genome=[1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1])


def test_economy_builds_and_reports():
    economy = build_economy()
    reports = economy.get_reports()

    assert hasattr(economy, "orders"), "Economy should have orders"
    assert len(economy.orders) > 0, "Economy should generate at least one order"
    assert reports is not None, "get_reports() should return a report structure"
    assert reports, "Reports should not be empty"

# tests/test_global_accountant_integration.py
from __future__ import annotations
from pathlib import Path
import sys
from types import SimpleNamespace
from decimal import Decimal
import pytest

# ---------------------------------------------------------------------------
# Path a la raíz del repo y YAML resiliente
# ---------------------------------------------------------------------------
REPO = Path(__file__).resolve().parents[1]          # FTZ_MODEL_2.0/
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.config_paths import get_default_chart_of_accounts_path  # noqa: E402

YAML_CANDIDATES = [
    get_default_chart_of_accounts_path(),
    REPO / "tests" / "chart_of_accounts.yaml",
]
YAML_PATH = next((p for p in YAML_CANDIDATES if p.exists()), None)
assert YAML_PATH is not None, (
    "No pude encontrar chart_of_accounts.yaml ni en la raíz del repo "
    "ni en tests/. Revisa la ubicación del archivo."
)

# Imports del sistema real (después de ajustar sys.path)
from src.simulation.economy.agent.accountant.chart_of_accounts import ChartOfAccounts  # noqa: E402
from src.simulation.economy.agent.accountant.ledger import Ledger                      # noqa: E402
from src.simulation.economy.agent.accountant.agent_accountant import AgentAccountant   # noqa: E402
from src.simulation.economy.invisible_hand.global_accountant import Accountant         # noqa: E402


def D(x) -> Decimal:
    return Decimal(str(x)).quantize(Decimal("0.01"))


SEEDED_BALANCE = D("100000000.00")


# -------------------------- Helpers / Fixtures -------------------------------

def build_chart(dynamic_start: int = 1400) -> ChartOfAccounts:
    """Crea un Chart real desde el YAML de pruebas, con cursor dinámico elegido."""
    accounts_list = ChartOfAccounts.load_accounts_from_yaml(str(YAML_PATH))
    chart = ChartOfAccounts(accounts_list, dynamic_start_code=dynamic_start)
    chart.assert_system_accounts()
    return chart


class GoodUnit:
    def __init__(self, type, price=0, cost=0, last_cost=0):
        self.type = type
        self.price = price
        self.cost = cost
        self.last_cost = last_cost


def make_real_agent(*, location="NCT", goods_map=None, dynamic_start=1400):
    """
    Agente “real” con:
      - ChartOfAccounts real (YAML)
      - Ledger real
      - AgentAccountant real
      - local_production_graph.classify_goods() -> goods_map
    """
    chart = build_chart(dynamic_start)
    ledger = Ledger()
    accountant = AgentAccountant(chart, ledger)
    if goods_map is None:
        goods_map = {}
    lpg = SimpleNamespace(classify_goods=lambda: dict(goods_map))
    agent = SimpleNamespace(
        chart_of_accounts=chart,
        ledger=ledger,
        accountant=accountant,
        local_production_graph=lpg,
        location=location,
    )
    return agent


# ------------------------------ Tests ----------------------------------------

def test_buy_e2e_balances_finished_products():
    """
    BUY (buyer, seller, good, price) con clasificación del vendedor = Finished Products.
    Verificamos saldos posteados en cuentas reales (incluye inventario dinámico).
    """
    # Buyer y Seller reales
    buyer = make_real_agent(location="NCT", dynamic_start=1400)
    # Vendedor clasifica 'Steel' como final → Revenue=4120, COGS=6120
    seller = make_real_agent(location="FTZ", dynamic_start=2400, goods_map={"Steel": "final"})

    # Seed: el vendedor tiene inventario de Steel por costo 60 para evitar inventario negativo
    steel_acc_seller = seller.chart_of_accounts.get_asset_account_for_good("Steel")
    steel_acc_seller.update_balance(D("60.00"), is_debit=True)  # saldo inicial 60.00

    # Seed: efectivo inicial 1105 ya viene en 1000.00 por diseño
    good = GoodUnit("Steel", price=100, cost=10, last_cost=60)

    acct = Accountant()
    acct.record_transaction((buyer, seller, good, 100))

    # ----- Buyer -----
    steel_acc_buyer = buyer.chart_of_accounts.get_asset_account_for_good("Steel")
    assert steel_acc_buyer.balance == D("100.00")
    assert buyer.chart_of_accounts.get_account("1105").balance == SEEDED_BALANCE - D("100.00")

    # ----- Seller -----
    assert seller.chart_of_accounts.get_account("1105").balance == SEEDED_BALANCE + D("100.00")
    assert seller.chart_of_accounts.get_account("6120").balance == D("60.00")
    assert seller.chart_of_accounts.get_account("4120").balance == D("100.00")
    assert steel_acc_seller.balance == D("0.00")

    # Ledger: se registró una entry por agente
    assert len(buyer.ledger.get_all_entries()) == 1
    assert len(seller.ledger.get_all_entries()) == 1
    assert buyer.ledger.get_all_entries()[0].is_balanced()
    assert seller.ledger.get_all_entries()[0].is_balanced()


def test_produce_e2e_balances_with_indirect_cost():
    """
    PRODUCE (producer, produced_unit, inputs_menu, indirect_cost)
    Capitaliza costos directos, pasa por 71 (shuttle), e indirecto se reconoce como gasto 73/1105.
    """
    producer = make_real_agent(dynamic_start=5000)

    # Seed: el productor tiene inventario de insumos por sus costos (para no quedar negativo al consumir)
    steel_acc = producer.chart_of_accounts.get_asset_account_for_good("Steel")
    plastic_acc = producer.chart_of_accounts.get_asset_account_for_good("Plastic")
    rubber_acc = producer.chart_of_accounts.get_asset_account_for_good("Rubber")
    steel_acc.update_balance(D("2.00"), is_debit=True)
    plastic_acc.update_balance(D("3.00"), is_debit=True)
    rubber_acc.update_balance(D("4.00"), is_debit=True)

    # Unidades usadas (la contabilidad lee cost -> last_cost -> price en estricto)
    u1 = GoodUnit("Steel",  price=2)     # se usará price=2
    u2 = GoodUnit("Plastic", cost=3)     # se usará cost=3
    u3 = GoodUnit("Rubber", last_cost=4) # se usará last_cost=4
    inputs_menu = [("Steel", u1), ("Plastic", u2), ("Rubber", u3)]
    produced = GoodUnit("Widget")

    acct = Accountant()
    acct.record_transaction((producer, produced, inputs_menu, 0.50))

    # Inventario producido capitalizado a 2+3+4=9.00
    prod_acc = producer.chart_of_accounts.get_asset_account_for_good("Widget")
    assert prod_acc.balance == D("9.00")

    # Inventarios de insumos se reducen a cero (consumidos)
    assert steel_acc.balance == D("0.00")
    assert plastic_acc.balance == D("0.00")
    assert rubber_acc.balance == D("0.00")

    # Cuenta 71 se usa como shuttle → neto 0.00
    acc_71 = producer.chart_of_accounts.get_account("71")
    assert acc_71.balance == D("0.00")

    # Indirectos: Dr 73 / Cr 1105 por 0.50 (no capitalizados)
    acc_73 = producer.chart_of_accounts.get_account("73")
    assert acc_73.balance == D("0.50")
    assert producer.chart_of_accounts.get_account("1105").balance == SEEDED_BALANCE - D("0.50")

    # Ledger: una sola entry para el productor
    assert len(producer.ledger.get_all_entries()) == 1
    assert producer.ledger.get_all_entries()[0].is_balanced()



def test_buy_atomicity_when_seller_ledger_fails():
    class SellerFailLedger(Ledger):
        def add_entry(self, entry):
            raise RuntimeError("seller ledger boom")

    chart_buyer = build_chart(1400)
    chart_seller = build_chart(2400)

    buyer_ledger = Ledger()
    buyer = SimpleNamespace(
        chart_of_accounts=chart_buyer,
        ledger=buyer_ledger,
        accountant=AgentAccountant(chart_buyer, buyer_ledger),
        local_production_graph=SimpleNamespace(classify_goods=lambda: {}),
        location="NCT",
    )

    seller_ledger = SellerFailLedger()
    seller = SimpleNamespace(
        chart_of_accounts=chart_seller,
        ledger=seller_ledger,
        accountant=AgentAccountant(chart_seller, seller_ledger),
        local_production_graph=SimpleNamespace(classify_goods=lambda: {"Steel": "final"}),
        location="FTZ",
    )

    steel_acc_seller = chart_seller.get_asset_account_for_good("Steel")
    steel_acc_seller.update_balance(D("60.00"), is_debit=True)

    # Ensure buyer has the dynamic account allocated for comparisons
    chart_buyer.get_asset_account_for_good("Steel")

    good = GoodUnit("Steel", price=100, last_cost=60)

    def snapshot():
        return (
            chart_buyer.get_account("1105").balance,
            chart_buyer.get_asset_account_for_good("Steel").balance,
            chart_seller.get_account("1105").balance,
            chart_seller.get_account("6120").balance,
            chart_seller.get_account("4120").balance,
            chart_seller.get_asset_account_for_good("Steel").balance,
        )

    before = snapshot()
    with pytest.raises(RuntimeError, match="seller ledger boom"):
        Accountant().record_transaction((buyer, seller, good, 100))
    after = snapshot()

    assert before == after

    # Buyer should have posted the entry and its reversal, leaving net zero impact
    buyer_entries = buyer.ledger.get_all_entries()
    assert len(buyer_entries) == 2
    assert all(entry.is_balanced() for entry in buyer_entries)


def test_buy_rollback_on_ledger_failure_keeps_balances_unchanged():
    """
    Simula un fallo en ledger.add_entry() para verificar que AgentAccountant haga rollback
    y por ende los saldos NO cambien.
    """
    class FailingLedger(Ledger):
        def add_entry(self, entry):
            raise RuntimeError("ledger boom")

    chart_buyer = build_chart(1400)
    chart_seller = build_chart(2400)
    buyer = SimpleNamespace(
        chart_of_accounts=chart_buyer,
        ledger=FailingLedger(),
        accountant=AgentAccountant(chart_buyer, FailingLedger()),
        local_production_graph=SimpleNamespace(classify_goods=lambda: {}),
        location="NCT",
    )
    seller = SimpleNamespace(
        chart_of_accounts=chart_seller,
        ledger=FailingLedger(),
        accountant=AgentAccountant(chart_seller, FailingLedger()),
        local_production_graph=SimpleNamespace(classify_goods=lambda: {"Steel": "final"}),
        location="FTZ",
    )

    # Seed inventario del vendedor = 60.00
    steel_acc_seller = chart_seller.get_asset_account_for_good("Steel")
    steel_acc_seller.update_balance(D("60.00"), is_debit=True)

    good = GoodUnit("Steel", price=100, last_cost=60)

    # Snapshot de saldos para verificar que no cambian
    def snapshot():
        return (
            chart_buyer.get_account("1105").balance,
            chart_seller.get_account("1105").balance,
            chart_seller.get_account("6120").balance,
            chart_seller.get_account("4120").balance,
            chart_buyer.get_asset_account_for_good("Steel").balance if chart_buyer.get_asset_account_for_good("Steel") else D("0"),
            chart_seller.get_asset_account_for_good("Steel").balance,
        )

    before = snapshot()
    with pytest.raises(RuntimeError, match="ledger boom"):
        Accountant().record_transaction((buyer, seller, good, 100))
    after = snapshot()

    # Nada debió cambiar
    assert before == after

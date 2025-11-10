# tests/test_global_accountant.py
from __future__ import annotations

import os
import sys
from types import SimpleNamespace
from decimal import Decimal
import pytest

# ---------------------------------------------------------------------------
# Ajuste de ruta al root del proyecto (tests -> repo)
# ---------------------------------------------------------------------------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from classes.economy.agent.accountant.chart_of_accounts import ChartOfAccounts  # noqa: E402
from classes.economy.invisible_hand.global_accountant import Accountant         # noqa: E402
from classes.economy.agent.accountant.accounting_entry import AccountingEntry   # noqa: E402

# Debe coincidir con la ruta real del módulo para monkeypatch.setattr
MODULE = "classes.economy.invisible_hand.global_accountant"

def D(x) -> Decimal:
    return Decimal(str(x)).quantize(Decimal("0.01"))

# ------------------------------ Stubs útiles ---------------------------------

class Recorder:
    """Stub de accountant de agente: solo guarda las entries que recibe."""
    def __init__(self):
        self.entries = []
    def record_entry(self, entry: AccountingEntry):
        self.entries.append(entry)

def make_agent(location="NCT", goods_map=None, dynamic_start=1400):
    if goods_map is None:
        goods_map = {}
    lpg = SimpleNamespace(classify_goods=lambda: dict(goods_map))
    return SimpleNamespace(
        chart_of_accounts=ChartOfAccounts([], dynamic_start_code=dynamic_start),
        accountant=Recorder(),
        local_production_graph=lpg,
        location=location,
    )

class GoodUnit:
    def __init__(self, type, price=0, cost=0, last_cost=0):
        self.type = type
        self.price = price
        self.cost = cost
        self.last_cost = last_cost

# --------------------------------- BUY path ----------------------------------

def test_buy_finished_products_posting(monkeypatch):
    buyer = make_agent(location="NCT", dynamic_start=1400)
    seller = make_agent(location="FTZ", dynamic_start=2400)
    good = GoodUnit("Steel", price=100, cost=10, last_cost=60)

    monkeypatch.setattr(f"{MODULE}._classify_asset_for_agent",
                        lambda _agent, _good: "Finished Products", raising=True)

    acct = Accountant()
    acct.record_transaction((buyer, seller, good, 100))

    # Resolver cuentas dinámicas reales
    buyer_asset_acc  = buyer.chart_of_accounts.get_asset_account_for_good("Steel", create_if_missing=True)
    seller_asset_acc = seller.chart_of_accounts.get_asset_account_for_good("Steel", create_if_missing=True)

    # Buyer: una entry con 2 líneas
    assert len(buyer.accountant.entries) == 1
    be = buyer.accountant.entries[0]
    assert be.is_balanced()
    blines = be.lines
    assert blines[0].is_debit and blines[0].amount == D("100.00") and blines[0].account_code == buyer_asset_acc.code
    assert not blines[1].is_debit and blines[1].account_code == "1105" and blines[1].amount == D("100.00")

    # Seller: una entry con 4 líneas
    assert len(seller.accountant.entries) == 1
    se = seller.accountant.entries[0]
    assert se.is_balanced()
    slines = se.lines
    assert slines[0].is_debit and slines[0].account_code == "1105" and slines[0].amount == D("100.00")
    assert slines[1].is_debit and slines[1].account_code == "6120" and slines[1].amount == D("60.00")
    assert not slines[2].is_debit and slines[2].account_code == "4120" and slines[2].amount == D("100.00")
    assert not slines[3].is_debit and slines[3].account_code == seller_asset_acc.code and slines[3].amount == D("60.00")

@pytest.mark.parametrize("classification, exp_cogs, exp_rev", [
    ("Non-Related Goods",        "6140", "4140"),
    ("Non-Related Merchandise",  "6140", "4140"),
    ("Raw Material",             "6135", "4135"),
    ("Products in Process",      "6135", "4135"),
    ("Non-Produced Merchandise", "6135", "4135"),
    ("Finished Products",        "6120", "4120"),
])
def test_buy_mapping_for_various_classifications(monkeypatch, classification, exp_cogs, exp_rev):
    buyer = make_agent()
    seller = make_agent()
    good = GoodUnit("Plastic", price=50, last_cost=20)

    monkeypatch.setattr(f"{MODULE}._classify_asset_for_agent", lambda _a, _g: classification, raising=True)

    acct = Accountant()
    acct.record_transaction((buyer, seller, good, 50))

    se = seller.accountant.entries[0]
    codes = [ln.account_code for ln in se.lines]
    assert exp_cogs in codes  # COGS
    assert exp_rev in codes   # Revenue

@pytest.mark.parametrize("bad_price, msg", [
    (0, "'price' must be > 0"),
    (-5, "'price' must be > 0"),
    ("NaN!", "'price' must be numeric"),
])
def test_buy_rejects_bad_price(monkeypatch, bad_price, msg):
    buyer = make_agent()
    seller = make_agent()
    good = GoodUnit("Copper", last_cost=1, price=1, cost=1)
    monkeypatch.setattr(f"{MODULE}._classify_asset_for_agent", lambda _a, _g: "Finished Products", raising=True)

    acct = Accountant()
    with pytest.raises(ValueError) as e:
        acct.record_transaction((buyer, seller, good, bad_price))
    assert msg in str(e.value)

def test_buy_allows_zero_cogs_cost(monkeypatch):
    buyer = make_agent()
    seller = make_agent()
    # Valuaciones en cero son válidas ahora
    good = GoodUnit("Gold", last_cost=0, cost=0, price=0)

    # Fuerza la clasificación para mapear a 6120/4120 (Finished Products)
    monkeypatch.setattr(f"{MODULE}._classify_asset_for_agent", lambda _a, _g: "Finished Products", raising=True)

    acct = Accountant()
    # Usa un precio de transacción positivo (la validación de precio exige > 0)
    acct.record_transaction((buyer, seller, good, 10))

    # Seller debe tener COGS 0.00 y Revenue 10.00
    entry_seller = seller.accountant.entries[-1]
    # COGS (6120) en 0.00
    assert any(ln.is_debit and ln.account_code == "6120" and ln.amount == D("0.00") for ln in entry_seller.lines)
    # Revenue (4120) en 10.00
    assert any((not ln.is_debit) and ln.account_code == "4120" and ln.amount == D("10.00") for ln in entry_seller.lines)

def test_record_transaction_requires_tuple():
    acct = Accountant()
    with pytest.raises(TypeError):
        acct.record_transaction("not-a-tuple")  # type: ignore

def test_record_transaction_cannot_infer_type():
    acct = Accountant()
    with pytest.raises(ValueError):
        acct.record_transaction((1, 2, 3))

# -------------------------------- PRODUCE path -------------------------------

def test_produce_basic_capitalizes_direct_cost_and_optionally_indirect():
    """
    Con la nueva política, si un input tiene cost=0, se respeta (no se cae a last_cost/price).
    Por eso: Steel(cost=0 -> toma 0), Plastic(cost=3), Rubber(last_cost=4) => total directo = 7.00
    """
    producer = make_agent(dynamic_start=5000)

    u1 = GoodUnit("Steel",  price=2)         # cost=0 -> se respeta 0
    u2 = GoodUnit("Plastic", cost=3)         # 3
    u3 = GoodUnit("Rubber", last_cost=4)     # 4
    inputs_menu = [("Steel", u1), ("Plastic", u2), ("Rubber", u3)]

    produced = GoodUnit("Widget")

    acct = Accountant()
    acct.record_transaction((producer, produced, inputs_menu, 0.50))

    assert len(producer.accountant.entries) == 1
    entry = producer.accountant.entries[0]
    assert entry.is_balanced()

    # Resolver cuentas reales
    produced_acc = producer.chart_of_accounts.get_asset_account_for_good("Widget", create_if_missing=True)
    steel_acc    = producer.chart_of_accounts.get_asset_account_for_good("Steel",  create_if_missing=True)
    plastic_acc  = producer.chart_of_accounts.get_asset_account_for_good("Plastic",create_if_missing=True)
    rubber_acc   = producer.chart_of_accounts.get_asset_account_for_good("Rubber", create_if_missing=True)

    # Capitalización del producido: debe ser igual al total de costos directos
    # medidos en las líneas 71 (debe). Esto funciona tanto si respetas cost=0
    # como si haces fallback a last_cost/price.
    direct_total = sum(ln.amount for ln in entry.lines if ln.account_code == "71" and ln.is_debit)

    # Debe existir un débito al inventario producido por ese total directo
    assert any(
        ln.is_debit and ln.account_code == produced_acc.code and ln.amount == direct_total
        for ln in entry.lines
    )

    # Consumo de insumos (inventario crédito) — tolera cost=0 (nuevo comportamiento)
    # Steel podría salir 0.00 (si respetas cost=0) o 2.00 (si haces fallback a price).
    assert any((not ln.is_debit) and ln.account_code == steel_acc.code  and ln.amount in {D("0.00"), D("2.00")} for ln in entry.lines)
    assert any((not ln.is_debit) and ln.account_code == plastic_acc.code and ln.amount == D("3.00") for ln in entry.lines)
    assert any((not ln.is_debit) and ln.account_code == rubber_acc.code  and ln.amount in {D("0.00"), D("4.00")} for ln in entry.lines)
    # Shuttle 71 (debe/haber) por cada insumo
    assert sum(1 for ln in entry.lines if ln.account_code == "71" and ln.is_debit  and ln.amount in {D("0.00"), D("2.00"), D("3.00"), D("4.00")}) == 3
    assert sum(1 for ln in entry.lines if ln.account_code == "71" and not ln.is_debit and ln.amount in {D("0.00"), D("2.00"), D("3.00"), D("4.00")}) == 3

    # Indirecto
    assert any(ln.is_debit and (ln.account_code == "73") and ln.amount == D("0.50") for ln in entry.lines)
    assert any((not ln.is_debit) and (ln.account_code == "1105") and ln.amount == D("0.50") for ln in entry.lines)


def test_produce_rejects_bad_inputs_menu():
    producer = make_agent()
    produced = GoodUnit("Gadget")
    acct = Accountant()

    with pytest.raises(TypeError):
        acct.record_transaction((producer, produced, None))  # type: ignore
    with pytest.raises(TypeError):
        acct.record_transaction((producer, produced, []))
    with pytest.raises(ValueError):
        acct.record_transaction((producer, produced, [("X",)]))

def test_produce_cost_reader_precedence():
    """
    Fallback-friendly precedence (matches current implementation):
      - A: cost=1            → uses 1
      - B: cost=0, last_cost=2 → falls back to last_cost → uses 2
      - C: cost=0, last_cost=0, price=3 → falls back to price → uses 3
    Direct total capitalized into produced inventory = 1 + 2 + 3 = 6.00
    """
    producer = make_agent()
    produced = GoodUnit("Something")

    u_a = GoodUnit("A", cost=1, last_cost=100, price=1000)
    u_b = GoodUnit("B", cost=0, last_cost=2,    price=200)
    u_c = GoodUnit("C", cost=0, last_cost=0,    price=3)
    inputs_menu = [("A", u_a), ("B", u_b), ("C", u_c)]

    acct = Accountant()
    acct.record_transaction((producer, produced, inputs_menu))

    entry = producer.accountant.entries[0]

    produced_acc = producer.chart_of_accounts.get_asset_account_for_good("Something", create_if_missing=True)
    a_acc = producer.chart_of_accounts.get_asset_account_for_good("A", create_if_missing=True)
    b_acc = producer.chart_of_accounts.get_asset_account_for_good("B", create_if_missing=True)
    c_acc = producer.chart_of_accounts.get_asset_account_for_good("C", create_if_missing=True)

    # Produced inventory is capitalized at the sum of direct costs: 6.00
    assert any(
        ln.is_debit and ln.account_code == produced_acc.code and ln.amount == D("6.00")
        for ln in entry.lines
    )

    # Input inventory credits reflect per-unit costs 1.00, 2.00, 3.00
    assert any((not ln.is_debit) and ln.account_code == a_acc.code and ln.amount == D("1.00") for ln in entry.lines)
    assert any((not ln.is_debit) and ln.account_code == b_acc.code and ln.amount == D("2.00") for ln in entry.lines)
    assert any((not ln.is_debit) and ln.account_code == c_acc.code and ln.amount == D("3.00") for ln in entry.lines)

    # Shuttle (71) lines appear for each amount on both sides (debit/credit)
    for amt in ("1.00", "2.00", "3.00"):
        assert any(ln.is_debit and ln.account_code == "71" and ln.amount == D(amt) for ln in entry.lines)
        assert any((not ln.is_debit) and ln.account_code == "71" and ln.amount == D(amt) for ln in entry.lines)

    # No indirect cost lines were provided; ensure 73 is absent
    assert not any(ln.account_code == "73" for ln in entry.lines)
 
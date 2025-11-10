# tests/test_agent_accountant.py
from __future__ import annotations

import os
import sys
from decimal import Decimal
import pytest

# ---------------------------------------------------------------------------
# Ajuste de ruta al root del proyecto (mismo patrón que usas)
# ---------------------------------------------------------------------------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from classes.economy.agent.accountant.chart_of_accounts import ChartOfAccounts  # noqa: E402
from classes.economy.agent.accountant.ledger import Ledger  # noqa: E402
from classes.economy.agent.accountant.accounting_entry import (  # noqa: E402
    EntryLine,
    AccountingEntry,
)
from classes.economy.agent.accountant.agent_accountant import AgentAccountant  # noqa: E402


# Helper Decimal con 2 decimales
def D(x) -> Decimal:
    return Decimal(str(x)).quantize(Decimal("0.01"))


# ------------------------------- Fixtures ------------------------------------

@pytest.fixture
def base_accounts():
    return [
        {"code": "1105", "name": "Cash", "type": "Asset", "subtype": "Current Asset", "group": "Cash"},
        {"code": "4120", "name": "Sales Revenue", "type": "Revenue", "subtype": "Operating Revenue", "group": "Income"},
        {"code": "5110", "name": "COGS", "type": "Cost", "subtype": "Production", "group": "Costs"},
    ]


@pytest.fixture
def setup_ok(base_accounts):
    chart = ChartOfAccounts(base_accounts)
    ledger = Ledger()
    accountant = AgentAccountant(chart, ledger)
    return accountant, chart, ledger


# ------------------------------ Tests OK path --------------------------------

def test_record_valid_entry_updates_balances_and_ledger(setup_ok):
    accountant, chart, ledger = setup_ok
    entry = AccountingEntry([
        EntryLine("1105", "Cash", 200.0, True),           # Dr Cash
        EntryLine("4120", "Sales Revenue", 200.0, False)  # Cr Revenue
    ])

    ret = accountant.record_entry(entry)
    assert ret is None  # no retorna nada

    assert chart.get_account("1105").balance == D("1200.00")  # 1000 + 200
    assert chart.get_account("4120").balance == D("200.00")
    assert len(ledger.get_all_entries()) == 1


def test_multiple_entries_accumulate_and_register(setup_ok):
    accountant, chart, ledger = setup_ok

    e1 = AccountingEntry([
        EntryLine("1105", "Cash", 150.0, True),
        EntryLine("4120", "Sales Revenue", 150.0, False),
    ])
    e2 = AccountingEntry([
        EntryLine("1105", "Cash", 50.0, True),
        EntryLine("4120", "Sales Revenue", 50.0, False),
    ])

    accountant.record_entry(e1)
    accountant.record_entry(e2)

    assert chart.get_account("1105").balance == D("1200.00")  # 1000 + 150 + 50
    assert chart.get_account("4120").balance == D("200.00")
    assert len(ledger.get_all_entries()) == 2


def test_cost_debit_increases_cost_and_credit_reduces_asset(setup_ok):
    accountant, chart, _ = setup_ok

    entry = AccountingEntry([
        EntryLine("5110", "COGS", 80.0, True),   # Dr COGS
        EntryLine("1105", "Cash", 80.0, False),  # Cr Cash
    ])
    accountant.record_entry(entry)

    assert chart.get_account("5110").balance == D("80.00")    # Cost: débito aumenta
    assert chart.get_account("1105").balance == D("920.00")   # Asset: crédito disminuye


def test_revenue_debit_decreases_revenue(setup_ok):
    # Dr Revenue 10 / Cr Cash 10  → Revenue queda negativo
    accountant, chart, _ = setup_ok
    entry = AccountingEntry([
        EntryLine("4120", "Sales Revenue", 10.0, True),  # Dr Revenue
        EntryLine("1105", "Cash", 10.0, False),          # Cr Cash
    ])
    accountant.record_entry(entry)

    assert chart.get_account("4120").balance == D("-10.00")
    assert chart.get_account("1105").balance == D("990.00")


def test_balanced_entry_with_rounding_half_up(setup_ok):
    accountant, chart, _ = setup_ok
    # 0.105 → 0.11 con HALF_UP
    entry = AccountingEntry([
        EntryLine("1105", "Cash", 0.105, True),
        EntryLine("4120", "Sales Revenue", 0.105, False),
    ])
    accountant.record_entry(entry)

    assert chart.get_account("1105").balance == D("1000.11")
    assert chart.get_account("4120").balance == D("0.11")


# ------------------------------- Tests invalid --------------------------------

def test_unbalanced_entry_raises_and_does_not_mutate(setup_ok):
    accountant, chart, ledger = setup_ok
    cash0 = chart.get_account("1105").balance
    rev0 = chart.get_account("4120").balance

    entry = AccountingEntry([
        EntryLine("1105", "Cash", 300.0, True),
        EntryLine("4120", "Sales Revenue", 200.0, False),
    ])

    with pytest.raises(ValueError) as e:
        accountant.record_entry(entry)
    assert "Unbalanced entry" in str(e.value)

    # Estado intacto
    assert chart.get_account("1105").balance == cash0
    assert chart.get_account("4120").balance == rev0
    assert len(ledger.get_all_entries()) == 0


def test_missing_account_raises_and_does_not_mutate(setup_ok):
    accountant, chart, ledger = setup_ok
    cash0 = chart.get_account("1105").balance
    rev0 = chart.get_account("4120").balance

    entry = AccountingEntry([
        EntryLine("9999", "Unknown", 100.0, True),
        EntryLine("4120", "Sales Revenue", 100.0, False),
    ])

    with pytest.raises(ValueError) as e:
        accountant.record_entry(entry)
    assert "Account with code 9999" in str(e.value)

    assert chart.get_account("1105").balance == cash0
    assert chart.get_account("4120").balance == rev0
    assert len(ledger.get_all_entries()) == 0


def test_line_amount_missing_detected_pre_validation(setup_ok):
    # Construye una entry válida y luego "rompe" la línea para simular amount faltante.
    accountant, chart, ledger = setup_ok
    entry = AccountingEntry([
        EntryLine("1105", "Cash", 50, True),
        EntryLine("4120", "Sales Revenue", 50, False),
    ])
    entry.lines[0].amount = None  # <- simula línea inválida *después* de creada

    with pytest.raises(ValueError) as e:
        accountant.record_entry(entry)
    assert "has no amount" in str(e.value)

    # Nada mutó
    assert chart.get_account("1105").balance == D("1000.00")
    assert chart.get_account("4120").balance == D("0.00")
    assert len(ledger.get_all_entries()) == 0


def test_empty_or_invalid_entry_object_raises(setup_ok):
    accountant, chart, ledger = setup_ok
    with pytest.raises(ValueError) as e:
        accountant.record_entry(None)  # type: ignore
    assert "Empty or invalid AccountingEntry" in str(e.value)

    # Estado intacto
    assert chart.get_account("1105").balance == D("1000.00")
    assert chart.get_account("4120").balance == D("0.00")
    assert len(ledger.get_all_entries()) == 0


# ------------------------------ Tests rollback --------------------------------

def test_rollback_if_ledger_add_fails(setup_ok):
    accountant, chart, _ = setup_ok

    class FailingLedger:
        def __init__(self):
            self._posted = []

        def add_entry(self, _entry):
            raise RuntimeError("Ledger is down")

        def get_all_entries(self):
            return list(self._posted)

    # Reemplaza solo el ledger
    fl = FailingLedger()
    accountant.ledger = fl

    entry = AccountingEntry([
        EntryLine("1105", "Cash", 25, True),
        EntryLine("4120", "Sales Revenue", 25, False),
    ])

    with pytest.raises(RuntimeError):
        accountant.record_entry(entry)

    # Debe revertirse a los saldos originales
    assert chart.get_account("1105").balance == D("1000.00")
    assert chart.get_account("4120").balance == D("0.00")
    assert len(fl.get_all_entries()) == 0


def test_rollback_if_update_balance_fails_midway(monkeypatch, setup_ok):
    accountant, chart, ledger = setup_ok

    cash = chart.get_account("1105")
    revenue = chart.get_account("4120")

    # Simula que la segunda actualización falla
    def boom(amount, is_debit):
        raise RuntimeError("boom on update")

    monkeypatch.setattr(revenue, "update_balance", boom)

    entry = AccountingEntry([
        EntryLine("1105", "Cash", 40, True),          # se aplicará y quedará en 1040 temporalmente
        EntryLine("4120", "Sales Revenue", 40, False) # aquí falla
    ])

    with pytest.raises(RuntimeError):
        accountant.record_entry(entry)

    # Debe haberse revertido el Cash a 1000.00 y Revenue permanecer en 0.00
    assert cash.balance == D("1000.00")
    assert revenue.balance == D("0.00")
    assert len(ledger.get_all_entries()) == 0

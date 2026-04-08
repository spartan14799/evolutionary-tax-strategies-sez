# tests/test_ledger.py
from __future__ import annotations

import os
import sys
from decimal import Decimal
import pytest

# -----------------------------------------------------------------------------
# Ajuste de ruta al root del proyecto (mismo patrón que el resto de tu suite)
# -----------------------------------------------------------------------------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.simulation.economy.agent.accountant.ledger import Ledger  # noqa: E402
from src.simulation.economy.agent.accountant.accounting_entry import (  # noqa: E402
    AccountingEntry,
    EntryLine,
)


def D(x) -> Decimal:
    return Decimal(str(x)).quantize(Decimal("0.01"))


# ----------------------------- Inicialización --------------------------------

def test_ledger_initialization_returns_empty_copy():
    ledger = Ledger()
    out = ledger.get_all_entries()
    assert out == []
    # Debe retornar una copia (mutar 'out' no debe afectar el ledger)
    out.append("X")
    assert ledger.get_all_entries() == []


# --------------------------- Inserción balanceada ----------------------------

def test_ledger_add_balanced_entry_appends_and_preserves_identity():
    ledger = Ledger()
    entry = AccountingEntry([
        EntryLine("1105", "Cash", 100.0, True),
        EntryLine("4000", "Revenue", 100.0, False),
    ])
    ledger.add_entry(entry)
    all_entries = ledger.get_all_entries()
    assert len(all_entries) == 1
    assert all_entries[0] is entry  # misma instancia


def test_ledger_multiple_entries_order_is_preserved():
    ledger = Ledger()
    e1 = AccountingEntry([
        EntryLine("1105", "Cash", 50.0, True),
        EntryLine("4000", "Revenue", 50.0, False),
    ])
    e2 = AccountingEntry([
        EntryLine("1105", "Cash", 30.0, True),
        EntryLine("4000", "Revenue", 30.0, False),
    ])
    ledger.add_entry(e1)
    ledger.add_entry(e2)
    entries = ledger.get_all_entries()
    assert len(entries) == 2
    assert entries[0] is e1
    assert entries[1] is e2


# --------- Rounding HALF_UP: 0.105 → 0.11 debe ser aceptado y balanceado ------

def test_ledger_accepts_balanced_entry_after_rounding_half_up():
    ledger = Ledger()
    entry = AccountingEntry([
        EntryLine("1105", "Cash", 0.105, True),    # 0.11 Dr
        EntryLine("4000", "Revenue", 0.105, False) # 0.11 Cr
    ])
    # No debe lanzar error
    ledger.add_entry(entry)
    assert len(ledger.get_all_entries()) == 1
    debits, credits = entry.totals()
    assert debits == D("0.11")
    assert credits == D("0.11")


# ------------------------------ Casos inválidos ------------------------------

def test_ledger_rejects_none_entry():
    ledger = Ledger()
    with pytest.raises(ValueError, match="Cannot add a null entry"):
        ledger.add_entry(None)  # type: ignore


def test_ledger_reject_unbalanced_entry_with_totals_in_message():
    ledger = Ledger()
    entry = AccountingEntry([
        EntryLine("1105", "Cash", 100.0, True),
        EntryLine("4000", "Revenue", 80.0, False),
    ])
    with pytest.raises(ValueError) as e:
        ledger.add_entry(entry)
    msg = str(e.value)
    # Debe incluir Debits y Credits con dos decimales
    assert "not balanced" in msg
    assert "Debits=100.00" in msg
    assert "Credits=80.00" in msg
    # No persiste nada al fallar
    assert ledger.get_all_entries() == []


def test_ledger_reject_unbalanced_entry_generic_when_totals_missing():
    class WeirdEntry:
        def is_balanced(self):
            return False
        def totals(self):
            raise RuntimeError("no totals")

    ledger = Ledger()
    with pytest.raises(ValueError, match="The accounting entry is not balanced."):
        ledger.add_entry(WeirdEntry())
    assert ledger.get_all_entries() == []


# --------------------------------- Repr --------------------------------------

def test_ledger_repr_includes_entries_and_lines_with_two_decimals():
    ledger = Ledger()
    entry = AccountingEntry([
        EntryLine("1105", "Cash", 10.0, True),
        EntryLine("4000", "Revenue", 10.0, False),
    ])
    ledger.add_entry(entry)
    rs = repr(ledger)
    assert "Ledger(entries=" in rs
    # Tu EntryLine.__repr__ actual usa dos decimales y 'Debit/Credit'
    assert "EntryLine(1105 - Cash, Debit 10.00)" in rs
    assert "EntryLine(4000 - Revenue, Credit 10.00)" in rs

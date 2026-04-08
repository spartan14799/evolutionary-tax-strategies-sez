# tests/test_accounting_entry.py
from __future__ import annotations

import os
import sys
from decimal import Decimal
import pytest

# ---------------------------------------------------------------------------
# Ajuste de ruta al root del proyecto (igual estilo que tus otros tests)
# ---------------------------------------------------------------------------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.simulation.economy.agent.accountant.accounting_entry import (  # noqa: E402
    EntryLine,
    AccountingEntry,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def D(x: str | float | int) -> Decimal:
    """Coerce a Decimal with 2 decimals for assertions."""
    return Decimal(str(x)).quantize(Decimal("0.01"))


# ---------------------------------------------------------------------------
# EntryLine tests
# ---------------------------------------------------------------------------

def test_entryline_creation_quantizes_and_types():
    ln = EntryLine("1105", "Cash", 100, True)
    assert ln.account_code == "1105"
    assert ln.account_name == "Cash"
    assert isinstance(ln.amount, Decimal)
    assert ln.amount == D("100.00")
    assert ln.is_debit is True


@pytest.mark.parametrize(
    "raw,expected",
    [
        (0.105, D("0.11")),  # ROUND_HALF_UP
        (0.104, D("0.10")),
        ("100", D("100.00")),
        (Decimal("3.335"), D("3.34")),  # clásico HALF_UP
    ],
)
def test_entryline_amount_round_half_up(raw, expected):
    ln = EntryLine("1105", "Cash", raw, True)
    assert ln.amount == expected


def test_entryline_truthy_falsy_is_debit():
    assert EntryLine("1105", "Cash", 1, 1).is_debit is True
    assert EntryLine("1105", "Cash", 1, 0).is_debit is False


def test_entryline_allows_negative_amount_and_preserves_sign():
    # El modelo no prohíbe negativos aquí; se documenta el comportamiento.
    ln = EntryLine("1105", "Cash", -5.2, True)
    assert ln.amount == D("-5.20")


def test_entryline_repr_formats_amount_and_direction():
    ln_debit = EntryLine("1105", "Cash", 100, True)
    ln_credit = EntryLine("4000", "Revenue", 100, False)
    assert repr(ln_debit) == "EntryLine(1105 - Cash, Debit 100.00)"
    assert repr(ln_credit) == "EntryLine(4000 - Revenue, Credit 100.00)"


# ---------------------------------------------------------------------------
# AccountingEntry tests
# ---------------------------------------------------------------------------

def test_accounting_entry_requires_at_least_one_line():
    with pytest.raises(ValueError):
        AccountingEntry([])


def test_accounting_entry_rejects_non_entryline_elements():
    with pytest.raises(TypeError) as e:
        AccountingEntry([object()])
    assert "is not an EntryLine" in str(e.value)


def test_accounting_entry_defensive_requantization():
    # Si alguien altera amount antes de construir la entry, se re-cuantiza dentro.
    ln = EntryLine("1105", "Cash", 1.234, True)
    # Fuerza un valor "no on-grid"
    ln.amount = Decimal("1.2357")
    entry = AccountingEntry([ln])
    # Totales deben reflejar 1.24 tras re-cuantización interna
    total_debits, total_credits = entry.totals()
    assert total_debits == D("1.24")
    assert total_credits == D("0.00")


def test_accounting_entry_totals_and_balance_true():
    lines = [
        EntryLine("1105", "Cash", 100, True),
        EntryLine("4000", "Revenue", 100, False),
    ]
    entry = AccountingEntry(lines)
    debits, credits = entry.totals()
    assert debits == D("100.00")
    assert credits == D("100.00")
    assert entry.is_balanced() is True


def test_accounting_entry_balance_false():
    lines = [
        EntryLine("1105", "Cash", 100, True),
        EntryLine("4000", "Revenue", 80, False),
    ]
    entry = AccountingEntry(lines)
    assert entry.is_balanced() is False


def test_accounting_entry_balance_true_after_rounding():
    # Ambos 0.105 → 0.11, por lo que debe quedar balanceado.
    lines = [
        EntryLine("1105", "Cash", 0.105, True),
        EntryLine("4000", "Revenue", 0.105, False),
    ]
    entry = AccountingEntry(lines)
    assert entry.totals() == (D("0.11"), D("0.11"))
    assert entry.is_balanced() is True


def test_accounting_entry_totals_type_are_decimal():
    lines = [
        EntryLine("1105", "Cash", 1, True),
        EntryLine("4000", "Revenue", 1, False),
    ]
    entry = AccountingEntry(lines)
    debits, credits = entry.totals()
    assert isinstance(debits, Decimal)
    assert isinstance(credits, Decimal)


def test_accounting_entry_zero_lines_balance_rules():
    # Dos líneas cero, una débito y otra crédito → balanceado.
    lines = [
        EntryLine("1105", "Cash", 0, True),
        EntryLine("4000", "Revenue", 0, False),
    ]
    entry = AccountingEntry(lines)
    assert entry.is_balanced() is True
    assert entry.totals() == (D("0.00"), D("0.00"))


def test_accounting_entry_repr_lists_lines_with_quantized_amounts():
    lines = [
        EntryLine("1105", "Cash", 50, True),
        EntryLine("4000", "Revenue", 50, False),
    ]
    entry = AccountingEntry(lines)
    expected = (
        "AccountingEntry(lines=[EntryLine(1105 - Cash, Debit 50.00), "
        "EntryLine(4000 - Revenue, Credit 50.00)])"
    )
    assert repr(entry) == expected


def test_accounting_entry_many_lines_accumulates_totals_correctly():
    lines = [
        EntryLine("1105", "Cash", 10.005, True),   # 10.01
        EntryLine("1105", "Cash", 9.994, True),    # 9.99
        EntryLine("4000", "Revenue", 20.00, False),
        EntryLine("4000", "Revenue", 0.005, False) # 0.01
    ]
    entry = AccountingEntry(lines)
    debits, credits = entry.totals()
    assert debits == D("20.00")    # 10.01 + 9.99
    assert credits == D("20.01")   # 20.00 + 0.01
    assert entry.is_balanced() is False

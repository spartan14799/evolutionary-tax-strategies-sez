# tests/test_accounting_entry_prop.py
from __future__ import annotations

import os
import sys
from decimal import Decimal, ROUND_HALF_UP
import pytest

# --- Ajuste de ruta al root del proyecto (mismo patrón que usas) ---
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.simulation.economy.agent.accountant.accounting_entry import (  # noqa: E402
    EntryLine,
    AccountingEntry,
)

# --- Helper de dinero (2 decimales, HALF_UP) ---
def D(x) -> Decimal:
    return Decimal(str(x)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


# =========================
# Tests deterministas extra
# =========================

@pytest.mark.parametrize(
    "raw,expected",
    [
        ("10.5", D("10.50")),  # string
        (10,     D("10.00")),  # int
        (10.015, D("10.02")),  # rounding HALF_UP
        (Decimal("3.335"), D("3.34")),
        (-2.1,   D("-2.10")),  # negativo permitido por la clase
    ],
)
def test_entryline_accepts_mixed_numeric_types_and_quantizes(raw, expected):
    ln = EntryLine("1105", "Cash", raw, True)
    assert ln.amount == expected


def test_accounting_entry_large_numbers():
    lines = [
        EntryLine("1105", "Cash", 10_000_000_000.99, True),
        EntryLine("4000", "Revenue", 10_000_000_000.99, False),
    ]
    entry = AccountingEntry(lines)
    debits, credits = entry.totals()
    assert debits == credits == D("10000000000.99")
    assert entry.is_balanced() is True


def test_accounting_entry_unbalanced_by_one_cent():
    lines = [
        EntryLine("1105", "Cash", 100, True),
        EntryLine("4000", "Revenue", 99.99, False),
    ]
    entry = AccountingEntry(lines)
    assert entry.totals() == (D("100.00"), D("99.99"))
    assert entry.is_balanced() is False


def test_entry_repr_shows_two_decimals_always():
    ln = EntryLine("1105", "Cash", 7, True)
    assert repr(ln) == "EntryLine(1105 - Cash, Debit 7.00)"


def test_accounting_entry_accepts_entryline_subclass():
    class MyLine(EntryLine):
        pass
    m = MyLine("1105", "Cash", 5, True)
    entry = AccountingEntry([m, EntryLine("4000", "Revenue", 5, False)])
    assert entry.is_balanced() is True


# =========================
# Property-based (opcional)
# =========================

# Importa hypothesis solo si está disponible; si no, deja definidos tests deterministas arriba
try:
    import hypothesis  # type: ignore
    from hypothesis import given, strategies as st  # type: ignore
except Exception:  # ModuleNotFoundError u otros
    hypothesis = None

if hypothesis is None:
    # Señal clara en el reporte, pero sin abortar el archivo completo
    @pytest.mark.skip(reason="hypothesis no está instalada; se omiten tests property-based.")
    def test_property_based_requires_hypothesis():
        pass
else:
    # Genera importes ya en grilla de 2 decimales (evita ambigüedad de redondeo)
    amounts_2dp = st.decimals(min_value=0, max_value=10_000, places=2)

    @given(st.lists(amounts_2dp, min_size=1, max_size=8))
    def test_balanced_when_credit_equals_sum_of_debits(amts):
        # N débitos + 1 crédito igual a la suma → balance exacto al centavo
        debit_lines = [EntryLine("1105", "Cash", a, True) for a in amts]
        total_debits = sum(Decimal(a) for a in amts)
        credit_line = EntryLine("4000", "Revenue", total_debits, False)

        entry = AccountingEntry(debit_lines + [credit_line])
        debits, credits = entry.totals()
        assert debits == D(total_debits)
        assert credits == D(total_debits)
        assert entry.is_balanced() is True

    @given(st.lists(amounts_2dp, min_size=1, max_size=8))
    def test_unbalanced_if_credit_off_by_one_cent(amts):
        total_debits = sum(Decimal(a) for a in amts)
        debit_lines = [EntryLine("1105", "Cash", a, True) for a in amts]
        # crédito “casi” igual pero 0.01 menos
        credit_line = EntryLine("4000", "Revenue", total_debits - Decimal("0.01"), False)

        entry = AccountingEntry(debit_lines + [credit_line])
        assert entry.is_balanced() is False

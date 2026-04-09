import sys
import os
from decimal import Decimal
import pytest

# Ensure project root is on sys.path so "classes.*" imports resolve when running pytest from anywhere
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.simulation.economy.agent.accountant.account import Account  # noqa: E402


# Helper to express money amounts as Decimal with two decimals
def D(x: str | float | int) -> Decimal:
    """Coerce to Decimal and quantize to 0.01 for stable money comparisons."""
    return Decimal(str(x)).quantize(Decimal("0.01"))


SEEDED_BALANCE = D("100000000.00")


@pytest.mark.parametrize("code,name,atype,expected_balance", [
    ("1105", "Cash", "Asset", SEEDED_BALANCE),            # special seeded account
    ("3115", "Owner's Equity", "Equity", SEEDED_BALANCE), # special seeded account
    ("4000", "Sales", "Revenue", D("0.00")),            # regular account
    ("2505", "Accounts Payable", "Liability", D("0.00"))# regular account
])
def test_account_initial_balance(code, name, atype, expected_balance):
    acc = Account(code, name, atype, "Some Subtype", "Some Group")
    assert acc.balance == expected_balance


def test_debit_increases_asset():
    # Starts at 1000.00 by seeding rule for code 1105
    acc = Account("1105", "Cash", "Asset", "Current Asset", "Cash and Cash Equivalents")
    acc.update_balance(100, is_debit=True)
    assert acc.balance == SEEDED_BALANCE + D("100.00")


def test_credit_decreases_asset():
    acc = Account("1105", "Cash", "Asset", "Current Asset", "Cash and Cash Equivalents")
    acc.update_balance(100, is_debit=True)   # 1100.00
    acc.update_balance(40, is_debit=False)   # 1060.00
    assert acc.balance == SEEDED_BALANCE + D("60.00")


def test_debit_decreases_revenue():
    acc = Account("4000", "Sales", "Revenue", "Operating Revenue", "Income")
    acc.update_balance(100, is_debit=True)   # debit reduces revenue
    assert acc.balance == D("-100.00")


def test_credit_increases_revenue():
    acc = Account("4000", "Sales", "Revenue", "Operating Revenue", "Income")
    acc.update_balance(100, is_debit=False)
    assert acc.balance == D("100.00")


def test_credit_increases_equity():
    acc = Account("3115", "Owner's Equity", "Equity", "Equity", "Equity Group")
    acc.update_balance(500, is_debit=False)  # credit increases equity
    assert acc.balance == SEEDED_BALANCE + D("500.00")


def test_debit_decreases_equity():
    acc = Account("3115", "Owner's Equity", "Equity", "Equity", "Equity Group")
    acc.update_balance(400, is_debit=True)   # debit decreases equity
    assert acc.balance == SEEDED_BALANCE - D("400.00")


def test_raises_on_invalid_type():
    with pytest.raises(ValueError):
        Account("9999", "Weird", "NotAType", "x", "y")


def test_quantizes_to_two_decimals_round_half_up():
    # Use a non-seeded asset code to start at 0.00
    acc = Account("1400", "Inventory X", "Asset", "Current Asset", "Inventory")
    acc.update_balance(0.105, is_debit=True)   # ROUND_HALF_UP → 0.11
    assert acc.balance == D("0.11")

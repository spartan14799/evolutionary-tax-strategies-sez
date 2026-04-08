"""
Module: accounting_entry.py
Description: Defines the basic structures for an accounting entry and its movement lines.
"""

from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP
from dataclasses import dataclass
from typing import List, Tuple

# ---------------------------------------------------------------------------
# Monetary utilities
# ---------------------------------------------------------------------------

# Money quantum: two decimal places (typical fiat currencies).
_MONEY_PLACES = Decimal("0.01")


def _to_money(value) -> Decimal:
    """
    Coerces any numeric-like value to Decimal and quantizes to _MONEY_PLACES
    using ROUND_HALF_UP, which is the usual rule in accounting contexts.

    This guarantees that all amounts (line-level and totals) live on the same
    monetary grid (0.01). It also prevents binary floating-point artifacts.
    """
    return Decimal(str(value)).quantize(_MONEY_PLACES, rounding=ROUND_HALF_UP)


# ---------------------------------------------------------------------------
# EntryLine
# ---------------------------------------------------------------------------

@dataclass
class EntryLine:
    """
    Represents a single movement line in an accounting entry.

    Attributes:
        account_code (str): The unique code for the account.
        account_name (str): The human-readable name for the account.
        amount (Decimal): The movement amount, stored as Decimal and quantized
                          to two decimals (see _to_money).
        is_debit (bool): True if the movement is a debit, False if it is a credit.

    Notes:
        - 'amount' is coerced and quantized at construction time to avoid any
          later surprises when computing totals.
        - 'is_debit' is stored as boolean; callers may pass truthy/falsy values.
    """
    account_code: str
    account_name: str
    amount: Decimal
    is_debit: bool

    def __init__(self, account_code: str, account_name: str, amount, is_debit: bool):
        # Store immutable identifiers as provided.
        self.account_code = account_code
        self.account_name = account_name

        # Normalize monetary value to Decimal with two decimals (ROUND_HALF_UP).
        self.amount = _to_money(amount)

        # Normalize direction to bool.
        self.is_debit = bool(is_debit)

    def __repr__(self):
        typ = "Debit" if self.is_debit else "Credit"
        return f"EntryLine({self.account_code} - {self.account_name}, {typ} {self.amount})"


# ---------------------------------------------------------------------------
# AccountingEntry
# ---------------------------------------------------------------------------

class AccountingEntry:
    """
    Represents an accounting entry composed of one or more movement lines.

    Invariants enforced by this class:
        - Each line's amount is a Decimal quantized to two decimals.
        - Totals are computed as Decimals on the same monetary grid.
        - Balance checks are exact at the cent (no floating tolerance).
    """

    def __init__(self, lines: List[EntryLine]):
        """
        Initialize the accounting entry with a list of EntryLine objects.

        Args:
            lines (list): List of EntryLine instances representing the entry movements.

        Raises:
            ValueError: If 'lines' is empty.
            TypeError:  If any element is not an EntryLine.
        """
        if not lines:
            raise ValueError("AccountingEntry requires at least one line.")

        # Defensive copy + normalization to guarantee all amounts are on-grid.
        # If a caller passed a subclass or modified EntryLine.amount externally,
        # we re-quantize here to preserve invariants.
        normalized: List[EntryLine] = []
        for idx, ln in enumerate(lines):
            if not isinstance(ln, EntryLine):
                raise TypeError(f"lines[{idx}] is not an EntryLine (got {type(ln)!r}).")
            ln.amount = _to_money(ln.amount)
            normalized.append(ln)

        self.lines = normalized

    # ----- Public helpers -----------------------------------------------------

    def totals(self) -> Tuple[Decimal, Decimal]:
        """
        Returns (total_debits, total_credits) as Decimals quantized to two decimals.

        Using Decimal with an explicit starting value avoids implicit coercions.
        """
        debits = sum((ln.amount for ln in self.lines if ln.is_debit), start=Decimal("0.00"))
        credits = sum((ln.amount for ln in self.lines if not ln.is_debit), start=Decimal("0.00"))
        # Quantize again to ensure the result strictly lives on the monetary grid.
        return _to_money(debits), _to_money(credits)

    def is_balanced(self) -> bool:
        """
        Check if the entry is balanced (i.e., total debits equal total credits)
        with exactness at the cent (no arbitrary float tolerance).

        Returns:
            bool: True if balanced, False otherwise.

        Rationale:
            - Accounting requires cent-accurate balancing.
            - By working with Decimals that have been quantized at input time,
              equality at two decimals is meaningful and robust.
        """
        total_debits, total_credits = self.totals()
        return total_debits == total_credits

    # ----- Dunder -------------------------------------------------------------

    def __repr__(self):
        return f"AccountingEntry(lines={self.lines})"

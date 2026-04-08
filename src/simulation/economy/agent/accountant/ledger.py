"""
Module: ledger.py
Description: Implements the ledger that records all accounting entries.
"""

from __future__ import annotations

from decimal import Decimal
from typing import List


class Ledger:
    """
    The ledger (general ledger) that stores all accounting entries.

    Design notes:
        - The ledger assumes each entry object exposes:
            * is_balanced() -> bool      (cent-accurate check using Decimal)
            * totals() -> (Decimal, Decimal)  (debits, credits) for diagnostics
        - Entries are appended only if they pass the balance check.
        - Error messages include debits/credits to aid debugging.
    """

    def __init__(self):
        # Internal storage for posted entries. Order is the posting order.
        self.entries: List[object] = []

    def add_entry(self, entry) -> None:
        """
        Add an accounting entry to the ledger after validating that it is balanced.

        Args:
            entry: The accounting entry to be added. Expected to be compatible
                   with the AccountingEntry interface (is_balanced/totals).

        Raises:
            ValueError: If the entry is not balanced.

        Implementation details:
            - Uses entry.is_balanced() with cent-accurate Decimal arithmetic.
            - On failure, includes (Debits, Credits) in the error for traceability.
        """
        if entry is None:
            raise ValueError("Cannot add a null entry to the ledger.")

        # Cent-accurate validation (AccountingEntry uses Decimal internally).
        if not entry.is_balanced():
            try:
                debits, credits = entry.totals()
            except Exception:
                # If totals() is not available for any reason, emit a generic error.
                raise ValueError("The accounting entry is not balanced.")
            # Provide diagnostic details to speed up investigation.
            raise ValueError(
                f"The accounting entry is not balanced. Debits={debits}, Credits={credits}"
            )

        # If validation passes, persist the entry in posting order.
        self.entries.append(entry)

    def get_all_entries(self) -> List[object]:
        """
        Retrieve all accounting entries.

        Returns:
            list: A shallow copy of all stored accounting entries.

        Rationale:
            - Returns a copy to prevent accidental mutation of the ledger's
              internal list by external callers, while keeping a simple API.
        """
        return list(self.entries)

    def __repr__(self):
        return f"Ledger(entries={self.entries})"

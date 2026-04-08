from __future__ import annotations

"""
Module: account.py
Description: Defines the core Account entity used by the Chart of Accounts.

This version stores monetary values as Decimal and enforces two-decimal
quantization (ROUND_HALF_UP). It preserves the original intent:
- Valid account types: asset, cost, liability, equity, revenue, expense
- Special initial seeding for accounts 1105 and 3115 (intentional)
"""

from decimal import Decimal, ROUND_HALF_UP
from typing import Final


# ---------------------------------------------------------------------------
# Monetary utilities
# ---------------------------------------------------------------------------

# Money quantum: two decimal places for standard fiat accounting.
_MONEY_PLACES: Final[Decimal] = Decimal("0.01")


def _to_money(value) -> Decimal:
    """
    Coerces any numeric-like value to Decimal and quantizes it to _MONEY_PLACES
    using ROUND_HALF_UP (typical accounting rounding).

    Rationale:
        - Prevents binary floating-point artifacts.
        - Keeps all balances and amounts on the same monetary grid (0.01).
    """
    return Decimal(str(value)).quantize(_MONEY_PLACES, rounding=ROUND_HALF_UP)


class Account:
    """
    Represents a single accounting account with all the information from the YAML file.

    Attributes:
        code (str): The unique code for the account.
        name (str): The name of the account.
        type (str): The account type (e.g., "Asset", "Cost", "Revenue", "Liability", "Equity", "Expense").
                    Internally validated case-insensitively against VALID_ACCOUNT_TYPES.
        subtype (str): A finer classification within the type (e.g., "Current Asset", "Cost of Production").
        group (str): A broader grouping category for presentation purposes (e.g., "Cash and Cash Equivalents").
        balance (Decimal): The current monetary balance of the account, stored as Decimal and quantized
                           to two decimals. It is initialized to 0.00 by default, or to 1000.00 for the
                           special seeded accounts (1105, 3115) as per the original design intent.

    Purpose:
        This class is the building block of the Chart of Accounts for each agent.
        When an accounting entry affects an account, the `update_balance()` method
        is used to adjust the balance accordingly.
    """

    # --- Valid account types allowed in the system (lowercase for comparison) ---
    VALID_ACCOUNT_TYPES = {"asset", "cost", "liability", "equity", "revenue", "expense"}

    def __init__(self, code: str, name: str, type: str, subtype: str, group: str):
        """
        Initializes the Account instance with its basic properties.

        Special behavior (intentional):
            - If the account code is "1105" (Cash) or "3115" (Owner's Equity),
              it is initialized with a balance of 100000000.00 monetary units.
            - Otherwise, it starts with a balance of 0.00.

        Args:
            code (str): The unique identifier for the account.
            name (str): The human-readable name of the account.
            type (str): The main classification of the account (case-insensitive).
            subtype (str): The sub-classification of the account.
            group (str): The group/category where the account belongs.

        Notes:
            - Type is stored as provided by the yaml file, but all logic
              validates against a lowercase version.
            - Balance is stored as Decimal; amounts are always quantized to 0.01.
        """
        self.code = str(code)
        self.name = name
        self.type = type  # keep original casing for presentation; logic will lowercase on use
        self.subtype = subtype
        self.group = group

        # Initialize balance (Decimal) respecting the original "magical seeding" intent.
        if self.code in {"1105", "3115"}:
            self.balance = _to_money("100000000.00")
        else:
            self.balance = _to_money("0.00")

        # Validate account type early to catch YAML/data issues as soon as possible.
        account_type = (self.type or "").lower()
        if account_type not in self.VALID_ACCOUNT_TYPES:
            raise ValueError(
                f"Unknown account type '{self.type}' for account '{self.code} - {self.name}'. "
                "Cannot initialize account."
            )

    def update_balance(self, amount, is_debit: bool) -> None:
        """
        Updates the account balance based on a transaction.

        Behavior:
            - For 'asset', 'cost', and 'expense' accounts:
                * Debit increases the balance.
                * Credit decreases the balance.
            - For 'liability', 'equity', and 'revenue' accounts:
                * Debit decreases the balance.
                * Credit increases the balance.

        Args:
            amount (numeric-like): The monetary value of the movement. It is coerced to Decimal
                                   and quantized to two decimals.
            is_debit (bool): True if it is a debit, False if it is a credit.

        Raises:
            ValueError: If the account type is not recognized (ensures system integrity).

        Notes:
            - Balance is always re-quantized after the operation to keep it on the 0.01 grid.
            - No implicit validation against negative amounts is performed here to preserve
              original behavior; enforce upstream if required by policy.
        """
        account_type = (self.type or "").lower()

        # Validate the account type before proceeding
        if account_type not in self.VALID_ACCOUNT_TYPES:
            raise ValueError(
                f"Unknown account type '{self.type}' for account '{self.code} - {self.name}'. "
                "Cannot update balance."
            )

        amt = _to_money(amount)

        # Apply accounting rules based on type
        if account_type in {"asset", "cost", "expense"}:
            # Assets, costs and expenses: debit increases, credit decreases
            new_balance = self.balance + (amt if is_debit else -amt)
        else:
            # liability, equity, revenue: debit decreases, credit increases
            new_balance = self.balance + (-amt if is_debit else amt)


        # Re-quantize to ensure the balance remains on the monetary grid.
        self.balance = _to_money(new_balance)

    def __repr__(self):
        """
        Returns a formal string representation of the account, useful for debugging.

        Example:
            Account(code='1105', name='Cash', type='Asset', subtype='Current Asset',
                    group='Cash and Cash Equivalents', balance=1000.00)
        """
        return (f"Account(code='{self.code}', name='{self.name}', type='{self.type}', "
                f"subtype='{self.subtype}', group='{self.group}', balance={self.balance})")

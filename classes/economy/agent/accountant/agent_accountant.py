from .accounting_entry import AccountingEntry, EntryLine


class AgentAccountant:
    """
    Processes and records accounting entries for a single agent.

    This class is responsible for updating the agent's Chart of Accounts and Ledger.
    """

    def __init__(self, chart_of_accounts, ledger):
        """
        Initialize with the agent's Chart of Accounts and Ledger.

        Args:
            chart_of_accounts (ChartOfAccounts): The agent's chart.
            ledger (Ledger): The agent's ledger for storing entries.
        """
        self.chart_of_accounts = chart_of_accounts
        self.ledger = ledger

    def record_entry(self, entry: AccountingEntry) -> None:
        """
        Atomically applies an accounting entry (Option A: pre-validate + rollback).

        This implementation enforces three phases:
            1) Pre-validation (no state mutation):
               - Resolves and validates every referenced account.
               - Verifies that the entry is balanced.
            2) Application with rollback capability:
               - Applies each movement line while tracking applied changes.
               - If any exception occurs, it reverts all applied changes (rollback).
            3) Ledger registration:
               - Persists the entry in the ledger.
               - If the ledger rejects the entry, it triggers a rollback.

        If any step fails, all balances remain exactly as before the call.

        Raises:
            ValueError: If the entry is invalid, unbalanced, or references a missing account.
            Exception:  Any underlying exception from account updates or ledger posting
                        is re-raised after rollback.
        """
        # ---- 1) PRE-VALIDATION (NO STATE CHANGES) ----
        if entry is None or not getattr(entry, "lines", None):
            raise ValueError("Empty or invalid AccountingEntry.")

        resolved_accounts = []
        for idx, line in enumerate(entry.lines):
            # Resolve target account
            account = self.chart_of_accounts.get_account(line.account_code)
            if account is None:
                raise ValueError(
                    f"Account with code {line.account_code} "
                    f"not found in the Chart of Accounts (line {idx})."
                )

            # Validate amount presence (type/constraints can be added if required)
            if getattr(line, "amount", None) is None:
                raise ValueError(f"Line {idx} has no amount.")

            resolved_accounts.append(account)

        # Validate balance before mutating any state
        if not entry.is_balanced():
            # If available, include totals to aid diagnostics
            try:
                debits, credits = entry.totals()
                raise ValueError(
                    f"Unbalanced entry. Debits={debits}, Credits={credits}."
                )
            except Exception:
                # If totals() is not available, provide a generic message
                raise ValueError("Unbalanced entry.")

        # ---- 2) APPLY WITH ROLLBACK CAPABILITY ----
        applied = []  # list of (account, amount, is_debit) already applied
        try:
            for account, line in zip(resolved_accounts, entry.lines):
                # Apply the movement
                account.update_balance(line.amount, line.is_debit)
                # Track the applied change for potential rollback
                applied.append((account, line.amount, line.is_debit))

            # ---- 3) LEDGER REGISTRATION ----
            # If this fails, the except block below will rollback all applied changes.
            self.ledger.add_entry(entry)

        except Exception:
            # ROLLBACK: revert all applied changes in reverse order
            for account, amount, was_debit in reversed(applied):
                # Reverting equals applying the opposite side (debit <-> credit)
                account.update_balance(amount, not was_debit)
            # Re-raise the original exception to the caller
            raise


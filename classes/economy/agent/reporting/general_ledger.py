from __future__ import annotations
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List

_MONEY_PLACES = Decimal("0.01")

def _to_money(value: Any) -> Decimal:
    """Coerce to Decimal and quantize to 2 decimals (ROUND_HALF_UP)."""
    return Decimal(str(value)).quantize(_MONEY_PLACES, rounding=ROUND_HALF_UP)

def _get_ledger(agent: Any):
    """
    Minimal ledger resolver:
      - agent.ledger
      - agent._accounting_agent.ledger (fallback)
    """
    led = getattr(agent, "ledger", None)
    if led is not None:
        return led
    acc_agent = getattr(agent, "_accounting_agent", None)
    if acc_agent is not None:
        led = getattr(acc_agent, "ledger", None)
        if led is not None:
            return led
    raise AttributeError("Ledger not found on agent (expected 'agent.ledger' or '_accounting_agent.ledger').")

def generate_general_ledger(agent: Any) -> List[Dict[str, Any]]:
    """
    Produce a flat list of postings from the agent's ledger.

    Each row:
      - entry_index: position of the entry in the ledger
      - account_code
      - account_name
      - amount: Decimal with 2 decimals
      - debit_or_credit: 'Debit' | 'Credit'
    """
    ledger = _get_ledger(agent)

    rows: List[Dict[str, Any]] = []
    for idx, entry in enumerate(ledger.get_all_entries()):
        for line in getattr(entry, "lines", []):
            rows.append({
                "entry_index": idx,
                "account_code": getattr(line, "account_code", ""),
                "account_name": getattr(line, "account_name", ""),
                "amount": _to_money(getattr(line, "amount", 0)),
                "debit_or_credit": "Debit" if getattr(line, "is_debit", False) else "Credit",
            })
    return rows

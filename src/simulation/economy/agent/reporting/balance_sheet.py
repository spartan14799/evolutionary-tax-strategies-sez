# -*- coding: utf-8 -*-
# Authors: ANDRES LEGUIZAMON / CARLOS SANCHEZ / GERMAN RODRIGUEZ / SOFIA OCAMPO
# Year: 2025
# TODO: LICENSE / COPYRIGHT
# -----------------------------------------------------------------------------
"""
balance_sheet.py

Minimal, safe Balance Sheet helpers.

Design goals
------------
• Keep it functional and straightforward.
• All arithmetic in Decimal (quantized a 2 decimales).
• Usar la vista de compatibilidad `.accounts` del Chart of Accounts.
• Opcionalmente incluir “Net Profit” en Liabilities+Equity (estilo cierre simple),
  tomando `agent.net_profit` si existe o recomputándolo con el income statement.

API
---
get_final_balances(agent, as_float=False) -> dict[code, Decimal|float]
generate_balance_sheet(
    agent,
    include_net_profit=True,
    recompute_net_profit_if_missing=True,
    as_float=False,
) -> dict
"""

from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, Optional


# ============================== Monetary helpers =============================

_MONEY_PLACES = Decimal("0.01")

def _to_money(value: Any) -> Decimal:
    """
    Coerce to Decimal and quantize to 2 decimals using ROUND_HALF_UP.
    Accepts Decimal, int, float, str.
    """
    return Decimal(str(value)).quantize(_MONEY_PLACES, rounding=ROUND_HALF_UP)


# ================================ Accessors ==================================

def _get_accounts_dict(agent: Any):
    """
    Resolve the accounts mapping ({code: Account}) from the agent.

    Tries:
      1) agent.chart_of_accounts.accounts
      2) agent._accounting_agent.chart_of_accounts.accounts
    """
    chart = getattr(agent, "chart_of_accounts", None)
    if chart is not None and hasattr(chart, "accounts"):
        return chart.accounts

    acc_agent = getattr(agent, "_accounting_agent", None)
    if acc_agent is not None:
        chart = getattr(acc_agent, "chart_of_accounts", None)
        if chart is not None and hasattr(chart, "accounts"):
            return chart.accounts

    raise AttributeError(
        "Could not find 'chart_of_accounts.accounts' on the agent. "
        "Ensure the ChartOfAccounts exposes a `.accounts` Mapping."
    )


# ================================ Public API =================================

def get_final_balances(agent: Any, *, as_float: bool = False) -> Dict[str, Decimal] | Dict[str, float]:
    """
    Retrieve the final balance of every account in the agent's Chart of Accounts.

    Returns:
        dict: {account_code -> balance} using Decimal by default (or float if as_float=True).
    """
    accounts = _get_accounts_dict(agent)
    out: Dict[str, Decimal] = {}
    for code, account in accounts.items():
        out[str(code)] = _to_money(getattr(account, "balance", Decimal("0.00")))
    if as_float:
        return {c: float(v) for c, v in out.items()}
    return out


def generate_balance_sheet(
    agent: Any,
    *,
    include_net_profit: bool = True,
    recompute_net_profit_if_missing: bool = True,
    as_float: bool = False,
) -> Dict[str, Any]:
    """
    Generate a basic Balance Sheet for the agent.

    Structure:
      {
        "Assets":       { "code - name": Decimal, ... },
        "Liabilities":  { "code - name": Decimal, ... },
        "Equity":       { "code - name": Decimal, ... },
        "Totals": {
            "Assets": Decimal,
            "Liabilities": Decimal,
            "Equity": Decimal,
            "Net Profit": Decimal,                 # only if include_net_profit
            "LiabilitiesPlusEquity": Decimal       # includes Net Profit if requested
        }
      }

    Notes
    -----
    • Only types 'asset' | 'liability' | 'equity' are shown in sections.
      Revenue/Cost/Expense are de estado de resultados y no se muestran aquí.
    • “LiabilitiesPlusEquity” suma pasivo + patrimonio (+ net profit si include_net_profit=True).
      Esto modela el caso donde utilidades no están cerradas a una cuenta de equity todavía.
    """
    accounts = _get_accounts_dict(agent)

    sheet: Dict[str, Dict[str, Decimal]] = {
        "Assets": {},
        "Liabilities": {},
        "Equity": {},
    }

    # ---- Classify accounts by type (only A/L/E in a BS) ----
    for code, account in accounts.items():
        acct_type = (getattr(account, "type", "") or "").lower().strip()
        if acct_type not in ("asset", "liability", "equity"):
            continue  # ignore revenue/cost/expense/etc for the balance sheet view

        label = f"{code} - {getattr(account, 'name', str(code))}"
        bal = _to_money(getattr(account, "balance", Decimal("0.00")))

        if acct_type == "asset":
            sheet["Assets"][label] = bal
        elif acct_type == "liability":
            sheet["Liabilities"][label] = bal
        elif acct_type == "equity":
            sheet["Equity"][label] = bal

    # ---- Totals (Decimal-only math) ----
    total_assets      = _to_money(sum(sheet["Assets"].values()))
    total_liabilities = _to_money(sum(sheet["Liabilities"].values()))
    total_equity      = _to_money(sum(sheet["Equity"].values()))

    # ---- Net Profit handling (optional) ----
    net_profit: Optional[Decimal] = None
    if include_net_profit:
        np_attr = getattr(agent, "net_profit", None)
        if np_attr is not None:
            # Use the persisted value (income_statement sets Decimal)
            try:
                net_profit = _to_money(np_attr)
            except Exception:
                net_profit = None

        # If missing and allowed, try to compute on the fly (lazy import)
        if net_profit is None and recompute_net_profit_if_missing:
            try:
                # Local import avoids hard dependency/circular import issues.
                from .income_statement import generate_income_statement  # type: ignore
                stmt = generate_income_statement(agent, persist=False, as_float=False)
                net_profit = _to_money(stmt.get("Net Profit", Decimal("0.00")))
            except Exception:
                # Last resort: assume zero; do not fail the balance sheet.
                net_profit = Decimal("0.00")
        elif net_profit is None:
            # Explicitly requested not to recompute: treat as zero.
            net_profit = Decimal("0.00")

    # ---- Build Totals section ----
    totals: Dict[str, Decimal] = {
        "Assets": total_assets,
        "Liabilities": total_liabilities,
        "Equity": total_equity,
    }

    if include_net_profit:
        totals["Net Profit"] = _to_money(net_profit or Decimal("0.00"))
        liabilities_plus_equity = total_liabilities + total_equity + totals["Net Profit"]
    else:
        liabilities_plus_equity = total_liabilities + total_equity

    totals["LiabilitiesPlusEquity"] = _to_money(liabilities_plus_equity)

    result: Dict[str, Any] = {
        "Assets": sheet["Assets"],
        "Liabilities": sheet["Liabilities"],
        "Equity": sheet["Equity"],
        "Totals": totals,
    }

    # ---- Optional float conversion for UI/serialization ----
    if as_float:
        def _flt_map(d: Dict[str, Decimal]) -> Dict[str, float]:
            return {k: float(v) for k, v in d.items()}

        result = {
            "Assets": _flt_map(result["Assets"]),
            "Liabilities": _flt_map(result["Liabilities"]),
            "Equity": _flt_map(result["Equity"]),
            "Totals": {k: float(v) for k, v in result["Totals"].items()},
        }

    return result

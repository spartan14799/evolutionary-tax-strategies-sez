# -*- coding: utf-8 -*-
# Authors: ANDRES LEGUIZAMON / CARLOS SANCHEZ / GERMAN RODRIGUEZ / SOFIA OCAMPO
# Year: 2025
# TODO: LICENSE / COPYRIGHT
# -----------------------------------------------------------------------------
"""
income_statement.py

Single-function façade to generate an Income Statement for an agent.

This version introduces a first-class **tax policy layer** (Strategy pattern).
The income statement generators are intentionally **agnostic** to fiscal rules:
they compute financial aggregates (e.g., Income Before Taxes, IBT) and delegate
tax computation to a pluggable **TaxPolicy**. This enables evolution from a
single flat tax regime to multi-regime, source-sensitive taxation (e.g., ZF flat
20% today; ZF mixed 20% exports / 35% domestic tomorrow) **without modifying**
the report generators.

Key changes
-----------
• Adds a `TaxPolicy` protocol with three policies: `StandardPolicy` (35%),
  `ZFFlatPolicy` (20%), and `ZFMixedPolicy` (20% for export-sourced IBT,
  35% otherwise).
• Extends the standard generator to compute **COGS by market** (national vs.
  international), when the Chart of Accounts provides such labels, so that
  mixed policies can compute **IBT by source** accurately (revenue − COGS),
  avoiding revenue-only prorations.
• Removes the legacy `colombia_simple_tax_policy` helper. All tax logic now
  flows through the policy layer.

Goals
-----
• Keep reporting simple: one public function `generate_income_statement(agent, ...)`.
• Choose between two built-in presentation strategies ("standard" | "alternative").
• Maintain numerical consistency: all computations use `Decimal` (quantized to cents).
• Be integration-friendly: can persist `agent.net_profit` (Decimal) and store a
  copy of the statement in `agent.last_income_statement`.
• Decouple fiscal rules: income statements remain stable as tax policies evolve.

How strategy is chosen
----------------------
We try, in order:
  1) agent.get_economic_context().income_statement_type
  2) agent.income_statement_type
Fallback: "standard"

Returned structure
------------------
A dict[str, Decimal] with well-labeled totals. Keys differ slightly by strategy,
but both include "Net Profit" so downstream code can rely on it.

Safety & Compatibility
----------------------
• Access accounts through `agent.chart_of_accounts.accounts` (read-only Mapping)
  with the compatibility view exposed by ChartOfAccounts.
• If floats are needed for UI, pass `as_float=True` to convert at the very end.
• To switch ZF from flat (20%) to mixed (20% exports / 35% rest), either:
    - set `DEFAULT_ZF_POLICY_NAME = "mixed"` in this module, or
    - set `agent.tax_policy_name = "zf_mixed"` (or configure in its economic context).

Example
-------
stmt = generate_income_statement(agent, persist=True)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, Optional, Protocol


# ============================== Monetary helpers =============================

_MONEY_PLACES = Decimal("0.01")


def _to_money(value: Any) -> Decimal:
    """
    Coerce to Decimal and quantize to 2 decimals using ROUND_HALF_UP.
    Accepts Decimal, int, float, str.
    """
    return Decimal(str(value)).quantize(_MONEY_PLACES, rounding=ROUND_HALF_UP)


# ============================== Generic helpers ==============================

def _safe_lower(value: Optional[str]) -> str:
    """Safely lowercase a string (None -> '')."""
    return (value or "").lower()


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


def _resolve_income_statement_type(agent: Any, default: str = "standard") -> str:
    """
    Resolve which presentation type to use.
    Order:
      1) agent.get_economic_context().income_statement_type
      2) agent.income_statement_type
      3) default
    """
    stype = None
    if hasattr(agent, "get_economic_context"):
        ctx = agent.get_economic_context()
        stype = getattr(ctx, "income_statement_type", None)
    if stype is None:
        stype = getattr(agent, "income_statement_type", None)
    return (stype or default).lower()


# ================================ Tax policy =================================
# Design overview:
# - The tax policy is encapsulated behind a Protocol (Strategy pattern).
# - Generators compute IBT and minimal source splits (revenues and optionally COGS).
# - Policy decides applicable rates (e.g., ZF 20% vs. standard 35%) and
#   how to allocate IBT across sources when needed (e.g., mixed ZF regime).
# - Negative IBT is permitted and yields negative tax expense unless a policy
#   implements alternative rules.

# Canonical corporate rates (can be reused across policies).
DEFAULT_TAX_RATE = Decimal("0.35")  # Standard regime
ZF_TAX_RATE = Decimal("0.20")       # ZF incentive rate

# Module default for ZF policy selection. Change to "mixed" when export-sensitive
# taxation is enabled system-wide (unless overridden per agent/context).
DEFAULT_ZF_POLICY_NAME = "flat"     # "flat" | "mixed"



# ====== Explicit code-based routing (robust to YAML label drift) ======
# Deprecated: kept for backward-compat / future migrations; not used by generators.
INTL_COGS_CODES: set[str] = {"6130"}                  # Internacional (COGS)
NAT_COGS_CODES:  set[str] = {"6120", "6135"}          # Nacional (COGS)

INTL_REV_CODES:  set[str] = {"4130", "4185"}          # Internacional (Revenues)
NAT_REV_CODES:   set[str] = {"4120", "4135"}          # Nacional (Revenues)


@dataclass(frozen=True)
class TaxBreakdown:
    """
    Minimal information required by a TaxPolicy to compute the tax expense.

    Attributes
    ----------
    ibt : Decimal
        Income Before Taxes (already quantized to money).
    revenue_by_source : Dict[str, Decimal]
        Operating revenue decomposition by source, e.g.:
        {
            "international_operating": Decimal(...),
            "national_operating": Decimal(...),
            # optionally others, e.g., "other": Decimal(...)
        }
        Policies may use this to allocate IBT proportionally when needed.
    cogs_by_source : Dict[str, Decimal]
        Optional Cost of Sales decomposition aligned with `revenue_by_source`.
        When provided, mixed policies (e.g., export-sensitive) can compute IBT
        by source without relying on revenue-only prorations.
    """
    ibt: Decimal
    revenue_by_source: Dict[str, Decimal]
    cogs_by_source: Dict[str, Decimal] = field(default_factory=dict)


@dataclass(frozen=True)
class TaxComputation:
    """
    Result of a tax policy computation.

    Attributes
    ----------
    tax_expense : Decimal
        Total tax expense to be recognized for the period.
    by_source : Dict[str, Decimal]
        Optional diagnostic/detail of tax expense contributions by source key.
    """
    tax_expense: Decimal
    by_source: Dict[str, Decimal]


class TaxPolicy(Protocol):
    """
    Protocol for pluggable tax policies.

    Any implementation must accept a `TaxBreakdown` and return a `TaxComputation`.
    """
    def compute(self, bd: TaxBreakdown) -> TaxComputation:
        ...


def _split_ibt_by_revenue(ibt: Decimal, rev_by_src: Dict[str, Decimal]) -> Dict[str, Decimal]:
    """
    Allocate IBT across revenue sources using proportional weights.

    Rationale
    ---------
    When total IBT is available but policy needs to apply distinct rates to
    parts of the business (e.g., international vs. national operations),
    IBT is apportioned according to each source's share of **operating revenue**.
    This is a practical proxy; if the system later tracks a true taxable base
    by source, policies can be updated to use it instead of a proxy.

    Edge cases
    ----------
    • If total operating revenue is zero, the full IBT is assigned to 'other'.
    """
    total_rev = _to_money(sum((v or Decimal("0")) for v in rev_by_src.values()))
    if total_rev == 0:
        return {"other": ibt}
    shares = {k: (v / total_rev) for k, v in rev_by_src.items()}
    return {k: _to_money(ibt * s) for k, s in shares.items()}


class StandardPolicy:
    """
    General corporate regime: single flat rate (35%) on total IBT.

    This policy is intentionally simple and serves as the default for non-ZF
    agents unless an override is configured.
    """
    RATE = DEFAULT_TAX_RATE

    def compute(self, bd: TaxBreakdown) -> TaxComputation:
        tax = _to_money(self.RATE * bd.ibt)
        return TaxComputation(tax_expense=tax, by_source={"all": tax})


class ZFFlatPolicy:
    """
    Current ZF regime (today): flat 20% on total IBT.

    This keeps behavior aligned with the present regulatory assumption where
    ZF incentive is a global rate, independent of revenue composition.
    """
    RATE = ZF_TAX_RATE

    def compute(self, bd: TaxBreakdown) -> TaxComputation:
        tax = _to_money(self.RATE * bd.ibt)
        return TaxComputation(tax_expense=tax, by_source={"all": tax})


class ZFMixedPolicy:
    """
    Export-sensitive ZF regime:

    • 20% on IBT sourced by "international_operating".
    • 35% on the remainder (e.g., "national_operating" and any residual "other").

    Implementation notes
    --------------------
    • If `cogs_by_source` is provided, IBT by source is computed as
      (revenue_by_source - cogs_by_source) with no truncation at zero, so
      losses (negative IBT) propagate as negative tax (credits).
    • Residual = bd.ibt - sum(ibt_by_source_raw) is taxed once at 35%.
    • Quantization is applied only when producing taxes and final totals to
      avoid artificial residuals caused by early rounding.
    """
    RATES = {
        "international_operating": ZF_TAX_RATE,     # 20%
        "national_operating":     DEFAULT_TAX_RATE, # 35%
        "other":                  DEFAULT_TAX_RATE, # 35% fallback
    }

    def compute(self, bd: TaxBreakdown) -> TaxComputation:
        parts: Dict[str, Decimal] = {}

        if bd.cogs_by_source:
            # 1) Raw IBT by source (no early rounding, no clamping).
            keys = set(bd.revenue_by_source) | set(bd.cogs_by_source)
            ibt_by_src_raw: Dict[str, Decimal] = {
                k: (bd.revenue_by_source.get(k, Decimal("0"))
                    - bd.cogs_by_source.get(k, Decimal("0")))
                for k in keys
            }

            # 2) Compute residual once, then quantize for taxation.
            sourced_sum_raw = sum(ibt_by_src_raw.values())
            residual = _to_money(bd.ibt - sourced_sum_raw)

            # 3) Tax each sourced chunk at its specific rate (allowing negatives).
            for k, ibt_k_raw in ibt_by_src_raw.items():
                rate = self.RATES.get(k, DEFAULT_TAX_RATE)
                parts[k] = _to_money(rate * _to_money(ibt_k_raw))

            # 4) Tax residual at general rate (35%) once.
            if residual != 0:
                parts["other"] = parts.get("other", Decimal("0.00")) + _to_money(DEFAULT_TAX_RATE * residual)

            return TaxComputation(tax_expense=_to_money(sum(parts.values())), by_source=parts)

        # Fallback: allocate total IBT by revenue shares if no COGS split was provided.
        ibt_by_src = _split_ibt_by_revenue(bd.ibt, bd.revenue_by_source)
        for k, ibt_k in ibt_by_src.items():
            rate = self.RATES.get(k, DEFAULT_TAX_RATE)
            parts[k] = _to_money(rate * ibt_k)

        return TaxComputation(tax_expense=_to_money(sum(parts.values())), by_source=parts)


def _is_zf_agent(agent: Any) -> bool:
    """
    Heuristic detector for ZF agents.

    It considers common attributes and naming conventions without coupling to a
    specific class hierarchy. If the system later adds a formal flag, prefer that.
    """
    cand = {
        str(getattr(agent, "type", "")),
        str(getattr(agent, "agent_type", "")),
        str(getattr(agent, "name", "")),
        getattr(agent, "__class__", type("X", (object,), {})).__name__,
    }
    normalized = {c.strip().upper() for c in cand if c}
    keywords = ("ZF", "FTZ", "FREE TRADE ZONE")
    return any(any(keyword in value for keyword in keywords) for value in normalized)


def _resolve_tax_policy(agent: Any) -> TaxPolicy:
    """
    Resolution precedence (strongest first):
      1) agent.<policy-name-like>
      2) agent.get_economic_context().<policy-name-like>
      3) Heuristic ZF with module default
      4) Standard
    """
    def norm(x):
        if not isinstance(x, str):
            return None
        return x.strip().lower().replace(" ", "_").replace("-", "_")

    # --- 1) Look on agent first (several synonyms) ---
    for attr in ("tax_policy_name", "tax_policy", "policy_name", "policy", "tax_regime"):
        val = norm(getattr(agent, attr, None))
        if val:
            if val in {"zf_mixed", "mixed_zf"}:
                return ZFMixedPolicy()
            if val in {"zf_flat"}:
                return ZFFlatPolicy()
            if val in {"standard", "std"}:
                return StandardPolicy()
            # If user hints ZF but not flat/mixed, default to module default
            if "zf" in val:
                return ZFMixedPolicy() if norm(DEFAULT_ZF_POLICY_NAME) == "mixed" else ZFFlatPolicy()

    # --- 2) Else, look on context (same synonyms) ---
    if hasattr(agent, "get_economic_context"):
        ctx = agent.get_economic_context()
        for attr in ("tax_policy_name", "tax_policy", "policy_name", "policy", "tax_regime"):
            val = norm(getattr(ctx, attr, None))
            if val:
                if val in {"zf_mixed", "mixed_zf"}:
                    return ZFMixedPolicy()
                if val in {"zf_flat"}:
                    return ZFFlatPolicy()
                if val in {"standard", "std"}:
                    return StandardPolicy()
                if "zf" in val:
                    return ZFMixedPolicy() if norm(DEFAULT_ZF_POLICY_NAME) == "mixed" else ZFFlatPolicy()

    # --- 3) Heuristic ZF by agent hints (keep your current heuristic) ---
    if _is_zf_agent(agent):
        return ZFMixedPolicy() if norm(DEFAULT_ZF_POLICY_NAME) == "mixed" else ZFFlatPolicy()

    # --- 4) Fallback ---
    return StandardPolicy()




# ============================ Strategy: STANDARD ==============================

def _generate_standard(agent: Any) -> Dict[str, Decimal]:
    """
    Standard income statement (detailed scheme).

    This function aggregates an agent’s accounts into a conventional income
    statement and delegates tax computation to a pluggable TaxPolicy. It
    classifies revenues, costs, and expenses using semantic labels exposed by
    the chart of accounts (`type`, `subtype`, `group`). Using the same
    label-based buckets for both presentation and taxation guarantees that
    mixed tax regimes (e.g., ZF mixed) react correctly to accounting labels
    and avoid drift from hard-coded account codes.

    Key characteristics
    -------------------
    • Label-based classification for:
        - Operating revenues (national vs. international)
        - Cost of sales (national vs. international vs. unspecified)
        - Production/indirect costs, selling/admin/overheads, financial,
          non-operating revenues/expenses
    • Computes the financial aggregates:
        GROSS PROFIT → OPERATING PROFIT → INCOME BEFORE TAXES (IBT).
    • Determines tax using the selected TaxPolicy. For ZF mixed, tax is applied
      to source-based operating profits:
          profit_intl = (international operating revenues) − (international COGS)
          profit_nat  = (national operating revenues)       − (national COGS)
          residual    = IBT − (profit_intl + profit_nat)
      with 20% on international, 35% on national, and 35% on residual.
    • All values are handled as Decimal and quantized to two decimals at the end.

    Returns
    -------
    Dict[str, Decimal]
        Labeled totals including "Net Profit".
    """

    accounts = _get_accounts_dict(agent)

    # ------------------------------- Buckets ---------------------------------
    # Operating revenues by market
    national_oper: Dict[str, Decimal] = {}
    international_oper: Dict[str, Decimal] = {}

    # Cost of sales buckets (by market + unspecified)
    cost_sales_unspec: Dict[str, Decimal] = {}
    cost_sales_nat: Dict[str, Decimal] = {}
    cost_sales_intl: Dict[str, Decimal] = {}

    # Other cost/expense buckets
    cost_prod: Dict[str, Decimal] = {}       # “Cost of Production”
    indirect_costs: Dict[str, Decimal] = {}  # “Indirect” costs

    selling: Dict[str, Decimal] = {}
    admin: Dict[str, Decimal] = {}
    overheads: Dict[str, Decimal] = {}
    financial: Dict[str, Decimal] = {}

    nonop_exp: Dict[str, Decimal] = {}
    nonop_rev: Dict[str, Decimal] = {}
    nonop_costs: Dict[str, Decimal] = {}

    # ---------------------------- Classification -----------------------------
    # Notes:
    # • All string fields are lowered for robust matching.
    # • Amounts are aggregated by magnitude; the chart should provide balances
    #   with appropriate signs at the account level.
    for code, account in accounts.items():
        t = _safe_lower(getattr(account, "type", None))
        subtype = _safe_lower(getattr(account, "subtype", None))
        group = _safe_lower(getattr(account, "group", None))
        name = getattr(account, "name", str(code))

        bal = _to_money(getattr(account, "balance", Decimal("0.00")))
        amount = bal
        label = f"{code} - {name}"

        # ------------------------- Revenue classification ---------------------
        if t == "revenue":
            # IMPORTANT: check 'international' BEFORE 'national' to avoid the
            # substring trap (e.g., "internatio[nal] operating revenue").
            if "international operating revenue" in group:
                international_oper[label] = amount
            elif "national operating revenue" in group:
                national_oper[label] = amount
            else:
                # Revenues outside operating scope are treated as non-operating.
                nonop_rev[label] = amount

        # ---------------------- Cost / Expense classification -----------------
        elif t == "cost":
            # Cost of Sales (by market or unspecified)
            if "sales" in subtype:
                if "international" in group:
                    cost_sales_intl[label] = amount
                elif "national" in group:
                    cost_sales_nat[label] = amount
                else:
                    cost_sales_unspec[label] = amount
            # Production and indirect costs (not part of COGS)
            elif "production" in subtype:
                cost_prod[label] = amount
            elif "indirect" in subtype:
                indirect_costs[label] = amount
            else:
                # Any remaining cost classified as non-operating cost
                nonop_costs[label] = amount

        elif t == "expense":
            # Overheads (by group) or indirect expenses (by subtype)
            if "production overheads" in group or "indirect" in subtype:
                overheads[label] = amount
            # Some ledgers may register COGS as expenses; handle safely by group
            elif "cost of sales" in group:
                if "international" in group:
                    cost_sales_intl[label] = amount
                elif "national" in group:
                    cost_sales_nat[label] = amount
                else:
                    cost_sales_unspec[label] = amount
            # Selling / administrative / financial operating expenses
            elif "sales" in group:
                selling[label] = amount
            elif "administrative" in group:
                admin[label] = amount
            elif "financial" in group:
                financial[label] = amount
            else:
                # Miscellaneous operating expenses fall here; non-operating.
                nonop_exp[label] = amount

    # -------------------------------- Totals ---------------------------------
    total_oper_rev = sum(national_oper.values()) + sum(international_oper.values())

    total_cogs_nat = sum(cost_sales_nat.values())
    total_cogs_intl = sum(cost_sales_intl.values())
    total_cogs_unspec = sum(cost_sales_unspec.values())

    # “TOTAL COSTS OF SALES” includes unspecified COGS and any production/indirect
    # costs if the reporting policy defines them within COGS. Adjust if needed.
    total_cost_sales = (
        total_cogs_unspec
        + total_cogs_nat
        + total_cogs_intl
        + sum(cost_prod.values())
        + sum(indirect_costs.values())
    )

    total_oper_exp = sum(selling.values()) + sum(admin.values()) + sum(overheads.values())

    total_nonop_rev = sum(nonop_rev.values())
    total_nonop_exp = sum(nonop_costs.values()) + sum(nonop_exp.values())

    total_fin_exp = sum(financial.values())

    # ------------------------------ Profits ----------------------------------
    gross_profit = _to_money(total_oper_rev - total_cost_sales)
    operating_profit = _to_money(gross_profit - total_oper_exp)

    income_before_taxes = _to_money(
        operating_profit
        + total_nonop_rev
        - total_nonop_exp
        - total_fin_exp
    )

    # ------------------------ Tax (label-based, deterministic) ---------------
    # The policy is resolved from the agent; ZF mixed is computed directly from
    # the same buckets used for presentation to guarantee alignment.
    policy = _resolve_tax_policy(agent)

    # Operating profits by market (quantized to money once)
    rev_nat_val  = _to_money(sum(national_oper.values()))
    rev_int_val  = _to_money(sum(international_oper.values()))
    cogs_nat_val = _to_money(total_cogs_nat)
    cogs_int_val = _to_money(total_cogs_intl)

    profit_nat  = _to_money(rev_nat_val - cogs_nat_val)
    profit_intl = _to_money(rev_int_val - cogs_int_val)

    # Residual captures everything not in the two operating markets:
    # non-operating items, financial expenses, unspecified COGS, etc.
    residual_profit = _to_money(income_before_taxes - _to_money(profit_nat + profit_intl))

    if isinstance(policy, ZFMixedPolicy):
        tax_nat   = _to_money(DEFAULT_TAX_RATE * profit_nat)       # 35% national
        tax_intl  = _to_money(ZF_TAX_RATE      * profit_intl)      # 20% international
        tax_other = _to_money(DEFAULT_TAX_RATE * residual_profit)  # 35% residual
        tax_expense = _to_money(tax_nat + tax_intl + tax_other)
    elif isinstance(policy, ZFFlatPolicy):
        tax_expense = _to_money(ZF_TAX_RATE * income_before_taxes)         # 20% on total IBT
    else:  # StandardPolicy
        tax_expense = _to_money(DEFAULT_TAX_RATE * income_before_taxes)     # 35% on total IBT

    # ------------------------------ Final lines ------------------------------
    taxable_income = income_before_taxes
    net_profit = _to_money(income_before_taxes - tax_expense)

    return {
        "TOTAL OPERATING REVENUES": _to_money(total_oper_rev),
        "TOTAL COSTS OF SALES": _to_money(total_cost_sales),
        "TOTAL COSTS OF SALES (National)": _to_money(total_cogs_nat),
        "TOTAL COSTS OF SALES (International)": _to_money(total_cogs_intl),
        "GROSS PROFIT": gross_profit,
        "TOTAL OPERATING EXPENSES": _to_money(total_oper_exp),
        "OPERATING PROFIT": operating_profit,
        "TOTAL NON-OPERATING REVENUES": _to_money(total_nonop_rev),
        "TOTAL NON-OPERATING EXPENSES": _to_money(total_nonop_exp),
        "Financial Expenses": _to_money(total_fin_exp),
        "Income Before Taxes": income_before_taxes,
        "Taxable Income": taxable_income,   # may be negative
        "Tax Expense": tax_expense,         # may be negative
        "Net Profit": net_profit,
    }




# =========================== Strategy: ALTERNATIVE ============================

def _generate_alternative(agent: Any) -> Dict[str, Decimal]:
    """
    Compact scheme:

    - Total Revenues
    - Total Outflows (Costs + Expenses)
    - Income Before Taxes
    - Tax Expense
    - Net Profit

    Tax neutrality
    --------------
    This function also delegates tax calculation to the policy layer.
    Unlike the standard strategy, it does not compute an operating revenue
    or COGS split; policies that require source decomposition will fall back
    to revenue-based prorations and residual handling.
    """
    accounts = _get_accounts_dict(agent)

    total_revenues = Decimal("0.00")
    total_outflows = Decimal("0.00")  # costs + expenses

    for _, account in accounts.items():
        t = _safe_lower(getattr(account, "type", None))
        bal = _to_money(getattr(account, "balance", Decimal("0.00")))
        if t == "revenue":
            total_revenues += bal
        elif t in ("cost", "expense"):
            total_outflows += bal

    income_before_taxes = _to_money(total_revenues - total_outflows)

    # --------------------------- TAX (policy-driven) --------------------------
    policy = _resolve_tax_policy(agent)

    # With the compact scheme there is no source split; everything is treated as 'other'.
    bd = TaxBreakdown(
        ibt=income_before_taxes,
        revenue_by_source={"other": _to_money(total_revenues)},
        # cogs_by_source left empty; policy will handle via fallback logic.
    )
    tax_result = policy.compute(bd)

    taxable_income = income_before_taxes
    tax_expense = tax_result.tax_expense
    net_profit = _to_money(income_before_taxes - tax_expense)

    return {
        "Total Revenues": _to_money(total_revenues),
        "Total Outflows (Costs+Expenses)": _to_money(total_outflows),
        "Income Before Taxes": income_before_taxes,
        "Taxable Income": taxable_income,  # may be negative
        "Tax Expense": tax_expense,        # may be negative
        "Net Profit": net_profit,          # keep same key as standard for downstream compatibility
        # "__Tax By Source": tax_result.by_source,
    }


# ================================ Public API =================================

def generate_income_statement(
    agent: Any,
    *,
    persist: bool = True,
    as_float: bool = False,
) -> Dict[str, Decimal] | Dict[str, float]:
    """
    Generate an income statement for the given agent.

    Parameters
    ----------
    agent : Any
        The agent whose accounts will be summarized.
    persist : bool
        If True, store `agent.net_profit` (Decimal) and `agent.last_income_statement`.
        This plays nicely with the Balance Sheet implementation.
    as_float : bool
        If True, convert all values to float at the end (UI/serialization convenience).
        Internally everything is still computed with Decimal to preserve accuracy.

    Returns
    -------
    Dict[str, Decimal] | Dict[str, float]
        Income statement totals. Always contains "Net Profit".

    Notes
    -----
    • This façade remains stable while tax policies evolve independently.
    • To switch ZF from flat (20%) to mixed (20% exports / 35% rest), either:
        - set `DEFAULT_ZF_POLICY_NAME = "mixed"` at module level, or
        - set `agent.tax_policy_name = "zf_mixed"` (or on its economic context).
    """
    stype = _resolve_income_statement_type(agent, default="standard")
    if stype == "alternative":
        result = _generate_alternative(agent)
    else:
        result = _generate_standard(agent)

    # Persist, if requested (ensures Balance Sheet can pick up a reliable Decimal)
    if persist:
        try:
            setattr(agent, "net_profit", result["Net Profit"])
            setattr(agent, "last_income_statement", result.copy())
        except Exception:
            # Do not hard-fail reporting if the agent is a lightweight stub
            pass

    if as_float:
        return {k: float(v) for k, v in result.items()}

    return result

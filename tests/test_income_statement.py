# -*- coding: utf-8 -*-
"""
Comprehensive tests for income_statement reporting and tax policies.

Coverage:
- Common pitfalls (mislabelled groups, wrong subtype for COGS, quantization).
- Classification sanity checks (international revenue, international COGS).
- Policies: Standard, ZF Flat, ZF Mixed (with and without COGS by source).
- Public façade behaviors: persist + as_float.

Style: pytest
"""

from __future__ import annotations
from classes.economy.agent.accountant.account import Account
from decimal import Decimal
from pathlib import Path
import sys
import pytest

# ---------------------------------------------------------------------------
# Ruta al root del repo (tests/.. -> repo)
# ---------------------------------------------------------------------------
REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# ---------------------------------------------------------------------------
# Imports del sistema real
# ---------------------------------------------------------------------------
from classes.economy.agent.accountant.chart_of_accounts import ChartOfAccounts  # noqa: E402

# Import flexible del income_statement:
# 1) Ubicación actual (agent.reporting)
# 2) Compatibilidad con la ruta antigua (accountant.reporting)
try:
    from classes.economy.agent.reporting.income_statement import (  # noqa: E402
        generate_income_statement,
        TaxBreakdown,
        StandardPolicy,
        ZFFlatPolicy,
        ZFMixedPolicy,
    )
except ModuleNotFoundError:
    from classes.economy.agent.reporting.income_statement import (  # noqa: E402
        generate_income_statement,
        TaxBreakdown,
        StandardPolicy,
        ZFFlatPolicy,
        ZFMixedPolicy,
    )

# ------------------------------ Test utilities -------------------------------

def D(x: str | float | int) -> Decimal:
    return Decimal(str(x))


def assert_money(x: Decimal, expected: str | Decimal) -> None:
    """Exact 2-decimal comparison (quantized in the module under test)."""
    assert isinstance(x, Decimal), f"Expected Decimal, got {type(x)}"
    # Ensure two-decimal quantization
    assert x.as_tuple().exponent == -2, f"Expected 2 decimals, got {x}"
    if isinstance(expected, (int, float, str)):
        expected = D(expected)
    assert x == expected, f"Expected {expected}, got {x}"


class AgentStub:
    """Minimal agent compatible with income_statement.generate_income_statement."""
    def __init__(self, chart_or_accounts, *, tax_policy_name=None, istype="standard"):
        # Si ya me pasaron un ChartOfAccounts, úsalo tal cual (no reconstruyas)
        if hasattr(chart_or_accounts, "accounts"):
            self.chart_of_accounts = chart_or_accounts
        else:
            # Si me pasaron dicts/lista, lo construyo aquí
            self.chart_of_accounts = ChartOfAccounts(chart_or_accounts)
        self.tax_policy_name = tax_policy_name
        self.income_statement_type = istype  # 'standard' o 'alternative'

    # Contexto económico opcional (para overrides de política/strategy)
    def get_economic_context(self):
        class Ctx: ...
        ctx = Ctx()
        ctx.tax_policy_name = getattr(self, "tax_policy_name", None)
        ctx.income_statement_type = getattr(self, "income_statement_type", None)
        return ctx


def credit(acc: Account, amount) -> None:
    """Increase balance for revenue/equity/liability by credit; decrease for asset/cost/expense."""
    acc.update_balance(amount, is_debit=False)


def debit(acc: Account, amount) -> None:
    """Increase balance for asset/cost/expense by debit; decrease for revenue/equity/liability."""
    acc.update_balance(amount, is_debit=True)


# ------------------------------ Fixtures/helpers -----------------------------

def make_accounts_for_international_case(
    *,
    intl_rev_group: str = "International Operating Revenue",
    intl_cogs_subtype: str = "Cost of Sales",
    intl_cogs_group: str = "Cost of Sales - International",
) -> ChartOfAccounts:
    """
    Build a minimal Chart of Accounts for international revenue & COGS scenarios.

    Parameters allow deliberate mislabelling to test pitfalls.
    """
    accs = [
        # Seeded accounts (Assets/Equity) — they don't affect P&L buckets
        {"code": "1105", "name": "Cash", "type": "Asset", "subtype": "Current Asset", "group": "Cash and Cash Equivalents"},
        {"code": "3115", "name": "Owner's Equity", "type": "Equity", "subtype": "Equity", "group": "Equity"},

        # International revenue (group may be mislabelled via parameter)
        {"code": "4180", "name": "Sales Revenue (International)", "type": "Revenue", "subtype": "Operating Revenue", "group": intl_rev_group},

        # International COGS (subtype/group may be adjusted via parameters)
        {"code": "6136", "name": "COGS (International)", "type": "Cost", "subtype": intl_cogs_subtype, "group": intl_cogs_group},
    ]
    return ChartOfAccounts(accs)


def post_balances_for_basic_ibf_scenario(coa: ChartOfAccounts, rev_amt, cogs_amt) -> None:
    """
    Post a simple international sale: increase revenue (credit), increase COGS (debit).
    """
    rev = coa.get_account("4180")
    cogs = coa.get_account("6136")
    assert rev and cogs
    credit(rev, rev_amt)
    debit(cogs, cogs_amt)


# --------------------------------- Pitfalls ----------------------------------

@pytest.mark.parametrize("attr", ["type", "agent_type", "name"])
def test_heuristic_detects_free_trade_zone_variants(attr):
    coa = make_accounts_for_international_case()
    post_balances_for_basic_ibf_scenario(coa, "100.00", "40.00")
    agent = AgentStub(coa, tax_policy_name=None, istype="standard")
    # Reset attributes that the heuristic inspects
    agent.type = ""
    agent.agent_type = ""
    agent.name = ""
    hints = {"type": "ftz manufacturer", "agent_type": "Free Trade Zone Operator", "name": "Acme ZF Holdings"}
    setattr(agent, attr, hints[attr])

    stmt = generate_income_statement(agent, persist=False, as_float=False)
    # Heuristic should pick the 20% policy (default for ZF)
    assert_money(stmt["Tax Expense"], "12.00")


def test_pitfall_mislabelled_international_revenue_goes_to_nonoperating_and_changes_tax():
    """
    If group is mistakenly 'International Op Revenue' (missing 'Operating'),
    it should NOT be classified as international operating revenue. It should
    fall into non-operating revenue, and ZF Mixed tax will change accordingly.
    """
    # Correct labelling case first (control)
    coa_ok = make_accounts_for_international_case(
        intl_rev_group="International Operating Revenue"
    )
    post_balances_for_basic_ibf_scenario(coa_ok, rev_amt="100.00", cogs_amt="40.00")
    agent_ok = AgentStub(coa_ok, tax_policy_name="zf_mixed", istype="standard")
    stmt_ok = generate_income_statement(agent_ok, persist=False, as_float=False)

    # Expect tax under ZF Mixed with proper labelling: (100 - 40)*20% = 12.00
    assert_money(stmt_ok["Tax Expense"], "12.00")
    # Non-operating revenue should be zero in the correct case
    assert_money(stmt_ok["TOTAL NON-OPERATING REVENUES"], "0.00")

    # Mislabelled case: falls to non-operating revenue; tax basis split changes
    coa_bad = make_accounts_for_international_case(
        intl_rev_group="International Op Revenue"  # <- missing 'Operating'
    )
    post_balances_for_basic_ibf_scenario(coa_bad, rev_amt="100.00", cogs_amt="40.00")
    agent_bad = AgentStub(coa_bad, tax_policy_name="zf_mixed", istype="standard")
    stmt_bad = generate_income_statement(agent_bad, persist=False, as_float=False)

    # Even though IBT should remain 60, the mixed policy split changes:
    # intl component becomes negative (0 - 40), residual absorbs +100.
    # Tax = (-40)*20% + (100)*35% = -8 + 35 = 27.00
    assert_money(stmt_bad["Tax Expense"], "27.00")
    # Now non-operating revenue should reflect the mislabelled 100
    assert_money(stmt_bad["TOTAL NON-OPERATING REVENUES"], "100.00")
    # COGS (International) should still reflect 40.00
    assert_money(stmt_bad["TOTAL COSTS OF SALES (International)"], "40.00")


def test_pitfall_cogs_wrong_subtype_not_counted_as_market_cogs_and_affects_tax():
    """
    If an international COGS line is mistakenly defined with subtype other than
    'Cost of Sales' (e.g., 'Cost of Production'), it won't be counted under
    'COGS (International)' bucket; ZF Mixed will treat it as residual.
    """
    # Correct subtype (Cost of Sales) — control
    coa_ok = make_accounts_for_international_case(
        intl_cogs_subtype="Cost of Sales",
        intl_cogs_group="Cost of Sales - International",
    )
    post_balances_for_basic_ibf_scenario(coa_ok, "100.00", "40.00")
    agent_ok = AgentStub(coa_ok, tax_policy_name="zf_mixed", istype="standard")
    stmt_ok = generate_income_statement(agent_ok, persist=False, as_float=False)
    # With proper subtype, intl COGS bucket captures 40.00, tax = 12.00
    assert_money(stmt_ok["TOTAL COSTS OF SALES (International)"], "40.00")
    assert_money(stmt_ok["Tax Expense"], "12.00")

    # Wrong subtype (Cost of Production) — the 40 goes to production, not intl COGS
    coa_bad = make_accounts_for_international_case(
        intl_cogs_subtype="Cost of Production",   # <- mistake
        intl_cogs_group="Cost of Sales - International",
    )
    post_balances_for_basic_ibf_scenario(coa_bad, "100.00", "40.00")
    agent_bad = AgentStub(coa_bad, tax_policy_name="zf_mixed", istype="standard")
    stmt_bad = generate_income_statement(agent_bad, persist=False, as_float=False)

    # The per-market COGS line should show 0.00 for International
    assert_money(stmt_bad["TOTAL COSTS OF SALES (International)"], "0.00")
    # IBT is still 60, but mixed policy taxes 100 at 20% and residual -40 at 35%:
    # Tax = 20.00 - 14.00 = 6.00
    assert_money(stmt_bad["Tax Expense"], "6.00")


def test_pitfall_quantization_of_programmatic_inputs_round_half_up():
    """
    The module quantizes balances to two decimals using ROUND_HALF_UP.
    Using amounts with >2 decimal places should yield cent-accurate results.
    """
    coa = make_accounts_for_international_case()
    # Post amounts with >2 decimals: 100.005 rounds to 100.01
    post_balances_for_basic_ibf_scenario(coa, rev_amt="100.005", cogs_amt="0.00")
    agent = AgentStub(coa, tax_policy_name="zf_flat", istype="standard")
    stmt = generate_income_statement(agent, persist=False, as_float=False)

    # IBT should be 100.01 (rounded HALF_UP), tax (20%) = 20.002 -> 20.00
    assert_money(stmt["Income Before Taxes"], "100.01")
    assert_money(stmt["Tax Expense"], "20.00")
    assert_money(stmt["Net Profit"], "80.01")


# ------------------------------ Classification -------------------------------

def test_classification_international_revenue_affects_mixed_policy_as_international():
    """
    A revenue with group="International Operating Revenue" must behave as
    international in the mixed policy (20% branch).
    """
    coa = make_accounts_for_international_case(
        intl_rev_group="International Operating Revenue",
        intl_cogs_subtype="Cost of Sales",
        intl_cogs_group="Cost of Sales - International",
    )
    post_balances_for_basic_ibf_scenario(coa, "100.00", "40.00")
    agent = AgentStub(coa, tax_policy_name="zf_mixed", istype="standard")
    stmt = generate_income_statement(agent, persist=False, as_float=False)

    # International COGS captured, non-operating revenues zero, tax at 20% on (100-40)
    assert_money(stmt["TOTAL COSTS OF SALES (International)"], "40.00")
    assert_money(stmt["TOTAL NON-OPERATING REVENUES"], "0.00")
    assert_money(stmt["Tax Expense"], "12.00")


def test_classification_cogs_international_bucket():
    """
    A COGS with subtype="Cost of Sales" and group="Cost of Sales - International"
    must be counted in the international COGS total.
    """
    coa = make_accounts_for_international_case()
    post_balances_for_basic_ibf_scenario(coa, "10.00", "4.00")
    agent = AgentStub(coa, tax_policy_name="zf_mixed", istype="standard")
    stmt = generate_income_statement(agent, persist=False, as_float=False)
    assert_money(stmt["TOTAL COSTS OF SALES (International)"], "4.00")


# --------------------------------- Policies ----------------------------------

def test_policy_standard_positive_and_negative_ibt():
    pol = StandardPolicy()
    bd_pos = TaxBreakdown(ibt=D("100.00"), revenue_by_source={"other": D("100.00")})
    bd_neg = TaxBreakdown(ibt=D("-100.00"), revenue_by_source={"other": D("100.00")})

    tax_pos = pol.compute(bd_pos)
    tax_neg = pol.compute(bd_neg)

    assert_money(tax_pos.tax_expense, "35.00")
    assert_money(tax_neg.tax_expense, "-35.00")


def test_policy_zf_flat_positive_and_negative_ibt():
    pol = ZFFlatPolicy()
    bd_pos = TaxBreakdown(ibt=D("100.00"), revenue_by_source={"other": D("100.00")})
    bd_neg = TaxBreakdown(ibt=D("-100.00"), revenue_by_source={"other": D("100.00")})

    tax_pos = pol.compute(bd_pos)
    tax_neg = pol.compute(bd_neg)

    assert_money(tax_pos.tax_expense, "20.00")
    assert_money(tax_neg.tax_expense, "-20.00")


def test_policy_zf_mixed_with_cogs_by_source_and_residual():
    """
    Scenario from the documentation example:
    - rev intl = 1000, rev nat = 500
    - cogs intl = 600,  cogs nat = 300
    - IBT total = 450  (due to additional unassigned items: prod/indirect/selling)
    -> ibt_by_src: intl=400, nat=200; residual=-150
    -> tax: 400*20% + 200*35% + (-150)*35% = 80 + 70 - 52.5 = 97.50
    """
    pol = ZFMixedPolicy()
    bd = TaxBreakdown(
        ibt=D("450.00"),
        revenue_by_source={
            "international_operating": D("1000.00"),
            "national_operating": D("500.00"),
        },
        cogs_by_source={
            "international_operating": D("600.00"),
            "national_operating": D("300.00"),
        },
    )
    tax = pol.compute(bd)
    assert_money(tax.tax_expense, "97.50")


def test_policy_zf_mixed_without_cogs_by_source_uses_revenue_shares():
    """
    Without COGS by source, the policy prorates IBT by revenue shares.
    Example: IBT=100, rev intl=60, rev nat=40 => tax = 60*20% + 40*35% = 26.00
    """
    pol = ZFMixedPolicy()
    bd = TaxBreakdown(
        ibt=D("100.00"),
        revenue_by_source={
            "international_operating": D("60.00"),
            "national_operating": D("40.00"),
        },
    )
    tax = pol.compute(bd)
    assert_money(tax.tax_expense, "26.00")


# ---------------------------------- Façade -----------------------------------

def test_facade_persist_sets_agent_fields():
    """
    persist=True should set agent.net_profit and agent.last_income_statement.
    """
    coa = make_accounts_for_international_case()
    post_balances_for_basic_ibf_scenario(coa, "100.00", "40.00")
    agent = AgentStub(coa, tax_policy_name="zf_flat", istype="standard")

    stmt = generate_income_statement(agent, persist=True, as_float=False)

    # Net Profit should be present and persisted (IBT=60, tax=12 -> net=48)
    assert "Net Profit" in stmt
    assert hasattr(agent, "net_profit")
    assert hasattr(agent, "last_income_statement")
    assert_money(agent.net_profit, "48.00")
    assert "Net Profit" in agent.last_income_statement


def test_facade_as_float_returns_floats_and_keeps_keys():
    coa = make_accounts_for_international_case()
    post_balances_for_basic_ibf_scenario(coa, "10.00", "4.00")
    agent = AgentStub(coa, tax_policy_name="zf_flat", istype="standard")

    stmt = generate_income_statement(agent, persist=False, as_float=True)

    # All values should be float, keys should include Net Profit
    assert "Net Profit" in stmt
    assert all(isinstance(v, float) for v in stmt.values())

# tests/test_sign_probe.py
import os, sys
from decimal import Decimal

# --- bootstrap path ---
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tests.test_income_statement import (
    make_accounts_for_international_case,
    post_balances_for_basic_ibf_scenario,
    AgentStub,
)
from classes.economy.agent.reporting.income_statement import generate_income_statement

def D(x) -> Decimal:
    return Decimal(str(x)).quantize(Decimal("0.01"))

def _sl(s):  # safe lower
    return (s or "").lower()

def test_signs_and_ibt_snapshot():
    # Control: all international, revenue=100, COGS=40
    coa = make_accounts_for_international_case(
        intl_rev_group="International Operating Revenue",
        intl_cogs_subtype="Cost of Sales",
        intl_cogs_group="Cost of Goods Sold - International",
    )
    post_balances_for_basic_ibf_scenario(coa, "100.00", "40.00")
    agent = AgentStub(coa, tax_policy_name="zf_mixed", istype="standard")

    # Aggregate by labels (NOT by hard-coded codes)
    rev_intl = D("0")
    cogs_intl = D("0")

    for code, acc in coa.accounts.items():
        t = _sl(getattr(acc, "type", None))
        subtype = _sl(getattr(acc, "subtype", None))
        group = _sl(getattr(acc, "group", None))
        bal = D(getattr(acc, "balance", 0))

        # Revenues: international operating
        if t == "revenue" and "international operating revenue" in group:
            rev_intl += abs(bal)

        # COGS: type=Cost, subtype contains "sales", group mentions "international"
        if t == "cost" and "sales" in subtype and "international" in group:
            cogs_intl += abs(bal)

    # Sanity: should be exactly 100 and 40 in this scenario
    assert rev_intl == D("100.00"), f"Expected intl revenue 100.00, got {rev_intl}"
    assert cogs_intl == D("40.00"), f"Expected intl COGS 40.00, got {cogs_intl}"

    # Magnitude-based IBT expected: 100 - 40 = 60
    ibt_norm = D(rev_intl - cogs_intl)

    # What the statement produces
    stmt = generate_income_statement(agent, persist=False, as_float=False)
    ibt_stmt = stmt["Income Before Taxes"]

    assert ibt_stmt == ibt_norm, f"IBT mismatch: normalized={ibt_norm} vs statement={ibt_stmt}"

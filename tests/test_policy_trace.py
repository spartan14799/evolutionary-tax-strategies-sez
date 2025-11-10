# tests/test_policy_trace.py
import os, sys
from decimal import Decimal

# Bootstrap del path del proyecto
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import classes.economy.agent.reporting.income_statement as mod

from tests.test_income_statement import (
    make_accounts_for_international_case,
    post_balances_for_basic_ibf_scenario,
    AgentStub,
)
from classes.economy.agent.reporting.income_statement import (
    generate_income_statement, ZFMixedPolicy, ZFFlatPolicy, StandardPolicy
)

def test_trace_policy_used_inside_generate(monkeypatch):
    # Escenario control: todo internacional, Rev=100, COGS=40
    coa = make_accounts_for_international_case(
        intl_rev_group="International Operating Revenue",
        intl_cogs_subtype="Cost of Sales",
        intl_cogs_group="Cost of Goods Sold - International",
    )
    post_balances_for_basic_ibf_scenario(coa, "100.00", "40.00")

    agent = AgentStub(coa, tax_policy_name="zf_mixed", istype="standard")

    # Guardamos el original
    original_resolver = mod._resolve_tax_policy

    # Variable para capturar la política usada dentro de generate_income_statement
    used = {"class": None}

    def wrapper_resolver(a):
        pol = original_resolver(a)  # lo que resolvería normalmente
        used["class"] = pol.__class__.__name__
        return pol

    # Monkeypatch: todo lo que llame _resolve_tax_policy dentro del módulo usará el wrapper
    monkeypatch.setattr(mod, "_resolve_tax_policy", wrapper_resolver)

    # Ejecutar el reporte
    stmt = generate_income_statement(agent, persist=False, as_float=False)

    # Afirmaciones mínimas
    assert stmt["Income Before Taxes"] == Decimal("60.00"), f"IBT debería ser 60.00 y es {stmt['Income Before Taxes']}"

    # Ver qué política usó realmente el reporte
    policy_used = used["class"]  # e.g., 'ZFMixedPolicy', 'ZFFlatPolicy', 'StandardPolicy'
    assert policy_used is not None, "No se pudo capturar la política usada dentro del reporte"

    # Si es ZF mixta, el impuesto debe ser 12.00. Si es Standard, será 21.00; si es ZF flat, 12.00.
    if policy_used == "ZFMixedPolicy":
        assert stmt["Tax Expense"] == Decimal("12.00"), f"Con ZF mixta debería ser 12.00; salió {stmt['Tax Expense']}"
    elif policy_used == "ZFFlatPolicy":
        # No sería tu caso esperado, pero dejarlo explícito ayuda a entender
        assert stmt["Tax Expense"] == Decimal("12.00"), f"Con ZF flat (20%) debería ser 12.00; salió {stmt['Tax Expense']}"
    elif policy_used == "StandardPolicy":
        assert stmt["Tax Expense"] == Decimal("21.00"), f"Con Standard (35%) debería ser 21.00; salió {stmt['Tax Expense']}"
    else:
        raise AssertionError(f"Política inesperada usada: {policy_used}")

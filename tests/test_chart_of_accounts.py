# tests/test_chart_of_accounts.py
from __future__ import annotations

import os
import sys
from types import SimpleNamespace
from decimal import Decimal
import textwrap
import pytest

# ---------------------------------------------------------------------------
# Ajuste de ruta al root del proyecto (mismo patrón que usas en tus tests)
# ---------------------------------------------------------------------------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.simulation.economy.agent.accountant.chart_of_accounts import (  # noqa: E402
    ChartOfAccounts,
    ensure_asset_accounts_for_graph,
)

# ---------------------------------------------------------------------------
# Fixtures de ejemplo
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_account_list():
    return [
        {
            "code": "1105",
            "name": "Cash",
            "type": "Asset",
            "subtype": "Current Asset",
            "group": "Cash and Cash Equivalents",
        },
        {
            "code": "4120",
            "name": "Sales Revenue",
            "type": "Revenue",
            "subtype": "Operating Revenue",
            "group": "Operating Revenues",
        },
    ]


@pytest.fixture
def sample_yaml_accounts_full():
    """
    Returns a comprehensive list of accounts reflecting the current YAML file.

    Includes:
    - National operating revenues: 4120, 4135
    - International operating revenues: 4121, 4136
    - COGS National: 6120, 6135
    - COGS International: 6121, 6136
    - Production costs: 71, 711
    - Indirect/overheads: 73 (Expense)
    - Non-operating: 4140 (Rev), 6140 (Cost)
    - Equity: 3115
    - Assets baseline: 1105
    """
    return [
        # -------------------------- Assets --------------------------
        {"code": "1105", "name": "Cash", "type": "Asset",
         "subtype": "Current Asset", "group": "Cash and Cash Equivalents"},

        # ================= 1. OPERATING REVENUES ====================
        # National
        {"code": "4135", "name": "Wholesale and Retail Trade Revenue (National)",
         "type": "Revenue", "subtype": "Operating Revenue",
         "group": "National Operating Revenue"},
        {"code": "4120", "name": "Manufacturing Revenue (National)",
         "type": "Revenue", "subtype": "Operating Revenue",
         "group": "National Operating Revenue"},

        # International (exports)
        {"code": "4136", "name": "Wholesale and Retail Trade Revenue (International)",
         "type": "Revenue", "subtype": "Operating Revenue",
         "group": "International Operating Revenue"},
        {"code": "4121", "name": "Manufacturing Revenue (International)",
         "type": "Revenue", "subtype": "Operating Revenue",
         "group": "International Operating Revenue"},

        # =================== 2. COSTS / COGS ========================
        # Cost of Sales - National
        {"code": "6135", "name": "Cost of Wholesale and Retail Trade (National)",
         "type": "Cost", "subtype": "Cost of Sales",
         "group": "Cost of Sales - National"},
        {"code": "6120", "name": "Cost of Manufacturing (National)",
         "type": "Cost", "subtype": "Cost of Sales",
         "group": "Cost of Sales - National"},

        # Cost of Sales - International
        {"code": "6136", "name": "Cost of Wholesale and Retail Trade (International)",
         "type": "Cost", "subtype": "Cost of Sales",
         "group": "Cost of Sales - International"},
        {"code": "6121", "name": "Cost of Manufacturing (International)",
         "type": "Cost", "subtype": "Cost of Sales",
         "group": "Cost of Sales - International"},

        # Cost of Production
        {"code": "71", "name": "Direct Materials Cost",
         "type": "Cost", "subtype": "Cost of Production",
         "group": "Cost of Production"},
        {"code": "711", "name": "Imported Direct Materials Cost",
         "type": "Cost", "subtype": "Cost of Production",
         "group": "Cost of Production"},

        # Indirect / Overheads
        {"code": "73", "name": "Indirect Cost",
         "type": "Expense", "subtype": "Indirect",
         "group": "Production Overheads"},

        # ================= Non-operating ============================
        {"code": "4140", "name": "Sale of Non-Inventory Assets",
         "type": "Revenue", "subtype": "Non-Operating Revenues",
         "group": "Non-Operating Revenues"},
        {"code": "6140", "name": "Cost of Goods Sold (Non-Core)",
         "type": "Cost", "subtype": "Non-Operating Expenses",
         "group": "Non-Operating Expenses"},

        # -------------------------- Equity --------------------------
        {"code": "3115", "name": "Owner's Equity",
         "type": "Equity", "subtype": "Equity", "group": "Equity"},
    ]


# ---------------------------------------------------------------------------
# Tests básicos de inicialización y acceso
# ---------------------------------------------------------------------------

def test_chart_initialization(sample_account_list):
    chart = ChartOfAccounts(sample_account_list)
    assert len(chart.accounts) == 2
    assert "1105" in chart.accounts
    assert chart.accounts["1105"].name == "Cash"


def test_get_account_and_by_name(sample_account_list):
    chart = ChartOfAccounts(sample_account_list)
    acc = chart.get_account("4120")
    assert acc is not None
    assert acc.name == "Sales Revenue"
    assert acc.type == "Revenue"

    acc_by_name = chart.get_account_by_name("Cash")
    assert acc_by_name.code == "1105"


def test_get_account_not_found(sample_account_list):
    chart = ChartOfAccounts(sample_account_list)
    assert chart.get_account("9999") is None
    assert chart.get_account_by_name("No existe") is None


def test_accounts_view_is_read_only(sample_account_list):
    chart = ChartOfAccounts(sample_account_list)
    # Mapping de solo-lectura: asignar debe fallar
    with pytest.raises(TypeError):
        chart.accounts["1105"] = "X"  # type: ignore


# ---------------------------------------------------------------------------
# Dinámica de cuentas de inventario para bienes
# ---------------------------------------------------------------------------

def test_get_asset_account_for_good_creates_and_is_idempotent():
    chart = ChartOfAccounts([], dynamic_start_code=1400)
    acc1 = chart.get_asset_account_for_good("Steel")
    acc2 = chart.get_asset_account_for_good("Steel")
    assert acc1 is acc2
    assert acc1.name == "Steel Asset"
    assert acc1.type.lower() == "asset"
    assert acc1.code == "1400"  # primer código dinámico al partir de 1400


def test_dynamic_code_bumps_above_existing_numeric_codes():
    # Si ya hay códigos numéricos >= start, el cursor se mueve a max+1
    existing = [
        {"code": "1399", "name": "Legacy A", "type": "Asset", "subtype": "x", "group": "y"},
        {"code": "1500", "name": "Legacy B", "type": "Asset", "subtype": "x", "group": "y"},
    ]
    chart = ChartOfAccounts(existing, dynamic_start_code=1400)
    new_acc = chart.get_asset_account_for_good("Copper")
    assert int(new_acc.code) >= 1501  # debe ser 1501 exacto si no hay huecos
    assert new_acc.name == "Copper Asset"


def test_casefold_normalization_on_and_off():
    chart_on = ChartOfAccounts([], dynamic_start_code=1400, normalize_goods_casefold=True)
    a1 = chart_on.get_asset_account_for_good("Steel")
    a2 = chart_on.get_asset_account_for_good("steel")
    assert a1 is a2  # misma cuenta gracias a casefold

    chart_off = ChartOfAccounts([], dynamic_start_code=2000, normalize_goods_casefold=False)
    b1 = chart_off.get_asset_account_for_good("Steel")
    b2 = chart_off.get_asset_account_for_good("steel")
    assert b1 is not b2
    assert b1.name == "Steel Asset"
    assert b2.name == "steel Asset"


def test_good_resolution_from_object_variants():
    chart = ChartOfAccounts([], dynamic_start_code=3000)

    obj_with_type = SimpleNamespace(type="Gold")
    obj_with_name = SimpleNamespace(name="Silver")

    acc_gold = chart.get_asset_account_for_good(obj_with_type)
    acc_silver = chart.get_asset_account_for_good(obj_with_name)
    assert acc_gold.name == "Gold Asset"
    assert acc_silver.name == "Silver Asset"

    class Weird:
        def __str__(self) -> str:
            return "Wood"

    acc_wood = chart.get_asset_account_for_good(Weird())
    assert acc_wood.name == "Wood Asset"


def test_preexisting_asset_name_is_reused():
    # Si ya existe "{good} Asset" en el YAML, se debe reutilizar (sin crear nueva)
    existing = [
        {"code": "1450", "name": "Steel Asset", "type": "Asset",
         "subtype": "Current Asset", "group": "Inventory"},
    ]
    chart = ChartOfAccounts(existing, dynamic_start_code=1400)
    acc = chart.get_asset_account_for_good("steel")  # casefold debe mapear al existente
    assert acc.code == "1450"
    assert len(chart.accounts) == 1  # no se creó otra


def test_get_asset_account_for_good_create_if_missing_false_returns_none():
    chart = ChartOfAccounts([], dynamic_start_code=4000)
    acc = chart.get_asset_account_for_good("Unobtainium", create_if_missing=False)
    assert acc is None


# ---------------------------------------------------------------------------
# ensure_asset_accounts_for_graph y YAML loader
# ---------------------------------------------------------------------------

def test_ensure_asset_accounts_for_graph_populates_accounts():
    chart = ChartOfAccounts([], dynamic_start_code=1500)
    mock_graph = SimpleNamespace(get_nodes=lambda: ["Gold", "Copper"])
    ensure_asset_accounts_for_graph(mock_graph, chart)
    ag = chart.get_asset_account_for_good("Gold")
    ac = chart.get_asset_account_for_good("Copper")
    assert ag is not None and ac is not None
    assert ag.code == "1500"
    assert ac.code == "1501"


def test_load_accounts_from_yaml(tmp_path):
    yaml_text = textwrap.dedent(
        """
        accounts:
          - code: "1105"
            name: "Cash"
            type: "Asset"
            subtype: "Current Asset"
            group: "Cash and Cash Equivalents"
          - code: "4120"
            name: "Sales Revenue"
            type: "Revenue"
            subtype: "Operating Revenue"
            group: "Operating Revenues"
        """
    ).strip()
    p = tmp_path / "coa.yml"
    p.write_text(yaml_text, encoding="utf-8")

    # Importamos aquí para no forzar dependencia de yaml si no se usa en otros tests
    from src.simulation.economy.agent.accountant.chart_of_accounts import (  # noqa: E402
        ChartOfAccounts as _COA,
    )
    data = _COA.load_accounts_from_yaml(str(p))
    chart = _COA(data)
    assert chart.get_account("1105").name == "Cash"
    assert chart.get_account("4120").type == "Revenue"


# ---------------------------------------------------------------------------
# Guardrails del sistema (validate/assert)
# ---------------------------------------------------------------------------

def test_validate_and_assert_system_accounts_success(sample_yaml_accounts_full):
    chart = ChartOfAccounts(sample_yaml_accounts_full)
    assert chart.validate_system_accounts() == []
    # No debe lanzar
    chart.assert_system_accounts()


def test_assert_system_accounts_raises_on_missing(sample_yaml_accounts_full):
    broken = [acc for acc in sample_yaml_accounts_full if acc["code"] != "4120"]
    chart = ChartOfAccounts(broken)
    issues = chart.validate_system_accounts()
    assert any("Missing required account 4120" in x for x in issues)
    with pytest.raises(ValueError) as e:
        chart.assert_system_accounts()
    assert "Missing required account 4120" in str(e.value)

# ---------------------------------------------------------------------------
# Verificación de que los Assets dinámicos arrancan en 14xx
# incluso si hay códigos 61xx (Cost) en el YAML.
# ---------------------------------------------------------------------------

def test_assets_start_at_1400_even_with_61xx_in_yaml(sample_yaml_accounts_full):
    """
    Dado un YAML con cuentas Cost 61xx y un único Asset (1105),
    el primer Asset dinámico debe ser 1400 (no 61xx).
    """
    chart = ChartOfAccounts(sample_yaml_accounts_full, dynamic_start_code=1400)
    acc = chart.get_asset_account_for_good("Steel")
    assert acc is not None
    assert acc.type.lower() == "asset"
    assert acc.name == "Steel Asset"
    # Debe iniciar en el rango 1400–1499, y en este caso exacto 1400
    assert 1400 <= int(acc.code) <= 1499
    assert acc.code == "1400"


def test_all_dynamic_asset_codes_are_in_14xx_range():
    """
    Al crear varios bienes seguidos, todos los Assets dinámicos deben
    quedar dentro del rango 1400–1499 y ser consecutivos desde 1400.
    """
    chart = ChartOfAccounts([], dynamic_start_code=1400)
    goods = ["Gold", "Copper", "Aluminum", "Silver", "Nickel"]
    created = [chart.get_asset_account_for_good(g) for g in goods]

    codes = [int(a.code) for a in created]
    # Todos en 14xx
    assert all(1400 <= c <= 1499 for c in codes)
    # Consecutivos empezando en 1400
    assert codes == list(range(1400, 1400 + len(goods)))


def test_non_asset_codes_do_not_push_asset_cursor(sample_yaml_accounts_full):
    """
    Aun si hay códigos altos no-Asset (p. ej., 61xx Cost) en el YAML,
    el cursor de Assets solo debe considerar Assets y respetar el start.
    """
    chart = ChartOfAccounts(sample_yaml_accounts_full, dynamic_start_code=1400)
    a1 = chart.get_asset_account_for_good("Titanium")
    a2 = chart.get_asset_account_for_good("Cobalt")
    c1, c2 = int(a1.code), int(a2.code)

    assert 1400 <= c1 <= 1499
    assert 1400 <= c2 <= 1499
    assert c2 == c1 + 1  # consecutivos

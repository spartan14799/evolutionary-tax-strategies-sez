import pytest
from decimal import Decimal

# importa Accountant desde tu ruta real
from classes.economy.invisible_hand.global_accountant import Accountant


class U:
    """Stub de GoodUnit con atributos opcionales."""
    def __init__(self, cost=None, last_cost=None, price=None):
        if cost is not ...:
            self.cost = cost
        if last_cost is not ...:
            self.last_cost = last_cost
        if price is not ...:
            self.price = price


def D(x) -> Decimal:
    return Decimal(str(x)).quantize(Decimal("0.01"))


# ----------------------------------------------------------------------
# Casos principales
# ----------------------------------------------------------------------

def test_priority_uses_cost_when_positive():
    u = U(cost=5, last_cost=4, price=3)
    assert Accountant._get_cost_strict(u) == D("5")


def test_fallback_to_last_cost_when_cost_is_none():
    u = U(cost=None, last_cost=4, price=3)
    assert Accountant._get_cost_strict(u) == D("4")


def test_fallback_to_price_when_cost_and_last_cost_none():
    u = U(cost=None, last_cost=None, price=2)
    assert Accountant._get_cost_strict(u) == D("2")


def test_zero_at_cost_prefers_positive_later_last_cost():
    # Cost=0 → no se acepta, se busca positivo en last_cost
    u = U(cost=0, last_cost=4, price=1)
    assert Accountant._get_cost_strict(u) == D("4")


def test_zero_at_cost_and_last_cost_prefers_positive_price():
    # Cost=0, last_cost=0 → se busca en price
    u = U(cost=0, last_cost=0, price=2)
    assert Accountant._get_cost_strict(u) == D("2")


def test_all_zero_returns_zero():
    u = U(cost=0, last_cost=0, price=0)
    assert Accountant._get_cost_strict(u) == D("0.00")


def test_only_zero_cost_and_others_missing_returns_zero():
    v = U(cost=0)
    # eliminar explícitamente los otros atributos
    delattr(v, "last_cost")
    delattr(v, "price")
    assert Accountant._get_cost_strict(v) == D("0.00")


def test_missing_all_attributes_raises():
    v = U()
    for a in ("cost", "last_cost", "price"):
        if hasattr(v, a):
            delattr(v, a)
    with pytest.raises(ValueError):
        Accountant._get_cost_strict(v)


def test_negative_at_any_considered_attribute_raises_even_if_later_positive():
    u = U(cost=-1, last_cost=2, price=3)
    with pytest.raises(ValueError):
        Accountant._get_cost_strict(u)


@pytest.mark.parametrize("val", [1, 1.2, "1.234", Decimal("1.235")])
def test_accepts_multiple_numeric_types_and_rounds_half_up(val):
    u = U(cost=None, last_cost=None, price=val)
    got = Accountant._get_cost_strict(u)
    if str(val) == "1.234":
        assert got == D("1.23")
    elif str(val) == "1.235":
        assert got == D("1.24")
    else:
        assert got == D(val)

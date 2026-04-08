
##### Está roto, por lo que se cambió la forma en la que economy trabaja, no con ordenes sino con tuplas, hay que revisar ######
##### Igual no importa, global_accountant tiene su propia lógica de clasificación. 



# tests/test_order_classifier.py
from __future__ import annotations

import os
import sys
from types import SimpleNamespace
import pytest

# -------------------------------------------------------------
# Ubicar raíz del proyecto para importar los módulos del paquete
# -------------------------------------------------------------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from simulation.economy.order_book.utils.order_classifier import (  # noqa: E402
    OrderClassifier,
)

# Ruta del símbolo importado dentro del módulo a monkeypatchear
ORDER_CLASSIFIER_MODULE = "classes.economy.order_book.utils.order_classifier"


# ------------------------------ Helpers --------------------------------------

def make_agent(location: str, goods_map: dict[str, str] | None = None):
    """
    Crea un agente mínimo con `location` y un façade `local_production_graph`
    que expone `classify_goods() -> dict`.
    """
    if goods_map is None:
        goods_map = {}
    lpg = SimpleNamespace(classify_goods=lambda: dict(goods_map))
    return SimpleNamespace(location=location, local_production_graph=lpg)


# --------------------------- classify_agents() --------------------------------

def test_classify_agents_with_legacy_order_nct_ftz():
    buyer = make_agent("NCT")
    seller = make_agent("FTZ")
    order = SimpleNamespace(involved_agents=[buyer, seller], involved_goods=["Steel"])
    assert OrderClassifier.classify_agents(order) == "NCT-FTZ"


def test_classify_agents_with_tuple_ftz_ftz():
    buyer = make_agent("FTZ")
    seller = make_agent("FTZ")

    class Goodish:
        def __init__(self, t): self.type = t

    tup = (buyer, seller, Goodish("Copper"), 123.45)
    assert OrderClassifier.classify_agents(tup) == "FTZ-FTZ"


def test_classify_agents_unknown_when_missing_location():
    buyer = SimpleNamespace()  # sin location
    seller = make_agent("NCT")
    order = SimpleNamespace(involved_agents=[buyer, seller], involved_goods=["X"])
    assert OrderClassifier.classify_agents(order) == "UNKNOWN"


def test_classify_agents_unknown_when_bad_shape():
    bad = object()
    assert OrderClassifier.classify_agents(bad) == "UNKNOWN"


# --------------------------- classify_assets() --------------------------------

def test_classify_assets_with_multiple_goods_and_legacy_order(monkeypatch):
    # Monkeypatch del clasificador para no depender de implementación real
    def fake_classify(cat_initial: str):
        return f"CAT[{cat_initial}]"

    monkeypatch.setattr(f"{ORDER_CLASSIFIER_MODULE}.classify_asset_category",
                        fake_classify, raising=True)

    buyer = make_agent("NCT", {"Steel": "raw", "Plastic": "final"})
    seller = make_agent("FTZ", {"Steel": "final", "Plastic": "final"})
    order = SimpleNamespace(involved_agents=[buyer, seller],
                            involved_goods=["Steel", "Plastic"])

    out = OrderClassifier.classify_assets(order)
    assert out == {
        "Steel": {"buyer": "CAT[raw]", "seller": "CAT[final]"},
        "Plastic": {"buyer": "CAT[final]", "seller": "CAT[final]"},
    }


def test_classify_assets_with_tuple_buy_like(monkeypatch):
    def fake_classify(cat_initial: str):
        return f"CAT[{cat_initial}]"

    monkeypatch.setattr(f"{ORDER_CLASSIFIER_MODULE}.classify_asset_category",
                        fake_classify, raising=True)

    # Buyer clasifica "Gold" como raw; seller como final
    buyer = make_agent("NCT", {"Gold": "raw"})
    seller = make_agent("FTZ", {"Gold": "final"})

    class Goodish:
        def __init__(self, t): self.type = t

    tup = (buyer, seller, Goodish("Gold"), 999)

    out = OrderClassifier.classify_assets(tup)
    assert out == {"Gold": {"buyer": "CAT[raw]", "seller": "CAT[final]"}}


def test_classify_assets_raises_if_agent_lacks_local_production_graph(monkeypatch):
    def fake_classify(cat_initial: str):
        return cat_initial

    monkeypatch.setattr(f"{ORDER_CLASSIFIER_MODULE}.classify_asset_category",
                        fake_classify, raising=True)

    buyer = SimpleNamespace(location="NCT")   # sin local_production_graph ni contexto
    seller = make_agent("FTZ", {"Steel": "final"})
    order = SimpleNamespace(involved_agents=[buyer, seller], involved_goods=["Steel"])

    with pytest.raises(AttributeError):
        OrderClassifier.classify_assets(order)


def test_classify_assets_with_fallback_get_economic_context(monkeypatch):
    def fake_classify(cat_initial: str):
        return f"CAT[{cat_initial}]"

    monkeypatch.setattr(f"{ORDER_CLASSIFIER_MODULE}.classify_asset_category",
                        fake_classify, raising=True)

    # Buyer sin façade directo, pero con contexto económico que sí lo tiene
    ctx = SimpleNamespace(local_production_graph=SimpleNamespace(
        classify_goods=lambda: {"Steel": "raw"}))
    buyer = SimpleNamespace(location="NCT", get_economic_context=lambda: ctx)

    seller = make_agent("FTZ", {"Steel": "final"})
    order = SimpleNamespace(involved_agents=[buyer, seller], involved_goods=["Steel"])

    out = OrderClassifier.classify_assets(order)
    assert out == {"Steel": {"buyer": "CAT[raw]", "seller": "CAT[final]"}}


def test_classify_assets_raises_when_less_than_two_agents():
    buyer = make_agent("NCT", {"X": "raw"})
    order = SimpleNamespace(involved_agents=[buyer], involved_goods=["X"])
    with pytest.raises(ValueError):
        OrderClassifier.classify_assets(order)


def test_classify_assets_accepts_goods_as_strings_in_order(monkeypatch):
    def fake_classify(cat_initial: str):
        return f"CAT[{cat_initial}]"

    monkeypatch.setattr(f"{ORDER_CLASSIFIER_MODULE}.classify_asset_category",
                        fake_classify, raising=True)

    buyer = make_agent("NCT", {"Steel": "raw"})
    seller = make_agent("FTZ", {"Steel": "final"})
    # involved_goods como strings simples (no objetos Good)
    order = SimpleNamespace(involved_agents=[buyer, seller], involved_goods=["Steel"])

    out = OrderClassifier.classify_assets(order)
    assert out == {"Steel": {"buyer": "CAT[raw]", "seller": "CAT[final]"}}

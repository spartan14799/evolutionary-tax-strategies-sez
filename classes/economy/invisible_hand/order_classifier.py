"""
Module: order_classifier.py
Description: Provides classification utilities for orders, separating concerns
             from accounting logic.

Compatibility goals:
  - Works with the legacy "Order" objects (having `involved_agents` and `involved_goods`).
  - Can also accept a BUY-like tuple: (buyer_agent, seller_agent, good_instance, price),
    extracting agents and good type safely.
  - Does NOT depend on OrderType; BUY vs PRODUCE should be handled by the caller.
"""

from typing import Any, Dict, List, Tuple

from classes.economy.invisible_hand.transaction_classification import (
    classify_asset_category,
    AssetCategory,
)


class OrderClassifier:
    """
    Classifies:
      - Agents' locations pair (e.g., 'NCT-FTZ') for reporting/routing.
      - Asset categories for buyer and seller given a set of goods.
    """

    # --------------------------- Public API ---------------------------------

    @staticmethod
    def classify_agents(order_or_tuple: Any) -> str:
        """
        Determine a location pair for the two agents involved in a BUY-like object.

        Returns:
            'NCT-NCT', 'NCT-FTZ', 'FTZ-FTZ', or 'UNKNOWN' if locations/agents are missing.

        This method tolerates:
          - Legacy order objects with `involved_agents`
          - Tuples shaped like (buyer, seller, good_instance, price)
        """
        try:
            buyer, seller, _ = OrderClassifier._extract_agents_and_goods(order_or_tuple)
        except Exception:
            return "UNKNOWN"

        loc1 = getattr(buyer, "location", None)
        loc2 = getattr(seller, "location", None)

        if loc1 == "NCT" and loc2 == "NCT":
            return "NCT-NCT"
        if loc1 == "NCT" and loc2 == "FTZ":
            return "NCT-FTZ"
        if loc1 == "FTZ" and loc2 == "FTZ":
            return "FTZ-FTZ"
        return "UNKNOWN"

    @staticmethod
    def classify_assets(order_or_tuple: Any) -> Dict[str, Dict[str, AssetCategory]]:
        """
        Classify each good involved for BUY context into refined categories
        from the perspective of buyer and seller.

        Returns:
            Dict like:
            {
              "Steel": {"buyer": AssetCategory.RAW_MATERIAL,
                        "seller": AssetCategory.FINISHED_PRODUCTS}
            }

        Notes:
          - This function expects two agents (BUY context). If only one agent is present
            (e.g., PRODUCE), the caller should not use this helper.
          - It reads each agent's LOCAL production graph classification as source of truth.
        """
        buyer, seller, goods = OrderClassifier._extract_agents_and_goods(order_or_tuple)

        buyer_class_map = OrderClassifier._classify_goods_in_agent(buyer)
        seller_class_map = OrderClassifier._classify_goods_in_agent(seller)

        out: Dict[str, Dict[str, AssetCategory]] = {}
        for good in goods:
            buyer_initial = buyer_class_map.get(good, "unknown")
            seller_initial = seller_class_map.get(good, "unknown")
            out[good] = {
                "buyer": classify_asset_category(buyer_initial),
                "seller": classify_asset_category(seller_initial),
            }
        return out

    # --------------------------- Internals ----------------------------------

    @staticmethod
    def _extract_agents_and_goods(order_or_tuple: Any) -> Tuple[Any, Any, List[str]]:
        """
        Normalize different input shapes to (buyer, seller, [good_types]).
        Supports:
          - Legacy order objects: .involved_agents (len>=2), .involved_goods
          - Tuple: (buyer, seller, good_instance, price) -> [good_instance.type]
        """
        # Tuple path
        if isinstance(order_or_tuple, tuple) and len(order_or_tuple) >= 3:
            buyer = order_or_tuple[0]
            seller = order_or_tuple[1]
            good_obj = order_or_tuple[2]
            good_type = getattr(good_obj, "type", good_obj)
            return buyer, seller, [str(good_type)]

        # Order-like object path
        if hasattr(order_or_tuple, "involved_agents") and hasattr(order_or_tuple, "involved_goods"):
            agents = list(getattr(order_or_tuple, "involved_agents"))
            if len(agents) < 2:
                raise ValueError("Expected at least two agents (buyer, seller).")
            buyer, seller = agents[0], agents[1]
            goods_raw = list(getattr(order_or_tuple, "involved_goods") or [])
            goods: List[str] = [
                getattr(g, "type", g) if g is not None else "unknown"
                for g in goods_raw
            ]
            return buyer, seller, [str(x) for x in goods]

        raise TypeError(
            "Unsupported order shape. Expected an order with involved_agents/involved_goods "
            "or a tuple (buyer, seller, good, ...)."
        )

    @staticmethod
    def _classify_goods_in_agent(agent: Any) -> Dict[str, str]:
        """
        Read the agent's local production graph classification map.
        Accepts both façade `agent.local_production_graph` and EconomicContext accessor.
        """
        # Preferred façade path
        lg = getattr(agent, "local_production_graph", None)
        if lg is None and hasattr(agent, "get_economic_context"):
            # Fallback via EconomicContext
            ctx = agent.get_economic_context()
            lg = getattr(ctx, "local_production_graph", None) or getattr(ctx, "get_local_production_graph", lambda: None)()
        if lg is None or not hasattr(lg, "classify_goods"):
            raise AttributeError("Agent does not expose a local production graph with 'classify_goods()'.")
        return dict(lg.classify_goods())

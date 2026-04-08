"""
Module: transaction_classification.py
Description: Contains the refined asset classification logic used to determine
             which accounts to move in an accounting entry.

Notes:
  - This module is intentionally tiny and stable: other modules can depend on
    its enum and pure function without dragging accounting code.
  - It accepts 'non produced' / 'non-produced' and 'non related' / 'non-related'
    to match upstream graph labels.
"""

from enum import Enum
from typing import Optional, Dict


class AssetCategory(Enum):
    """
    Enumeration of refined asset categories for accounting purposes.

    The categories help downstream accounting map COGS/Revenue accounts.
    """
    RAW_MATERIAL = "Raw Material"
    NON_PRODUCED_MERCHANDISE = "Non-Produced Merchandise"
    NON_RELATED_GOODS = "Non-Related Goods"
    FINISHED_PRODUCTS = "Finished Products"
    PRODUCTS_IN_PROCESS = "Products in Process"
    NON_RELATED_MERCHANDISE = "Non-Related Merchandise"


def classify_asset_category(initial_classification: str, context: Optional[Dict] = None) -> AssetCategory:
    """
    Map the production-graph label to a refined asset category.

    Args:
        initial_classification: One of "primary", "intermediate", "final",
                                "non related", "non produced" (variants accepted).
        context: Optional future hook (e.g., firm produces or not).

    Returns:
        AssetCategory: The refined category used by accounting mappings.

    Policy:
        - "primary"          -> RAW_MATERIAL
        - "intermediate"     -> PRODUCTS_IN_PROCESS
        - "final"            -> FINISHED_PRODUCTS
        - "non produced/*"   -> NON_PRODUCED_MERCHANDISE
        - "non related/*"    -> NON_RELATED_MERCHANDISE
        - unknown            -> NON_RELATED_GOODS
    """
    label = (initial_classification or "").strip().lower()

    if label == "primary":
        return AssetCategory.RAW_MATERIAL
    if label == "intermediate":
        return AssetCategory.PRODUCTS_IN_PROCESS
    if label == "final":
        return AssetCategory.FINISHED_PRODUCTS
    if label in ("non produced", "non-produced"):
        return AssetCategory.NON_PRODUCED_MERCHANDISE
    if label in ("non related", "non-related"):
        return AssetCategory.NON_RELATED_MERCHANDISE

    # Fallback for unknown or upstream typos.
    return AssetCategory.NON_RELATED_GOODS


__all__ = ["AssetCategory", "classify_asset_category"]

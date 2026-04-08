import pytest
from simulation.economy.agent.accountant.transaction_classification import (
    AssetCategory,
    classify_asset_category
)


@pytest.mark.parametrize("input_str,expected_category", [
    ("primary", AssetCategory.RAW_MATERIAL),
    ("intermediate", AssetCategory.PRODUCTS_IN_PROCESS),
    ("final", AssetCategory.FINISHED_PRODUCTS),
    ("non related", AssetCategory.NON_RELATED_MERCHANDISE),
])
def test_valid_classifications(input_str, expected_category):
    result = classify_asset_category(input_str)
    assert result == expected_category



@pytest.mark.parametrize("input_str", ["Primary", "PRIMARY", "PrImArY"])
def test_case_insensitivity_primary(input_str):
    result = classify_asset_category(input_str)
    assert result == AssetCategory.RAW_MATERIAL



@pytest.mark.parametrize("input_str", ["extraterrestrial", "random", "classified", "not listed"])
def test_unknown_classification_fallback(input_str):
    result = classify_asset_category(input_str)
    assert result == AssetCategory.NON_RELATED_GOODS



@pytest.mark.parametrize("input_str", [None, "", "   "])
def test_empty_or_none_input(input_str):
    result = classify_asset_category(input_str or "")
    assert result == AssetCategory.NON_RELATED_GOODS



def test_context_does_not_affect_yet():
    """
    Test to document that context currently has no effect on classification,
    even though it's passed as an argument.
    """
    result_with_context = classify_asset_category("primary", context={"produced_by_firm": False})
    result_without_context = classify_asset_category("primary")
    assert result_with_context == result_without_context == AssetCategory.RAW_MATERIAL

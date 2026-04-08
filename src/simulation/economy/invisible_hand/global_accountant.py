# -*- coding: utf-8 -*-
### Author: ANDRES LEGUIZAMON / CARLOS SANCHEZ / GERMAN RODRIGUEZ / SOFIA OCAMPO
### 2025
### TODO : LICENSE / COPYRIGHT

"""
global_accountant.py

Tuple-based Accountant compatible with `Economy` settlement outputs.

This module receives **already-settled** transactions from the Economy layer as tuples
and posts the corresponding journal entries to each involved agent’s ledger. It does
**not** mutate inventory (stock in/out must be handled by the business layer that
produced the tuple). Its sole responsibility is to translate a settled fact into
balanced accounting entries.

# Key design points

• Tuple detection by number of agents:
    - 2 agent-like objects → BUY
      Shape: (buyer_agent, seller_agent, good_instance, price)
    - 1 agent-like object  → PRODUCE
      Shape: (producer_agent, produced_good_instance, inputs_menu[, indirect_cost])

• Fixed inputs contract for PRODUCE:
    - inputs_menu: list of (input_type: str, qty: int) emitted by Economy.
    - Unit costs for consumed inputs come from produced_good_instance.production_report["inputs_consumed"].
      If missing or inconsistent with inputs_menu, the accountant raises a clear error.

• Read-only valuation fields:
    - This accountant never writes to Good.price, Good.cost or Good.last_cost.

• Cost policy:
    - BUY: Buyer debits inventory at transaction price; Seller’s COGS uses last_cost ⇒ cost ⇒ price.
    - PRODUCE: Capitalizes total direct cost from inputs (cost ⇒ last_cost ⇒ price) + optional indirect expense (73),
      without capitalizing indirects.

# Public contract (stable)

• `record_transaction(payload: tuple) -> None`
    - Accepts a tuple shaped as described above.
    - Automatically detects BUY vs PRODUCE.
    - Posts entries to the agent(s)’ ledgers via `agent.accountant.record_entry(entry)`.

The module assumes the surrounding system exposes these attributes (directly or via façade):
    - `agent.accountant` with `record_entry(AccountingEntry)`
    - `agent.chart_of_accounts` exposing `get_asset_account_for_good(good_type, create_if_missing=True)`
    - `agent.local_production_graph` exposing `classify_goods()` and `generate_direct_inputs()`
    - (optional) `agent.location` (used for potential policy/tax routing; not used in JEs here)

"""

from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP
from typing import Any, List, Tuple

from src.simulation.economy.agent.accountant.accounting_entry import AccountingEntry, EntryLine
from src.simulation.economy.invisible_hand.transaction_classification import classify_asset_category


# =============================================================================
# Monetary helpers
# =============================================================================

_MONEY_PLACES = Decimal("0.01")

def _to_money(value: Any) -> Decimal:
    """
    Coerce a value into Decimal (via string to avoid binary float artifacts) and
    quantize to cents using ROUND_HALF_UP, which matches typical accounting rounding.
    """
    return Decimal(str(value)).quantize(_MONEY_PLACES, rounding=ROUND_HALF_UP)


# =============================================================================
# Agent / Context resolution helpers (façade-friendly)
# =============================================================================

def _is_agent(obj: Any) -> bool:
    """
    Heuristic to detect agent-like objects.
    We do not import a concrete `Agent` type here to keep the accountant decoupled.
    """
    return any([
        hasattr(obj, "chart_of_accounts"),
        hasattr(obj, "_accounting_agent"),
        hasattr(obj, "get_economic_context"),
        hasattr(obj, "get_inventory"),
        hasattr(obj, "inventory"),
    ])

def _get_chart(agent: Any):
    """
    Resolve the Chart of Accounts whether it is exposed directly (façade) or
    nested under an internal accounting agent wrapper.
    """
    if hasattr(agent, "chart_of_accounts"):
        return agent.chart_of_accounts
    acc = getattr(agent, "_accounting_agent", None)
    if acc is not None and hasattr(acc, "chart_of_accounts"):
        return acc.chart_of_accounts
    raise AttributeError("Agent does not expose a Chart of Accounts.")

def _get_accountant(agent: Any):
    """
    Resolve the AgentAccountant (the object that accepts record_entry).
    """
    if hasattr(agent, "accountant"):
        return agent.accountant
    acc = getattr(agent, "_accounting_agent", None)
    if acc is not None and hasattr(acc, "accountant"):
        return acc.accountant
    raise AttributeError("Agent does not expose an AgentAccountant.")

def _get_inventory(agent: Any):
    """
    Resolve an inventory handle (if ever needed for cost measurement fallback).
    NOTE: This accountant never mutates inventory — reads only.
    """
    if hasattr(agent, "inventory"):
        return agent.inventory
    if hasattr(agent, "get_inventory"):
        return agent.get_inventory()
    acc = getattr(agent, "_accounting_agent", None)
    if acc is not None and hasattr(acc, "inventory"):
        return acc.inventory
    raise AttributeError("Agent does not expose an Inventory.")

def _get_local_pgraph(agent: Any):
    """
    Resolve the agent's local production graph.
    """
    if hasattr(agent, "local_production_graph"):
        return agent.local_production_graph
    if hasattr(agent, "get_economic_context"):
        ctx = agent.get_economic_context()
        if ctx is not None and hasattr(ctx, "get_local_production_graph"):
            return ctx.get_local_production_graph()
    raise AttributeError("Agent does not expose a local production graph.")


# =============================================================================
# Classification helpers
# =============================================================================

def _classify_asset_for_agent(agent: Any, good_type: str) -> str:
    """
    Reads the agent's local classification map (e.g. 'primary'/'intermediate'/'final'/
    'non related'/'non produced'), normalizes labels, and refines it using the
    `classify_asset_category` helper. Returns the final Enum.value string, such as:
        'Raw Material', 'Finished Products', 'Non-Produced Merchandise', etc.

    Normalization detail:
        Some graphs may emit 'non-related' with a hyphen; others use 'non related'.
        We normalize by replacing '-' with ' ' before refinement.
    """
    pgraph = _get_local_pgraph(agent)
    initial_map = pgraph.classify_goods()  # dict[str -> str]
    initial = (initial_map.get(good_type, "non related") or "").replace("-", " ")
    refined = classify_asset_category(initial)  # Enum instance
    return getattr(refined, "value", str(refined))

def _classify_agents_location(a1: Any, a2: Any) -> str:
    """
    Returns a simple location pair string like 'NCT-FTZ'. We keep this helper so
    the accountant can be extended later for tax/policy routing, though we do not
    alter journal lines by location at this time.
    """
    def loc(a: Any) -> str:
        if hasattr(a, "location") and getattr(a, "location") is not None:
            return str(getattr(a, "location"))
        if hasattr(a, "type") and getattr(a, "type") is not None:
            return str(getattr(a, "type"))
        return "UNKNOWN"
    return f"{loc(a1)}-{loc(a2)}"


# =============================================================================
# Accountant (public API)
# =============================================================================

class Accountant:
    """
    Tuple-based accountant.

    Public API:
        record_transaction(payload: tuple) -> None
            Detects transaction type (BUY/PRODUCE) and posts balanced entries.
    """

    # ---------------------------------------------------------------------
    # Dispatcher
    # ---------------------------------------------------------------------
    def record_transaction(self, payload: tuple) -> None:
        """
        Route a settled transaction tuple to the appropriate handler and post
        the resulting entries to the involved agents’ ledgers.

        FIXED tuple contracts (as emitted by Economy):

        BUY (2 agents):
            (buyer_agent, seller_agent, good_instance, price)
            → Emits two journal entries (buyer, seller).

        PRODUCE (1 agent):
            (producer_agent, produced_good_instance, inputs_menu[, indirect_cost])
            where:
            - inputs_menu: list of (input_type: str, qty: int) as returned by Economy
            - indirect_cost: optional numeric; if absent, try produced_good_instance.production_report['indirect_cost'] or 0

        Notes
        -----
        • Detection still uses "number of agent-like objects". This remains simple and robust.
        • Cost discovery for production uses produced_good_instance.production_report (authoritative).
        """
        if not isinstance(payload, tuple):
            raise TypeError("Expected a tuple emitted by Economy.")

        agents = [obj for obj in payload if _is_agent(obj)]
        if len(agents) == 2:
            buyer_entry, seller_entry, buyer, seller = self._handle_buy_tuple(payload)
            buyer_acc = _get_accountant(buyer)
            seller_acc = _get_accountant(seller)
            posted: List[Tuple[Any, AccountingEntry]] = []
            try:
                buyer_acc.record_entry(buyer_entry)
                posted.append((buyer_acc, buyer_entry))
                seller_acc.record_entry(seller_entry)
                posted.append((seller_acc, seller_entry))
            except Exception:
                for acc, entry in reversed(posted):
                    try:
                        acc.record_entry(self._build_reversal_entry(entry))
                    except Exception:
                        pass
                raise
            return

        if len(agents) == 1:
            entry, producer = self._handle_production_tuple(payload)
            _get_accountant(producer).record_entry(entry)
            return

        raise ValueError("Cannot determine transaction type from tuple; expected 1 or 2 agents.")


    # ---------------------------------------------------------------------
    # BUY path
    # ---------------------------------------------------------------------
    def _handle_buy_tuple(
        self,
        payload: Tuple[Any, ...]
    ) -> Tuple[AccountingEntry, AccountingEntry, Any, Any]:
        """
        Construct buyer and seller journal entries for a BUY tuple.

        Expected core shape:
            (buyer_agent, seller_agent, good_instance, price[, ...])

        Accounting semantics:
            • Buyer:  Dr Inventory(dynamic) / Cr Cash(1105) @ transaction price
                      (La activación de inventario se hace al precio de la transacción.)
            • Seller: Dr Cash(1105) / Dr COGS(...) / Cr Revenue(...) / Cr Inventory(dynamic)
                      (El COGS se reconoce al **costo histórico del vendedor** = last_cost ⇒ cost ⇒ price.)
        """
        buyer, seller, good_instance, price_raw = payload[:4]

        # --- Data extraction and normalization ---
        good_type = self._require_attr(good_instance, "type", ctx="good_instance")
        price_dec = _to_money(self._ensure_positive_number(price_raw, "price"))

        # Costo para COGS del vendedor: last_cost ⇒ cost ⇒ price (read-only; nunca escribe).
        unit_cost_dec = self._get_cogs_cost(good_instance)

        # --- Classification (seller) to route COGS/Revenue ---
        classification_seller_key = _classify_asset_for_agent(seller, good_type)
        # location_pair = _classify_agents_location(buyer, seller)  # kept for future use

        # --- Dynamic inventory accounts ---
        buyer_asset_acc  = self._get_dynamic_asset_account(buyer,  good_type, role="buyer")
        seller_asset_acc = self._get_dynamic_asset_account(seller, good_type, role="seller")

        # --- Mapping: refined classification → (COGS, Revenue) accounts ---
        # Centralized for auditability. If you need policy by location, split by
        # classification *and* location_pair.
        seller_accounts_map = {
            "Raw Material":             ("6135", "Cost of Wholesale and Retail Trade", "4135", "Wholesale and Retail Trade Revenue"),
            "Non-Produced Merchandise": ("6135", "Cost of Wholesale and Retail Trade", "4135", "Wholesale and Retail Trade Revenue"),
            "Products in Process":      ("6135", "Cost of Wholesale and Retail Trade", "4135", "Wholesale and Retail Trade Revenue"),
            "Finished Products":        ("6120", "Cost of Goods Sold",               "4120", "Sales Revenue"),
            "Non-Related Goods":        ("6140", "Cost of Goods Sold (Non-Core)",    "4140", "Sale of Non-Inventory Assets"),
            "Non-Related Merchandise":  ("6140", "Cost of Goods Sold (Non-Core)",    "4140", "Sale of Non-Inventory Assets"),
        }
        if classification_seller_key not in seller_accounts_map:
            raise NotImplementedError(
                f"Seller classification '{classification_seller_key}' has no configured COGS/Revenue mapping."
            )
        debit_cogs_code, debit_cogs_name, credit_rev_code, credit_rev_name = seller_accounts_map[classification_seller_key]

        # --- Buyer journal entry ---
        buyer_entry = AccountingEntry([
            EntryLine(
                account_code=buyer_asset_acc.code,
                account_name=buyer_asset_acc.name,
                amount=price_dec,
                is_debit=True,   # Inventory increases (capitalized at transaction price)
            ),
            EntryLine(
                account_code="1105",
                account_name="Cash",
                amount=price_dec,
                is_debit=False,  # Cash decreases
            ),
        ])

        # --- Seller journal entry ---
        seller_entry = AccountingEntry([
            EntryLine(
                account_code="1105",
                account_name="Cash",
                amount=price_dec,
                is_debit=True,   # Cash increases
            ),
            EntryLine(
                account_code=debit_cogs_code,
                account_name=debit_cogs_name,
                amount=unit_cost_dec,
                is_debit=True,   # COGS recognized at historical cost (read from last_cost ⇒ cost ⇒ price)
            ),
            EntryLine(
                account_code=credit_rev_code,
                account_name=credit_rev_name,
                amount=price_dec,
                is_debit=False,  # Revenue recognized at transaction price
            ),
            EntryLine(
                account_code=seller_asset_acc.code,
                account_name=seller_asset_acc.name,
                amount=unit_cost_dec,
                is_debit=False,  # Inventory decreases at historical cost
            ),
        ])

        return buyer_entry, seller_entry, buyer, seller

    # ---------------------------------------------------------------------
    # PRODUCTION path
    # ---------------------------------------------------------------------
    def _handle_production_tuple(
        self,
        payload: Tuple[Any, ...],
        *,
        default_indirect_cost: Decimal = Decimal("0.00"),
    ) -> Tuple[AccountingEntry, Any]:
        """
        Build a production journal entry from a FIXED tuple shape emitted by Economy.

        Strict contract (explicit, no fallbacks, no production_report I/O):
            (producer, produced_unit, inputs_menu[, indirect_cost])

        Where:
            • inputs_menu: list of (input_type: str, unit: Good)
            (Exactly the instances the producer consumed; one row per unit.)
            • indirect_cost (optional): numeric; if present and > 0 posts Dr73 / Cr1105.
        
        Cost policy:
            • Direct inputs are measured per-unit using unit.cost ⇒ unit.last_cost ⇒ unit.price
            (read-only; never mutates valuation fields).
            • Produced inventory is capitalized at the total direct cost only.
            • Indirect costs (if any) are expensed as incurred (not capitalized).

        Posting pattern (explicit and traceable):
            For each input unit:
                Cr {input_type} Asset                  amount=unit_cost
                Dr 71 Direct Materials Cost           amount=unit_cost
                Cr 71 Direct Materials Cost           amount=unit_cost
            Then:
                Dr {produced_type} Asset              amount=Σ unit_cost (direct total)
            Optionally (if indirect_cost > 0):
                Dr 73 Indirect Cost                   amount=indirect_cost
                Cr 1105 Cash                          amount=indirect_cost
        """
        if len(payload) < 3:
            raise ValueError("PRODUCE payload must be (producer, produced_unit, inputs_menu[, indirect_cost]).")

        producer = payload[0]
        produced_unit = payload[1]
        produced_type = self._require_attr(produced_unit, "type", ctx="produced_unit")

        # inputs_menu must be a list of (input_type: str, unit: Good)
        inputs_menu = payload[2]
        if not isinstance(inputs_menu, (list, tuple)) or len(inputs_menu) == 0:
            raise TypeError("inputs_menu must be a non-empty list of (input_type: str, unit: Good).")

        # Optional indirect cost (only if 4th element provided and numeric)
        indirect_cost_dec = default_indirect_cost
        if len(payload) >= 4:
            try:
                indirect_cost_dec = _to_money(payload[3])
            except Exception:
                indirect_cost_dec = default_indirect_cost

        # --- Normalize input rows and compute per-unit costs (strict: exactly what tuple carries) ---
        inputs_with_cost: List[Tuple[str, Decimal, int]] = []  # (input_type, unit_cost, qty=1)
        for idx, row in enumerate(inputs_menu):
            if not (isinstance(row, (list, tuple)) and len(row) == 2 and isinstance(row[0], str)):
                raise ValueError(f"inputs_menu[{idx}] must be (input_type: str, unit: Good); got {row!r}")
            input_type, unit = row
            unit_cost_dec = self._get_cost_strict(unit)  # cost ⇒ last_cost ⇒ price (read-only)
            inputs_with_cost.append((input_type, unit_cost_dec, 1))

        # --- Total direct cost (sum of unit costs) ---
        total_direct_cost = _to_money(
            sum((cost * int(qty) for _, cost, qty in inputs_with_cost), start=Decimal("0.00"))
        )

        # --- Build journal entry lines (explicit shuttle 71 for traceability) ---
        lines: List[EntryLine] = []

        # a) Consume each input unit (credit input inventory; shuttle through 71)
        for input_type, unit_cost_dec, qty in inputs_with_cost:
            amount = _to_money(unit_cost_dec * int(qty))
            input_acc = _get_chart(producer).get_asset_account_for_good(input_type, create_if_missing=True)

            # Credit input inventory (inventory decreases)
            lines.append(EntryLine(
                account_code=input_acc.code,
                account_name=input_acc.name,
                amount=amount,
                is_debit=False,
            ))
            # Debit 71 (Direct Materials Cost) — temporary shuttle account
            lines.append(EntryLine(
                account_code="71",
                account_name="Direct Materials Cost",
                amount=amount,
                is_debit=True,
            ))
            # Credit 71 (mirror/clearance)
            lines.append(EntryLine(
                account_code="71",
                account_name="Direct Materials Cost",
                amount=amount,
                is_debit=False,
            ))

        # b) Capitalize produced inventory at total direct cost
        produced_acc = _get_chart(producer).get_asset_account_for_good(produced_type, create_if_missing=True)
        lines.append(EntryLine(
            account_code=produced_acc.code,
            account_name=produced_acc.name,
            amount=total_direct_cost,
            is_debit=True,   # produced inventory increases
        ))

        # c) Optional indirect costs (explicit: only if provided in the tuple and > 0)
        if indirect_cost_dec and indirect_cost_dec != 0:
            amount = _to_money(indirect_cost_dec)
            lines.append(EntryLine(
                account_code="73",
                account_name="Indirect Cost",
                amount=amount,
                is_debit=True,    # recognize indirect production cost
            ))
            lines.append(EntryLine(
                account_code="1105",
                account_name="Cash",
                amount=amount,
                is_debit=False,   # cash outflow
            ))

        entry = AccountingEntry(lines=lines)
        return entry, producer



    # ---------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------
    @staticmethod
    def _require_attr(obj: Any, attr: str, *, ctx: str = "") -> Any:
        """
        Retrieve an attribute and ensure it is present and not None.
        Provides contextual error messages to aid diagnostics upstream.
        """
        if not hasattr(obj, attr):
            raise ValueError(f"Missing attribute '{attr}' on {ctx or 'object'} ({obj!r}).")
        value = getattr(obj, attr)
        if value is None:
            raise ValueError(f"Attribute '{attr}' on {ctx or 'object'} is None.")
        return value

    @staticmethod
    def _ensure_positive_number(value: Any, name: str) -> Decimal:
        """
        Validate that a numeric value is strictly positive and return it as Decimal.
        Monetary amounts in these flows are expected to be > 0 by construction.
        """
        try:
            v = Decimal(str(value))
        except Exception as e:
            raise ValueError(f"'{name}' must be numeric; got {value!r}.") from e
        if not (v > 0):
            raise ValueError(f"'{name}' must be > 0; got {v}.")
        return v

    @staticmethod
    def _get_cogs_cost(good_instance: Any) -> Decimal:
        """
        Cost used for **COGS of the seller** in a sale:
        Prefer `last_cost`, then `cost`, then `price`. Treat 0.00 as "not available".
        This method is **read-only** and will not write to any attribute.
        """
        for attr in ("last_cost", "cost", "price"):
            if hasattr(good_instance, attr):
                try:
                    dec = _to_money(getattr(good_instance, attr))
                    if dec >= 0:
                        return dec
                except Exception:
                    pass
        raise ValueError("Unable to determine COGS cost (last_cost/cost/price missing or zero).")

    @staticmethod
    def _get_cost_strict(good_instance: Any) -> Decimal:
        """
        Unit Cost Logic
        The system follows a strict priority to determine the unit cost of an item.

        Strict Priority: The system looks for the cost in this order: cost, then last_cost, then price.
        It will return the first non-zero, positive value it finds in that sequence.
        If all properties are either missing, None, or are zero, the system will return 0.00.
        Any negative value found will cause a ValueError to be thrown.
        """
        zero_seen = False

        for attr in ("cost", "last_cost", "price"):
            if not hasattr(good_instance, attr):
                continue
            raw = getattr(good_instance, attr)
            if raw is None:
                continue
            dec = _to_money(raw)
            if dec < 0:
                raise ValueError(
                    f"Unit cost for attribute '{attr}' cannot be negative: {dec}."
                )
            if dec > 0:
                return dec
            zero_seen = True  # at least one 0

        if zero_seen:
            return _to_money(0)
        raise ValueError("GoodUnit lacks a usable cost: all None or missing.")

    @staticmethod
    def _get_dynamic_asset_account(agent: Any, good_type: str, *, role: str) -> Any:
        """
        Resolve the dynamic inventory account for a given good type from an agent’s chart of accounts.

        Requirements:
            • Returns an object with `.code` and `.name` used for journal entries.
        """
        chart = _get_chart(agent)
        account = chart.get_asset_account_for_good(good_type, create_if_missing=True)
        if account is None:
            raise ValueError(
                f"Dynamic asset account for '{good_type}' not found in {role}'s chart of accounts."
            )
        if not hasattr(account, "code") or not hasattr(account, "name"):
            raise ValueError(
                f"Resolved account for '{good_type}' on {role} lacks '.code' and/or '.name'."
            )
        return account

    @staticmethod
    def _build_reversal_entry(entry: AccountingEntry) -> AccountingEntry:
        """Create a reversing entry that offsets the provided entry."""
        lines = [
            EntryLine(
                account_code=line.account_code,
                account_name=line.account_name,
                amount=line.amount,
                is_debit=not line.is_debit,
            )
            for line in entry.lines
        ]
        return AccountingEntry(lines)

import yaml # type: ignore
from collections.abc import Mapping
from .account import Account



class _AccountsView(Mapping):
    """
    Read-only, dict-like facade over ChartOfAccounts' internal store.

    Why this exists
    ---------------
    Legacy reporting/utilities still expect `agent.chart_of_accounts.accounts`
    to behave like a `dict[str, Account]`. Internally we keep multiple indexes,
    but this facade preserves the old contract without exposing mutability.

    Design
    ------
    • Delegates lookups to the owning ChartOfAccounts' `_by_code` dict.
    • Implements the Mapping protocol: __getitem__, __iter__, __len__.
      That automatically enables `.keys()`, `.items()`, `.values()`, `in`, `dict(...)`, etc.
    • Read-only by construction: all mutations must go through ChartOfAccounts API.
    """

    def __init__(self, owner: "ChartOfAccounts") -> None:
        self._owner = owner

    def __getitem__(self, code: str) -> Account:
        return self._owner._by_code[str(code)]

    def __iter__(self):
        # Legacy semantics: iterate over account codes.
        return iter(self._owner._by_code)

    def __len__(self) -> int:
        return len(self._owner._by_code)

    def __repr__(self) -> str:
        return f"_AccountsView({list(self._owner._by_code.keys())})"


class ChartOfAccounts:
    """
    Central registry for all accounts of an agent, plus dynamic inventory accounts.

    Responsibilities
    ----------------
    • Keep a single source of truth for accounts by code and by name.
    • Dynamically create inventory accounts for goods on demand, with a stable
      naming convention: '{good} Asset'.
    • Provide a legacy-compatible, read-only `.accounts` property that behaves
      like a dict {code -> Account} for existing consumers (reports, helpers).

    Guarantees
    ----------
    • Idempotent dynamic creation: resolving the same good returns the same account.
    • Robust good key normalization (accepts strings or objects with .type/.name).
    • Safe dynamic code allocation: the internal cursor "bumps" above the highest
      numeric code found in static YAML to avoid collisions.
    """

    def __init__(
        self,
        accounts_list,
        *,
        dynamic_start_code: int = 1400,
        dynamic_type: str = "Asset",
        dynamic_subtype: str = "Current Asset",
        dynamic_group: str = "Inventory",
        normalize_goods_casefold: bool = True,
    ):
        # -------- Dynamic creation configuration --------
        self._normalize_goods_casefold = normalize_goods_casefold
        self._dynamic_cursor = int(dynamic_start_code)
        self._dynamic_type = dynamic_type
        self._dynamic_subtype = dynamic_subtype
        self._dynamic_group = dynamic_group

        # -------- Internal indexes --------
        self._by_code: dict[str, Account] = {}   # code -> Account
        self._by_name: dict[str, Account] = {}   # name -> Account
        self._good_to_code: dict[str, str] = {}  # normalized good key -> account code

        # -------- Load static accounts from YAML payload --------
        for acc_data in accounts_list or []:
            code = str(acc_data.get("code"))
            name = (acc_data.get("name") or "").strip()
            a_type = (acc_data.get("type") or "").strip()
            subtype = (acc_data.get("subtype") or acc_data.get("sub_type") or "").strip()
            group = (acc_data.get("group") or "").strip()

            if not code or not name:
                raise ValueError(f"Invalid account definition (missing code/name): {acc_data!r}")
            if code in self._by_code:
                raise ValueError(f"Duplicate account code in COA: {code}")

            acc = Account(code=code, name=name, type=a_type, subtype=subtype, group=group)
            self._by_code[code] = acc
            self._by_name[name] = acc

            # Pre-register dynamic mapping if the name matches '{good} Asset'
            good = self._extract_good_from_asset_name_if_applicable(name)
            if good:
                key = self._good_key(good)
                self._good_to_code.setdefault(key, code)

        # -------- Dynamic cursor bump (collision avoidance) --------
        # If the YAML already defines numeric codes >= dynamic_start_code,
        # push the cursor above the highest numeric code to guarantee uniqueness
        # for future dynamic accounts.
        try:
            dynamic_type = (self._dynamic_type or "").strip().lower()

            # Find the highest numeric code among *existing Asset accounts* only.
            max_numeric_asset = max(
                int(code)
                for code, acc in self._by_code.items()
                if str(code).isdigit()
                and isinstance(getattr(acc, "type", None), str)
                and getattr(acc, "type").strip().lower() == dynamic_type  # ← only Assets
            )

            # Start dynamic codes just after the highest Asset code we already have,
            # but never below the configured dynamic_start_code.
            self._dynamic_cursor = max(int(self._dynamic_cursor), max_numeric_asset + 1)

        except ValueError:
            # No numeric Asset codes are present; keep the configured dynamic_start_code.
            pass

        # -------- Legacy compatibility view (read-only dict-like) --------
        self._accounts_view = _AccountsView(self)

    # ---------- Public API ----------

    @property
    def accounts(self) -> Mapping[str, Account]:
        """
        Legacy-compatible, read-only dict-like view over accounts by code.

        Context
        -------
        Existing reporting code expects `agent.chart_of_accounts.accounts`
        to be a dict-ish mapping. Returning a Mapping keeps that contract,
        while preventing accidental mutation of internal structures.
        """
        return self._accounts_view

    def get_account(self, code: str) -> Account | None:
        """Retrieve an account by its code."""
        return self._by_code.get(str(code))

    def get_account_by_name(self, name: str) -> Account | None:
        """Retrieve an account by its exact name."""
        return self._by_name.get(name)

    def get_asset_account_for_good(self, good, *, create_if_missing: bool = True) -> Account | None:
        """
        Resolve the inventory account for the given good.

        Parameters
        ----------
        good : str | object
            Either a string (e.g., 'Steel') or an object with `.type` or `.name`.
        create_if_missing : bool
            When True (default), creates '{good} Asset' on demand if absent.

        Returns
        -------
        Account | None
        """
        good_text = self._coerce_good_text(good)
        if not good_text:
            raise ValueError(f"Invalid good identifier: {good!r}")

        key = self._good_key(good_text)

        # Fast path via cached mapping
        code = self._good_to_code.get(key)
        if code:
            return self._by_code[code]

        # Lookup by canonical name
        target_name = self._asset_name_from_good(good_text)
        acc = self._by_name.get(target_name)
        if acc:
            self._good_to_code[key] = acc.code
            return acc

        if not create_if_missing:
            return None

        new_acc = self._create_dynamic_asset_account(good_text)
        self._good_to_code[key] = new_acc.code
        return new_acc

    def __repr__(self):
        return f"ChartOfAccounts({list(self._by_code.keys())})"

    @staticmethod
    def load_accounts_from_yaml(file_path: str) -> list[dict]:
        """
        Load account definitions from a YAML file.

        Expected YAML shape:
        --------------------
        accounts:
          - code: "1105"
            name: "Cash"
            type: "Asset"
            subtype: "Current Asset"
            group: "Cash and Cash Equivalents"
          - ...
        """
        with open(file_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return data.get("accounts", [])

    # ---------- Internals ----------

    def _coerce_good_text(self, good) -> str:
        """
        Produce a robust string for a good identifier.

        If an object is provided, it tries `.type`, then `.name`, then `str(obj)`.
        """
        if isinstance(good, str):
            return good.strip()
        for attr in ("type", "name"):
            if hasattr(good, attr):
                val = getattr(good, attr)
                if isinstance(val, str) and val.strip():
                    return val.strip()
        return str(good).strip()

    def _good_key(self, good_text: str) -> str:
        """
        Normalize a good string to avoid duplicates due to case/whitespace.
        """
        s = good_text.strip()
        return s.casefold() if self._normalize_goods_casefold else s

    @staticmethod
    def _asset_name_from_good(good_text: str) -> str:
        return f"{good_text} Asset"

    @staticmethod
    def _extract_good_from_asset_name_if_applicable(name: str) -> str | None:
        """
        If the account name follows the '{good} Asset' convention, return 'good'.
        Otherwise, return None.
        """
        name = (name or "").strip()
        if not name.endswith(" Asset"):
            return None
        return name[:-len(" Asset")]

    def _create_dynamic_asset_account(self, good_text: str) -> Account:
        """
        Create a new dynamic inventory account with the next available code.

        The code comes from `_next_available_code()`, which guarantees no collision
        with existing numeric codes (thanks to the cursor bump in __init__).
        """
        code = self._next_available_code()
        acc = Account(
            code=code,
            name=self._asset_name_from_good(good_text),
            type=self._dynamic_type,
            subtype=self._dynamic_subtype,
            group=self._dynamic_group,
        )
        self._by_code[acc.code] = acc
        self._by_name[acc.name] = acc
        return acc

    def _next_available_code(self) -> str:
        """
        Find the next free numeric code for dynamic accounts.

        Implementation detail:
        ----------------------
        We rely on `_dynamic_cursor`, which was bumped in `__init__` to be strictly
        higher than any numeric code already present. We still guard against sparse
        dicts by skipping occupied codes in a loop.
        """
        while str(self._dynamic_cursor) in self._by_code:
            self._dynamic_cursor += 1
        out = str(self._dynamic_cursor)
        self._dynamic_cursor += 1
        return out


def ensure_asset_accounts_for_graph(production_graph, chart_of_accounts: ChartOfAccounts) -> None:
    """
    Ensure that every node (good) in the production graph has a '{good} Asset' account.

    Notes
    -----
    • Nodes are typically strings, but if they were objects with `.type`/`.name`,
      `ChartOfAccounts` will coerce them safely.
    • This function is idempotent and non-destructive.
    """
    nodes = getattr(production_graph, "get_nodes", lambda: [])()
    for node in nodes:
        chart_of_accounts.get_asset_account_for_good(node, create_if_missing=True)

# ------------------------- System guardrails (pre-flight) -------------------------

# Set of system-required accounts used by global_accountant and reporting.
# Only TYPE is validated.
SYSTEM_REQUIRED_ACCOUNTS = {
    # Assets / Cash
    "1105": {"type": "asset"},   # Cash

    # Production / Indirect costs
    "71":   {"type": "cost"},     # Direct Materials Cost
    "711":  {"type": "cost"},     # Imported Direct Materials Cost
    "73":   {"type": "expense"},  # Indirect Cost (overheads)

    # Operating Revenues (National / International)
    "4120": {"type": "revenue"},  # Manufacturing Revenue (National)
    "4135": {"type": "revenue"},  # Wholesale & Retail Revenue (National)
    "4121": {"type": "revenue"},  # Manufacturing Revenue (International)
    "4136": {"type": "revenue"},  # Wholesale & Retail Revenue (International)

    # Non-Operating Revenue
    "4140": {"type": "revenue"},  # Sale of Non-Inventory Assets

    # COGS (National / International / Non-Core)
    "6120": {"type": "cost"},     # Cost of Manufacturing (National)
    "6135": {"type": "cost"},     # Cost of Wholesale & Retail (National)
    "6121": {"type": "cost"},     # Cost of Manufacturing (International)
    "6136": {"type": "cost"},     # Cost of Wholesale & Retail (International)
    "6140": {"type": "cost"},     # Cost of Goods Sold (Non-Core)

    # Equity
    "3115": {"type": "equity"},   # Owner's Equity
}



def _safe_lower(s: str | None) -> str:
    return (s or "").strip().lower()

class ChartOfAccounts(ChartOfAccounts):  # extend class with guard methods
    def validate_system_accounts(self) -> list[str]:
        """
        Returns a list of human-readable issues if any required system account
        is missing or has a mismatched type. Empty list means 'all good'.
        """
        issues: list[str] = []
        for code, req in SYSTEM_REQUIRED_ACCOUNTS.items():
            acc = self.get_account(code)
            if acc is None:
                issues.append(f"Missing required account {code}.")
                continue
            expected_type = _safe_lower(req.get("type"))
            actual_type = _safe_lower(getattr(acc, "type", None))
            if actual_type != expected_type:
                issues.append(
                    f"Account {code} type mismatch: got '{acc.type}', expected '{expected_type}'."
                )
        return issues

    def assert_system_accounts(self) -> None:
        """
        Raises ValueError with a consolidated message if validation fails.
        Use this early (e.g., at AccAgent init).
        """
        issues = self.validate_system_accounts()
        if issues:
            bullets = "\n  - " + "\n  - ".join(issues)
            raise ValueError(
                "ChartOfAccounts pre-flight failed. Please fix your YAML so it includes the "
                "system-required accounts with correct types:" + bullets
            )

from __future__ import annotations

from pathlib import Path


CANONICAL_CHART_OF_ACCOUNTS_RELATIVE_PATH = (
    Path("configs") / "chart_of_accounts" / "chart_of_accounts.yaml"
)


def find_repo_root(start: str | Path | None = None) -> Path:
    current = Path(start) if start is not None else Path(__file__)
    current = current.resolve()
    if current.is_file():
        current = current.parent

    for candidate in (current, *current.parents):
        if (candidate / "src").exists() and (candidate / "configs").exists():
            return candidate

    raise FileNotFoundError(
        "Could not determine repository root while resolving chart_of_accounts.yaml."
    )


def get_default_chart_of_accounts_path() -> Path:
    repo_root = find_repo_root()
    chart_path = repo_root / CANONICAL_CHART_OF_ACCOUNTS_RELATIVE_PATH
    if not chart_path.exists():
        raise FileNotFoundError(
            f"Canonical chart of accounts file was not found at: {chart_path}"
        )
    return chart_path


def resolve_chart_of_accounts_path(
    chart_path: str | Path | None,
    *,
    base_dir: str | Path | None = None,
) -> Path:
    canonical_path = get_default_chart_of_accounts_path()

    if chart_path is None or not str(chart_path).strip():
        return canonical_path

    raw_path = Path(chart_path)
    base_path = Path(base_dir).resolve() if base_dir is not None else None
    if base_path is not None and base_path.is_file():
        base_path = base_path.parent

    candidates: list[Path] = []
    if raw_path.is_absolute():
        candidates.append(raw_path)
    else:
        if base_path is not None:
            candidates.append((base_path / raw_path).resolve())
        candidates.append((Path.cwd() / raw_path).resolve())
        try:
            repo_root = find_repo_root()
            candidates.append((repo_root / raw_path).resolve())
        except FileNotFoundError:
            pass

    for candidate in candidates:
        if candidate.exists():
            return candidate

    if raw_path.name == canonical_path.name:
        return canonical_path

    tried = ", ".join(str(path) for path in candidates) or str(raw_path)
    raise FileNotFoundError(
        "Could not resolve chart_of_accounts.yaml. "
        f"Tried: {tried}. Canonical path is {canonical_path}."
    )

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Rebuilds a rich, wide CSV from per-run JSON artifacts produced by run_all.py.

It scans an input directory (per algorithm), loads each JSON file, and extracts:
- key metrics (best_final, aucs, runtime, evals, budget flags),
- diagnostics (improvements, plateau, volatility) when present,
- curve metadata (lengths, final mean/median/std/variance if present),
- genome summary (best genome and count of maximizers),
- hyperparameters actually used in the run,
- file path of the source JSON.

This allows post-hoc analysis without re-running experiments.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import csv


def _safe_get(d: Dict[str, Any], path: List[str], default: Any = None) -> Any:
    """Safely navigates nested dictionaries using a path of keys."""
    cur = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _best_curve_payload(d: Dict[str, Any]) -> List[float]:
    """Extracts the best curve list if present."""
    return _safe_get(d, ["raw", "curves", "best"], []) or _safe_get(d, ["curves", "best"], []) or []


def _curve_len(d: Dict[str, Any], name: str) -> int:
    """Returns the length of a named curve if present."""
    arr = _safe_get(d, ["raw", "curves", name], []) or _safe_get(d, ["curves", name], [])
    return len(arr) if isinstance(arr, list) else 0


def _per_gen_stat(d: Dict[str, Any], key: str, reducer: str = "final") -> Optional[float]:
    """
    Pulls a per-generation statistic from `per_generation`:
    - reducer="final": last value
    - reducer="mean": arithmetic mean over gens
    """
    table = _safe_get(d, ["per_generation"], None)
    if not isinstance(table, list) or not table:
        return None

    # Expect rows like {"gen": i, "best": ..., "mean": ..., "median": ..., "std": ..., "variance": ...}
    values = [row.get(key) for row in table if isinstance(row, dict) and key in row]
    values = [v for v in values if isinstance(v, (int, float))]
    if not values:
        return None

    if reducer == "final":
        return float(values[-1])
    if reducer == "mean":
        return float(sum(values) / len(values))
    return None


def _as_compact_json(obj: Any, max_len: int = 200) -> str:
    """Serializes small objects to a compact JSON string and truncates for CSV safety."""
    try:
        s = json.dumps(obj, separators=(",", ":"), ensure_ascii=False)
    except Exception:
        s = str(obj)
    if len(s) > max_len:
        s = s[: max_len - 3] + "..."
    return s


def _row_from_run(payload: Dict[str, Any], src_path: Path) -> Dict[str, Any]:
    """
    Builds one CSV row from a single run payload.
    Assumes payload is the JSON returned by run_all's worker.
    """
    algo = payload.get("algo", "")
    cid = payload.get("candidate_id", payload.get("hparams", {}).get("id"))
    seed = payload.get("seed")

    # If this run failed, just capture the error row.
    error = payload.get("error")
    row: Dict[str, Any] = {
        "algo": algo,
        "candidate_id": cid,
        "seed": seed,
        "error": error or "",
        "json_path": str(src_path),
    }
    if error:
        return row

    # Metrics
    row.update({
        "best_final": _safe_get(payload, ["metrics", "best_final"]),
        "auc_best": _safe_get(payload, ["metrics", "auc_best"]),
        "auc_best_norm": _safe_get(payload, ["metrics", "auc_best_norm"]),
        "runtime_sec": _safe_get(payload, ["metrics", "runtime_sec"]),
        "evals_total": _safe_get(payload, ["metrics", "evals_total"]),
        "budget_triggered": _safe_get(payload, ["metrics", "budget_triggered"]),
        "stop_reason": _safe_get(payload, ["metrics", "stop_reason"], ""),
    })

    # Hparams (keep all common keys; non-used remain blank)
    hp = payload.get("hparams", {})
    for k in ("cxpb", "mutpb", "mutation_rate", "parents_rate", "sel_mutation", "tail_mutation", "c1", "c2", "w",
              "generations", "popsize", "seed"):
        row[k] = hp.get(k, "")

    # Diagnostics (as produced by utils_metrics.curves_summary)
    diag = payload.get("diagnostics", {})
    best_diag = diag.get("best", {}) if isinstance(diag, dict) else {}
    row.update({
        "best_first": best_diag.get("first"),
        "best_last": best_diag.get("last"),
        "best_impr_abs": best_diag.get("improvement_abs"),
        "best_impr_pct": best_diag.get("improvement_pct"),
        "best_plateau_gen": best_diag.get("plateau_gen"),
        "best_volatility_std": best_diag.get("volatility_std"),
        "best_n_strict_improvements": best_diag.get("n_strict_improvements"),
    })

    # Curve metadata
    row.update({
        "len_best": _curve_len(payload, "best"),
        "len_mean": _curve_len(payload, "mean"),
        "len_median": _curve_len(payload, "median"),
    })

    # Per-generation summary: final and mean of std/variance if available
    row.update({
        "final_mean": _per_gen_stat(payload, "mean", "final"),
        "final_median": _per_gen_stat(payload, "median", "final"),
        "final_std": _per_gen_stat(payload, "std", "final"),
        "final_variance": _per_gen_stat(payload, "variance", "final"),
        "avg_std": _per_gen_stat(payload, "std", "mean"),
        "avg_variance": _per_gen_stat(payload, "variance", "mean"),
    })

    # Genomes
    best_genome = _safe_get(payload, ["genomes", "best_genome"])
    all_best = _safe_get(payload, ["genomes", "all_best_genomes"], [])
    row.update({
        "best_genome": _as_compact_json(best_genome, max_len=200),
        "n_best_genomes": (len(all_best) if isinstance(all_best, list) else 0),
    })

    return row


def _collect_rows(in_dir: Path) -> List[Dict[str, Any]]:
    """Scans a directory for *.json and returns a list of CSV rows."""
    rows: List[Dict[str, Any]] = []
    for p in sorted(in_dir.glob("*.json")):
        try:
            payload = json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            rows.append({
                "algo": "",
                "candidate_id": "",
                "seed": "",
                "error": f"LoadError: {type(e).__name__}: {e}",
                "json_path": str(p),
            })
            continue
        rows.append(_row_from_run(payload, p))
    return rows


def _write_csv(rows: List[Dict[str, Any]], out_csv: Path) -> None:
    """Writes rows to CSV with a stable union of keys as headers."""
    # Collect all keys present across rows to build header
    header_keys: List[str] = []
    seen = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                header_keys.append(k)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header_keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> None:
    """CLI entrypoint."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", type=Path, required=True, help="Directory with per-run JSON files (e.g., results/generic).")
    ap.add_argument("--out", type=Path, required=True, help="Output CSV path (e.g., results/generic_summary_wide.csv).")
    args = ap.parse_args()

    rows = _collect_rows(args.in_dir)
    _write_csv(rows, args.out)
    print(f"[INFO] Wrote {len(rows)} rows to: {args.out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Metric utilities for run logging and post-processing.

This module provides:
- auc_trapezoid: trapezoidal AUC for a 1D series.
- auc_best_and_norm: raw and length-normalized AUC for the "best" curve.
- curves_summary: compact diagnostics per curve (best/mean/median if present).
- per_generation_summary: per-generation table aligning available series and variance.

All functions are defensive against missing keys and unequal lengths.
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Any
import math


def _safe_isfinite(x: Any) -> bool:
    """Returns True if x is a finite float; False otherwise."""
    try:
        return math.isfinite(float(x))
    except Exception:
        return False


def _trim_to_common_length(curves: Dict[str, List[float]]) -> Dict[str, List[float]]:
    """
    Trims all list-valued curves to the minimum common length > 0.

    Args:
        curves (Dict[str, List[float]]): Mapping of name → list of values.

    Returns:
        Dict[str, List[float]]: Curves with equalized lengths (if possible).
    """
    series = {k: v for k, v in curves.items() if isinstance(v, list) and len(v) > 0}
    if not series:
        return {}
    m = min(len(v) for v in series.values())
    if m < 1:
        return {}
    return {k: v[:m] for k, v in series.items()}


def auc_trapezoid(y: List[float]) -> float:
    """
    Computes the trapezoidal area under the curve for a sequence y[0..G-1].

    Args:
        y (List[float]): Sequence of values ordered by generation.

    Returns:
        float: Trapezoidal area. Returns 0.0 when length < 2.
    """
    if not y or len(y) < 2:
        return 0.0
    area = 0.0
    for i in range(1, len(y)):
        y0 = float(y[i - 1])
        y1 = float(y[i])
        area += 0.5 * (y0 + y1)  # Δx = 1
    return area


def auc_best_and_norm(best_curve: List[float]) -> Tuple[float, float]:
    """
    Computes raw and normalized AUC for the best curve.

    Args:
        best_curve (List[float]): Best-utility-by-generation series.

    Returns:
        Tuple[float, float]: (auc_raw, auc_normalized), where the normalization
        divides by (len(best_curve) - 1). If len < 2, both values are 0.0.
    """
    if not best_curve or len(best_curve) < 2:
        return 0.0, 0.0
    raw = auc_trapezoid(best_curve)
    norm = raw / float(len(best_curve) - 1)
    return raw, norm


def _curve_stats(y: List[float]) -> Dict[str, float]:
    """
    Computes compact stats for a single curve.

    Args:
        y (List[float]): Series y[0..G-1].

    Returns:
        Dict[str, float]: start, end, delta_abs, delta_rel, max_val, argmax, auc_raw, auc_norm.
    """
    if not y:
        return {
            "start": float("nan"),
            "end": float("nan"),
            "delta_abs": float("nan"),
            "delta_rel": float("nan"),
            "max_val": float("nan"),
            "argmax": -1,
            "auc_raw": 0.0,
            "auc_norm": 0.0,
        }
    start = float(y[0])
    end = float(y[-1])
    delta_abs = end - start
    delta_rel = (delta_abs / abs(start)) if _safe_isfinite(start) and abs(start) > 0 else float("nan")
    max_val = max(float(v) for v in y)
    argmax = max(range(len(y)), key=lambda i: float(y[i]))
    auc_raw = auc_trapezoid(y)
    auc_norm = auc_raw / float(len(y) - 1) if len(y) > 1 else 0.0
    return {
        "start": start,
        "end": end,
        "delta_abs": delta_abs,
        "delta_rel": delta_rel,
        "max_val": max_val,
        "argmax": int(argmax),
        "auc_raw": auc_raw,
        "auc_norm": auc_norm,
    }


def curves_summary(curves: Dict[str, List[float]]) -> Dict[str, Dict[str, float]]:
    """
    Builds a compact diagnostic summary for each present curve among
    {'best','mean','median'}.

    Args:
        curves (Dict[str, List[float]]): Mapping curve name → values.

    Returns:
        Dict[str, Dict[str, float]]: Mapping curve name → stats dict.
    """
    names = ("best", "mean", "median")
    out: Dict[str, Dict[str, float]] = {}
    for k in names:
        y = curves.get(k)
        if isinstance(y, list) and len(y) > 0:
            out[k] = _curve_stats(y)
    return out


def per_generation_summary(curves: Dict[str, List[float]]) -> List[Dict[str, float]]:
    """
    Builds a per-generation table aligning available series. If 'variance' is missing
    but 'std' is present, variance is computed as std^2. Lengths are equalized by
    trimming to the minimum present length.

    Args:
        curves (Dict[str, List[float]]): Mapping curve name → values.

    Returns:
        List[Dict[str, float]]: One dict per generation with keys:
            'gen', 'best', 'mean', 'median', 'std', 'variance'.
    """
    allowed = {k: v for k, v in curves.items() if isinstance(v, list)}
    if not allowed:
        return []

    eq = _trim_to_common_length(allowed)
    if not eq:
        return []

    best = eq.get("best")
    mean = eq.get("mean")
    median = eq.get("median")
    std = eq.get("std")
    var = eq.get("variance")

    rows: List[Dict[str, float]] = []
    G = len(next(iter(eq.values())))
    for i in range(G):
        row: Dict[str, float] = {"gen": float(i)}
        if best:   row["best"] = float(best[i])
        if mean:   row["mean"] = float(mean[i])
        if median: row["median"] = float(median[i])
        if std is not None:
            row["std"] = float(std[i])
            # If variance is absent but std is present, derive it
            if var is None:
                row["variance"] = float(std[i]) ** 2
        if var is not None:
            row["variance"] = float(var[i])
        rows.append(row)
    return rows

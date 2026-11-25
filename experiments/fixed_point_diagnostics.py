"""Persist and summarize fixed-point diagnostics for paper experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def fixed_point_metrics(solver: Any, output_dir: str | Path) -> dict[str, Any]:
    diagnostics = list(getattr(solver, "diagnostics_history", []))
    output_path = Path(output_dir) / "fixed_point_diagnostics.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as handle:
        json.dump(
            {
                "criterion": "normalized_bellman_residual_linf",
                "solves": diagnostics,
            },
            handle,
            indent=2,
        )

    iterations = np.asarray(
        [float(item["total_iterations"]) for item in diagnostics], dtype=float
    )
    final_differences = np.asarray(
        [float(item.get("final_iterate_difference_linf", np.nan)) for item in diagnostics],
        dtype=float,
    )
    final_normalized_residuals = np.asarray(
        [float(item.get("final_normalized_residual_linf", item.get("final_linf", np.nan))) for item in diagnostics],
        dtype=float,
    )
    final_absolute_residuals = np.asarray(
        [float(item.get("final_residual_linf", np.nan)) for item in diagnostics],
        dtype=float,
    )
    failed_count = sum(not bool(item["converged"]) for item in diagnostics)
    retry_count = sum(bool(item["retry_used"]) for item in diagnostics)
    max_difference = (
        float(np.nanmax(final_differences)) if final_differences.size else float("nan")
    )
    max_normalized_residual = (
        float(np.nanmax(final_normalized_residuals)) if final_normalized_residuals.size else float("nan")
    )
    max_absolute_residual = (
        float(np.nanmax(final_absolute_residuals))
        if final_absolute_residuals.size and not np.all(np.isnan(final_absolute_residuals))
        else float("nan")
    )
    return {
        "fp_solve_count": int(len(diagnostics)),
        "fp_failed_solve_count": int(failed_count),
        "fp_retry_solve_count": int(retry_count),
        "fp_all_converged": int(bool(diagnostics) and failed_count == 0),
        "fp_median_iterations": (
            float(np.median(iterations)) if iterations.size else float("nan")
        ),
        "fp_max_iterations": (
            float(np.max(iterations)) if iterations.size else float("nan")
        ),
        "fp_max_final_iterate_difference_linf": max_difference,
        "fp_max_final_residual_linf": max_absolute_residual,
        "fp_max_final_normalized_residual_linf": max_normalized_residual,
        # Compatibility alias now refers to the actual convergence criterion.
        "fp_max_final_residual": max_normalized_residual,
    }

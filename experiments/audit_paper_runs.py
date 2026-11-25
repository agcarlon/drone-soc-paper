"""Audit fixed-point acceptance and boundary contact in paper experiment data."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.paper_result_filters import fixed_point_policy_status


PAPER_RESULT_DIRS = (
    "wind_aligned_comparison",
    "greedy_eig_comparison",
    "paper_pairwise_showcase",
    "paper_pairwise_discount_sweep",
    "scaling_M",
)


def audit_results(results_dir: str | Path) -> dict[str, Any]:
    root = Path(results_dir)
    failed: list[dict[str, Any]] = []
    retry_cases: list[dict[str, Any]] = []
    boundary_by_method: dict[str, list[float]] = defaultdict(list)
    invalid_metric_files: list[str] = []
    metric_files = 0
    fp_cases = 0

    for result_name in PAPER_RESULT_DIRS:
        result_root = root / result_name
        if not result_root.exists():
            continue
        for metrics_path in result_root.rglob("metrics.json"):
            metric_files += 1
            relative_path = str(metrics_path.relative_to(root))
            try:
                with metrics_path.open() as handle:
                    row = json.load(handle)
                if not isinstance(row, dict):
                    raise ValueError("top-level JSON value is not an object")
            except (OSError, ValueError, json.JSONDecodeError) as error:
                invalid_metric_files.append(f"{relative_path}: {error}")
                continue
            method = str(row.get("method", metrics_path.parent.name))
            if "boundary_dwell_fraction" in row:
                boundary_by_method[method].append(
                    float(row["boundary_dwell_fraction"])
                )
            fp_status = fixed_point_policy_status(row)
            if fp_status == "unverified":
                continue
            fp_cases += 1
            record = {
                "path": relative_path,
                "failed_solve_count": int(row.get("fp_failed_solve_count", 0)),
                "retry_solve_count": int(row.get("fp_retry_solve_count", 0)),
                "max_final_iterate_difference_linf": float(
                    row.get(
                        "fp_max_final_iterate_difference_linf",
                        row.get("fp_max_final_residual", float("nan")),
                    )
                ),
            }
            if record["retry_solve_count"] > 0:
                retry_cases.append(record)
            if fp_status == "rejected":
                failed.append(record)

    boundary_summary = {
        method: {
            "num_cases": len(values),
            "median_dwell_fraction": float(np.median(values)),
            "max_dwell_fraction": float(np.max(values)),
        }
        for method, values in sorted(boundary_by_method.items())
        if values
    }
    return {
        "metric_file_count": metric_files,
        "fixed_point_case_count": fp_cases,
        "failed_fixed_point_case_count": len(failed),
        "retry_case_count": len(retry_cases),
        "failed_fixed_point_cases": failed,
        "retry_cases": retry_cases,
        "invalid_metric_file_count": len(invalid_metric_files),
        "invalid_metric_files": invalid_metric_files,
        "boundary_summary_by_method": boundary_summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--out", default="results/paper_run_audit.json")
    args = parser.parse_args()

    report = audit_results(args.results_dir)
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n")
    print(
        f"Audited {report['metric_file_count']} cases: "
        f"{report['failed_fixed_point_case_count']} failed FP cases, "
        f"{report['retry_case_count']} cases used at least one retry."
    )
    print(f"Audit report saved to {output_path}")
    if report["invalid_metric_file_count"]:
        raise SystemExit(
            f"Audit found {report['invalid_metric_file_count']} invalid metrics files; "
            f"see {output_path}"
        )


if __name__ == "__main__":
    main()

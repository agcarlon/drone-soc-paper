"""Compute lightweight inferential statistics for the paper tables.

The script reads existing per-trial CSV files and writes derived confidence
interval tables. It does not run simulations or modify experiment outputs.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.paper_result_filters import (
    accepted_fixed_point_rows,
    fixed_point_policy_accepted,
)
from experiments.result_io import write_csv_atomic
from experiments.validate_paper_results import validate_paper_results


DEFAULT_OUT = ROOT / "results" / "paper_statistical_intervals"
BOOTSTRAP_REPS = 10_000
BOOTSTRAP_SEED = 20_260_612


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write for {path}")
    write_csv_atomic(path, rows)


def _float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def _int_flag(row: dict[str, str], key: str) -> int:
    return int(float(row[key]))


def _fixed_point_final_difference(row: dict[str, str]) -> float:
    """Read the precise diagnostic name, with support for archived rows."""
    value = row.get("fp_max_final_iterate_difference_linf", "")
    if value is None or str(value).strip() == "":
        value = row["fp_max_final_residual"]
    return float(value)


def _wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return math.nan, math.nan
    p = k / n
    denom = 1.0 + z**2 / n
    center = (p + z**2 / (2.0 * n)) / denom
    half = z * math.sqrt(p * (1.0 - p) / n + z**2 / (4.0 * n**2)) / denom
    return center - half, center + half


def _bca_interval(
    values: np.ndarray,
    statistic: Callable[[np.ndarray], float] = np.median,
    *,
    reps: int = BOOTSTRAP_REPS,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = values.size
    if n == 0:
        return math.nan, math.nan, math.nan

    theta_hat = float(statistic(values))
    if n == 1 or np.all(values == values[0]):
        return theta_hat, theta_hat, theta_hat

    indices = rng.integers(0, n, size=(reps, n))
    boot = np.asarray([statistic(values[idx]) for idx in indices], dtype=float)
    boot.sort()

    prop_less = np.mean(boot < theta_hat)
    eps = 1.0 / (2.0 * reps)
    prop_less = float(np.clip(prop_less, eps, 1.0 - eps))
    z0 = norm.ppf(prop_less)

    jack = np.empty(n, dtype=float)
    for i in range(n):
        jack[i] = statistic(np.delete(values, i))
    jack_mean = np.mean(jack)
    numerator = np.sum((jack_mean - jack) ** 3)
    denominator = 6.0 * np.sum((jack_mean - jack) ** 2) ** 1.5
    accel = 0.0 if denominator == 0.0 else float(numerator / denominator)

    def adjusted_quantile(alpha: float) -> float:
        z_alpha = norm.ppf(alpha)
        denom = 1.0 - accel * (z0 + z_alpha)
        if denom == 0.0:
            return alpha
        return float(norm.cdf(z0 + (z0 + z_alpha) / denom))

    q_low = np.clip(adjusted_quantile(0.025), 0.0, 1.0)
    q_high = np.clip(adjusted_quantile(0.975), 0.0, 1.0)
    low = float(np.quantile(boot, q_low, method="linear"))
    high = float(np.quantile(boot, q_high, method="linear"))
    return theta_hat, low, high


@dataclass(frozen=True)
class GroupSpec:
    experiment: str
    label: str
    rows: list[dict[str, str]]


def _rate_rows() -> list[dict[str, object]]:
    rows_out: list[dict[str, object]] = []

    baseline = _read_csv(ROOT / "results" / "greedy_eig_comparison" / "per_trial_results.csv")
    for snr in sorted({float(r["snr0_target"]) for r in baseline}):
        for method in ("order1", "greedy_eig", "patrol"):
            group = [r for r in baseline if float(r["snr0_target"]) == snr and r["method"] == method]
            k = sum(_int_flag(r, "success") for r in group)
            low, high = _wilson_interval(k, len(group))
            rows_out.append(
                {
                    "experiment": "baseline",
                    "group": f"snr0={snr:g}, method={method}",
                    "metric": "success_rate",
                    "count": k,
                    "n": len(group),
                    "estimate": k / len(group),
                    "ci_low": low,
                    "ci_high": high,
                }
            )

    showcase = _read_csv(ROOT / "results" / "paper_pairwise_showcase" / "per_trial_results.csv")
    for method in ("order1", "order2"):
        group = [r for r in showcase if r["method"] == method]
        for metric in ("stopped", "success_2m", "success_5m"):
            k = sum(_int_flag(r, metric) for r in group)
            low, high = _wilson_interval(k, len(group))
            rows_out.append(
                {
                    "experiment": "pairwise_showcase",
                    "group": f"method={method}",
                    "metric": metric,
                    "count": k,
                    "n": len(group),
                    "estimate": k / len(group),
                    "ci_low": low,
                    "ci_high": high,
                }
            )

    discount = accepted_fixed_point_rows(
        _read_csv(ROOT / "results" / "paper_pairwise_discount_sweep" / "per_trial_results.csv")
    )
    for discount_value in sorted({float(r["discount"]) for r in discount}):
        for method in ("order1", "order2"):
            group = [
                r
                for r in discount
                if float(r["discount"]) == discount_value and r["method"] == method
            ]
            for metric in ("stopped", "success_5m"):
                k = sum(_int_flag(r, metric) for r in group)
                low, high = _wilson_interval(k, len(group))
                rows_out.append(
                    {
                        "experiment": "discount_sweep",
                        "group": f"r={discount_value:g}, method={method}",
                        "metric": metric,
                        "count": k,
                        "n": len(group),
                        "estimate": k / len(group),
                        "ci_low": low,
                        "ci_high": high,
                    }
                )

    scaling = _read_csv(ROOT / "results" / "scaling_M" / "per_trial_results.csv")
    for num_drones in sorted({int(r["num_drones"]) for r in scaling}):
        group = [r for r in scaling if int(r["num_drones"]) == num_drones]
        k = sum(_int_flag(r, "stopped") for r in group)
        low, high = _wilson_interval(k, len(group))
        rows_out.append(
            {
                "experiment": "scaling_M",
                "group": f"M={num_drones}",
                "metric": "stopped",
                "count": k,
                "n": len(group),
                "estimate": k / len(group),
                "ci_low": low,
                "ci_high": high,
            }
        )

    return rows_out


def _median_groups() -> Iterable[tuple[str, str, str, np.ndarray]]:
    baseline = _read_csv(ROOT / "results" / "greedy_eig_comparison" / "per_trial_results.csv")
    for snr in sorted({float(r["snr0_target"]) for r in baseline}):
        for method in ("order1", "greedy_eig", "patrol"):
            group = [r for r in baseline if float(r["snr0_target"]) == snr and r["method"] == method]
            for metric in ("final_error", "energy"):
                yield "baseline", f"snr0={snr:g}, method={method}", metric, np.array(
                    [_float(r, metric) for r in group]
                )

    showcase = _read_csv(ROOT / "results" / "paper_pairwise_showcase" / "per_trial_results.csv")
    for method in ("order1", "order2"):
        group = [r for r in showcase if r["method"] == method]
        for metric in ("final_error", "final_trace_sigma", "energy", "travel_distance", "runtime_sec"):
            yield "pairwise_showcase", f"method={method}", metric, np.array(
                [_float(r, metric) for r in group]
            )

    discount = accepted_fixed_point_rows(
        _read_csv(ROOT / "results" / "paper_pairwise_discount_sweep" / "per_trial_results.csv")
    )
    for discount_value in sorted({float(r["discount"]) for r in discount}):
        for method in ("order1", "order2"):
            group = [
                r
                for r in discount
                if float(r["discount"]) == discount_value and r["method"] == method
            ]
            for metric in ("final_error", "final_trace_sigma", "energy", "travel_distance", "runtime_sec"):
                yield "discount_sweep", f"r={discount_value:g}, method={method}", metric, np.array(
                    [_float(r, metric) for r in group]
                )

    scaling = _read_csv(ROOT / "results" / "scaling_M" / "per_trial_results.csv")
    for num_drones in sorted({int(r["num_drones"]) for r in scaling}):
        group = [r for r in scaling if int(r["num_drones"]) == num_drones]
        for metric in ("final_error", "final_trace_sigma", "energy_per_drone"):
            yield "scaling_M", f"M={num_drones}", metric, np.array([_float(r, metric) for r in group])


def _median_rows() -> list[dict[str, object]]:
    rows_out: list[dict[str, object]] = []
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    for experiment, group, metric, values in _median_groups():
        estimate, low, high = _bca_interval(values, rng=rng)
        rows_out.append(
            {
                "experiment": experiment,
                "group": group,
                "metric": metric,
                "n": values.size,
                "median": estimate,
                "bca95_low": low,
                "bca95_high": high,
            }
        )
    return rows_out


def _paired_difference_rows() -> list[dict[str, object]]:
    rows = _read_csv(ROOT / "results" / "paper_pairwise_showcase" / "per_trial_results.csv")
    by_seed: dict[int, dict[str, dict[str, str]]] = {}
    for row in rows:
        by_seed.setdefault(int(row["seed"]), {})[row["method"]] = row

    metric_labels = {
        "final_error": "Final error [m]",
        "final_trace_sigma": "tr(Sigma) [m^2]",
        "energy": "Energy [m^2 s^-1]",
    }
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    out: list[dict[str, object]] = []
    for metric, label in metric_labels.items():
        diffs = []
        for seed, pair in sorted(by_seed.items()):
            if "order1" not in pair or "order2" not in pair:
                continue
            diffs.append(_float(pair["order1"], metric) - _float(pair["order2"], metric))
        arr = np.asarray(diffs, dtype=float)
        estimate, low, high = _bca_interval(arr, rng=rng)
        out.append(
            {
                "metric": metric,
                "label": label,
                "n_pairs": arr.size,
                "median_one_body_minus_pairwise": estimate,
                "bca95_low": low,
                "bca95_high": high,
                "seeds_favoring_one_body": int(np.sum(arr < 0.0)),
                "seeds_favoring_pairwise": int(np.sum(arr > 0.0)),
                "ties": int(np.sum(arr == 0.0)),
                "pairwise_better_error_and_energy": "",
                "pairwise_better_error_only": "",
                "pairwise_better_energy_only": "",
                "one_body_better_error_and_energy": "",
            }
        )

    error = np.asarray(
        [
            _float(pair["order1"], "final_error") - _float(pair["order2"], "final_error")
            for pair in by_seed.values()
            if "order1" in pair and "order2" in pair
        ],
        dtype=float,
    )
    energy = np.asarray(
        [
            _float(pair["order1"], "energy") - _float(pair["order2"], "energy")
            for pair in by_seed.values()
            if "order1" in pair and "order2" in pair
        ],
        dtype=float,
    )
    out.append(
        {
            "metric": "joint_error_energy",
            "label": "Joint final error and energy",
            "n_pairs": error.size,
            "median_one_body_minus_pairwise": math.nan,
            "bca95_low": math.nan,
            "bca95_high": math.nan,
            "seeds_favoring_one_body": int(np.sum((error < 0.0) & (energy < 0.0))),
            "seeds_favoring_pairwise": int(np.sum((error > 0.0) & (energy > 0.0))),
            "ties": int(error.size - np.sum((error < 0.0) & (energy < 0.0)) - np.sum((error > 0.0) & (energy > 0.0))),
            "pairwise_better_error_and_energy": int(np.sum((error > 0.0) & (energy > 0.0))),
            "pairwise_better_error_only": int(np.sum((error > 0.0) & (energy < 0.0))),
            "pairwise_better_energy_only": int(np.sum((error < 0.0) & (energy > 0.0))),
            "one_body_better_error_and_energy": int(np.sum((error < 0.0) & (energy < 0.0))),
        }
    )
    return out


def _discount_paired_difference_rows() -> list[dict[str, object]]:
    rows = accepted_fixed_point_rows(
        _read_csv(ROOT / "results" / "paper_pairwise_discount_sweep" / "per_trial_results.csv")
    )
    by_method_rate_seed = {
        (row["method"], float(row["discount"]), int(row["seed"])): row
        for row in rows
    }
    comparisons = ((0.2, 0.3), (0.1, 0.2))
    metrics = ("final_error", "final_trace_sigma", "energy", "travel_distance", "stop_time")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    out: list[dict[str, object]] = []

    for method in ("order1", "order2"):
        for lower_rate, higher_rate in comparisons:
            common_seeds = sorted(
                seed
                for candidate_method, rate, seed in by_method_rate_seed
                if candidate_method == method
                and rate == lower_rate
                and (method, higher_rate, seed) in by_method_rate_seed
            )
            for metric in metrics:
                differences = np.asarray(
                    [
                        _float(by_method_rate_seed[(method, lower_rate, seed)], metric)
                        - _float(by_method_rate_seed[(method, higher_rate, seed)], metric)
                        for seed in common_seeds
                    ],
                    dtype=float,
                )
                estimate, low, high = _bca_interval(differences, rng=rng)
                out.append(
                    {
                        "method": method,
                        "lower_rate": lower_rate,
                        "higher_rate": higher_rate,
                        "metric": metric,
                        "n_pairs": differences.size,
                        "median_lower_minus_higher": estimate,
                        "bca95_low": low,
                        "bca95_high": high,
                    }
                )
    return out


def _discount_fixed_point_comparison_rows() -> list[dict[str, object]]:
    """Describe accepted and rejected pairwise runs at the weakest discount."""
    rows = _read_csv(
        ROOT / "results" / "paper_pairwise_discount_sweep" / "per_trial_results.csv"
    )
    target = [
        row
        for row in rows
        if row["method"] == "order2" and float(row["discount"]) == 0.1
    ]
    groups = (
        ("accepted", [row for row in target if fixed_point_policy_accepted(row)]),
        ("rejected", [row for row in target if not fixed_point_policy_accepted(row)]),
    )
    out: list[dict[str, object]] = []
    for status, group in groups:
        if not group:
            out.append(
                {
                    "status": status,
                    "n": 0,
                    "diagnostic_stop_rate": math.nan,
                    "success_5m_rate": math.nan,
                    "median_final_error": math.nan,
                    "median_final_trace_sigma": math.nan,
                    "median_energy": math.nan,
                    "median_travel_distance": math.nan,
                    "median_runtime_sec": math.nan,
                    "median_final_iterate_difference_linf": math.nan,
                    "max_final_iterate_difference_linf": math.nan,
                }
            )
            continue
        iterate_differences = np.asarray(
            [_fixed_point_final_difference(row) for row in group],
            dtype=float,
        )
        out.append(
            {
                "status": status,
                "n": len(group),
                "diagnostic_stop_rate": np.mean([_int_flag(row, "stopped") for row in group]),
                "success_5m_rate": np.mean([_int_flag(row, "success_5m") for row in group]),
                "median_final_error": np.median([_float(row, "final_error") for row in group]),
                "median_final_trace_sigma": np.median(
                    [_float(row, "final_trace_sigma") for row in group]
                ),
                "median_energy": np.median([_float(row, "energy") for row in group]),
                "median_travel_distance": np.median(
                    [_float(row, "travel_distance") for row in group]
                ),
                "median_runtime_sec": np.median(
                    [_float(row, "runtime_sec") for row in group]
                ),
                "median_final_iterate_difference_linf": np.median(iterate_differences),
                "max_final_iterate_difference_linf": np.max(iterate_differences),
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute paper confidence intervals from per-trial CSVs.")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT), help="Directory for derived statistics CSVs")
    args = parser.parse_args()

    validation_errors = validate_paper_results(ROOT / "results")
    if validation_errors:
        details = "\n".join(f"  - {error}" for error in validation_errors)
        raise SystemExit(f"Cannot compute paper intervals from incomplete results:\n{details}")

    out_dir = Path(args.out_dir)
    _write_csv(out_dir / "rate_wilson_intervals.csv", _rate_rows())
    _write_csv(out_dir / "median_bca_intervals.csv", _median_rows())
    _write_csv(out_dir / "paired_difference_bca_intervals.csv", _paired_difference_rows())
    _write_csv(
        out_dir / "discount_paired_difference_bca_intervals.csv",
        _discount_paired_difference_rows(),
    )
    _write_csv(
        out_dir / "discount_fixed_point_acceptance_comparison.csv",
        _discount_fixed_point_comparison_rows(),
    )
    print(f"Saved derived statistics to {out_dir}")


if __name__ == "__main__":
    main()

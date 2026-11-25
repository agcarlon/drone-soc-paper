"""
Paper experiment: discount-rate sweep in the pairwise-showcase setting.

Uses the same prior-tuned cooperative scenario as exp_paper_pairwise_showcase.
The only swept parameter is the discount rate r [s^-1] in
{0.7, 0.5, 0.3, 0.2, 0.1}. With Delta=30 s, this order increases the
per-cycle factor beta=exp(-r Delta). All seeds and methods for one rate are
completed and persisted before the next rate starts.
Results for r=0.3 s^-1 are reused from results/paper_pairwise_showcase when
available (--no-reuse disables this).

Usage:
    python3 -u experiments/exp_paper_discount_sweep.py --num-seeds 100
    python3 -u experiments/exp_paper_discount_sweep.py --num-seeds 100 --discounts 0.2 0.1

Output (results/paper_pairwise_discount_sweep/):
  per_trial_results.csv  — one row per (discount, method, seed)
  summary_results.csv    — statistics over fixed-point-accepted rows

Run plots:
  python3 plotting/plot_discount_sweep_outcomes.py
  python3 plotting/plot_discount_sweep_summary.py
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from experiments.exp_pairwise_showcase import (  # noqa: E402
    DEFAULT_METHODS,
    RuntimeOverrides,
    _initial_condition_summary,
    _run_method,
    _sample_scenario,
)
from experiments.paper_result_filters import accepted_fixed_point_rows  # noqa: E402
from experiments.result_io import (  # noqa: E402
    load_json_if_complete,
    write_csv_atomic,
    write_json_atomic,
)


DEFAULT_OUT_DIR = os.path.join(ROOT, "results", "paper_pairwise_discount_sweep")
DEFAULT_REUSE_DIR = os.path.join(ROOT, "results", "paper_pairwise_showcase")
DEFAULT_VARIANT = "prior_tuned"
DEFAULT_DISCOUNTS = (0.7, 0.5, 0.3, 0.2, 0.1)
DEFAULT_REUSE_DISCOUNT = 0.3


def _discount_label(discount: float) -> str:
    return f"discount_{discount:.3f}".replace(".", "p")


def _parse_discount_label(label: str) -> float:
    if not label.startswith("discount_"):
        raise ValueError(f"Not a discount directory: {label}")
    return float(label[len("discount_") :].replace("p", "."))


def _metrics_path(out_dir: Path, discount: float, seed: int, method: str) -> Path:
    return out_dir / _discount_label(discount) / f"seed_{seed:03d}" / method / "metrics.json"


def _reuse_metrics_path(reuse_dir: Path, seed: int, method: str) -> Path:
    return reuse_dir / f"seed_{seed:03d}" / method / "metrics.json"


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    write_json_atomic(path, payload)


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    write_csv_atomic(path, rows)


def _row_key(row: Dict[str, Any]) -> tuple[float, str, int]:
    return (float(row["discount"]), str(row["method"]), int(row["seed"]))


def _collect_existing_rows(out_dir: Path) -> Dict[tuple[float, str, int], Dict[str, Any]]:
    rows: Dict[tuple[float, str, int], Dict[str, Any]] = {}
    for metrics_path in out_dir.glob("discount_*/seed_*/order*/metrics.json"):
        try:
            discount = _parse_discount_label(metrics_path.parents[2].name)
            seed = int(metrics_path.parents[1].name.split("_", 1)[1])
            method = metrics_path.parent.name
            row = _normalize_row(_load_json(metrics_path), discount=discount, method=method, seed=seed)
        except (ValueError, IndexError, KeyError, json.JSONDecodeError):
            continue
        rows[_row_key(row)] = row
    return rows


def _write_aggregate_outputs(out_dir: Path, rows_by_key: Dict[tuple[float, str, int], Dict[str, Any]]) -> None:
    rows = [rows_by_key[key] for key in sorted(rows_by_key)]
    _write_csv(out_dir / "per_trial_results.csv", rows)
    _write_csv(
        out_dir / "summary_results.csv",
        _aggregate(accepted_fixed_point_rows(rows)),
    )


def _normalize_row(row: Dict[str, Any], *, discount: float, method: str, seed: int, reused_from: str = "") -> Dict[str, Any]:
    """Attach sweep-level fields and fill default columns for legacy rows."""
    out = dict(row)
    out["variant"] = DEFAULT_VARIANT
    out["discount"] = float(discount)
    out["method"] = str(method)
    out["seed"] = int(seed)
    out["reused_from"] = reused_from if reused_from else out.get("reused_from", "")

    final_error = float(out.get("final_error", np.nan))
    out.setdefault("success_1m", int(final_error <= 1.0))
    out.setdefault("success_2m", int(final_error <= 2.0))
    out.setdefault("success_5m", int(final_error <= 5.0))
    out.setdefault("success_10m", int(final_error <= 10.0))
    out.setdefault("stopped", int(str(out.get("stop_reason", "")).startswith("confidence_criterion_met")))
    out.setdefault("safe", int(not bool(out.get("collision_failure", 0))))
    out.setdefault("no_fly_violation", 0)
    out.setdefault("boundary_dwell_fraction", 0.0)
    out.setdefault("final_q_abs_error", np.nan)
    return out


def _aggregate(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_key: Dict[tuple[float, str], List[Dict[str, Any]]] = {}
    for row in rows:
        by_key.setdefault((float(row["discount"]), str(row["method"])), []).append(row)

    summary: List[Dict[str, Any]] = []
    for (discount, method), group in sorted(by_key.items()):
        errors = np.asarray([float(g["final_error"]) for g in group], dtype=float)
        traces = np.asarray([float(g["final_trace_sigma"]) for g in group], dtype=float)
        summary.append(
            {
                "discount": float(discount),
                "method": method,
                "num_trials": len(group),
                "stop_rate": float(np.mean([float(g["stopped"]) for g in group])),
                "success_1m_rate": float(np.mean([float(g["success_1m"]) for g in group])),
                "success_2m_rate": float(np.mean([float(g["success_2m"]) for g in group])),
                "success_5m_rate": float(np.mean([float(g["success_5m"]) for g in group])),
                "success_10m_rate": float(np.mean([float(g["success_10m"]) for g in group])),
                "safe_rate": float(np.mean([float(g["safe"]) for g in group])),
                "no_fly_violation_rate": float(np.mean([float(g["no_fly_violation"]) for g in group])),
                "median_boundary_dwell_fraction": float(np.median([float(g["boundary_dwell_fraction"]) for g in group])),
                "max_boundary_dwell_fraction": float(np.max([float(g["boundary_dwell_fraction"]) for g in group])),
                "median_final_error": float(np.median(errors)),
                "q10_final_error": float(np.quantile(errors, 0.1)),
                "q90_final_error": float(np.quantile(errors, 0.9)),
                "median_trace_sigma": float(np.median(traces)),
                "q10_trace_sigma": float(np.quantile(traces, 0.1)),
                "q90_trace_sigma": float(np.quantile(traces, 0.9)),
                "median_q_abs_error": float(np.nanmedian([float(g["final_q_abs_error"]) for g in group])),
                "median_energy": float(np.median([float(g["energy"]) for g in group])),
                "q10_energy": float(np.quantile([float(g["energy"]) for g in group], 0.1)),
                "q90_energy": float(np.quantile([float(g["energy"]) for g in group], 0.9)),
                "median_travel_distance": float(np.median([float(g["travel_distance"]) for g in group])),
                "median_runtime_sec": float(np.median([float(g["runtime_sec"]) for g in group])),
            }
        )
    return summary


def _runtime() -> RuntimeOverrides:
    return RuntimeOverrides(
        t_end=600.0,
        dt_meas=30.0,
        dx=60.0,
        dx_secondary=120.0,
        num_belief_discr=3,
        fp_max_iter=12,
    )


def _can_reuse(discount: float, reuse_discount: float, reuse_dir: Path | None, no_reuse: bool) -> bool:
    return (not no_reuse) and reuse_dir is not None and abs(float(discount) - float(reuse_discount)) < 5e-10


def _run_or_load_case(
    *,
    out_dir: Path,
    reuse_dir: Path | None,
    reuse_discount: float,
    no_reuse: bool,
    overwrite: bool,
    discount: float,
    seed: int,
    method: str,
    runtime: RuntimeOverrides,
) -> Dict[str, Any]:
    metrics_path = _metrics_path(out_dir, discount, seed, method)
    if metrics_path.exists() and not overwrite:
        cached = load_json_if_complete(
            metrics_path,
            ("final_error", "final_trace_sigma", "energy", "runtime_sec"),
        )
        if cached is not None:
            row = _normalize_row(cached, discount=discount, method=method, seed=seed)
            print(f"Skipping existing discount={discount:.3f}, seed={seed}, method={method}", flush=True)
            return row
        print(f"Cached metrics are incomplete for discount={discount:.3f}, seed={seed}, method={method}; rerunning.", flush=True)

    if _can_reuse(discount, reuse_discount, reuse_dir, no_reuse):
        assert reuse_dir is not None
        reuse_path = _reuse_metrics_path(reuse_dir, seed, method)
        if reuse_path.exists() and not overwrite:
            reused = load_json_if_complete(
                reuse_path,
                ("final_error", "final_trace_sigma", "energy", "runtime_sec"),
            )
            if reused is not None:
                row = _normalize_row(
                    reused,
                    discount=discount,
                    method=method,
                    seed=seed,
                    reused_from=str(reuse_path),
                )
                _write_json(metrics_path, row)
                print(f"Reused discount={discount:.3f}, seed={seed}, method={method} from {reuse_path}", flush=True)
                return row
            print(f"Reuse metrics are incomplete at {reuse_path}; running the case instead.", flush=True)

    scenario = _sample_scenario(int(seed), DEFAULT_VARIANT)
    method_dir = metrics_path.parent
    method_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"Running discount={discount:.3f}, method={method}, seed={seed}, "
        f"source=({scenario.source_pos[0]:.1f}, {scenario.source_pos[1]:.1f}), q={scenario.q:.1f}",
        flush=True,
    )
    row = _run_method(
        scenario=scenario,
        discount=float(discount),
        method=str(method),
        out_dir=str(method_dir),
        variant=DEFAULT_VARIANT,
        make_plots=False,
        make_contours=False,
        runtime=runtime,
    )
    row = _normalize_row(row, discount=discount, method=method, seed=seed)
    _write_json(metrics_path, row)
    return row


def run_discount_sweep(
    *,
    out_dir: str,
    seeds: Sequence[int],
    discounts: Sequence[float],
    methods: Sequence[str],
    reuse_dir: str | None,
    reuse_discount: float,
    no_reuse: bool,
    overwrite: bool,
) -> List[Dict[str, Any]]:
    if not seeds:
        raise ValueError("At least one seed is required")
    if not discounts:
        raise ValueError("At least one discount rate is required")
    if not methods:
        raise ValueError("At least one method is required")
    unknown_methods = set(methods) - set(DEFAULT_METHODS)
    if unknown_methods:
        raise ValueError(f"Unknown methods: {sorted(unknown_methods)}")
    if any(float(discount) <= 0.0 for discount in discounts):
        raise ValueError("Discount rates must be positive")
    if len({float(discount) for discount in discounts}) != len(discounts):
        raise ValueError("Discount rates must be unique")
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    reuse_path = Path(reuse_dir) if reuse_dir else None
    runtime = _runtime()

    _initial_condition_summary(
        seed=int(seeds[0]),
        discount=float(discounts[0]),
        out_dir=str(out_path),
        variant=DEFAULT_VARIANT,
        runtime=runtime,
    )

    total = len(seeds) * len(discounts) * len(methods)
    print(
        "Paper discount sweep:",
        f"variant={DEFAULT_VARIANT}",
        f"discounts={list(discounts)}",
        f"num_seeds={len(seeds)}",
        f"methods={list(methods)}",
        f"runs={total}",
        f"output={out_path}",
        flush=True,
    )
    if not no_reuse and reuse_path is not None:
        print(f"Reuse enabled for r={reuse_discount}: {reuse_path}", flush=True)

    rows_by_key = _collect_existing_rows(out_path)
    if rows_by_key:
        print(f"Found {len(rows_by_key)} existing run metrics in {out_path}", flush=True)

    for discount in discounts:
        for seed in seeds:
            for method in methods:
                row = _run_or_load_case(
                    out_dir=out_path,
                    reuse_dir=reuse_path,
                    reuse_discount=float(reuse_discount),
                    no_reuse=bool(no_reuse),
                    overwrite=bool(overwrite),
                    discount=float(discount),
                    seed=int(seed),
                    method=str(method),
                    runtime=runtime,
                )
                rows_by_key[_row_key(row)] = row
                _write_aggregate_outputs(out_path, rows_by_key)
                print(
                    f"  Result: r={float(row['discount']):.3f}, method={row['method']}, seed={row['seed']}, "
                    f"error={float(row['final_error']):.2f}, trace={float(row['final_trace_sigma']):.2f}, "
                    f"energy={float(row['energy']):.1f}, stopped={row['stopped']}, "
                    f"runtime={float(row['runtime_sec']):.1f}s",
                    flush=True,
                )

    print(f"Saved per-trial results to {out_path / 'per_trial_results.csv'}", flush=True)
    print(f"Saved summary results to {out_path / 'summary_results.csv'}", flush=True)
    return [rows_by_key[key] for key in sorted(rows_by_key)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the paper discount sweep in the pairwise-showcase setting")
    parser.add_argument("--num-seeds", type=int, default=100, help="Number of seeds starting from --seed-offset")
    parser.add_argument("--seed-offset", type=int, default=0, help="First seed index")
    parser.add_argument("--discounts", nargs="+", type=float, default=list(DEFAULT_DISCOUNTS))
    parser.add_argument("--methods", nargs="+", choices=DEFAULT_METHODS, default=list(DEFAULT_METHODS))
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--reuse-dir", default=DEFAULT_REUSE_DIR, help="Existing r=0.3 s^-1 paper-showcase results to reuse")
    parser.add_argument("--reuse-discount", type=float, default=DEFAULT_REUSE_DISCOUNT)
    parser.add_argument("--no-reuse", action="store_true", help="Disable reuse of existing r=0.3 s^-1 results")
    parser.add_argument("--overwrite", action="store_true", help="Rerun cases even if metrics.json already exists")
    args = parser.parse_args()

    seeds = tuple(range(int(args.seed_offset), int(args.seed_offset) + int(args.num_seeds)))
    run_discount_sweep(
        out_dir=str(args.out_dir),
        seeds=seeds,
        discounts=tuple(float(r) for r in args.discounts),
        methods=tuple(str(m) for m in args.methods),
        reuse_dir=None if args.reuse_dir == "" else str(args.reuse_dir),
        reuse_discount=float(args.reuse_discount),
        no_reuse=bool(args.no_reuse),
        overwrite=bool(args.overwrite),
    )


if __name__ == "__main__":
    main()

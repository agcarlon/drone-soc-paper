"""
Paper experiment: §5.5 — wind-aligned vs. free-φ HJB computational savings.

Compares single-drone planning with wind_aligned=True (φ fixed to α, 5-D HJB)
against wind_aligned=False (φ free, 6-D HJB).  Localisation accuracy and
energy are expected to be statistically indistinguishable; the 6-D grid
requires ~num_belief_discr times more HJB points and proportionally more compute.

Usage:
    python3 experiments/exp_wind_aligned_comparison.py --num-seeds 100

Output (results/wind_aligned_comparison/):
  wind_aligned_comparison.csv  — one row per (seed, wind_aligned flag)

Run plots:
  python3 plotting/plot_wind_aligned_comparison.py
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
import types
from typing import Any, Dict, List

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from _drone_soc_internal import construct_hjb_grid_axes  # noqa: E402
from drone_soc import find_pollutant_source  # noqa: E402
from experiments.exp_pairwise_showcase import (  # noqa: E402
    RuntimeOverrides,
    _build_case,
    _sample_scenario,
)
from infinite_horizon import InfiniteHorizonSingleDronePlanner  # noqa: E402
from experiments.fixed_point_diagnostics import fixed_point_metrics  # noqa: E402

DEFAULT_OUT_DIR = os.path.join(ROOT, "results", "wind_aligned_comparison")
DEFAULT_VARIANT = "prior_tuned"
DEFAULT_DISCOUNT = 0.3


def _runtime(num_belief_discr: int = 3) -> RuntimeOverrides:
    return RuntimeOverrides(
        t_end=600.0,
        dt_meas=30.0,
        dx=60.0,
        dx_secondary=120.0,
        num_belief_discr=num_belief_discr,
        fp_max_iter=12,
    )


def _make_timed_planner(wind_aligned: bool, fp_solver) -> InfiniteHorizonSingleDronePlanner:
    """Wrap a planner to record wall-clock time of each plan() call in .planning_times."""
    planner = InfiniteHorizonSingleDronePlanner(
        wind_aligned=wind_aligned, fp_solver=fp_solver
    )
    planner.planning_times: List[float] = []
    original_plan = planner.plan

    def _timed_plan(t_now, par, problem, s):
        t0 = time.perf_counter()
        result = original_plan(t_now, par, problem, s)
        planner.planning_times.append(time.perf_counter() - t0)
        return result

    planner.plan = types.MethodType(
        lambda self, t_now, par, problem, s: _timed_plan(t_now, par, problem, s),
        planner,
    )
    return planner


def _run_seed(
    seed: int,
    *,
    wind_aligned: bool,
    variant: str,
    discount: float,
    runtime: RuntimeOverrides,
    output_dir: str,
) -> Dict[str, Any]:
    scenario = _sample_scenario(seed, variant)
    par, problem, state, fp_solver = _build_case(
        scenario=scenario,
        discount=discount,
        out_dir=output_dir,
        variant=variant,
        runtime=runtime,
    )

    planner = _make_timed_planner(wind_aligned=wind_aligned, fp_solver=fp_solver)

    grid_axes = construct_hjb_grid_axes(par, problem, state, num_drones=1, wind_aligned=wind_aligned)
    grid_size = int(np.prod([len(a) for a in grid_axes]))

    traj_arr, log = find_pollutant_source(par, problem, state, planner=planner)

    theta_true = np.asarray(problem.pollutant.source, dtype=float)
    mu_final = np.asarray(log["beliefs"][-1].mu, dtype=float)
    dist_final = float(np.linalg.norm(mu_final - theta_true))

    energy = float(log.get("energy_total", 0.0))
    stopped = log.get("stop_reason", "").startswith("confidence_criterion_met")

    planning_times = getattr(planner, "planning_times", [])
    first_plan_time = planning_times[0] if planning_times else float("nan")
    total_plan_time = sum(planning_times)

    row = dict(
        seed=seed,
        wind_aligned=wind_aligned,
        dist_final=dist_final,
        energy=energy,
        stopped=stopped,
        first_plan_time=first_plan_time,
        total_plan_time=total_plan_time,
        num_plan_calls=len(planning_times),
        grid_size=grid_size,
        hjb_dims=len(grid_axes),
    )
    row.update(fixed_point_metrics(fp_solver, output_dir))
    return row


def _summarise(rows: List[Dict[str, Any]], label: str) -> None:
    if not rows:
        raise ValueError(f"Cannot summarize an empty group: {label}")
    n = len(rows)
    stop_rate = np.mean([r["stopped"] for r in rows])
    dists = [r["dist_final"] for r in rows]
    energies = [r["energy"] for r in rows]
    t1s = [r["first_plan_time"] for r in rows]
    gs = rows[0]["grid_size"] if rows else "?"
    dims = rows[0]["hjb_dims"] if rows else "?"
    print(f"\n{label}  (n={n})")
    print(f"  HJB dims / grid size : {dims}-D  /  {gs:,} points")
    print(f"  Stop rate            : {stop_rate:.0%}")
    print(f"  Dist to source [m]   : median={np.median(dists):.1f}  "
          f"q25={np.quantile(dists,0.25):.1f}  q75={np.quantile(dists,0.75):.1f}")
    print(f"  Energy [m]           : median={np.median(energies):.0f}  "
          f"q25={np.quantile(energies,0.25):.0f}  q75={np.quantile(energies,0.75):.0f}")
    print(f"  First plan time [s]  : median={np.median(t1s):.2f}  "
          f"q25={np.quantile(t1s,0.25):.2f}  q75={np.quantile(t1s,0.75):.2f}")


def _write_rows(path: str, rows_by_key: Dict[tuple[int, bool], Dict[str, Any]]) -> None:
    """Atomically persist all completed controller cases for reliable resume."""
    rows = [rows_by_key[key] for key in sorted(rows_by_key)]
    if not rows:
        raise ValueError("No wind-alignment rows are available to write")
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    temporary_path = f"{path}.tmp"
    with open(temporary_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary_path, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-seeds", type=int, default=100)
    parser.add_argument("--variant", type=str, default=DEFAULT_VARIANT)
    parser.add_argument("--discount", type=float, default=DEFAULT_DISCOUNT)
    parser.add_argument("--num-belief-discr", type=int, default=3,
                        help="Grid points per belief axis. 6-D grid has this factor more points.")
    parser.add_argument("--out-dir", type=str, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    if args.num_seeds <= 0:
        parser.error("--num-seeds must be positive")
    if args.num_belief_discr < 2:
        parser.error("--num-belief-discr must be at least 2")
    if args.discount <= 0.0:
        parser.error("--discount must be positive")

    os.makedirs(args.out_dir, exist_ok=True)
    runtime = _runtime(num_belief_discr=args.num_belief_discr)

    csv_path = os.path.join(args.out_dir, "wind_aligned_comparison.csv")
    rows_by_key: Dict[tuple[int, bool], Dict[str, Any]] = {}
    if os.path.exists(csv_path):
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                # Restore types lost in CSV round-trip: numeric fields are read as strings.
                for key in ("dist_final", "energy", "first_plan_time", "runtime_sec",
                            "hjb_dims", "grid_size", "snr0"):
                    if key in row:
                        try:
                            row[key] = float(row[key])
                        except (ValueError, TypeError):
                            pass
                for key in ("stopped", "success"):
                    if key in row:
                        row[key] = row[key].strip().lower() in ("true", "1", "yes")
                seed = int(row["seed"])
                aligned_text = str(row.get("wind_aligned", "")).strip().lower()
                if aligned_text not in ("true", "1", "false", "0"):
                    raise ValueError(
                        f"Invalid wind_aligned value for seed {seed}: {aligned_text!r}"
                    )
                aligned = aligned_text in ("true", "1")
                key = (seed, aligned)
                if key in rows_by_key:
                    raise ValueError(f"Duplicate cached result for seed={seed}, wind_aligned={aligned}")
                rows_by_key[key] = row
        done_seeds = {
            seed
            for seed in range(args.num_seeds)
            if (seed, True) in rows_by_key and (seed, False) in rows_by_key
        }
        if done_seeds:
            print(f"Resuming: {len(done_seeds)}/{args.num_seeds} seeds already completed")

    for seed in range(args.num_seeds):
        for aligned, directory in ((True, "wind_aligned"), (False, "full_angle")):
            if (seed, aligned) in rows_by_key:
                continue
            print(f"Seed {seed:3d}  wind_aligned={str(aligned):5s} ...", end="  ", flush=True)
            row = _run_seed(
                seed, wind_aligned=aligned,
                variant=args.variant, discount=args.discount, runtime=runtime,
                output_dir=os.path.join(args.out_dir, f"seed_{seed:03d}", directory),
            )
            rows_by_key[(seed, aligned)] = row
            _write_rows(csv_path, rows_by_key)
            print(f"dist={row['dist_final']:.0f}m  E={row['energy']:.0f}m  "
                  f"t1={row['first_plan_time']:.2f}s  "
                  f"stopped={'yes' if row['stopped'] else 'no'}")

    _write_rows(csv_path, rows_by_key)
    print(f"\nSaved CSV → {csv_path}")

    rows_wa = [rows_by_key[(seed, True)] for seed in range(args.num_seeds)]
    rows_nwa = [rows_by_key[(seed, False)] for seed in range(args.num_seeds)]

    _summarise(rows_wa, "wind_aligned=True  (5-D HJB)")
    _summarise(rows_nwa, "wind_aligned=False (6-D HJB)")

    t_wa = np.median([r["first_plan_time"] for r in rows_wa])
    t_nwa = np.median([r["first_plan_time"] for r in rows_nwa])
    print(f"\nPlan-time speedup (wind-align vs no-align): {t_nwa/t_wa:.2f}×  "
          f"(expected ~{args.num_belief_discr}× from grid size)")

    print("\nRun  python3 plotting/plot_wind_aligned_comparison.py  to generate the figure.")


if __name__ == "__main__":
    main()

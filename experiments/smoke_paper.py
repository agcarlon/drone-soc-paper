"""Reduced end-to-end execution of every paper controller family."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from drone_soc import find_pollutant_source
from experiments.baseline_scenario import (
    BaselineScenarioSpec,
    PatrolOnlyPlanner,
    _sample_scenario as sample_baseline,
    make_baseline_case,
)
from experiments.exp_pairwise_showcase import RuntimeOverrides, _build_case, _sample_scenario
from experiments.fixed_point_diagnostics import fixed_point_metrics
from greedy_eig import GreedyEIGPlanner
from infinite_horizon import InfiniteHorizonSingleDronePlanner, InfiniteHorizonTwoDronesPlanner


def _prepare_short_run(par, out_dir: Path, *, dt: float) -> None:
    par.t_end = float(par.meas_times[-1])
    par.dt = float(dt)
    par.dyn_noise_std = 0.0
    par.stop_when_confident = False
    par.output_dir = str(out_dir)
    par.verbose_planning = False


def _metrics(method: str, par, problem, state, planner, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    traj, log = find_pollutant_source(par, problem, state, planner)
    final = log["beliefs"][-1]
    row = {
        "method": method,
        "final_error": float(np.linalg.norm(final.mu - problem.pollutant.source)),
        "energy": float(log["energy_total"]),
        "steps": int(len(log["controls"])),
        "measurement_count": int(len(log["measurement_times"])),
    }
    solver = getattr(planner, "fp_solver", None)
    if solver is not None:
        row.update(fixed_point_metrics(solver, out_dir))
    with (out_dir / "metrics.json").open("w") as handle:
        json.dump(row, handle, indent=2)
    return row


def _baseline(out_dir: Path) -> list[dict]:
    spec = BaselineScenarioSpec(output_dir=str(out_dir), snr0_levels=(100.0,), seeds=(0,))
    scenario = sample_baseline(0, 100.0, spec)
    rows = []
    for name in ("order1", "greedy_eig", "patrol"):
        case_dir = out_dir / name
        par, problem, state, solver = make_baseline_case(scenario, str(case_dir))
        par.meas_times = np.array([0.0, 5.0, 10.0])
        par.dt_hjb = 2.5
        par.dx = 250.0
        par.num_belief_discr = 3
        solver.max_iter = 8
        _prepare_short_run(par, case_dir, dt=0.5)
        planner = {
            "order1": InfiniteHorizonSingleDronePlanner(wind_aligned=True, fp_solver=solver),
            "greedy_eig": GreedyEIGPlanner(num_restarts=2),
            "patrol": PatrolOnlyPlanner(),
        }[name]
        rows.append({"experiment": "baseline", **_metrics(name, par, problem, state, planner, case_dir)})
    return rows


def _cooperative(out_dir: Path) -> list[dict]:
    runtime = RuntimeOverrides(
        t_end=60.0,
        dt_meas=30.0,
        dx=240.0,
        dx_secondary=360.0,
        num_belief_discr=3,
        fp_max_iter=8,
        dt_hjb=10.0,
    )
    rows = []
    for discount in (0.3, 0.1):
        for method in ("order1", "order2"):
            case_dir = out_dir / f"discount_{discount:g}" / method
            scenario = _sample_scenario(0, "prior_tuned")
            par, problem, state, solver = _build_case(
                scenario=scenario,
                discount=discount,
                out_dir=str(case_dir),
                variant="prior_tuned",
                runtime=runtime,
            )
            _prepare_short_run(par, case_dir, dt=1.0)
            planner = (
                InfiniteHorizonSingleDronePlanner(wind_aligned=True, fp_solver=solver)
                if method == "order1"
                else InfiniteHorizonTwoDronesPlanner(wind_aligned=True, fp_solver=solver)
            )
            rows.append({
                "experiment": "cooperative",
                "discount": discount,
                **_metrics(method, par, problem, state, planner, case_dir),
            })
    return rows


def _write_outputs(out_dir: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (out_dir / "smoke_results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.2))
    labels = [f"{row['experiment']}:{row['method']}" for row in rows]
    x = np.arange(len(rows))
    axes[0].bar(x, [row["final_error"] for row in rows])
    axes[0].set_ylabel("Final error [m]")
    axes[1].bar(x, [row["energy"] for row in rows])
    axes[1].set_ylabel(r"Control-effort proxy [m$^2$s$^{-1}$]")
    for ax in axes:
        ax.set_xticks(x, labels, rotation=55, ha="right")
        ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "smoke_summary.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "smoke_summary.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="results/smoke")
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = _baseline(out_dir / "baseline") + _cooperative(out_dir / "cooperative")
    _write_outputs(out_dir, rows)
    print(f"Smoke reproduction complete: {out_dir}")


if __name__ == "__main__":
    main()

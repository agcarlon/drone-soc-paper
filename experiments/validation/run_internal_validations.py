"""Run internal model and numerical validations outside the paper workflow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fixed_point import PicardFixedPointSolver
from hjb import OCProblem, solve_hjb


def _periodic_vs_no_continuation() -> dict:
    beta = np.exp(-0.3 * 5.0)
    jump_reward = 2.0

    def bellman(value):
        previous = 0.0 if value is None else float(np.asarray(value)[0])
        return np.array([beta * (previous - jump_reward)])

    fixed, errors = PicardFixedPointSolver(tol=1e-12, max_iter=100).solve(bellman)
    no_continuation = bellman(None)[0]
    exact = -beta * jump_reward / (1.0 - beta)
    return {
        "fixed_point": float(fixed[0]),
        "analytic_fixed_point": float(exact),
        "no_continuation": float(no_continuation),
        "absolute_error": float(abs(fixed[0] - exact)),
        "iterations": len(errors),
    }


def _block_vs_full_information(rng: np.random.Generator, samples: int) -> dict:
    differences = []
    for _ in range(samples):
        A = rng.normal(size=(3, 3))
        covariance = A @ A.T + np.eye(3)
        J = rng.normal(size=(4, 3))
        full = 0.5 * np.linalg.slogdet(np.eye(3) + covariance @ (J.T @ J))[1]
        block_cov = np.zeros_like(covariance)
        block_cov[:2, :2] = covariance[:2, :2]
        block_cov[2, 2] = covariance[2, 2]
        block_info = np.zeros((3, 3))
        block_info[:2, :2] = (J.T @ J)[:2, :2]
        block_info[2, 2] = (J.T @ J)[2, 2]
        block = 0.5 * np.linalg.slogdet(np.eye(3) + block_cov @ block_info)[1]
        differences.append(block - full)
    values = np.asarray(differences)
    return {
        "samples": samples,
        "median_block_minus_full": float(np.median(values)),
        "max_absolute_difference": float(np.max(np.abs(values))),
    }


def _grid_sensitivity() -> dict:
    def solve(dx: float, dt: float) -> tuple[np.ndarray, np.ndarray]:
        axis = np.arange(-1.0, 1.0 + 1e-12, dx)
        problem = OCProblem(
            dynamics=lambda x, u, t: 0.0 * x,
            running_cost=lambda x, u, t: x[0] ** 2,
            terminal_cost=lambda x: 0.5 * x[0] ** 2,
            grid_axes=(axis,),
            dt=dt,
            T=1.0,
            discount=0.2,
            analytic_minimizer=lambda mesh, grad, t: (np.zeros_like(mesh), None),
            num_spatial_axes=1,
        )
        return axis, solve_hjb(problem)[1][0]

    coarse_axis, coarse = solve(0.5, 0.25)
    fine_axis, fine = solve(0.25, 0.125)
    restricted = np.interp(coarse_axis, fine_axis, fine)
    return {"max_absolute_difference": float(np.max(np.abs(coarse - restricted)))}


def _stopping_calibration(rng: np.random.Generator, samples: int) -> dict:
    # Under a calibrated Gaussian observation model, the normalized squared
    # residual has mean one. Report the false-alarm probability for threshold 3.
    scores = rng.normal(size=(samples, 3)) ** 2
    batch_scores = np.mean(scores, axis=1)
    return {
        "samples": samples,
        "mean_score": float(np.mean(batch_scores)),
        "fraction_above_3": float(np.mean(batch_scores > 3.0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--out-dir", default="results/internal_validation")
    args = parser.parse_args()
    samples = 100 if args.profile == "smoke" else 100_000
    rng = np.random.default_rng(20260810)
    report = {
        "profile": args.profile,
        "periodic_vs_no_continuation": _periodic_vs_no_continuation(),
        "block_vs_full_information": _block_vs_full_information(rng, samples),
        "grid_sensitivity": _grid_sensitivity(),
        "stopping_calibration": _stopping_calibration(rng, samples),
    }
    out_dir = Path(args.out_dir) / args.profile
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "validation_results.json"
    out_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Internal validation complete: {out_path}")


if __name__ == "__main__":
    main()

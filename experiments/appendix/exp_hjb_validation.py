"""
Experiment: HJB Validation Plots
================================

Generate plots that mirror the unit tests for the HJB solver, so they can
be included in the report. In particular, we reproduce the diffusion
grid-refinement test: solve a simple 1D stochastic control problem on a
coarse and a refined grid, then compare the value functions at t=0.
"""

import os
import sys
from typing import Tuple

import numpy as np
from matplotlib import pyplot as plt

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hjb import OCProblem, solve_hjb


def _quad_analytic_minimizer(mesh: np.ndarray, gradV: np.ndarray, t: float):
    """
    Analytic minimizer for quadratic control effort:
    running_cost ~ 0.5 * ||u||^2 and dynamics x_dot = u.
    The minimizer is u* = -gradV.
    """
    a_opt = -gradV
    return a_opt, None


def _make_diffusion_problem(dx: float, dt: float) -> OCProblem:
    """
    Build a small 1D HJB problem with diffusion, matching tests/test_hjb.py.
    """
    x_axis = np.arange(-1.0, 1.0 + 1e-9, dx)
    axes = (x_axis,)

    def dynamics(x, u, t):
        # Simple integrator x_dot = u
        return u

    def running_cost(x, u, t):
        # Quadratic effort plus a mild state penalty
        return 0.5 * np.sum(u**2, axis=0) + 0.5 * x[0] ** 2

    def terminal_cost(x):
        return 0.5 * x[0] ** 2

    return OCProblem(
        dynamics=dynamics,
        running_cost=running_cost,
        terminal_cost=terminal_cost,
        grid_axes=axes,
        dt=dt,
        T=1.0,
        discount=0.1,
        diffusion_coeff=0.05,
        analytic_minimizer=_quad_analytic_minimizer,
        measurement_times=None,
        measurement_jump=None,
        num_spatial_axes=1,
    )


def run_hjb_grid_refinement_experiment(output_dir: str = os.path.join("results", "hjb_validation")) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Reproduce the diffusion grid-refinement comparison and save a plot.

    Returns:
        x_coarse: spatial grid for the coarse problem
        V0_coarse: value function at t=0 on coarse grid
        V0_fine_on_coarse: refined solution restricted to coarse grid nodes
    """
    os.makedirs(output_dir, exist_ok=True)

    # Coarse and fine problems with aligned domains (as in tests/test_hjb.py)
    prob_coarse = _make_diffusion_problem(dx=0.5, dt=0.2)
    prob_fine = _make_diffusion_problem(dx=0.25, dt=0.1)

    _, V_coarse, axes_coarse = solve_hjb(prob_coarse)
    _, V_fine, axes_fine = solve_hjb(prob_fine)

    x_coarse = axes_coarse[0]
    x_fine = axes_fine[0]

    # Restrict fine solution to coarse grid nodes
    V0_coarse = V_coarse[0]
    V0_fine_on_coarse = V_fine[0][::2]

    # Plot both solutions and their difference
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.plot(x_coarse, V0_coarse, "o-", label="coarse grid")
    ax1.plot(x_coarse, V0_fine_on_coarse, "s--", label="refined (restricted)")
    ax1.set_xlabel(r"$x$")
    ax1.set_ylabel(r"$V(t=0,x)$")
    ax1.set_title("HJB value function on coarse vs refined grid")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    diff = V0_coarse - V0_fine_on_coarse
    ax2.plot(x_coarse, diff, "k-o")
    ax2.set_xlabel(r"$x$")
    ax2.set_ylabel(r"Difference")
    ax2.set_title("Difference between coarse and refined values")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path = os.path.join(output_dir, "hjb_grid_refinement.pdf")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved HJB grid-refinement plot to {out_path}")

    return x_coarse, V0_coarse, V0_fine_on_coarse


if __name__ == "__main__":
    run_hjb_grid_refinement_experiment()


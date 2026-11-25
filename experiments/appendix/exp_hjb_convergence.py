"""
Experiment: HJB Discretisation Convergence
==========================================

Study the convergence of the 1D diffusive HJB solution with respect to
spatial grid size dx and time step dt_hjb. We use a simple stochastic
control problem with quadratic running/terminal costs and dynamics x_dot = u,
and estimate convergence rates via Richardson extrapolation.
"""

import os
import sys
from typing import Dict, Tuple, List

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
    Build a small 1D HJB problem with diffusion, mirroring tests/test_hjb.py.
    """
    x_axis = np.arange(-1.0, 1.0 + 1e-9, dx)
    axes = (x_axis,)

    def dynamics(x, u, t):
        return u

    def running_cost(x, u, t):
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


def _solve_value(dx: float, dt: float) -> Tuple[np.ndarray, np.ndarray]:
    """Solve the HJB and return (x_grid, V(t=0,x))."""
    prob = _make_diffusion_problem(dx=dx, dt=dt)
    _, V_time, axes = solve_hjb(prob)
    x = axes[0]
    V0 = V_time[0]
    return x, V0


def _restrict_to_grid(x_fine: np.ndarray, V_fine: np.ndarray, x_coarse: np.ndarray) -> np.ndarray:
    """Restrict a fine-grid solution to a coarser grid via nearest-neighbour matching."""
    indices = [int(np.argmin(np.abs(x_fine - xc))) for xc in x_coarse]
    return V_fine[indices]


def _compute_spatial_convergence(
    output_dir: str,
    dt_ref: float = 0.05,
) -> Tuple[List[float], List[float], List[float]]:
    """
    Convergence of V(t=0,·) with respect to dx, keeping dt fixed.
    Returns (dx_list, errors_vs_ref, rates_richardson).
    """
    # Geometric grid sizes; finest first
    dx_min = 1/32  # 1/16
    levels = 6
    dx_list = [dx_min * (2 ** k) for k in range(levels)]  # 1/16, 1/8, 1/4, 1/2

    x_solutions: Dict[float, np.ndarray] = {}
    V_solutions: Dict[float, np.ndarray] = {}

    for dx in dx_list:
        x, V0 = _solve_value(dx=dx, dt=dt_ref)
        x_solutions[dx] = x
        V_solutions[dx] = V0

    # Use finest grid as reference
    dx_ref = dx_list[0]
    x_ref = x_solutions[dx_ref]
    V_ref = V_solutions[dx_ref]

    errors = []
    for dx in dx_list:
        x_c = x_solutions[dx]
        V_c = V_solutions[dx]
        V_ref_on_c = _restrict_to_grid(x_ref, V_ref, x_c)
        err = float(np.max(np.abs(V_c - V_ref_on_c)))
        errors.append(err)

    # Richardson rate estimates using consecutive differences
    rates = [np.nan] * len(dx_list)
    for k in range(2, len(dx_list)):
        dx_k = dx_list[k]
        dx_km1 = dx_list[k - 1]
        dx_km2 = dx_list[k - 2]
        x_k = x_solutions[dx_k]
        x_km1 = x_solutions[dx_km1]
        x_km2 = x_solutions[dx_km2]
        V_k = V_solutions[dx_k]
        V_km1 = V_solutions[dx_km1]
        V_km2 = V_solutions[dx_km2]
        # Differences between consecutive levels, evaluated on the coarser grid
        Vk_on_km1 = _restrict_to_grid(x_k, V_k, x_km1)
        diff1 = float(np.max(np.abs(Vk_on_km1 - V_km1)))
        Vkm1_on_km2 = _restrict_to_grid(x_km1, V_km1, x_km2)
        diff2 = float(np.max(np.abs(Vkm1_on_km2 - V_km2)))
        rates[k] = np.log(diff2 / diff1) / np.log(2.0)

    # Plot error vs dx (log-log)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.loglog(dx_list[1:], errors[1:], "o-", label=r"$\|V_{\Delta x} - V_{\mathrm{ref}}\|_\infty$")
    ax.set_xlabel(r"Grid size $\Delta x$")
    ax.set_ylabel(r"Error in $V(t=0,\cdot)$")
    ax.set_title("Spatial convergence of diffusive HJB")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    path_err = os.path.join(output_dir, "hjb_spatial_convergence.pdf")
    fig.savefig(path_err, dpi=150)
    plt.close(fig)

    # Plot estimated rates vs dx (skip first two levels)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(dx_list[2:], rates[2:], "s-")
    ax.set_xscale("log")
    ax.set_xlabel(r"Grid size $\Delta x$")
    ax.set_ylabel("Estimated convergence rate $p$")
    ax.set_title("Estimated spatial convergence rate via Richardson")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    path_rate = os.path.join(output_dir, "hjb_spatial_rate.pdf")
    fig.savefig(path_rate, dpi=150)
    plt.close(fig)

    print(f"Saved spatial convergence plots to {path_err} and {path_rate}")
    return dx_list, errors, rates


def _compute_temporal_convergence(
    output_dir: str,
    dx_ref: float = 0.0625,
) -> Tuple[List[float], List[float], List[float]]:
    """
    Convergence of V(t=0,·) with respect to dt_hjb, keeping dx fixed.
    Returns (dt_list, errors_vs_ref, rates_richardson).
    """
    # Geometric time steps with T=1, so that T/dt is integer
    dt_min = 1.0 / 512.0
    levels = 6
    dt_list = [dt_min * (2 ** k) for k in range(levels)]  # 1/64,1/32,1/16,1/8

    x_solutions: Dict[float, np.ndarray] = {}
    V_solutions: Dict[float, np.ndarray] = {}

    for dt in dt_list:
        x, V0 = _solve_value(dx=dx_ref, dt=dt)
        x_solutions[dt] = x
        V_solutions[dt] = V0

    # Use finest time step as reference
    dt_ref = dt_list[0]
    x_ref = x_solutions[dt_ref]
    V_ref = V_solutions[dt_ref]

    errors = []
    for dt in dt_list:
        x_c = x_solutions[dt]
        V_c = V_solutions[dt]
        V_ref_on_c = _restrict_to_grid(x_ref, V_ref, x_c)
        err = float(np.max(np.abs(V_c - V_ref_on_c)))
        errors.append(err)

    # Richardson rate estimates using consecutive differences
    rates = [np.nan] * len(dt_list)
    for k in range(2, len(dt_list)):
        dt_k = dt_list[k]
        dt_km1 = dt_list[k - 1]
        dt_km2 = dt_list[k - 2]
        x_k = x_solutions[dt_k]
        x_km1 = x_solutions[dt_km1]
        x_km2 = x_solutions[dt_km2]
        V_k = V_solutions[dt_k]
        V_km1 = V_solutions[dt_km1]
        V_km2 = V_solutions[dt_km2]
        Vk_on_km1 = _restrict_to_grid(x_k, V_k, x_km1)
        diff1 = float(np.max(np.abs(Vk_on_km1 - V_km1)))
        Vkm1_on_km2 = _restrict_to_grid(x_km1, V_km1, x_km2)
        diff2 = float(np.max(np.abs(Vkm1_on_km2 - V_km2)))
        rates[k] = np.log(diff2 / diff1) / np.log(2.0)

    # Plot error vs dt (log-log)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.loglog(dt_list[1:], errors[1:], "o-", label=r"$\|V_{\Delta t} - V_{\mathrm{ref}}\|_\infty$")
    ax.set_xlabel(r"Time step $\Delta t_{\mathrm{HJB}}$")
    ax.set_ylabel(r"Error in $V(t=0,\cdot)$")
    ax.set_title("Temporal convergence of diffusive HJB")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    path_err = os.path.join(output_dir, "hjb_temporal_convergence.pdf")
    fig.savefig(path_err, dpi=150)
    plt.close(fig)

    # Plot estimated rates vs dt (skip first two levels)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(dt_list[2:], rates[2:], "s-")
    ax.set_xscale("log")
    ax.set_xlabel(r"Time step $\Delta t_{\mathrm{HJB}}$")
    ax.set_ylabel("Estimated convergence rate $p$")
    ax.set_title("Estimated temporal convergence rate via Richardson")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    path_rate = os.path.join(output_dir, "hjb_temporal_rate.pdf")
    fig.savefig(path_rate, dpi=150)
    plt.close(fig)

    print(f"Saved temporal convergence plots to {path_err} and {path_rate}")
    return dt_list, errors, rates


def run_hjb_convergence_experiments(output_dir: str = os.path.join("results", "hjb_validation")):
    os.makedirs(output_dir, exist_ok=True)
    print("=== Spatial convergence study ===")
    _compute_spatial_convergence(output_dir=output_dir)
    print("=== Temporal convergence study ===")
    _compute_temporal_convergence(output_dir=output_dir)


if __name__ == "__main__":
    run_hjb_convergence_experiments()

"""
Greedy one-step EIG maximizer for gas source localization.

At each planning cycle this planner solves:

    X_target = argmax_{||X_i - X_i_cur|| <= v_max * Delta} EIG(X_target; pi^-)

where EIG is the linearized-Gaussian expected information gain from
Proposition 1 of the paper.  Once target positions are found each drone
is steered straight toward its target at v_max, with the same pairwise
collision-avoidance correction used by the order-1 controller (Eq. 22).

This serves as a comparison baseline that isolates the value of multi-step
HJB lookahead versus single-step greedy optimization.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.optimize import minimize

from drone_soc import (
    ControlPlanner,
    DroneProblemState,
    Params,
    ProblemSpec,
    compute_collision_gradients,
)


def _compute_eig(
    X_abs: np.ndarray,
    *,
    belief_mu: np.ndarray,
    belief_sigma: np.ndarray,
    belief_q_log: Optional[float],
    belief_q_log_var: Optional[float],
    problem: ProblemSpec,
    par: Params,
) -> float:
    """
    Compute the linearized-Gaussian EIG (Eq. 12 in the paper) for a given
    team configuration X_abs (shape: (M, 2)).

    Returns a scalar EIG value.
    """
    eps = 1e-30
    inv_noise_var = 1.0 / max(float(par.noise_std) ** 2, eps)
    intensity = float(np.exp(belief_q_log)) if belief_q_log is not None else None

    J_theta = problem.pollutant.get_concentration_gradient_wrt_theta(
        X_abs, 0.0, theta=belief_mu, intensity=intensity
    )
    J_theta = np.asarray(J_theta, dtype=float).reshape(X_abs.shape[0], 2)

    # Fisher information for theta (2x2 matrix)
    F_theta = inv_noise_var * (J_theta.T @ J_theta)

    Sigma = np.asarray(belief_sigma, dtype=float)
    det_inner = np.linalg.det(np.eye(2) + Sigma @ F_theta)
    eig_theta = 0.5 * np.log(max(det_inner, eps))

    eig_zeta = 0.0
    if belief_q_log_var is not None:
        dC_dlogq = problem.pollutant.get_concentration_derivative_wrt_logq(
            X_abs, 0.0, theta=belief_mu, intensity=intensity
        )
        dC_dlogq = np.asarray(dC_dlogq, dtype=float).ravel()
        F_zeta = float(inv_noise_var * np.sum(dC_dlogq ** 2))
        eig_zeta = 0.5 * np.log(max(1.0 + float(belief_q_log_var) * F_zeta, eps))

    return float(eig_theta + eig_zeta)


def _neg_eig_and_grad(
    x_flat: np.ndarray,
    *,
    belief_mu: np.ndarray,
    belief_sigma: np.ndarray,
    belief_q_log: Optional[float],
    belief_q_log_var: Optional[float],
    problem: ProblemSpec,
    par: Params,
    num_drones: int,
) -> tuple[float, np.ndarray]:
    """Return (-EIG, -grad_EIG) for use with scipy minimize."""
    X_abs = x_flat.reshape(num_drones, 2)
    h = 1e-4
    eig0 = _compute_eig(
        X_abs,
        belief_mu=belief_mu,
        belief_sigma=belief_sigma,
        belief_q_log=belief_q_log,
        belief_q_log_var=belief_q_log_var,
        problem=problem,
        par=par,
    )
    grad = np.zeros_like(x_flat)
    for k in range(len(x_flat)):
        x_plus = x_flat.copy()
        x_plus[k] += h
        eig_plus = _compute_eig(
            x_plus.reshape(num_drones, 2),
            belief_mu=belief_mu,
            belief_sigma=belief_sigma,
            belief_q_log=belief_q_log,
            belief_q_log_var=belief_q_log_var,
            problem=problem,
            par=par,
        )
        grad[k] = (eig_plus - eig0) / h
    return -eig0, -grad


def _greedy_eig_targets(
    *,
    current_positions: np.ndarray,
    belief_mu: np.ndarray,
    belief_sigma: np.ndarray,
    belief_q_log: Optional[float],
    belief_q_log_var: Optional[float],
    problem: ProblemSpec,
    par: Params,
    meas_interval: float,
    num_restarts: int = 8,
    rng: Optional[np.random.Generator] = None,
    return_diagnostics: bool = False,
) -> np.ndarray | tuple[np.ndarray, list[dict]]:
    """
    Maximize EIG over reachable target positions for each drone.

    Returns optimal target positions (M, 2).
    """
    num_drones = current_positions.shape[0]
    v_max = float(par.v_max)
    radius = v_max * meas_interval

    if rng is None:
        rng = np.random.default_rng()

    kwargs = dict(
        belief_mu=belief_mu,
        belief_sigma=belief_sigma,
        belief_q_log=belief_q_log,
        belief_q_log_var=belief_q_log_var,
        problem=problem,
        par=par,
        num_drones=num_drones,
    )

    best_val = np.inf
    best_x = current_positions.copy()
    diagnostics: list[dict] = []
    domain = np.asarray(problem.domain, dtype=float)
    bounds = [tuple(domain[axis]) for _ in range(num_drones) for axis in range(2)]

    def _reachable(x_flat: np.ndarray) -> np.ndarray:
        offsets = x_flat.reshape(num_drones, 2) - current_positions
        return radius**2 - np.sum(offsets**2, axis=1)

    def _reachable_jac(x_flat: np.ndarray) -> np.ndarray:
        offsets = x_flat.reshape(num_drones, 2) - current_positions
        jac = np.zeros((num_drones, 2 * num_drones), dtype=float)
        for drone_idx in range(num_drones):
            jac[drone_idx, 2 * drone_idx : 2 * drone_idx + 2] = -2.0 * offsets[drone_idx]
        return jac

    for restart in range(num_restarts):
        if restart == 0:
            x0 = current_positions.copy().ravel()
        else:
            # Random start within reachable ball for each drone
            angles = rng.uniform(0, 2 * np.pi, num_drones)
            rs = rng.uniform(0, radius, num_drones)
            offsets = np.stack([rs * np.cos(angles), rs * np.sin(angles)], axis=1)
            x0_positions = np.clip(current_positions + offsets, domain[:, 0], domain[:, 1])
            # Domain clipping can only shorten an offset when the current point is feasible.
            x0 = x0_positions.ravel()

        def _objective(x_flat: np.ndarray) -> tuple[float, np.ndarray]:
            return _neg_eig_and_grad(x_flat, **kwargs)

        result = minimize(
            _objective,
            x0,
            jac=True,
            method="SLSQP",
            bounds=bounds,
            constraints={"type": "ineq", "fun": _reachable, "jac": _reachable_jac},
            options={"maxiter": 200, "ftol": 1e-9},
        )
        feasible = bool(
            np.all(np.isfinite(result.x))
            and np.all(_reachable(np.asarray(result.x, dtype=float)) >= -1e-7)
        )
        diagnostics.append({
            "restart": int(restart),
            "success": bool(result.success),
            "status": int(result.status),
            "message": str(result.message),
            "iterations": int(getattr(result, "nit", 0)),
            "objective": float(result.fun) if np.isfinite(result.fun) else float("nan"),
            "feasible": feasible,
        })

        if result.success and feasible and np.isfinite(result.fun) and result.fun < best_val:
            best_val = result.fun
            best_x = result.x.reshape(num_drones, 2)

    if return_diagnostics:
        return best_x, diagnostics
    return best_x


def _clip_to_vmax(u: np.ndarray, v_max: float) -> np.ndarray:
    speed = np.linalg.norm(u)
    if speed > v_max:
        return u * v_max / speed
    return u


@dataclass
class GreedyEIGPlanner(ControlPlanner):
    """
    One-step greedy EIG planner.

    At each measurement time, maximizes expected information gain over
    reachable target positions and then steers toward those targets.
    """
    num_restarts: int = 8
    _target_positions: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _current_positions_at_plan: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _meas_interval: float = field(default=30.0, init=False, repr=False)
    _rng: Optional[np.random.Generator] = field(default=None, init=False, repr=False)
    optimizer_diagnostics: list[dict] = field(default_factory=list, init=False)

    def planning_summary(self) -> str:
        return "Solving greedy-EIG one-step problem"

    def plan(self, t_now: float, par: Params, problem: ProblemSpec, s: DroneProblemState):
        meas = np.asarray(par.meas_times, dtype=float)
        idx0 = int(np.searchsorted(meas, float(t_now) - 1e-12, side="left"))
        idx0 = min(max(idx0, 0), len(meas) - 2)
        meas_interval = float(meas[idx0 + 1] - meas[idx0])
        self._meas_interval = meas_interval

        if self._rng is None:
            self._rng = np.random.default_rng(par.seed)

        current_positions = np.asarray(s.drone_positions, dtype=float)
        self._current_positions_at_plan = current_positions.copy()

        self._target_positions, diagnostics = _greedy_eig_targets(
            current_positions=current_positions,
            belief_mu=np.asarray(s.belief.mu, dtype=float),
            belief_sigma=np.asarray(s.belief.sigma, dtype=float),
            belief_q_log=s.belief.q_log,
            belief_q_log_var=s.belief.q_log_var,
            problem=problem,
            par=par,
            meas_interval=meas_interval,
            num_restarts=int(self.num_restarts),
            rng=self._rng,
            return_diagnostics=True,
        )
        self.optimizer_diagnostics.append({"time": float(t_now), "restarts": diagnostics})

    def compute_alpha(
        self, t: float, par: Params, problem: ProblemSpec, s: DroneProblemState
    ) -> np.ndarray:
        if self._target_positions is None:
            # Fallback: stay still
            return np.zeros((len(s.drones), 2), dtype=float)

        v_max = float(par.v_max)
        targets = self._target_positions
        num_drones = len(s.drones)
        controls = np.zeros((num_drones, 2), dtype=float)

        for i, drone in enumerate(s.drones):
            direction = targets[i] - drone.x
            dist = np.linalg.norm(direction)
            if dist < 1e-8:
                u_i = np.zeros(2)
            else:
                u_i = direction / dist * v_max
            controls[i] = u_i

        # Apply pairwise collision-avoidance correction (same as order-1, Eq. 22)
        c1c2 = float(problem.c1) + float(problem.c2)
        if problem.collision is not None and c1c2 > 0:
            current_positions_now = np.asarray(s.drone_positions, dtype=float)
            grads = compute_collision_gradients(current_positions_now, problem.collision)
            for i in range(num_drones):
                controls[i] = _clip_to_vmax(
                    controls[i] - (float(problem.tau_col) * float(problem.c3) / c1c2) * grads[i],
                    v_max,
                )

        return controls

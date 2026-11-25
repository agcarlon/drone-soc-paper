# drone_soc.py
"""
Core API for source localization with drones.

Public API:
- `Params`
- `ProblemSpec`
- `Belief`
- `DroneProblemState`
- `Drone`
- `ControlPlanner`
- `find_pollutant_source`
- `Pollutant`
- `SinusoidalPatrolling`
- `SigmoidCollisionAvoidance`
- `CircularNoFlyZone`
"""
from __future__ import annotations

import numpy as np
from numpy.linalg import solve
from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple, Callable, Optional
import json
import hashlib
import importlib.metadata
import subprocess
import sys

from measurement import take_measurement, Pollutant, get_plume_axis_angle
from scipy.optimize import minimize
import time
import os

from _drone_soc_internal import (
    covariance_from_eigen,
    eigen_from_covariance,
    construct_hjb_grid_axes,
    default_fixed_state_indices,
    _apply_measurement_jump,
    spatial_grid_resolution_metadata,
)

@dataclass
class Belief:
    mu: np.ndarray
    sigma: np.ndarray
    q_log: Optional[float] = None
    q_log_var: Optional[float] = None
    q_log_bounds: Optional[Tuple[float, float]] = None
    diagnostics: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        self.mu = np.asarray(self.mu, dtype=float)
        self.sigma = np.asarray(self.sigma, dtype=float)
        if self.mu.shape != (2,):
            raise ValueError("Belief.mu must have shape (2,).")
        if self.sigma.shape != (2, 2):
            raise ValueError("Belief.sigma must have shape (2, 2).")
        if self.q_log_bounds is not None:
            if len(self.q_log_bounds) != 2:
                raise ValueError("Belief.q_log_bounds must be a pair (lower, upper).")
            q_lo, q_hi = float(self.q_log_bounds[0]), float(self.q_log_bounds[1])
            if not q_lo < q_hi:
                raise ValueError("Belief.q_log_bounds must satisfy lower < upper.")
            self.q_log_bounds = (q_lo, q_hi)

    def copy(self) -> "Belief":
        return Belief(
            mu=np.array(self.mu, copy=True),
            sigma=np.array(self.sigma, copy=True),
            q_log=self.q_log,
            q_log_var=self.q_log_var,
            q_log_bounds=self.q_log_bounds,
            diagnostics=None if self.diagnostics is None else dict(self.diagnostics),
        )

    def update(self, new: "Belief") -> "Belief":
        """
        Update the belief, preserving intensity fields
        when the update does not provide them.
        """
        return Belief(
            mu=np.array(new.mu, copy=True),
            sigma=np.array(new.sigma, copy=True),
            q_log=new.q_log if new.q_log is not None else self.q_log,
            q_log_var=new.q_log_var if new.q_log_var is not None else self.q_log_var,
            q_log_bounds=new.q_log_bounds if new.q_log_bounds is not None else self.q_log_bounds,
            diagnostics=new.diagnostics,
        )

    @property
    def intensity_enabled(self) -> bool:
        return (self.q_log is not None) and (self.q_log_var is not None)

@dataclass
class Params:
    """
    Runtime and numerical settings only.

    This class intentionally excludes world/objective/policy definitions.
    Those belong in `ProblemSpec`.
    """
    t_0: float = 0.0              # start time
    t_end: float = 100.0          # end time (s)
    dt: float = 0.2               # integration step (s)
    meas_times: np.ndarray = field(default_factory=lambda: np.arange(0.0, 100.0, 10.0))  # seconds
    dt_hjb: float = 0.2         # integration step for the HJB
    dx: float = 0.5             # discretization on space
    # Optional coarser spatial discretization for "other" drone coordinates in
    # ANOVA order-2 (pairwise) HJBs. If None, uses `dx` for all drones.
    # This can reduce cost substantially since the 4D grid scales like
    # O(Nx1*Ny1*Nx2*Ny2). Typical choice: dx_secondary = 2*dx or 4*dx.
    dx_secondary: Optional[float] = None
    noise_std: float = 1e-4         # sigma_epsilon
    dyn_noise_std: float = 0.0      # standard deviation of drone motion noise (per coordinate)
    num_belief_discr: int = 5          # grid size for the belief uncertainty axes
    belief_discr_factor: float = 0.5  # span for belief discretisation (e.g., mu ± f * mu)
    v_max : float = 5.              # maximum speed of each drone
    tol_sigma: float = 1e-2    # tolerance on trace of Sigma
    tol_residual: Optional[float] = None  # optional threshold on predictive mean squared normalized residual
    tol_q_log_var: Optional[float] = None  # optional threshold on posterior variance of log-intensity
    seed: int = 0
    save_videos: bool = False
    num_obs_in_hjb: int = 1  # number of measurement times included in each HJB planning window
    output_dir: Optional[str] = None
    # Infinite-horizon / discounting
    discount: float = 0.0           # exponential discount rate r
    # Bayesian update mode: 'full_batch' uses (mu_prior, Sigma_prior) with ALL data;
    # 'sequential' uses current (mu, Sigma) with only the latest batch
    update_mode: str = "full_batch"
    # Early-stop control: stop iterations once covariance trace is sufficiently small for several measurements
    stop_when_confident: bool = True
    min_stop_updates: int = 0
    confident_streak: int = 3
    verbose_planning: bool = True

    def __post_init__(self):
        self.meas_times = np.asarray(self.meas_times, dtype=float)
        if self.dt <= 0.0:
            raise ValueError("Params.dt must be positive.")
        if self.dt_hjb <= 0.0:
            raise ValueError("Params.dt_hjb must be positive.")
        if self.noise_std <= 0.0:
            raise ValueError("Params.noise_std must be positive.")
        if self.tol_residual is not None and self.tol_residual < 0.0:
            raise ValueError("Params.tol_residual must be nonnegative or None.")
        if self.tol_q_log_var is not None and self.tol_q_log_var < 0.0:
            raise ValueError("Params.tol_q_log_var must be nonnegative or None.")
        if self.min_stop_updates < 0:
            raise ValueError("Params.min_stop_updates must be nonnegative.")
        if self.num_obs_in_hjb < 1:
            raise ValueError("Params.num_obs_in_hjb must be >= 1.")
        if self.meas_times.ndim != 1:
            raise ValueError("Params.meas_times must be a 1D array.")
        if self.meas_times.size == 0:
            raise ValueError("Params.meas_times must not be empty.")
        if np.any(np.diff(self.meas_times) < 0.0):
            raise ValueError("Params.meas_times must be sorted in nondecreasing order.")
        if self.meas_times[0] < self.t_0:
            raise ValueError("Params.meas_times must satisfy meas_times >= t_0.")
        if self.meas_times[-1] > self.t_end:
            raise ValueError("Params.meas_times must satisfy meas_times <= t_end.")
        measurement_steps = (self.meas_times - self.t_0) / self.dt
        if not np.allclose(measurement_steps, np.rint(measurement_steps), rtol=1e-10, atol=1e-10):
            raise ValueError("Params.meas_times must align with the simulation timestep.")
        self.update_mode = _validate_update_mode(self.update_mode)

@dataclass
class ProblemSpec:
    """
    Defines one optimization instance: world, objective terms, and policy fields.

    Ownership:
    - World/objective: pollutant, domain, wind, no-fly, c1..c4.
    - Policy fields used by costs/constraints: patrol, collision.
    """
    pollutant: Pollutant
    domain: np.ndarray = field(default_factory=lambda: np.array([[-5.0, 5.0], [-5.0, 5.0]], dtype=float))
    wind: np.ndarray = field(default_factory=lambda: np.array([-0.003, 0.0], dtype=float))
    no_fly_zones: Optional[List["NoFlyZone"]] = None
    c1: float = 1e-5
    c2: float = 1e-3
    c3: float = 1e-3
    c3_terminal: float = 1e-2
    c4: float = 1.0
    patrol: Optional[Callable[[np.ndarray, float], np.ndarray]] = None
    collision: Optional["CollisionAvoidance"] = None
    tau_col: float = 1.0  # seconds; runtime collision-gradient time scale

    def __post_init__(self):
        self.domain = np.asarray(self.domain, dtype=float)
        self.wind = np.asarray(self.wind, dtype=float)
        if self.domain.shape != (2, 2):
            raise ValueError("ProblemSpec.domain must have shape (2, 2).")
        if self.wind.shape != (2,):
            raise ValueError("ProblemSpec.wind must have shape (2,).")
        if self.patrol is not None and not callable(self.patrol):
            raise TypeError("ProblemSpec.patrol must be callable or None.")
        if not np.isfinite(self.tau_col) or self.tau_col <= 0.0:
            raise ValueError("ProblemSpec.tau_col must be finite and positive (seconds).")


@dataclass
class DroneProblemState:
    """
    State of the drone source seeking problem.
    """
    prior: Belief
    belief: Belief
    drones: List["Drone"]

    @property
    def drone_positions(self) -> np.ndarray:
        return np.stack([d.x for d in self.drones])

    def update_positions(self, new_pos: np.ndarray) -> None:
        for drone, new_drone_pos in zip(self.drones, new_pos):
            drone.x = new_drone_pos


class Drone:
    """
    Concrete drone model with optional patrol-direction field.
    """
    def __init__(
        self,
        x0: np.ndarray,
        v_max: float = 1.0,
        patrol_dir_fn: Optional[Callable[[np.ndarray, float], np.ndarray]] = None,
        dt_patrol: float = 0.1,
    ):
        self.x0 = x0
        self.x = np.copy(x0)
        self.v_max = v_max
        self._patrol_dir_fn = patrol_dir_fn
        self._dt_patrol = dt_patrol

    def patrolling_trajectory(self, t: float):
        if self._patrol_dir_fn is None:
            return self.x.copy()
        if t <= 0:
            return self.x0.copy()
        n_steps = int(max(1, np.floor(t / self._dt_patrol)))
        rem = t - n_steps * self._dt_patrol
        pos = self.x0.copy()
        dt = self._dt_patrol
        for k in range(n_steps):
            u = self._patrol_dir_fn(pos, k * dt)
            un = np.linalg.norm(u) + 1e-12
            if un > self.v_max:
                u = u * (self.v_max / un)
            pos = pos + dt * u
        if rem > 0:
            u = self._patrol_dir_fn(pos, t)
            un = np.linalg.norm(u) + 1e-12
            if un > self.v_max:
                u = u * (self.v_max / un)
            pos = pos + rem * u
        return pos

    def integrate_dynamics(self, alpha_drone: np.ndarray, par: Params, problem: ProblemSpec):
        new_pos = self.x + alpha_drone * par.dt
        if par.dyn_noise_std > 0.0:
            new_pos = new_pos + par.dyn_noise_std * np.sqrt(par.dt) * np.random.randn(*new_pos.shape)
        new_pos[0] = np.clip(new_pos[0], *problem.domain[0])
        new_pos[1] = np.clip(new_pos[1], *problem.domain[1])
        self.x = new_pos
        return


class ControlPlanner:
    """
    Base class for planning feedback controls over an interval, and computing
    per-drone controls at a given time t using the current state and problem.
    """
    def planning_summary(self) -> str:
        """
        Short status text describing what `plan(...)` does for this planner.
        """
        return "Updating planner"

    def plan(self, t_now: float, par: Params, problem: ProblemSpec, s: DroneProblemState):
        raise NotImplementedError

    def compute_alpha(self, t: float, par: Params, problem: ProblemSpec, s: DroneProblemState) -> np.ndarray:
        """
        Return per-drone controls with shape (num_drones, 2) at time t.
        `plan` must be called first to set internal feedback.
        """
        raise NotImplementedError

class CollisionAvoidance:
    """
    Collision avoidance interface providing scalar potential f(r)
    and its derivative f'(r) with r = ||x - y||.
    """
    def function(self, d: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def derivative(self, d: np.ndarray) -> np.ndarray:
        raise NotImplementedError


def _stable_inverse_logistic(x: np.ndarray) -> np.ndarray:
    """
    Compute 1 / (1 + exp(x)) without overflow using the identity
    exp(-log(1 + exp(x))) = exp(-logaddexp(0, x)).
    """
    return np.exp(-np.logaddexp(0.0, x))

@dataclass
class SigmoidCollisionAvoidance(CollisionAvoidance):
    d_min: float = 1.
    d_max: float = 2.
    max_value: float = 100.
    confidence: float = 0.99

    def __post_init__(self):
        self.factor = 2*np.log(self.confidence/(1-self.confidence)) / (self.d_max - self.d_min)
        self.shift = self.factor * (self.d_max + self.d_min) / 2

    def function(self, d: np.ndarray) -> np.ndarray:
        arg = self.factor * d - self.shift
        return self.max_value * _stable_inverse_logistic(arg)
    
    def derivative(self, d: np.ndarray) -> np.ndarray:
        arg = self.factor * d - self.shift
        sig = _stable_inverse_logistic(arg)
        return - self.max_value * self.factor * sig * (1 - sig)


class NoFlyZone:
    """
    Base class for no-fly zones. A zone provides a smooth penalty function
    f(r) and derivative f'(r) as a function of the distance r to its boundary.
    """
    def function(self, pos: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def derivative(self, pos: np.ndarray) -> np.ndarray:
        """
        Return ∂/∂r f(r) evaluated at the given positions (shape (..., 2)).
        """
        raise NotImplementedError


@dataclass
class CircularNoFlyZone(NoFlyZone):
    center: np.ndarray
    radius: float
    smooth_width: float = 1.0
    max_value: float = 100.0
    confidence: float = 0.99

    def __post_init__(self):
        # Use a logistic barrier that is ~max_value at the boundary and decays smoothly over smooth_width
        r_far = self.radius + self.smooth_width
        self.factor = 2 * np.log(self.confidence / (1 - self.confidence)) / (r_far - self.radius)
        self.shift = self.factor * (r_far + self.radius) / 2.0
        self.center = np.asarray(self.center, dtype=float)

    def function(self, pos: np.ndarray) -> np.ndarray:
        """
        pos: array-like (..., 2) of absolute positions.
        Returns the penalty value at each position.
        """
        diff = pos - self.center
        r = np.sqrt(np.sum(diff * diff, axis=-1))
        arg = self.factor * r - self.shift
        return self.max_value * _stable_inverse_logistic(arg)

    def derivative(self, pos: np.ndarray) -> np.ndarray:
        """
        pos: array-like (..., 2) of absolute positions.
        Returns df/dr evaluated at each position.
        """
        diff = pos - self.center
        r = np.sqrt(np.sum(diff * diff, axis=-1)) + 1e-12
        arg = self.factor * r - self.shift
        sig = _stable_inverse_logistic(arg)
        return - self.max_value * self.factor * sig * (1 - sig)


def compute_collision_gradients(positions: np.ndarray, collision: "CollisionAvoidance", eps: float = 1e-8) -> np.ndarray:
    """
    Compute ∂/∂x_i of V_ca = ∑_{i<j} f(||x_i - x_j||).
    Returns array of shape (num_drones, 2) with gradients for each drone.
    """
    num_drones = positions.shape[0]
    grads = np.zeros_like(positions, dtype=float)
    for i in range(num_drones):
        diff = positions[i] - positions  # (N,2)
        mask = np.arange(num_drones) != i
        if not np.any(mask):
            continue
        r = np.sqrt(np.sum(diff[mask] * diff[mask], axis=1))
        r_safe = np.maximum(r, eps)
        scale = collision.derivative(r_safe) / r_safe  # (N-1,)
        grads[i] = np.sum(scale[:, None] * diff[mask], axis=0)
    return grads


def collision_potential_from_state(x_state: np.ndarray, num_drones: int, collision: Optional["CollisionAvoidance"]) -> np.ndarray:
    """
    Compute pairwise collision-avoidance potential over the HJB state grid.
    Expects x_state shaped as stacked relative positions (z1x, z1y, z2x, z2y, ...).
    Returns an array broadcastable to the grid shape, or zeros if disabled.
    """
    if (collision is None) or (num_drones < 2):
        return 0.0 * x_state[0]
    total = 0.0 * x_state[0]
    for i in range(num_drones):
        zi_x = x_state[2 * i]
        zi_y = x_state[2 * i + 1]
        for j in range(i + 1, num_drones):
            zj_x = x_state[2 * j]
            zj_y = x_state[2 * j + 1]
            dx = zi_x - zj_x
            dy = zi_y - zj_y
            r = np.sqrt(dx * dx + dy * dy)
            total = total + collision.function(r)
    return total


def no_fly_potential_from_state(x_state: np.ndarray, num_drones: int, zones: Optional[List["NoFlyZone"]], offset: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Compute no-fly zone penalty over the HJB state grid.
    The state x_state is assumed to contain relative positions; offset is the
    reference point (e.g., current belief mean) to convert to absolute coordinates.
    """
    if zones is None or len(zones) == 0:
        return 0.0 * x_state[0]
    if offset is None:
        offset = np.zeros(2, dtype=float)
    total = 0.0 * x_state[0]
    for i in range(num_drones):
        zi_x = x_state[2 * i]
        zi_y = x_state[2 * i + 1]
        # Absolute position on the grid
        xi_abs = zi_x + offset[0]
        yi_abs = zi_y + offset[1]
        pos_abs = np.stack([xi_abs, yi_abs], axis=-1)
        for z in zones:
            total = total + z.function(pos_abs)
    return total

@dataclass
class PatrollingBase:
    def patrolling_trajectory(self, x:np.ndarray, t: float):
        raise NotImplementedError
    
    def __call__(self, x:np.ndarray, t: float):
        return self.patrolling_trajectory(x, t)


@dataclass
class SinusoidalPatrolling(PatrollingBase):
    wind: np.ndarray
    v_max: float
    period: float = 40.
    y_scale: float = 0.99
    def patrolling_trajectory(self, x:np.ndarray, t: float):
        wind_angle = get_plume_axis_angle(self.wind)
        t = np.reshape(t, ())
        omega = 2 * np.pi / self.period
        # get_plume_axis_angle points opposite to the stored downwind vector,
        # so a positive axial component moves the patrol upwind.
        path = np.array([np.sqrt(1 - self.y_scale**2), self.y_scale * np.sin(omega * t)]) * self.v_max
        rotation_matrix = np.array([
            [np.cos(wind_angle), -np.sin(wind_angle)], 
            [np.sin(wind_angle), np.cos(wind_angle)]])
        return rotation_matrix @ path


def _validate_update_mode(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("update_mode must be a string.")
    if value not in {"full_batch", "sequential"}:
        raise ValueError("update_mode must be one of {'full_batch', 'sequential'}")
    return value


def _dense_hess_inv_or_none(hess_inv: Any, dim: int) -> Optional[np.ndarray]:
    """
    Convert SciPy's optimizer-specific inverse-Hessian representation to a dense
    square array when possible.

    Across SciPy versions, `res.hess_inv` for L-BFGS-B may be a dense array,
    a matrix-like object with `todense()`, or an `LbfgsInvHessProduct` that
    exposes `matmat`. Returning `None` triggers the conservative covariance
    fallback used elsewhere in this module.
    """
    if hess_inv is None:
        return None

    arr = None
    if hasattr(hess_inv, "todense"):
        arr = np.asarray(hess_inv.todense(), dtype=float)
    elif hasattr(hess_inv, "matmat"):
        arr = np.asarray(hess_inv.matmat(np.eye(dim)), dtype=float)
    else:
        try:
            arr = np.asarray(hess_inv, dtype=float)
        except (TypeError, ValueError):
            return None

    if arr.shape != (dim, dim):
        return None
    return arr


def _fisher_laplace_covariance(
    *,
    x: np.ndarray,
    t: float,
    theta_map: np.ndarray,
    q_log_map: Optional[float],
    prior_sigma: np.ndarray,
    prior_q_log_var: Optional[float],
    par: Params,
    problem: ProblemSpec,
    use_intensity: bool,
) -> Tuple[np.ndarray, Optional[float], Dict[str, Any]]:
    """
    Gauss-Newton/Laplace covariance at the MAP using plume sensitivities.

    This uses H = P^{-1} + J^T R^{-1} J, rather than the optimizer's
    quasi-Newton inverse-Hessian approximation.
    """
    x_arr = np.atleast_2d(np.asarray(x, dtype=float))
    theta_map = np.asarray(theta_map, dtype=float).reshape(2)
    intensity = float(np.exp(q_log_map)) if use_intensity and q_log_map is not None else None
    inv_noise_var = 1.0 / max(float(par.noise_std) ** 2, 1e-30)

    J_theta = problem.pollutant.get_concentration_gradient_wrt_theta(
        x_arr,
        t,
        theta_map,
        intensity=intensity,
    )
    J_theta = np.asarray(J_theta, dtype=float).reshape(x_arr.shape[0], 2)

    if use_intensity:
        J_q = problem.pollutant.get_concentration_derivative_wrt_logq(
            x_arr,
            t,
            theta_map,
            intensity=intensity,
        )
        J_q = np.asarray(J_q, dtype=float).reshape(x_arr.shape[0], 1)
        J = np.hstack([J_theta, J_q])

        prior_precision = np.zeros((3, 3), dtype=float)
        prior_precision[:2, :2] = np.linalg.pinv(np.asarray(prior_sigma, dtype=float))
        if prior_q_log_var is not None and float(prior_q_log_var) > 0.0:
            prior_precision[2, 2] = 1.0 / float(prior_q_log_var)
    else:
        J = J_theta
        prior_precision = np.linalg.pinv(np.asarray(prior_sigma, dtype=float))

    H = prior_precision + inv_noise_var * (J.T @ J)
    H = 0.5 * (H + H.T)
    B_full = np.linalg.pinv(H)
    B_full = 0.5 * (B_full + B_full.T)

    if use_intensity:
        Sigma_theta = B_full[:2, :2]
        q_log_var = float(np.clip(B_full[2, 2], 1e-9, None))
    else:
        Sigma_theta = B_full
        q_log_var = prior_q_log_var

    eigvals, eigvecs = np.linalg.eigh(Sigma_theta)
    unclipped_eigvals = eigvals.copy()
    eigvals = np.clip(eigvals, a_min=1e-6, a_max=None)
    sigma_post = eigvecs @ np.diag(eigvals) @ eigvecs.T
    diagnostics = {
        "information_condition_number": float(np.linalg.cond(H)),
        "pseudoinverse_used": True,
        "covariance_eigenvalues_clipped": int(np.sum(unclipped_eigvals < 1e-6)),
    }
    return sigma_post, q_log_var, diagnostics


def _wind_relative_energy(alpha_drones: np.ndarray, dt: float, wind: np.ndarray) -> float:
    """
    Discrete wind-relative control energy over one integration step:

        sum_i ||alpha_i - w||^2 dt.
    """
    air_relative = np.asarray(alpha_drones, dtype=float) - np.asarray(wind, dtype=float).reshape(1, 2)
    return float(np.sum(air_relative**2) * float(dt))


def find_pollutant_source(
    par: Params,
    problem: ProblemSpec,
    s: DroneProblemState,
    planner: ControlPlanner,
    on_after_measurement: Optional[
        Callable[[np.ndarray, np.ndarray, float, Params, ProblemSpec, DroneProblemState, Belief, Belief, bool], None]
    ] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Run the source-localization loop with periodic measurements and replanning.

    The planner is solved at `t_0` (when needed) and after each measurement time.
    Belief updates use `par.update_mode`:
    - `full_batch`: fixed prior + accumulated measurements.
    - `sequential`: rolling prior + latest batch.
    """
    np.random.seed(par.seed)
    traj = [s.drone_positions]
    log: Dict[str, Any] = {
        "times": [par.t_0],
        "beliefs": [s.belief.copy()],
        "controls": [],
        "control_times": [],
        "residual_scores": [],
        "energy_steps": [],
        "energy_cycles": [],
        "energy_cumulative_by_cycle": [0.0],
        "energy_total": 0.0,
        "measurement_times": [],
        "measurement_positions": [],
        "measurement_values": [],
        "bayesian_diagnostics": [],
        "stop_reason": "reached_time_horizon",
    }

    _save_run_config(par, problem, s, planner)

    meas_idx = 0
    meas_times = par.meas_times
    mode = _validate_update_mode(par.update_mode)

    if meas_times[0] > par.t_0:
        if par.verbose_planning:
            print(f"{planner.planning_summary()} at t={float(par.t_0)}")
        t0 = time.time()
        planner.plan(float(par.t_0), par=par, problem=problem, s=s)
        if par.verbose_planning:
            print(f"Time taken: {time.time()-t0}")

    positions_list: List[np.ndarray] = []
    measurement_list: List[np.ndarray] = []
    duration = float(par.t_end - par.t_0)
    Nt = int(round(duration / par.dt))
    if Nt < 1 or not np.isclose(Nt * par.dt, duration, rtol=1e-10, atol=1e-12):
        raise ValueError("The simulation horizon must be an integer multiple of Params.dt.")
    ts = np.linspace(par.t_0, par.t_end, Nt + 1)

    break_counter = 0
    current_cycle_energy = 0.0
    cumulative_energy = 0.0
    for t in ts:
        if meas_idx < len(meas_times) and np.isclose(t, meas_times[meas_idx]):
            log["energy_cycles"].append(float(current_cycle_energy))
            current_cycle_energy = 0.0
            y = take_measurement(s.drone_positions, t=t, pol=problem.pollutant)
            x_event = np.asarray(s.drone_positions, dtype=float)
            log["measurement_times"].append(float(t))
            log["measurement_positions"].append(x_event.copy())
            log["measurement_values"].append(np.asarray(y, dtype=float).copy())
            if s.belief.intensity_enabled:
                intensity_pred = float(np.exp(s.belief.q_log))
            else:
                intensity_pred = getattr(problem.pollutant, "q", 1.0)
            y_pred = problem.pollutant.get_pollutant_concentration(
                x_event,
                float(t),
                theta=np.asarray(s.belief.mu, dtype=float),
                intensity=float(intensity_pred),
            )
            residual_score = float(np.mean(((np.asarray(y, dtype=float) - np.asarray(y_pred, dtype=float)) / par.noise_std) ** 2))
            log["residual_scores"].append(residual_score)

            positions_list.append(s.drone_positions.copy())
            measurement_list.append(y)
            if mode == "full_batch":
                # Full-batch Laplace always uses the fixed initial prior.
                prior = s.prior.copy()
                # Full-batch likelihood uses all measurements accumulated so far.
                y_for_event = np.concatenate(measurement_list)
                positions_for_event = np.vstack(positions_list)
                result = bayesian_update(
                    y=y_for_event,
                    x=positions_for_event,
                    t=t,
                    par=par,
                    problem=problem,
                    s=s,
                )
                used_all = True
            else:
                # Sequential update uses the current belief as prior.
                prior = s.belief.copy()
                # Sequential likelihood uses the latest batch only.
                y_for_event = y
                positions_for_event = np.asarray(s.drone_positions, dtype=float)
                result = bayesian_update_sequential(
                    y=y,
                    x=positions_for_event,
                    t=t,
                    par=par,
                    problem=problem,
                    s=s,
                )
                used_all = False

            s.belief = s.belief.update(result)
            if result.diagnostics is not None:
                log["bayesian_diagnostics"].append(dict(result.diagnostics))

            if par.stop_when_confident:
                num_updates_after_assimilation = len(log["measurement_times"])
                residual_ok = (par.tol_residual is None) or (residual_score < par.tol_residual)
                sigma_ok = np.trace(s.belief.sigma) < par.tol_sigma
                q_var_ok = (
                    (par.tol_q_log_var is None)
                    or (s.belief.q_log_var is not None and s.belief.q_log_var < par.tol_q_log_var)
                )
                enough_updates = num_updates_after_assimilation >= par.min_stop_updates
                if sigma_ok and residual_ok and q_var_ok and enough_updates:
                    break_counter += 1
                else:
                    break_counter = 0
                if break_counter >= par.confident_streak:
                    log["stop_reason"] = (
                        f"confidence_criterion_met:"
                        f" trace_sigma={float(np.trace(s.belief.sigma)):.3f} < {float(par.tol_sigma):.3f},"
                        f" residual={float(residual_score):.3f} < {float(par.tol_residual) if par.tol_residual is not None else float('nan'):.3f},"
                        f" q_log_var={float(s.belief.q_log_var) if s.belief.q_log_var is not None else float('nan'):.3f}"
                        f" < {float(par.tol_q_log_var) if par.tol_q_log_var is not None else float('nan'):.3f},"
                        f" updates={num_updates_after_assimilation} >= {int(par.min_stop_updates)}"
                    )
                    log["beliefs"].append(s.belief.copy())
                    log["energy_cumulative_by_cycle"].append(float(cumulative_energy))
                    break

            if on_after_measurement is not None:
                on_after_measurement(
                    np.asarray(y_for_event),
                    np.asarray(positions_for_event),
                    float(t),
                    par,
                    problem,
                    s,
                    prior,
                    result,
                    used_all,
                )

            t_now = float(meas_times[meas_idx])
            meas_idx += 1
            log["beliefs"].append(s.belief.copy())
            log["energy_cumulative_by_cycle"].append(float(cumulative_energy))
            # Assimilate a scheduled terminal measurement, but do not solve a
            # controller that can never be used.
            if np.isclose(t, par.t_end):
                break
            if par.verbose_planning:
                print(f"{planner.planning_summary()} at t={t_now}")
            t0 = time.time()
            planner.plan(t_now=t_now, par=par, problem=problem, s=s)
            if par.verbose_planning:
                print(f"Time taken: {time.time()-t0}")

        # There are Nt+1 state/event times but only Nt integration intervals.
        if np.isclose(t, par.t_end):
            break

        alpha_drones = planner.compute_alpha(t, par=par, problem=problem, s=s)
        step_energy = _wind_relative_energy(alpha_drones, par.dt, problem.wind)
        current_cycle_energy += step_energy
        cumulative_energy += step_energy
        log["energy_steps"].append(step_energy)
        log["energy_total"] = float(cumulative_energy)
        for d, alpha_drone in zip(s.drones, alpha_drones):
            d.integrate_dynamics(alpha_drone, par, problem)

        traj.append(s.drone_positions)
        log["times"].append(float(t + par.dt))
        log["control_times"].append(float(t))
        log["controls"].append(alpha_drones.copy())

    traj_arr = np.stack(traj, axis=-1)
    _save_run_results(par, traj_arr, log)
    return traj_arr, log



def bayesian_update(
    y: np.ndarray,
    x: np.ndarray,
    t: float,
    par: Params,
    problem: ProblemSpec,
    s: DroneProblemState,
) -> Belief:
    """
    Laplace update - returns updated belief.

    Args:
        y: Measurement vector.
        x: Measurement Positions (len(y),2)
        t: Time.
        par: Simulation parameters.
        s: Current state.
        problem: Problem specification including pollutant.
    """
    use_intensity = s.belief.intensity_enabled

    def neg_log_posterior(z):
        z = np.array(z)
        theta = z[:2]
        res = theta - s.prior.mu
        prior_term = 0.5 * res.T @ solve(s.prior.sigma, res)
        if use_intensity:
            zeta = z[2]
            res_zeta = zeta - s.prior.q_log
            prior_term += 0.5 * (res_zeta ** 2) / s.prior.q_log_var
            intensity = np.exp(zeta)
        else:
            intensity = getattr(problem.pollutant, "q", 1.0)
        model_prediction = problem.pollutant.get_pollutant_concentration(x, t, theta, intensity=intensity)
        likelihood_term = 0.5 / par.noise_std ** 2 * np.sum((y - model_prediction) ** 2)
        return prior_term + likelihood_term

    theta_bounds = [
        (float(problem.domain[0, 0]), float(problem.domain[0, 1])),
        (float(problem.domain[1, 0]), float(problem.domain[1, 1])),
    ]
    if use_intensity:
        bounds = [*theta_bounds, s.prior.q_log_bounds if s.prior.q_log_bounds is not None else (None, None)]
    else:
        bounds = theta_bounds

    rng = np.random.default_rng(par.seed + int(round(1000.0 * float(t))) + 1001)
    num_random_starts = 256
    top_k_starts = 8
    theta_starts = np.column_stack([
        rng.uniform(problem.domain[0, 0], problem.domain[0, 1], size=num_random_starts),
        rng.uniform(problem.domain[1, 0], problem.domain[1, 1], size=num_random_starts),
    ])
    theta_candidates = [np.asarray(s.belief.mu, dtype=float), np.asarray(s.prior.mu, dtype=float), *theta_starts]
    if use_intensity:
        zeta_start = float(s.belief.q_log)
        z_candidates = np.array(
            [np.concatenate([theta, [zeta_start]]) for theta in theta_candidates],
            dtype=float,
        )
    else:
        z_candidates = np.array(theta_candidates, dtype=float)

    start_vals = np.array([neg_log_posterior(z) for z in z_candidates], dtype=float)
    best_order = np.argsort(start_vals)[: min(top_k_starts, len(z_candidates))]
    best_res = None
    best_val = float("inf")
    for idx in best_order:
        res_try = minimize(neg_log_posterior, z_candidates[idx], method='L-BFGS-B', bounds=bounds)
        if np.isfinite(res_try.fun) and float(res_try.fun) < best_val:
            best_res = res_try
            best_val = float(res_try.fun)
    if best_res is None:
        fallback_idx = int(np.argmin(start_vals))
        best_res = minimize(neg_log_posterior, z_candidates[fallback_idx], method='L-BFGS-B', bounds=bounds)
    res = best_res
    z_map = res.x
    theta_map = z_map[:2]
    if use_intensity:
        q_log_map = z_map[2]
    else:
        q_log_map = s.belief.q_log
    sigma_post, q_log_var, covariance_diagnostics = _fisher_laplace_covariance(
        x=x,
        t=t,
        theta_map=theta_map,
        q_log_map=q_log_map,
        prior_sigma=s.prior.sigma,
        prior_q_log_var=s.prior.q_log_var,
        par=par,
        problem=problem,
        use_intensity=use_intensity,
    )
    active_bounds = []
    for value, bound in zip(z_map, bounds):
        low, high = bound
        active_bounds.append(bool(
            (low is not None and np.isclose(value, low))
            or (high is not None and np.isclose(value, high))
        ))
    diagnostics = {
        "optimizer_success": bool(res.success),
        "optimizer_status": int(res.status),
        "optimizer_message": str(res.message),
        "optimizer_iterations": int(getattr(res, "nit", 0)),
        "active_bounds": active_bounds,
        **covariance_diagnostics,
    }
    if q_log_map is None or q_log_var is None:
        return Belief(mu=theta_map, sigma=sigma_post, q_log_bounds=s.prior.q_log_bounds, diagnostics=diagnostics)
    return Belief(
        mu=theta_map,
        sigma=sigma_post,
        q_log=float(q_log_map),
        q_log_var=float(q_log_var),
        q_log_bounds=s.prior.q_log_bounds,
        diagnostics=diagnostics,
    )


def bayesian_update_sequential(
    y: np.ndarray,
    x: np.ndarray,
    t: float,
    par: Params,
    problem: ProblemSpec,
    s: DroneProblemState,
) -> Belief:
    """
    Sequential Laplace update using current belief as prior and only the provided batch (y, x).
    """
    use_intensity = s.belief.intensity_enabled

    def neg_log_posterior(z):
        z = np.array(z)
        theta = z[:2]
        res = theta - s.belief.mu
        prior_term = 0.5 * res.T @ solve(s.belief.sigma, res)
        if use_intensity:
            zeta = z[2]
            res_zeta = zeta - s.belief.q_log
            prior_term += 0.5 * (res_zeta ** 2) / s.belief.q_log_var
            intensity = np.exp(zeta)
        else:
            intensity = getattr(problem.pollutant, "q", 1.0)
        model_prediction = problem.pollutant.get_pollutant_concentration(x, t, theta, intensity=intensity)
        likelihood_term = 0.5 / par.noise_std ** 2 * np.sum((y - model_prediction) ** 2)
        return prior_term + likelihood_term

    theta_bounds = [
        (float(problem.domain[0, 0]), float(problem.domain[0, 1])),
        (float(problem.domain[1, 0]), float(problem.domain[1, 1])),
    ]
    if use_intensity:
        bounds = [*theta_bounds, s.belief.q_log_bounds if s.belief.q_log_bounds is not None else (None, None)]
    else:
        bounds = theta_bounds

    rng = np.random.default_rng(par.seed + int(round(1000.0 * float(t))) + 2003)
    num_random_starts = 256
    top_k_starts = 8
    theta_starts = np.column_stack([
        rng.uniform(problem.domain[0, 0], problem.domain[0, 1], size=num_random_starts),
        rng.uniform(problem.domain[1, 0], problem.domain[1, 1], size=num_random_starts),
    ])
    theta_candidates = [np.asarray(s.belief.mu, dtype=float), *theta_starts]
    if use_intensity:
        zeta_start = float(s.belief.q_log)
        z_candidates = np.array(
            [np.concatenate([theta, [zeta_start]]) for theta in theta_candidates],
            dtype=float,
        )
    else:
        z_candidates = np.array(theta_candidates, dtype=float)

    start_vals = np.array([neg_log_posterior(z) for z in z_candidates], dtype=float)
    best_order = np.argsort(start_vals)[: min(top_k_starts, len(z_candidates))]
    best_res = None
    best_val = float("inf")
    for idx in best_order:
        res_try = minimize(neg_log_posterior, z_candidates[idx], method='L-BFGS-B', bounds=bounds)
        if np.isfinite(res_try.fun) and float(res_try.fun) < best_val:
            best_res = res_try
            best_val = float(res_try.fun)
    if best_res is None:
        fallback_idx = int(np.argmin(start_vals))
        best_res = minimize(neg_log_posterior, z_candidates[fallback_idx], method='L-BFGS-B', bounds=bounds)
    res = best_res
    z_map = res.x
    theta_map = z_map[:2]
    if use_intensity:
        q_log_map = z_map[2]
    else:
        q_log_map = s.belief.q_log
    sigma_post, q_log_var, covariance_diagnostics = _fisher_laplace_covariance(
        x=x,
        t=t,
        theta_map=theta_map,
        q_log_map=q_log_map,
        prior_sigma=s.belief.sigma,
        prior_q_log_var=s.belief.q_log_var,
        par=par,
        problem=problem,
        use_intensity=use_intensity,
    )
    active_bounds = []
    for value, bound in zip(z_map, bounds):
        low, high = bound
        active_bounds.append(bool(
            (low is not None and np.isclose(value, low))
            or (high is not None and np.isclose(value, high))
        ))
    diagnostics = {
        "optimizer_success": bool(res.success),
        "optimizer_status": int(res.status),
        "optimizer_message": str(res.message),
        "optimizer_iterations": int(getattr(res, "nit", 0)),
        "active_bounds": active_bounds,
        **covariance_diagnostics,
    }
    if q_log_map is None or q_log_var is None:
        return Belief(mu=theta_map, sigma=sigma_post, q_log_bounds=s.belief.q_log_bounds, diagnostics=diagnostics)
    return Belief(
        mu=theta_map,
        sigma=sigma_post,
        q_log=float(q_log_map),
        q_log_var=float(q_log_var),
        q_log_bounds=s.belief.q_log_bounds,
        diagnostics=diagnostics,
    )




def _to_serialisable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    return obj


def _save_run_config(par: "Params", problem: "ProblemSpec", s: "DroneProblemState", planner: ControlPlanner):
    if not getattr(par, 'output_dir', None):
        return
    cfg = {}
    package_names = ("numpy", "scipy", "matplotlib", "pandas")
    versions = {}
    for package_name in package_names:
        try:
            versions[package_name] = importlib.metadata.version(package_name)
        except importlib.metadata.PackageNotFoundError:
            versions[package_name] = None
    environment_payload = {
        "python": sys.version.split()[0],
        "packages": versions,
    }
    environment_json = json.dumps(environment_payload, sort_keys=True)
    try:
        git_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        git_commit = None
    cfg["provenance"] = {
        "git_commit": git_commit,
        "environment": environment_payload,
        "environment_sha256": hashlib.sha256(environment_json.encode("utf-8")).hexdigest(),
    }
    # Params
    cfg["params"] = {k: _to_serialisable(getattr(par, k)) for k in par.__dict__.keys()}
    # ``dx`` selects a node count; linspace then includes both domain endpoints.
    # Persist the resulting spacing so numerical resolution is unambiguous.
    cfg["hjb_grid_resolution"] = list(
        spatial_grid_resolution_metadata(par, problem).values()
    )
    # Problem definition
    problem_cfg = {
        "domain": _to_serialisable(problem.domain),
        "wind": _to_serialisable(problem.wind),
        "c1": _to_serialisable(problem.c1),
        "c2": _to_serialisable(problem.c2),
        "c3": _to_serialisable(problem.c3),
        "c3_terminal": _to_serialisable(problem.c3_terminal),
        "tau_col": _to_serialisable(problem.tau_col),
        "c4": _to_serialisable(problem.c4),
    }
    zones = getattr(problem, "no_fly_zones", None)
    if zones is not None:
        problem_cfg["no_fly_zones"] = []
        for z in zones:
            try:
                problem_cfg["no_fly_zones"].append(
                    {"type": z.__class__.__name__, **{k: _to_serialisable(v) for k, v in z.__dict__.items()}}
                )
            except Exception:
                continue
    # Drones
    cfg["drones"] = [{
        "type": d.__class__.__name__,
        "x0": _to_serialisable(d.x0),
        "v_max": _to_serialisable(d.v_max)
    } for d in s.drones]
    # Pollutant metadata
    pol_dict = {
        k: _to_serialisable(v)
        for k, v in getattr(problem.pollutant, "__dict__", {}).items()
    }
    pol_dict["type"] = problem.pollutant.__class__.__name__
    problem_cfg["pollutant"] = pol_dict
    cfg["problem"] = problem_cfg
    # Planner
    pl = {"type": planner.__class__.__name__}
    for attr in ("order", "pair_weight"):
        if hasattr(planner, attr):
            pl[attr] = _to_serialisable(getattr(planner, attr))
    cfg["planner"] = pl
    # Problem extras (best effort serialization)
    try:
        collision_obj = getattr(problem, "collision", None)
        if collision_obj is not None:
            coll_dict = {k: _to_serialisable(v) for k, v in collision_obj.__dict__.items()}
            coll_dict["type"] = collision_obj.__class__.__name__
            cfg["problem"]["collision"] = coll_dict
    except Exception:
        pass
    try:
        patrol_obj = getattr(problem, "patrol", None)
        if patrol_obj is not None:
            patrol_dict = {"type": patrol_obj.__class__.__name__}
            if hasattr(patrol_obj, "__dict__"):
                patrol_dict.update({k: _to_serialisable(v) for k, v in patrol_obj.__dict__.items()})
            cfg["problem"]["patrol"] = patrol_dict
    except Exception:
        pass
    try:
        os.makedirs(par.output_dir, exist_ok=True)
        with open(os.path.join(par.output_dir, "run_config.json"), "w") as f:
            json.dump(cfg, f, indent=2)
    except Exception as e:
        print(f"Warning: failed to save run config: {e}")


def _save_run_results(par: "Params", traj: np.ndarray, log: Dict[str, Any]):
    if not getattr(par, 'output_dir', None):
        return
    try:
        os.makedirs(par.output_dir, exist_ok=True)
        times = np.array(log.get("times", []), dtype=float)
        # controls: list of (M,2) -> (T,M,2)
        controls_list = log.get("controls", [])
        if len(controls_list) > 0:
            controls = np.stack(controls_list, axis=0)
        else:
            controls = np.empty((0, 0, 0))
        # beliefs: mus and eigen params from sigmas
        beliefs = log.get("beliefs", [])
        if len(beliefs) > 0:
            mus = np.stack([b.mu for b in beliefs], axis=0)
            vals_logs = []
            angs = []
            for b in beliefs:
                Sigma = b.sigma
                vlog, ang = eigen_from_covariance(Sigma)
                vals_logs.append(vlog)
                angs.append(ang)
            vals_logs = np.stack(vals_logs, axis=0)
            angs = np.array(angs, dtype=float)
            q_logs = np.array([b.q_log if b.q_log is not None else np.nan for b in beliefs], dtype=float)
            q_log_vars = np.array([b.q_log_var if b.q_log_var is not None else np.nan for b in beliefs], dtype=float)
        else:
            mus = np.empty((0, 2))
            vals_logs = np.empty((0, 2))
            angs = np.empty((0,))
            q_logs = np.empty((0,))
            q_log_vars = np.empty((0,))
        np.savez_compressed(
            os.path.join(par.output_dir, "run_results.npz"),
            times=times,
            traj=traj,
            controls=controls,
            control_times=np.array(log.get("control_times", []), dtype=float),
            mus=mus,
            eig_vals_log=vals_logs,
            eig_angle=angs,
            meas_times=np.array(par.meas_times, dtype=float),
            q_logs=q_logs,
            q_log_vars=q_log_vars,
            residual_scores=np.array(log.get("residual_scores", []), dtype=float),
            measurement_times=np.array(log.get("measurement_times", []), dtype=float),
            measurement_positions=np.array(log.get("measurement_positions", []), dtype=float),
            measurement_values=np.array(log.get("measurement_values", []), dtype=float),
            energy_steps=np.array(log.get("energy_steps", []), dtype=float),
            energy_cycles=np.array(log.get("energy_cycles", []), dtype=float),
            energy_cumulative_by_cycle=np.array(log.get("energy_cumulative_by_cycle", []), dtype=float),
            bayesian_diagnostics_json=np.array(
                [json.dumps(item, sort_keys=True) for item in log.get("bayesian_diagnostics", [])]
            ),
        )
    except Exception as e:
        print(f"Warning: failed to save run results: {e}")

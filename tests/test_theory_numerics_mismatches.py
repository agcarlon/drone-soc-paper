"""Regression checks for previously identified theory/numerics mismatches.

How to run (in an environment with numpy/scipy installed):
  python3 -u tests/debug_detect_theory_numerics_mismatches.py

This script prints diagnostic values and exits non-zero if it detects:
  1) eigenvalue-axis swapping in `_apply_measurement_jump`, and/or
  2) an unexpected measurement jump being applied at the start of a finite-horizon
     planning window (t = t0) due to including the current measurement time.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import List

from pathlib import Path


# Ensure repo root is on sys.path when running as a script (e.g. via `conda run`).
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _print_header(title: str) -> None:
    bar = "=" * 80
    print(bar)
    print(title)
    print(bar)


def check_eigen_axes_swapped() -> bool:
    """
    Detect whether `_apply_measurement_jump` maps posterior eigenvalues to the
    correct grid axes.

    Strategy:
      - Build a tiny HJB grid where the value function is linear in the two
        eigenvalue-log axes: V = 1000 * logλ_max_axis + logλ_min_axis.
      - Apply the measurement jump with a simple constant-gradient pollutant so
        Fisher information is known and the posterior covariance update is easy.
      - Reconstruct V_after by adding IG back (since the jump returns V_after - IG).
      - Compare V_after to the "correct mapping" vs "swapped mapping" predictions.
    """
    import numpy as np

    from drone_soc import Belief, Drone, DroneProblemState, Params, ProblemSpec
    from measurement import Pollutant
    from _drone_soc_internal import _apply_measurement_jump

    @dataclass
    class ConstantGradientPollutant(Pollutant):
        grad_theta: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0], dtype=float))

        def get_pollutant_concentration(self, x, t, theta, intensity=None):
            x = np.atleast_2d(x)
            return np.zeros((x.shape[0],), dtype=float)

        def get_concentration_gradient_wrt_theta(self, x, t, theta, intensity=None):
            x = np.atleast_2d(x)
            g = np.asarray(self.grad_theta, dtype=float).reshape(1, 2)
            return np.repeat(g, x.shape[0], axis=0)

    # Choose wind so plume-axis angle is 0 (wind_aligned jump uses this angle).
    wind = np.array([-1.0, 0.0], dtype=float)

    # Keep Fisher info small so posterior eigenvalues stay within axis bounds.
    noise_std = 10.0  # invR = 0.01

    pol = ConstantGradientPollutant(
        source=np.array([0.0, 0.0], dtype=float),
        q=1.0,
        wind=wind,
        sigma_e=noise_std,
        grad_theta=np.array([1.0, 0.0], dtype=float),
    )
    problem = ProblemSpec(pollutant=pol, wind=wind, domain=np.array([[-1.0, 1.0], [-1.0, 1.0]], dtype=float))

    par = Params(
        t_end=5.0,
        dt=1.0,
        dt_hjb=1.0,
        dx=1.0,
        noise_std=float(noise_std),
        meas_times=np.array([0.0, 5.0], dtype=float),
        num_obs_in_hjb=1,
        save_videos=False,
        stop_when_confident=False,
    )

    # Belief fields used by the jump: mu, optional intensity fields.
    belief = Belief(mu=np.array([0.0, 0.0], dtype=float), sigma=np.eye(2))
    prior = belief.copy()
    drones = [Drone(x0=np.array([0.0, 0.0], dtype=float))]
    s = DroneProblemState(prior=prior, belief=belief, drones=drones)

    # HJB grid axes (wind_aligned=True => no angle axis).
    log9 = float(np.log(9.0))
    log4 = float(np.log(4.0))
    eig0_axis = np.array([log9 - 0.2, log9], dtype=float)  # intended: log λ_max
    eig1_axis = np.array([log4 - 0.2, log4], dtype=float)  # intended: log λ_min
    z_axis = np.array([-1.0, 1.0], dtype=float)
    grid_axes = (z_axis, z_axis, eig0_axis, eig1_axis)

    mesh = np.meshgrid(*grid_axes, indexing="ij")
    Vnext = 1000.0 * mesh[2] + 1.0 * mesh[3]

    out = _apply_measurement_jump(
        grid_axes=grid_axes,
        Vnext=Vnext,
        t_now=0.0,
        s=s,
        par=par,
        problem=problem,
        num_drones=1,
        wind_aligned=True,
    )

    # Probe a single gridpoint: z=(first indices), eigen logs = (max indices).
    idx = (0, 0, 1, 1)
    V_jump_out = float(out[idx])

    # Recompute the scalars analytically at this gridpoint.
    log_lambda_max = float(eig0_axis[1])
    log_lambda_min = float(eig1_axis[1])
    lam_max = float(np.exp(log_lambda_max))  # along x due to angle=0
    lam_min = float(np.exp(log_lambda_min))  # along y due to angle=0

    invR = 1.0 / (noise_std**2)  # 0.01
    # g = [1,0] => F = invR * [[1,0],[0,0]]
    lam_x_plus = 1.0 / (1.0 / lam_max + invR)
    lam_y_plus = lam_min
    log_lam_x_plus = float(np.log(lam_x_plus))
    log_lam_y_plus = float(np.log(lam_y_plus))

    # IG for diagonal Σ and rank-1 F: det(I + ΣF) = 1 + lam_max*invR
    IG = 0.5 * float(np.log(1.0 + lam_max * invR))

    V_after = V_jump_out + IG
    # Compare two hypotheses, *including the same axis-clipping behaviour* as
    # `_apply_measurement_jump` (clipping is crucial here because the swapped
    # mapping can silently hit axis bounds).
    def _clip(val: float, axis: np.ndarray) -> float:
        return float(np.clip(val, float(axis[0]), float(axis[-1])))

    # Hypothesis A (correct): eig0 axis holds logλ_max+, eig1 holds logλ_min+.
    c0_correct = _clip(log_lam_x_plus, eig0_axis)
    c1_correct = _clip(log_lam_y_plus, eig1_axis)
    V_expected_correct = 1000.0 * c0_correct + 1.0 * c1_correct

    # Hypothesis B (swapped): eig0 axis holds logλ_min+, eig1 holds logλ_max+.
    c0_swapped = _clip(log_lam_y_plus, eig0_axis)
    c1_swapped = _clip(log_lam_x_plus, eig1_axis)
    V_expected_swapped = 1000.0 * c0_swapped + 1.0 * c1_swapped

    err_correct = abs(V_after - V_expected_correct)
    err_swapped = abs(V_after - V_expected_swapped)

    _print_header("CHECK 1/2: eigenvalue-axis mapping in _apply_measurement_jump")
    print(f"Probe gridpoint idx={idx}")
    print(f"Prior axes (intended): eig0=logλ_max={log_lambda_max:.6f}, eig1=logλ_min={log_lambda_min:.6f}")
    print(f"Posterior (expected): logλ_x+={log_lam_x_plus:.6f}, logλ_y+={log_lam_y_plus:.6f}")
    print(f"IG={IG:.12f}")
    print(f"V_jump_out (returned)={V_jump_out:.12f}")
    print(f"V_after (=out+IG)={V_after:.12f}")
    print(f"V_expected_correct={V_expected_correct:.12f}  |err|={err_correct:.3e}")
    print(f"  (coords used: eig0={c0_correct:.6f}, eig1={c1_correct:.6f})")
    print(f"V_expected_swapped={V_expected_swapped:.12f}  |err|={err_swapped:.3e}")
    print(f"  (coords used: eig0={c0_swapped:.6f}, eig1={c1_swapped:.6f})")

    detected = err_swapped + 1e-9 < err_correct
    if detected:
        print("DETECTED: posterior eigenvalues appear mapped to the wrong eigen axes (swapped).")
    else:
        print("NOT DETECTED: eigen-axis mapping does not look swapped under this probe.")
    return detected


def check_extra_jump_at_t0_finite_horizon() -> bool:
    """
    Detect whether the finite-horizon planning window includes the current
    measurement time and therefore causes the HJB solver to apply a measurement
    jump at t = t0.

    We instrument `measurement_jump` to record the times it is invoked by
    `solve_hjb`. If t0 appears, that indicates an (often unintended) jump at
    the start of the window.
    """
    import numpy as np

    from hjb import OCProblem, solve_hjb
    from drone_soc import Params
    from finite_horizon import _planning_window

    par = Params(
        t_end=10.0,
        dt=1.0,
        dt_hjb=1.0,
        dx=1.0,
        noise_std=1.0,
        meas_times=np.array([0.0, 5.0, 10.0], dtype=float),
        num_obs_in_hjb=1,
        save_videos=False,
        stop_when_confident=False,
    )
    t_now = 0.0
    (T0, T1), meas_times_window = _planning_window(t_now, par)

    # Minimal 1D grid (needs >= 3 points because solve_hjb uses edge_order=2).
    x_axis = np.array([-1.0, 0.0, 1.0], dtype=float)
    grid_axes = (x_axis,)
    called_times: List[float] = []

    def terminal_cost(s):
        return 0.0 * s[0]

    def running_cost(s, a, t):
        return 0.0 * s[0]

    def dynamics(s, a, t):
        return 0.0 * s

    def analytic_minimizer(mesh, gradV, t_now_local):
        # Control has same shape as state dynamics expects (1, ...).
        a_opt = 0.0 * mesh
        return a_opt, None

    def measurement_jump(axes, Vnext, t_jump):
        called_times.append(float(t_jump))
        return Vnext

    oc = OCProblem(
        dynamics=dynamics,
        running_cost=running_cost,
        terminal_cost=terminal_cost,
        grid_axes=grid_axes,
        dt=float(par.dt_hjb),
        T0=float(T0),
        T=float(T1),
        discount=0.0,
        diffusion_coeff=0.0,
        analytic_minimizer=analytic_minimizer,
        measurement_times=np.asarray(meas_times_window, dtype=float),
        measurement_jump=measurement_jump,
        num_spatial_axes=1,
        boundary_mode="clip",
    )
    solve_hjb(oc)

    _print_header("CHECK 2/2: extra measurement jump at t=t0 in finite-horizon windows")
    print(f"t_now={t_now}  planning window=(T0={T0}, T1={T1})")
    print(f"meas_times_window={meas_times_window.tolist()}")
    print(f"measurement_jump called at times={called_times}")

    detected = any(abs(t - float(T0)) < 1e-12 for t in called_times)
    if detected:
        print("DETECTED: measurement_jump was invoked at t=T0 (start of window).")
    else:
        print("NOT DETECTED: no measurement_jump call at t=T0 under this probe.")
    return detected


def main() -> int:
    detected_any = False

    try:
        detected_any |= check_eigen_axes_swapped()
    except Exception as exc:
        print("ERROR while running eigen-axis check:", repr(exc))
        detected_any = True

    try:
        detected_any |= check_extra_jump_at_t0_finite_horizon()
    except Exception as exc:
        print("ERROR while running finite-horizon jump-t0 check:", repr(exc))
        detected_any = True

    _print_header("SUMMARY")
    if detected_any:
        print("One or more mismatches were detected (or checks errored).")
        print("Exit code: 1")
        return 1
    print("No mismatches detected by these probes.")
    print("Exit code: 0")
    return 0


def test_measurement_jump_preserves_eigenvalue_axis_order():
    assert not check_eigen_axes_swapped()


def test_finite_horizon_excludes_current_measurement_jump():
    assert not check_extra_jump_at_t0_finite_horizon()


if __name__ == "__main__":
    raise SystemExit(main())

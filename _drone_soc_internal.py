from __future__ import annotations

from typing import Dict, Tuple, TYPE_CHECKING

import numpy as np
from numpy.linalg import eigh

from hjb import n_linear_interpolate
from measurement import get_plume_axis_angle

if TYPE_CHECKING:
    from drone_soc import DroneProblemState, Params, ProblemSpec

def covariance_from_eigen(lambda1, lambda2, angle):
    """
    Vectorized function to compute covariance matrices from eigenvalues and angle.

    Args:
        lambda1: LOG of the largest eigenvalue(s).
        lambda2: LOG of the smallest eigenvalue(s).
        angle: Rotation angle(s) in radians.

    Returns:
        Covariance matrix or array of covariance matrices.
    """
    lambda1 = np.atleast_1d(lambda1)
    lambda2 = np.atleast_1d(lambda2)
    angle = np.atleast_1d(angle)
    lambda1_lin = np.exp(lambda1)
    lambda2_lin = np.exp(lambda2)
    lambda1_lin, lambda2_lin, angle = np.broadcast_arrays(lambda1_lin, lambda2_lin, angle)
    n = len(lambda1_lin)
    c = np.cos(angle)
    s = np.sin(angle)
    Q = np.stack([[c, -s], [s, c]], axis=0)
    Q = np.atleast_3d(Q)
    Q = np.moveaxis(Q, -1, 0)
    lambdas = np.stack([lambda1_lin, lambda2_lin], axis=-1).reshape(n, 2)
    result = np.einsum("kij,kj,kmj->kim", Q, lambdas, Q)
    return np.squeeze(result)


def eigen_from_covariance(Sigma: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Return LOG-eigenvalues and principal angle of the largest-eigenvalue eigenvector.
    """
    vals, vecs = eigh(Sigma)
    lam_min, lam_max = vals[0], vals[1]
    angle = np.arctan2(vecs[1, 1], vecs[0, 1])
    vals_reordered = np.array([lam_max, lam_min], dtype=float)
    vals_reordered = np.clip(vals_reordered, a_min=1e-16, a_max=None)
    return np.log(vals_reordered), angle


def construct_hjb_grid_axes(
    par: "Params",
    problem: "ProblemSpec",
    s: "DroneProblemState",
    num_drones: int,
    wind_aligned: bool,
) -> Tuple[np.ndarray, ...]:
    """
    Build the tuple of grid axes for the HJB state, including relative positions,
    covariance eigen-parameters, and optional intensity uncertainty axis.
    """
    axes: list[np.ndarray] = []
    dx_primary = float(par.dx)
    dx_secondary = float(par.dx_secondary) if getattr(par, "dx_secondary", None) is not None else dx_primary
    Lx = float(problem.domain[0][1] - problem.domain[0][0])
    Ly = float(problem.domain[1][1] - problem.domain[1][0])
    for k in range(num_drones):
        dx_k = dx_primary if k == 0 else dx_secondary
        Nx = max(2, int(round(Lx / max(dx_k, 1e-12))))
        Ny = max(2, int(round(Ly / max(dx_k, 1e-12))))
        z1_axis = np.linspace(problem.domain[0][0] - s.belief.mu[0], problem.domain[0][1] - s.belief.mu[0], Nx)
        z2_axis = np.linspace(problem.domain[1][0] - s.belief.mu[1], problem.domain[1][1] - s.belief.mu[1], Ny)
        axes.append(z1_axis)
        axes.append(z2_axis)

    vals_log, ang0 = eigen_from_covariance(s.belief.sigma)
    f = par.belief_discr_factor
    eig_0_grid = np.linspace(vals_log[0] - np.log1p(f), vals_log[0], par.num_belief_discr)
    eig_1_grid = np.linspace(vals_log[1] - np.log1p(f), vals_log[1], par.num_belief_discr)
    axes.append(eig_0_grid)
    axes.append(eig_1_grid)
    if not wind_aligned:
        eig_angle_grid = np.linspace(ang0 - np.pi / 4, ang0 + np.pi / 4, par.num_belief_discr)
        axes.append(eig_angle_grid)

    if s.belief.intensity_enabled:
        log_var_q = np.log(np.maximum(s.belief.q_log_var, 1e-12))
        intensity_axis = np.linspace(log_var_q - np.log1p(f), log_var_q, par.num_belief_discr)
        axes.append(intensity_axis)

    return tuple(axes)


def spatial_grid_resolution_metadata(par: "Params", problem: "ProblemSpec") -> dict:
    """Return nominal resolution, node counts, and actual endpoint spacing."""
    records = {}
    for label, nominal in (
        ("primary", float(par.dx)),
        ("secondary", float(par.dx_secondary) if par.dx_secondary is not None else float(par.dx)),
    ):
        counts = []
        spacings = []
        for bounds in np.asarray(problem.domain, dtype=float):
            length = float(bounds[1] - bounds[0])
            count = max(2, int(round(length / max(nominal, 1e-12))))
            counts.append(count)
            spacings.append(length / (count - 1))
        records[label] = {
            "role": label,
            "nominal_dx": nominal,
            "node_counts": counts,
            "actual_spacings": spacings,
        }
    return records


def default_fixed_state_indices(
    grid_axes: Tuple[np.ndarray, ...],
    spatial_axes: Tuple[int, int],
) -> Dict[int, int]:
    """
    Fix all axes except the ones being visualized to their middle index.
    """
    fixed: Dict[int, int] = {}
    for idx, axis in enumerate(grid_axes):
        if idx in spatial_axes:
            continue
        fixed[idx] = len(axis) // 2
    return fixed


def _apply_measurement_jump(
    grid_axes: Tuple[np.ndarray, ...],
    Vnext: np.ndarray,
    t_now: float,
    s: "DroneProblemState",
    par: "Params",
    problem: "ProblemSpec",
    num_drones: int,
    wind_aligned: bool = False,
) -> np.ndarray:
    eps = 1e-12
    mesh = np.meshgrid(*grid_axes, indexing="ij")
    grid_shape = mesh[0].shape

    intensity_enabled = s.belief.intensity_enabled
    offset = 1 if intensity_enabled else 0
    if wind_aligned:
        eig0_idx = len(grid_axes) - 2 - offset
        eig1_idx = len(grid_axes) - 1 - offset
        E1L = mesh[eig0_idx]
        E2L = mesh[eig1_idx]
        ang_val = get_plume_axis_angle(problem.wind)
        Sigma = covariance_from_eigen(E1L.ravel(), E2L.ravel(), ang_val).reshape((*grid_shape, 2, 2))
    else:
        eig0_idx = len(grid_axes) - 3 - offset
        eig1_idx = len(grid_axes) - 2 - offset
        angle_idx = len(grid_axes) - 1 - offset
        E1L = mesh[eig0_idx]
        E2L = mesh[eig1_idx]
        ANG = mesh[angle_idx]
        Sigma = covariance_from_eigen(E1L.ravel(), E2L.ravel(), ANG.ravel()).reshape((*grid_shape, 2, 2))

    F11 = np.zeros(grid_shape)
    F22 = np.zeros(grid_shape)
    F12 = np.zeros(grid_shape)
    Fq = np.zeros(grid_shape) if intensity_enabled else None
    invR = 1.0 / (par.noise_std ** 2)
    base_intensity = np.exp(s.belief.q_log) if intensity_enabled else getattr(problem.pollutant, "q", 1.0)
    for k in range(num_drones):
        Zx = mesh[2 * k]
        Zy = mesh[2 * k + 1]
        Xabs = Zx + s.belief.mu[0]
        Yabs = Zy + s.belief.mu[1]
        domain = np.stack([Xabs.ravel(), Yabs.ravel()], axis=-1)
        J = problem.pollutant.get_concentration_gradient_wrt_theta(
            domain,
            t_now,
            theta=s.belief.mu,
            intensity=base_intensity,
        )
        Jx = J[:, 0].reshape(grid_shape)
        Jy = J[:, 1].reshape(grid_shape)
        F11 += invR * (Jx * Jx)
        F22 += invR * (Jy * Jy)
        F12 += invR * (Jx * Jy)
        if intensity_enabled:
            dC_dlogq = problem.pollutant.get_concentration_derivative_wrt_logq(
                domain,
                t_now,
                theta=s.belief.mu,
                intensity=base_intensity,
            )
            dC_dlogq = dC_dlogq.reshape(grid_shape)
            Fq += invR * (dC_dlogq * dC_dlogq)

    detS = Sigma[..., 0, 0] * Sigma[..., 1, 1] - Sigma[..., 0, 1] * Sigma[..., 1, 0]
    detS = np.maximum(detS, eps)
    invS00 = Sigma[..., 1, 1] / detS
    invS11 = Sigma[..., 0, 0] / detS
    invS01 = -Sigma[..., 0, 1] / detS
    invS10 = -Sigma[..., 1, 0] / detS
    A00 = invS00 + F11
    A11 = invS11 + F22
    A01 = invS01 + F12
    A10 = invS10 + F12
    detA = A00 * A11 - A01 * A10
    detA = np.maximum(detA, eps)
    Splus00 = A11 / detA
    Splus11 = A00 / detA
    Splus01 = -A01 / detA
    Splus10 = -A10 / detA

    tr = Splus00 + Splus11
    det = Splus00 * Splus11 - Splus01 * Splus10
    det = np.maximum(det, eps)
    disc = np.sqrt(np.maximum(tr * tr - 4 * det, 0.0))
    # Eigenvalues of the posterior covariance (lam_max >= lam_min).
    lam_min = 0.5 * (tr - disc)
    lam_max = 0.5 * (tr + disc)
    lam_min = np.clip(lam_min, a_min=1e-16, a_max=None)
    lam_max = np.clip(lam_max, a_min=1e-16, a_max=None)

    log_lam_max = np.log(lam_max)
    log_lam_min = np.log(lam_min)

    axes = grid_axes
    coords = list(mesh)
    if wind_aligned:
        # Grid convention (see `eigen_from_covariance` / `construct_hjb_grid_axes`):
        # eig0 axis is log λ_max, eig1 axis is log λ_min.
        coords[eig0_idx] = np.clip(log_lam_max, grid_axes[eig0_idx][0], grid_axes[eig0_idx][-1])
        coords[eig1_idx] = np.clip(log_lam_min, grid_axes[eig1_idx][0], grid_axes[eig1_idx][-1])
    else:
        vx = Splus01
        vy = lam_max - Splus00
        ang_plus = np.arctan2(vy, vx)
        a0 = grid_axes[angle_idx][0]
        ang_wrapped = (ang_plus - a0) % np.pi + a0
        coords[eig0_idx] = np.clip(log_lam_max, grid_axes[eig0_idx][0], grid_axes[eig0_idx][-1])
        coords[eig1_idx] = np.clip(log_lam_min, grid_axes[eig1_idx][0], grid_axes[eig1_idx][-1])
        coords[angle_idx] = ang_wrapped
    if intensity_enabled:
        intensity_idx = -1
        var_q = np.exp(mesh[intensity_idx])
        invSq = 1.0 / np.maximum(var_q, eps)
        invSq_plus = invSq + Fq
        Sq_plus = 1.0 / np.maximum(invSq_plus, eps)
        log_var_q_plus = np.log(np.maximum(Sq_plus, eps))
        coords[intensity_idx] = np.clip(log_var_q_plus, grid_axes[intensity_idx][0], grid_axes[intensity_idx][-1])
    V_after = n_linear_interpolate(Vnext, axes, coords)

    IG_mat00 = Sigma[..., 0, 0] * F11 + Sigma[..., 0, 1] * F12
    IG_mat01 = Sigma[..., 0, 0] * F12 + Sigma[..., 0, 1] * F22
    IG_mat10 = Sigma[..., 1, 0] * F11 + Sigma[..., 1, 1] * F12
    IG_mat11 = Sigma[..., 1, 0] * F12 + Sigma[..., 1, 1] * F22
    det_I = (1.0 + IG_mat00) * (1.0 + IG_mat11) - IG_mat01 * IG_mat10
    det_I = np.maximum(det_I, eps)
    if intensity_enabled:
        var_q = np.exp(mesh[-1])
        det_I *= np.maximum(1.0 + var_q * Fq, eps)
    IG = 0.5 * np.log(np.maximum(det_I, 1e-12))
    return V_after - IG

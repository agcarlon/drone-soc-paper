# ── 0. imports ────────────────────────────────────────────────────────────────
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Protocol, Callable, Tuple, Sequence, Optional
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import map_coordinates


@dataclass(frozen=True)
class OCProblem:
    """All data that defines one deterministic optimal‑control problem."""

    dynamics      : Callable
    running_cost  : Callable
    terminal_cost : Callable

    # Discretisation
    grid_axes: Tuple[np.ndarray, ...]
    dt: float
    T: float
    T0: float = 0.0
    discount: float = 0.0

    # Control constraints
    u_min: Optional[np.ndarray] = None
    u_max: Optional[np.ndarray] = None

    # Spatial diffusion coefficient for noisy dynamics.
    # If the state dynamics are dX_t = f(X_t, u_t) dt + sigma dW_t,
    # then diffusion_coeff should be set to 0.5 * sigma**2, so that
    # the HJB includes the term diffusion_coeff * Δ_x V.
    diffusion_coeff: float = 0.0

    # Analytic minimizer: must be callable as (mesh, gradV, t) -> (a_opt, H_min)
    analytic_minimizer: Optional[Callable] = None

    # Optional measurement jump (impulse) at scheduled times
    measurement_times: Optional[np.ndarray] = None
    measurement_jump: Optional[Callable] = None

    # Explicit number of spatial axes (for gradient computation). If None, assume
    # all but the last 3 axes are spatial (legacy behaviour).
    num_spatial_axes: Optional[int] = 2
    # Boundary handling for semi-Lagrangian foot points:
    #   "clip" (default) clamps to domain bounds,
    #   "reflect" mirrors across the boundary (approx. Neumann),
    #   "extrapolate" leaves points out of bounds to be linearly extrapolated.
    boundary_mode: str = "clip"


def solve_hjb(problem: OCProblem) -> Tuple[Callable, np.ndarray, Tuple[np.ndarray, ...]]:
    """
    Solve the HJB and also return the full V(t, x) grid over the solver time grid.
    Returns:
        feedback: Callable f(t, x_state) -> alpha*
        V_time: ndarray with shape (Nt+1, *grid_shape)
        axes: the grid axes (tuple of 1D arrays)
    """
    T0 = problem.T0
    T = problem.T
    dt = problem.dt
    axes = problem.grid_axes

    mesh = np.meshgrid(*axes, indexing='ij')
    ds = [ax[1] - ax[0] for ax in axes]
    shape = tuple(len(ax) for ax in axes)
    duration = float(T - T0)
    Nt = int(round(duration / dt))
    if Nt < 1 or not np.isclose(Nt * dt, duration, rtol=1e-10, atol=1e-12):
        raise ValueError("The HJB interval length must be an integer multiple of dt.")
    step = duration / Nt
    s_dims = int(problem.num_spatial_axes)
    bmode = (problem.boundary_mode or "clip").lower()

    def _apply_boundary(coord: np.ndarray, ax: np.ndarray) -> np.ndarray:
        if bmode == "clip":
            return np.clip(coord, ax[0], ax[-1])
        if bmode == "reflect":
            low = ax[0]; high = ax[-1]
            coord_ref = coord.copy()
            coord_ref = np.where(coord_ref < low, 2 * low - coord_ref, coord_ref)
            coord_ref = np.where(coord_ref > high, 2 * high - coord_ref, coord_ref)
            return coord_ref
        # extrapolate: do nothing, letting the interpolator handle it
        return coord

    meas_idx_set = set()
    if problem.measurement_times is not None and problem.measurement_jump is not None:
        for tm in problem.measurement_times:
            k = int(round((tm - T0) / dt))
            if 0 <= k <= Nt:
                meas_idx_set.add(k)

    V_next = problem.terminal_cost(np.stack(mesh))
    if Nt in meas_idx_set:
        V_next = problem.measurement_jump(problem.grid_axes, V_next, T)

    V_time = np.empty((Nt + 1, *shape), dtype=float)
    V_time[-1] = V_next

    for k in reversed(range(Nt)):
        t_now = T0 + k * step
        if k in meas_idx_set:
            V_next = problem.measurement_jump(problem.grid_axes, V_next, t_now)
        Vn = V_next
        gradient_edge_order = 2 if all(len(axis) >= 3 for axis in axes) else 1
        gradV = np.stack(np.gradient(Vn, *ds, edge_order=gradient_edge_order), axis=0)

        if problem.diffusion_coeff != 0.0:
            lapV = np.zeros_like(Vn)
            for i in range(s_dims):
                edge_order = 2 if len(axes[i]) >= 3 else 1
                dVi = np.gradient(Vn, ds[i], axis=i, edge_order=edge_order)
                d2Vi = np.gradient(dVi, ds[i], axis=i, edge_order=edge_order)
                lapV += d2Vi
        else:
            lapV = 0.0

        if problem.analytic_minimizer is None:
            raise NotImplementedError("Provide analytic_minimizer")
        a_opt, H_min = problem.analytic_minimizer(np.stack(mesh), gradV, t_now)
        L = problem.running_cost(np.stack(mesh), a_opt, t_now)
        f = problem.dynamics(np.stack(mesh), a_opt, t_now)
        s_next: list[np.ndarray] = []
        for i in range(len(axes)):
            if i < s_dims:
                s_next.append(mesh[i] + step * f[i])
            else:
                s_next.append(mesh[i])
        s_next = [_apply_boundary(coord, ax) for coord, ax in zip(s_next, axes)]
        Vnext_interp = n_linear_interpolate(Vn, axes, s_next, boundary_mode=bmode)
        rate = float(problem.discount)
        continuation = float(np.exp(-rate * step))
        source_weight = step if rate == 0.0 else -float(np.expm1(-rate * step)) / rate
        V_next = continuation * Vnext_interp + source_weight * (
            L + problem.diffusion_coeff * lapV
        )
        V_time[k] = V_next

    ts = np.linspace(T0, T, Nt + 1)

    V_interp = RegularGridInterpolator((ts, *axes), V_time, bounds_error=False, fill_value=None)
    s_dims = int(problem.num_spatial_axes)

    def feedback(t, x) -> np.ndarray:
        assert T0 <= t <= T
        grad = np.zeros(len(axes), dtype=float)
        x_clipped = []
        for i, ax in enumerate(axes):
            x_clipped.append(float(_apply_boundary(np.array([x[i]], dtype=float), ax)[0]))
        x_clipped = list(x_clipped)
        for i in range(s_dims):
            hi = 0.01 * ds[i]
            lo = axes[i][0]; hi_max = axes[i][-1]
            xp = x_clipped.copy(); xm = x_clipped.copy()
            xp[i] = float(_apply_boundary(np.array([xp[i] + hi], dtype=float), axes[i])[0])
            xm[i] = float(_apply_boundary(np.array([xm[i] - hi], dtype=float), axes[i])[0])
            Vp = V_interp((t, *xp))
            Vm = V_interp((t, *xm))
            grad[i] = (Vp - Vm) / (2.0 * hi)
        mesh_point = [np.array([coord], dtype=float) for coord in x_clipped]
        gradV_point = grad.reshape((len(axes),) + (1,) * len(axes))
        a_opt, _ = problem.analytic_minimizer(np.stack(mesh_point), gradV_point, float(t))
        return np.squeeze(a_opt)

    return feedback, V_time, axes


def n_linear_interpolate(Vslice: np.ndarray,
                         axes: Tuple[np.ndarray, ...],
                         coords: Sequence[np.ndarray],
                         boundary_mode: str = "clip") -> np.ndarray:
    """
    Vectorised N-linear interpolation.

    For uniform axes, use `scipy.ndimage.map_coordinates`, which is
    substantially faster on the high-dimensional HJB grids used here. Fall back
    to `RegularGridInterpolator` for non-uniform rectilinear axes.

    Args:
        Vslice (ndarray): Scalar field tabulated on the rectilinear grid.
        axes (tuple): Grid nodes for each dimension.
        coords (iterable): Interpolation points for each axis.
        boundary_mode: Boundary handling for points outside the grid. Uniform
            grids use the fast `map_coordinates` path for "clip" and "reflect".
            The "extrapolate" mode preserves the previous
            `RegularGridInterpolator(fill_value=None)` behavior.

    Returns:
        ndarray: Interpolated values at the given coordinates.
    """
    if len(coords) != len(axes):
        raise ValueError(
            f"n_linear_interpolate: expected {len(axes)} coordinate arrays (one per axis), "
            f"got {len(coords)}. This usually means you're only providing spatial coordinates "
            f"for a higher-dimensional state grid."
        )
    mode = (boundary_mode or "clip").lower()
    if _axes_are_uniform(axes) and mode != "extrapolate":
        index_coords = []
        for ax, c in zip(axes, coords):
            dx = float(ax[1] - ax[0])
            index_coords.append((np.asarray(c, dtype=float) - float(ax[0])) / dx)
        stacked = np.stack(index_coords, axis=0)
        map_mode = "reflect" if mode == "reflect" else "nearest"
        return map_coordinates(Vslice, stacked, order=1, mode=map_mode)

    rgi = RegularGridInterpolator(axes, Vslice, bounds_error=False, fill_value=None)
    pts = np.zeros((len(Vslice.ravel()), len(Vslice.shape)))
    for i, c in enumerate(coords):
        pts[:, i] = c.ravel()
    vals = rgi(pts)
    return vals.reshape(coords[0].shape)


def _axes_are_uniform(axes: Tuple[np.ndarray, ...], rtol: float = 1e-10, atol: float = 1e-12) -> bool:
    for ax in axes:
        ax = np.asarray(ax, dtype=float)
        if ax.ndim != 1 or ax.size < 2:
            return False
        diffs = np.diff(ax)
        if not np.allclose(diffs, diffs[0], rtol=rtol, atol=atol):
            return False
    return True


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    c1 = 1.0  # effort weight

    def dynamics(s: np.ndarray, a: np.ndarray, t: float) -> np.ndarray:
        """Simple integrator: ẋ = u"""
        return a

    def running_cost(s: np.ndarray, a: np.ndarray, t: float) -> float:
        """Quadratic running cost."""
        return 0.5 * np.sum(a ** 2, axis=-1)

    def terminal_cost(s: np.ndarray) -> float:
        """Quadratic terminal cost."""
        return 0.5 * np.sum(s ** 2, axis=-1)

    def quad_minimizer(s, _u_dummy, t, gradV):
        """
        Analytic minimizer for quadratic Hamiltonian.
        Returns optimal control and minimum Hamiltonian.
        """
        a_opt = -gradV
        H_min = -0.5 * np.sum(gradV ** 2, axis=-1)
        return a_opt, H_min

    x_grid = np.linspace(-2.0, 2.0, 201)
    T = 1.0
    Nt = 50

    prob = OCProblem(
        dynamics=dynamics,
        running_cost=running_cost,
        terminal_cost=terminal_cost,
        grid_axes=(x_grid,),
        dt=T / Nt,
        T=T,
        discount=0.0,
        u_min=np.array([-1.0]),
        u_max=np.array([+1.0]),
        analytic_minimizer=quad_minimizer,
    )

    print("Solving toy 1‑D HJB…")
    feedback_interp = solve_hjb(prob)
    print("Done.  V(0,x) sample:", feedback_interp(0.0, np.array([0.0])))

    X, TT = np.meshgrid(x_grid, np.linspace(0, T, Nt + 1))
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, TT, feedback_interp(TT, X), cmap='viridis', linewidth=0)
    ax.set_xlabel('x')
    ax.set_ylabel('t')
    ax.set_zlabel('V(t,x)')
    ax.set_title('Value function surface for LQR toy problem')
    plt.tight_layout()
    plt.show()

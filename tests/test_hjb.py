import numpy as np

from hjb import OCProblem, solve_hjb, n_linear_interpolate


def _zero_analytic_minimizer(mesh: np.ndarray, gradV: np.ndarray, t: float):
    """
    Simple analytic minimizer that always returns zero control.
    """
    return np.zeros_like(mesh), None


def _quad_analytic_minimizer(mesh: np.ndarray, gradV: np.ndarray, t: float):
    """
    Analytic minimizer for quadratic control effort:
    running_cost ~ 0.5 * ||u||^2 and dynamics x_dot = u.
    The minimizer is u* = -gradV.
    """
    a_opt = -gradV
    return a_opt, None


def test_hjb_zero_problem_returns_zero_value_and_control():
    """
    For a problem with zero dynamics, zero running cost, and zero terminal cost,
    the value function should be identically zero and the optimal control zero.
    """
    axes = (np.linspace(-1.0, 1.0, 5),)
    problem = OCProblem(
        dynamics=lambda x, u, t: 0.0 * x,
        running_cost=lambda x, u, t: 0.0 * x[0],
        terminal_cost=lambda x: 0.0 * x[0],
        grid_axes=axes,
        dt=0.1,
        T=1.0,
        discount=0.0,
        diffusion_coeff=0.0,
        analytic_minimizer=_zero_analytic_minimizer,
        measurement_times=None,
        measurement_jump=None,
        num_spatial_axes=1,
    )

    feedback, V_time, axes_out = solve_hjb(problem)

    assert V_time.shape[1:] == (len(axes[0]),)
    assert np.allclose(V_time, 0.0)

    for x in (-1.0, 0.0, 0.5):
        a = feedback(0.0, np.array([x]))
        assert np.allclose(a, 0.0)


def test_hjb_measurement_jump_is_applied():
    """
    If a measurement jump adds a constant to V at the terminal time,
    and dynamics/running cost are zero, the whole V_time should be constant.
    """
    axes = (np.linspace(-1.0, 1.0, 5),)
    meas_times = np.array([1.0])

    def jump(grid_axes, Vnext, t_now):
        return Vnext + 1.0

    problem = OCProblem(
        dynamics=lambda x, u, t: 0.0 * x,
        running_cost=lambda x, u, t: 0.0 * x[0],
        terminal_cost=lambda x: 0.0 * x[0],
        grid_axes=axes,
        dt=0.5,
        T=1.0,
        discount=0.0,
        diffusion_coeff=0.0,
        analytic_minimizer=_zero_analytic_minimizer,
        measurement_times=meas_times,
        measurement_jump=jump,
        num_spatial_axes=1,
    )

    _, V_time, _ = solve_hjb(problem)
    # Jump of +1 applied at terminal, propagated unchanged backwards
    assert np.allclose(V_time, 1.0)


def test_hjb_uses_exact_exponential_discount_factor():
    axes = (np.linspace(-1.0, 1.0, 5),)
    rate = 0.7
    horizon = 1.0
    problem = OCProblem(
        dynamics=lambda x, u, t: 0.0 * x,
        running_cost=lambda x, u, t: 0.0 * x[0],
        terminal_cost=lambda x: np.ones_like(x[0]),
        grid_axes=axes,
        dt=0.25,
        T=horizon,
        discount=rate,
        analytic_minimizer=_zero_analytic_minimizer,
        num_spatial_axes=1,
    )
    _, values, _ = solve_hjb(problem)
    assert np.allclose(values[0], np.exp(-rate * horizon), rtol=1e-12, atol=1e-12)


def test_n_linear_interpolate_simple_1d_linear():
    """
    n_linear_interpolate should reproduce linear interpolation on a 1D grid.
    """
    x_axis = np.linspace(0.0, 1.0, 11)
    Vslice = x_axis.copy()  # V(x) = x on the grid
    axes = (x_axis,)
    # Interpolate exactly at the grid nodes using the same shape
    coords = (x_axis.copy(),)
    vals = n_linear_interpolate(Vslice, axes, coords)
    assert np.allclose(vals, x_axis, atol=1e-6)


def _make_diffusion_problem(dx: float, dt: float) -> OCProblem:
    """
    Build a small 1D HJB problem with diffusion for grid-refinement tests.
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


def test_hjb_diffusion_grid_refinement_stable():
    """
    With diffusion enabled, refining the spatial and temporal grid should
    not qualitatively change the value function.
    We compare coarse and refined solutions on the coarse grid.
    """
    # Coarse and fine problems with aligned domains
    prob_coarse = _make_diffusion_problem(dx=0.5, dt=0.2)
    prob_fine = _make_diffusion_problem(dx=0.25, dt=0.1)

    _, V_coarse, axes_coarse = solve_hjb(prob_coarse)
    _, V_fine, axes_fine = solve_hjb(prob_fine)

    x_coarse = axes_coarse[0]
    x_fine = axes_fine[0]

    # Coarse grid points should be a subset of the fine grid
    assert len(x_fine) == 2 * len(x_coarse) - 1
    assert np.allclose(x_fine[::2], x_coarse, atol=1e-12)

    V0_coarse = V_coarse[0]  # V(t=0, x) on coarse grid
    V0_fine_on_coarse = V_fine[0][::2]  # restrict fine solution to coarse nodes

    scale = max(1.0, float(np.max(np.abs(V0_fine_on_coarse))))
    diff = float(np.max(np.abs(V0_coarse - V0_fine_on_coarse)))

    # Require that the relative difference is modest; this is a soft stability check.
    # In practice this can vary with discretization choices and stabilizers in the scheme.
    assert diff / scale < 0.35

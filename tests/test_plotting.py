import numpy as np

from drone_soc import Params, ProblemSpec, DroneProblemState, Drone, Belief
from plotting.plotting import animate_value_function, default_after_measurement_callback
from plotting.plot_order1_vs_order2_distance_vs_energy import _interpolate_common_effort

# Keep a module-level reference to animations created in tests to avoid
# Matplotlib's warning about animations being deleted without rendering.
_ANIMATIONS = []


def test_tradeoff_curves_are_interpolated_on_common_effort_grid():
    grid, distances = _interpolate_common_effort([
        (np.array([0.0, 2.0, 4.0]), np.array([10.0, 6.0, 2.0])),
        (np.array([0.0, 1.0, 3.0]), np.array([9.0, 7.0, 3.0])),
    ], num_points=4)
    assert np.allclose(grid, [0.0, 1.0, 2.0, 3.0])
    assert distances.shape == (2, 4)
    assert np.allclose(distances[0], [10.0, 8.0, 6.0, 4.0])


def test_animate_value_function_smoke_no_save():
    """
    animate_value_function should create an animation object without saving to disk.
    """
    Nt_plus_1 = 4
    x_axis = np.linspace(-1.0, 1.0, 5)
    y_axis = np.linspace(-1.0, 1.0, 5)
    grid_axes = (x_axis, y_axis)
    V_time = np.zeros((Nt_plus_1, len(x_axis), len(y_axis)), dtype=float)
    for k in range(Nt_plus_1):
        V_time[k] = k * np.outer(x_axis**2, y_axis**2)

    anim = animate_value_function(
        V_time=V_time,
        grid_axes=grid_axes,
        spatial_axes=(0, 1),
        fixed_state_indices=None,
        times=np.linspace(0.0, 1.0, Nt_plus_1),
        save_path=None,
        video_length=2.0,
    )
    _ANIMATIONS.append(anim)

    # Basic checks: we have a FuncAnimation and a reasonable interval
    from matplotlib.animation import FuncAnimation

    assert isinstance(anim, FuncAnimation)
    assert anim._interval > 0

    # Avoid Matplotlib warning about animations being deleted without rendering:
    # mark the animation as if drawing had started.
    anim._draw_was_started = True


class _DummyDrone(Drone):
    def patrolling_trajectory(self, t: float):
        return self.x


class _DummyPollutant:
    def __init__(self):
        self.source = np.zeros(2)
        self.q = 1.0

    def get_pollutant_concentration(self, x, t, theta, intensity=None):
        x = np.asarray(x)
        return np.zeros(x.shape[0], dtype=float)


def test_default_after_measurement_callback_contract(monkeypatch):
    called = {"ok": False}

    def fake_plot(**kwargs):
        called["ok"] = True
        assert kwargs["y"].shape[0] == 1
        assert kwargs["x"].shape == (1, 2)

    monkeypatch.setattr("plotting.plotting.plot_prior_lkl_post_contours", fake_plot)

    par = Params(meas_times=np.array([0.0]), t_end=0.0)
    s = DroneProblemState(
        prior=Belief(np.zeros(2), np.eye(2)),
        belief=Belief(np.zeros(2), np.eye(2)),
        drones=[_DummyDrone(x0=np.zeros(2), v_max=1.0)],
    )
    pol = _DummyPollutant()
    problem = ProblemSpec(
        pollutant=pol,
        domain=np.array([[-1.0, 1.0], [-1.0, 1.0]], dtype=float),
        wind=np.zeros(2, dtype=float),
    )
    prior = Belief(mu=np.zeros(2), sigma=np.eye(2), q_log=0.0, q_log_var=1.0)
    post = Belief(mu=np.zeros(2), sigma=np.eye(2), q_log=0.1, q_log_var=0.9)

    default_after_measurement_callback(
        y=np.array([0.0]),
        positions=np.zeros((1, 2)),
        t=0.0,
        par=par,
        problem=problem,
        s=s,
        prior=prior,
        posterior=post,
        used_all_measurements=True,
    )

    assert called["ok"] is True

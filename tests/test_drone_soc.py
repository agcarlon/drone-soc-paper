import numpy as np
import json

from drone_soc import (
    Params,
    ProblemSpec,
    Belief,
    DroneProblemState,
    Drone,
    ControlPlanner,
    SigmoidCollisionAvoidance,
    SinusoidalPatrolling,
    find_pollutant_source,
    compute_collision_gradients,
    collision_potential_from_state,
)


class _TestDrone(Drone):
    """
    Minimal concrete Drone for testing integrate_dynamics.
    """

    def patrolling_trajectory(self, t: float):
        return self.x


def test_sinusoidal_patrol_moves_upwind_with_bounded_speed():
    wind = np.array([-5.0, 1.5])
    patrol = SinusoidalPatrolling(
        wind=wind,
        v_max=12.0,
        period=200.0,
        y_scale=0.8,
    )

    times = np.linspace(0.0, patrol.period, 101)
    controls = np.asarray([patrol(np.zeros(2), time) for time in times])

    assert np.all(controls @ wind < 0.0)
    assert np.all(np.linalg.norm(controls, axis=1) <= patrol.v_max + 1e-12)
    assert np.allclose(controls[0], controls[-1])


def test_integrate_dynamics_no_noise():
    """
    When dyn_noise_std == 0, integrate_dynamics should be deterministic.
    """
    par = Params(dt=0.1, dyn_noise_std=0.0)
    problem = ProblemSpec(
        pollutant=_DummyPollutant(),
        domain=np.array([[-1.0, 1.0], [-1.0, 1.0]], dtype=float),
        wind=np.zeros(2, dtype=float),
    )
    d = _TestDrone(x0=np.array([0.0, 0.0]), v_max=1.0)

    alpha = np.array([1.0, -2.0])
    d.integrate_dynamics(alpha, par, problem)
    expected = np.array([0.0, 0.0]) + alpha * par.dt

    assert np.allclose(d.x, expected)


def test_integrate_dynamics_with_noise(monkeypatch):
    """
    With dyn_noise_std > 0, integrate_dynamics should add a Gaussian perturbation
    scaled by dyn_noise_std * sqrt(dt).
    """
    par = Params(dt=0.1, dyn_noise_std=0.5)
    problem = ProblemSpec(
        pollutant=_DummyPollutant(),
        domain=np.array([[-1.0, 1.0], [-1.0, 1.0]], dtype=float),
        wind=np.zeros(2, dtype=float),
    )
    d = _TestDrone(x0=np.array([0.0, 0.0]), v_max=1.0)

    alpha = np.array([1.0, -2.0])

    def fake_randn(*shape):
        # Always return a fixed 2D vector
        return np.array([1.0, -2.0])

    monkeypatch.setattr("drone_soc.np.random.randn", fake_randn)

    d.integrate_dynamics(alpha, par, problem)
    expected = np.array([0.0, 0.0]) + alpha * par.dt
    expected += par.dyn_noise_std * np.sqrt(par.dt) * np.array([1.0, -2.0])

    assert np.allclose(d.x, expected)


def test_collision_gradients_symmetry():
    """
    For two drones aligned on the x-axis, the collision-avoidance gradients
    should be equal in magnitude and opposite in sign along x, and near zero in y.
    """
    positions = np.array([[0.0, 0.0], [1.0, 0.0]])
    collision = SigmoidCollisionAvoidance(d_min=0.5, d_max=2.0, max_value=10.0, confidence=0.99)

    grads = compute_collision_gradients(positions, collision)

    # x-components should be opposite
    assert np.allclose(grads[0, 0], -grads[1, 0], atol=1e-6)
    # y-components should be (approximately) zero and opposite
    assert np.allclose(grads[0, 1], -grads[1, 1], atol=1e-6)


def test_collision_potential_zero_when_disabled():
    """
    collision_potential_from_state should return zero when collision is None
    or when num_drones < 2.
    """
    par = Params()
    x_state_one = np.array([0.0, 0.0])
    pot1 = collision_potential_from_state(x_state_one, num_drones=1, collision=None)
    assert np.allclose(pot1, 0.0)

    collision = SigmoidCollisionAvoidance(d_min=0.5, d_max=2.0, max_value=10.0, confidence=0.99)
    pot2 = collision_potential_from_state(x_state_one, num_drones=1, collision=collision)
    assert np.allclose(pot2, 0.0)


def test_collision_potential_matches_pairwise_potential():
    """
    For two drones, collision_potential_from_state should match the pairwise
    potential evaluated at their distance.
    """
    collision = SigmoidCollisionAvoidance(d_min=0.5, d_max=2.0, max_value=10.0, confidence=0.99)
    # Two drones located at (0,0) and (1,0)
    x_state = np.array([0.0, 0.0, 1.0, 0.0])
    pot = collision_potential_from_state(x_state, num_drones=2, collision=collision)
    expected = collision.function(np.array([1.0]))
    assert np.allclose(pot, expected)


class _DummyPlanner(ControlPlanner):
    def __init__(self):
        self.plan_calls = []

    def plan(self, t_now, par, problem, s):
        self.plan_calls.append(float(t_now))

    def compute_alpha(self, t, par, problem, s):
        return np.zeros((len(s.drones), 2), dtype=float)


class _DummyPollutant:
    def __init__(self):
        self.source = np.zeros(2)
        self.q = 1.0
        self.sigma_e = 1e-3

    def get_pollutant_concentration(self, x, t, theta, intensity=None):
        x = np.asarray(x)
        return np.zeros(x.shape[0], dtype=float)


def _make_small_problem(update_mode: str = "full_batch"):
    par = Params(
        t_0=0.0,
        t_end=0.2,
        dt=0.1,
        meas_times=np.array([0.0, 0.2]),
        dt_hjb=0.1,
        dx=1.0,
        noise_std=1e-3,
        update_mode=update_mode,
    )
    s = DroneProblemState(
        prior=Belief(np.zeros(2), np.eye(2)),
        belief=Belief(np.zeros(2), np.eye(2)),
        drones=[_TestDrone(x0=np.zeros(2), v_max=1.0)],
    )
    pol = _DummyPollutant()
    problem = ProblemSpec(
        pollutant=pol,
        domain=np.array([[-1.0, 1.0], [-1.0, 1.0]], dtype=float),
        wind=np.zeros(2, dtype=float),
        c3=0.0,
        c4=0.0,
    )
    return par, problem, s


def test_find_pollutant_source_dispatches_t_now(monkeypatch):
    par, problem, s = _make_small_problem()
    planner = _DummyPlanner()

    def fake_bayes(**kwargs):
        s_local = kwargs["s"]
        return Belief(
            mu=s_local.belief.mu.copy(),
            sigma=s_local.belief.sigma.copy(),
            q_log=s_local.belief.q_log,
            q_log_var=s_local.belief.q_log_var,
        )

    monkeypatch.setattr("drone_soc.bayesian_update", fake_bayes)

    find_pollutant_source(par, problem, s, planner=planner)
    assert len(planner.plan_calls) > 0
    assert all(isinstance(call, float) for call in planner.plan_calls)


def test_find_pollutant_source_logs_belief_objects(monkeypatch):
    par, problem, s = _make_small_problem()
    planner = _DummyPlanner()

    def fake_bayes(**kwargs):
        s_local = kwargs["s"]
        return Belief(
            mu=s_local.belief.mu.copy(),
            sigma=s_local.belief.sigma.copy(),
            q_log=s_local.belief.q_log,
            q_log_var=s_local.belief.q_log_var,
        )

    monkeypatch.setattr("drone_soc.bayesian_update", fake_bayes)

    traj, log = find_pollutant_source(par, problem, s, planner=planner)
    assert traj.ndim == 3
    assert len(log["beliefs"]) >= 1
    assert all(isinstance(b, Belief) for b in log["beliefs"])
    assert traj.shape[-1] == 3
    assert np.allclose(log["times"], [0.0, 0.1, 0.2])
    assert np.allclose(log["control_times"], [0.0, 0.1])
    assert np.allclose(log["measurement_times"], [0.0, 0.2])
    assert len(log["controls"]) == 2


def test_callback_payload_full_batch_uses_accumulated_data(monkeypatch):
    par, problem, s = _make_small_problem(update_mode="full_batch")
    planner = _DummyPlanner()

    def fake_bayes(**kwargs):
        s_local = kwargs["s"]
        return Belief(
            mu=s_local.belief.mu.copy(),
            sigma=s_local.belief.sigma.copy(),
            q_log=s_local.belief.q_log,
            q_log_var=s_local.belief.q_log_var,
        )

    monkeypatch.setattr("drone_soc.bayesian_update", fake_bayes)
    callback_calls = []

    def callback(y, positions, t, par, problem, s, prior, posterior, used_all):
        callback_calls.append((y, positions, used_all))

    find_pollutant_source(par, problem, s, planner=planner, on_after_measurement=callback)

    assert len(callback_calls) == 2
    assert callback_calls[0][0].shape[0] == 1
    assert callback_calls[1][0].shape[0] == 2
    assert callback_calls[0][1].shape == (1, 2)
    assert callback_calls[1][1].shape == (2, 2)
    assert callback_calls[0][2] is True and callback_calls[1][2] is True


def test_callback_payload_sequential_uses_latest_batch(monkeypatch):
    par, problem, s = _make_small_problem(update_mode="sequential")
    planner = _DummyPlanner()

    def fake_bayes_seq(**kwargs):
        s_local = kwargs["s"]
        return Belief(
            mu=s_local.belief.mu.copy(),
            sigma=s_local.belief.sigma.copy(),
            q_log=s_local.belief.q_log,
            q_log_var=s_local.belief.q_log_var,
        )

    monkeypatch.setattr("drone_soc.bayesian_update_sequential", fake_bayes_seq)
    callback_calls = []

    def callback(y, positions, t, par, problem, s, prior, posterior, used_all):
        callback_calls.append((y, positions, used_all))

    find_pollutant_source(par, problem, s, planner=planner, on_after_measurement=callback)

    assert len(callback_calls) == 2
    assert callback_calls[0][0].shape[0] == 1
    assert callback_calls[1][0].shape[0] == 1
    assert callback_calls[0][1].shape == (1, 2)
    assert callback_calls[1][1].shape == (1, 2)
    assert callback_calls[0][2] is False and callback_calls[1][2] is False


def test_save_run_results_serializes_beliefs(tmp_path):
    par = Params(
        t_0=0.0,
        t_end=0.1,
        dt=0.1,
        meas_times=np.array([0.0]),
        output_dir=str(tmp_path),
    )
    traj = np.zeros((1, 2, 2), dtype=float)
    beliefs = [
        Belief(mu=np.array([0.0, 0.0]), sigma=np.eye(2), q_log=0.0, q_log_var=1.0),
        Belief(mu=np.array([1.0, -1.0]), sigma=2.0 * np.eye(2), q_log=0.1, q_log_var=0.8),
    ]
    log = {
        "times": [0.0, 0.1],
        "beliefs": beliefs,
        "controls": [np.zeros((1, 2), dtype=float), np.zeros((1, 2), dtype=float)],
    }

    from drone_soc import _save_run_results

    _save_run_results(par, traj, log)
    data = np.load(tmp_path / "run_results.npz")
    assert data["mus"].shape == (2, 2)
    assert data["eig_vals_log"].shape == (2, 2)
    assert data["q_logs"].shape == (2,)
    assert data["q_log_vars"].shape == (2,)


def test_params_rejects_moved_problem_fields():
    try:
        Params(domain=np.array([[-1.0, 1.0], [-1.0, 1.0]]))
    except TypeError:
        return
    raise AssertionError("Params must not accept problem-level fields like domain.")


def test_save_run_config_has_problem_section(tmp_path):
    par = Params(
        t_0=0.0,
        t_end=0.1,
        dt=0.1,
        meas_times=np.array([0.0]),
        output_dir=str(tmp_path),
    )
    pol = _DummyPollutant()
    problem = ProblemSpec(
        pollutant=pol,
        domain=np.array([[-1.0, 1.0], [-2.0, 2.0]], dtype=float),
        wind=np.array([0.3, -0.1], dtype=float),
        c1=1.0,
        c2=2.0,
        c3=3.0,
        c3_terminal=4.0,
        c4=5.0,
    )
    s = DroneProblemState(
        prior=Belief(np.zeros(2), np.eye(2)),
        belief=Belief(np.zeros(2), np.eye(2)),
        drones=[_TestDrone(x0=np.zeros(2), v_max=1.0)],
    )
    planner = _DummyPlanner()
    from drone_soc import _save_run_config

    _save_run_config(par, problem, s, planner)
    with open(tmp_path / "run_config.json", "r") as f:
        cfg = json.load(f)
    assert "params" in cfg
    assert "problem" in cfg
    assert "pollutant" in cfg["problem"]
    assert cfg["problem"]["domain"] == [[-1.0, 1.0], [-2.0, 2.0]]

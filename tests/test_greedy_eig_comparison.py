import json
import numpy as np

from experiments.exp_greedy_eig_comparison import _cached_patrol_matches
from drone_soc import Params, ProblemSpec
from greedy_eig import _greedy_eig_targets


def _write_config(path, *, period=40.0, y_scale=0.99, planner="PatrolOnlyPlanner"):
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({
        "problem": {"patrol": {"period": period, "y_scale": y_scale}},
        "planner": {"type": planner},
    }))


def test_cached_patrol_matches_requested_parameters(tmp_path):
    metrics_path = tmp_path / "patrol" / "metrics.json"
    _write_config(metrics_path.parent / "run_config.json")

    assert _cached_patrol_matches(
        metrics_path, expected_period=40.0, expected_y_scale=0.99
    )


def test_cached_patrol_rejects_stale_or_missing_configuration(tmp_path):
    metrics_path = tmp_path / "patrol" / "metrics.json"
    _write_config(metrics_path.parent / "run_config.json", period=200.0, y_scale=0.8)

    assert not _cached_patrol_matches(
        metrics_path, expected_period=40.0, expected_y_scale=0.99
    )
    (metrics_path.parent / "run_config.json").unlink()
    assert not _cached_patrol_matches(
        metrics_path, expected_period=40.0, expected_y_scale=0.99
    )


def test_greedy_targets_satisfy_exact_reachable_balls_and_domain():
    class Pollutant:
        q = 1.0

        def get_concentration_gradient_wrt_theta(self, x, t, theta, intensity=None):
            x = np.atleast_2d(x)
            return x - np.asarray(theta)

    current = np.array([[0.8, 0.0], [-0.8, 0.0]])
    par = Params(t_end=1.0, dt=0.5, dt_hjb=0.5, meas_times=np.array([0.0, 1.0]), v_max=0.5)
    problem = ProblemSpec(
        pollutant=Pollutant(),
        domain=np.array([[-1.0, 1.0], [-1.0, 1.0]]),
        wind=np.zeros(2),
    )
    targets, diagnostics = _greedy_eig_targets(
        current_positions=current,
        belief_mu=np.zeros(2),
        belief_sigma=np.eye(2),
        belief_q_log=None,
        belief_q_log_var=None,
        problem=problem,
        par=par,
        meas_interval=1.0,
        num_restarts=3,
        rng=np.random.default_rng(0),
        return_diagnostics=True,
    )
    assert np.all(np.linalg.norm(targets - current, axis=1) <= 0.5 + 1e-7)
    assert np.all(targets >= -1.0 - 1e-12)
    assert np.all(targets <= 1.0 + 1e-12)
    assert len(diagnostics) == 3

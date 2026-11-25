import numpy as np

from measurement import Pollutant
from drone_soc import (
    Params,
    ProblemSpec,
    Belief,
    DroneProblemState,
    Drone,
    bayesian_update,
    bayesian_update_sequential,
)


class _DummyDrone(Drone):
    def patrolling_trajectory(self, t: float):
        return self.x


class _DummyPollutant(Pollutant):
    def __init__(self, q: float = 1.0):
        super().__init__(
            source=np.zeros(2, dtype=float),
            q=float(q),
            wind=np.zeros(2, dtype=float),
            sigma_e=0.1,
        )

    def get_pollutant_concentration(self, x, t, theta, intensity=None):
        x = np.asarray(x)
        theta = np.asarray(theta)
        q = float(self.q if intensity is None else intensity)
        return np.full(x.shape[0], q * theta[0], dtype=float)


def _make_simple_bayes_problem(use_intensity: bool = False):
    q_log_prior = 0.0
    q_log_var_prior = 0.5
    par = Params(
        t_0=0.0,
        t_end=1.0,
        dt=1.0,
        meas_times=np.array([0.0, 1.0]),
        dx=1.0,
        noise_std=0.1,
    )
    s = DroneProblemState(
        prior=Belief(
            mu=np.zeros(2, dtype=float),
            sigma=10.0 * np.eye(2),
            q_log=q_log_prior if use_intensity else None,
            q_log_var=q_log_var_prior if use_intensity else None,
        ),
        belief=Belief(
            mu=np.zeros(2, dtype=float),
            sigma=10.0 * np.eye(2),
            q_log=q_log_prior if use_intensity else None,
            q_log_var=q_log_var_prior if use_intensity else None,
        ),
        drones=[_DummyDrone(x0=np.zeros(2), v_max=1.0)],
    )
    return par, s


def _linear_concentration(x, t, theta):
    """
    Simple linear "concentration" function used for testing Bayesian updates.
    C(x, t, theta) = theta[0] (independent of x, t)
    """
    x = np.asarray(x)
    theta = np.asarray(theta)
    return np.full(x.shape[0], theta[0], dtype=float)


def test_bayesian_update_reduces_variance_and_moves_towards_true():
    """
    bayesian_update should move the mean towards the true parameter
    and reduce the posterior covariance trace.
    """
    par, s = _make_simple_bayes_problem()
    true_theta = np.array([1.0, -0.5])
    x_meas = np.zeros((5, 2))
    t = 0.5
    y = _linear_concentration(x_meas, t, true_theta)
    pol = _DummyPollutant(q=1.0)
    problem = ProblemSpec(
        pollutant=pol,
        domain=np.array([[-1.0, 1.0], [-1.0, 1.0]], dtype=float),
        wind=np.zeros(2, dtype=float),
    )

    mu_prior = s.prior.mu.copy()
    sigma_prior = s.prior.sigma.copy()

    result = bayesian_update(
        y=y,
        x=x_meas,
        t=t,
        par=par,
        problem=problem,
        s=s,
    )

    assert isinstance(result, Belief)
    assert np.linalg.norm(result.mu - true_theta) < np.linalg.norm(mu_prior - true_theta)
    assert np.trace(result.sigma) < np.trace(sigma_prior)
    assert result.diagnostics is not None
    assert "optimizer_status" in result.diagnostics
    assert "information_condition_number" in result.diagnostics
    assert "pseudoinverse_used" in result.diagnostics
    assert "covariance_eigenvalues_clipped" in result.diagnostics


def test_bayesian_update_sequential_reduces_variance():
    """
    bayesian_update_sequential should reduce variance relative to the current belief.
    """
    par, s = _make_simple_bayes_problem()
    s.belief = Belief(
        mu=s.prior.mu.copy(),
        sigma=s.prior.sigma.copy(),
        q_log=s.prior.q_log,
        q_log_var=s.prior.q_log_var,
    )

    true_theta = np.array([0.5, 0.2])
    x_meas = np.zeros((3, 2))
    t = 0.5
    y = _linear_concentration(x_meas, t, true_theta)
    pol = _DummyPollutant(q=1.0)
    problem = ProblemSpec(
        pollutant=pol,
        domain=np.array([[-1.0, 1.0], [-1.0, 1.0]], dtype=float),
        wind=np.zeros(2, dtype=float),
    )

    result = bayesian_update_sequential(
        y=y,
        x=x_meas,
        t=t,
        par=par,
        problem=problem,
        s=s,
    )

    assert isinstance(result, Belief)
    assert np.trace(result.sigma) < np.trace(s.prior.sigma)


def test_bayesian_update_intensity_reduces_variance():
    """
    With intensity belief enabled, the posterior over log q should move toward the
    true log intensity and reduce its variance.
    """
    par, s = _make_simple_bayes_problem(use_intensity=True)
    true_log_q = np.log(2.5)
    x_meas = np.zeros((4, 2))
    t = 0.5

    def intensity_concentration(x, t, theta, intensity):
        x = np.asarray(x)
        return np.full(x.shape[0], intensity, dtype=float)

    y = np.full(x_meas.shape[0], np.exp(true_log_q))
    pol = _DummyPollutant(q=np.exp(true_log_q))
    pol.get_pollutant_concentration = intensity_concentration  # type: ignore[method-assign]
    problem = ProblemSpec(
        pollutant=pol,
        domain=np.array([[-1.0, 1.0], [-1.0, 1.0]], dtype=float),
        wind=np.zeros(2, dtype=float),
    )

    result = bayesian_update(
        y=y,
        x=x_meas,
        t=t,
        par=par,
        problem=problem,
        s=s,
    )

    assert isinstance(result, Belief)
    assert abs(result.q_log - true_log_q) < abs(s.prior.q_log - true_log_q)
    assert result.q_log_var < s.prior.q_log_var

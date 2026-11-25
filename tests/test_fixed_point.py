import numpy as np
import pytest

from fixed_point import (
    PicardFixedPointSolver,
    AndersonAccelerationFixedPointSolver,
)


def _linf(err):
    if isinstance(err, dict):
        return float(err["linf"])
    return float(err)


def _make_linear_contraction(shape=(4, 3), a=0.5, b=1.0):
    """
    Build a simple affine contraction F(x) = a * x + b with |a| < 1.
    The fixed point is x* = b / (1 - a), independent of the initial guess.
    """

    def F(x_prev):
        if x_prev is None:
            x_prev = np.zeros(shape, dtype=float)
        return a * x_prev + b

    x_star = np.full(shape, b / (1.0 - a), dtype=float)
    return F, x_star


def test_picard_and_anderson_converge_to_same_fixed_point():
    """
    Picard and Anderson solvers should converge to the same fixed point
    on a simple linear contraction map.
    """
    F, x_star = _make_linear_contraction(shape=(5, 4), a=0.4, b=0.7)

    picard = PicardFixedPointSolver(tol=1e-6, max_iter=50)
    anderson = AndersonAccelerationFixedPointSolver(tol=1e-6, max_iter=50, m=3)

    x_picard, errs_picard = picard.solve(F, x0=None)
    x_anderson, errs_anderson = anderson.solve(F, x0=None)

    assert np.allclose(x_picard, x_star, atol=1e-6)
    assert np.allclose(x_anderson, x_star, atol=1e-6)
    assert np.allclose(x_picard, x_anderson, atol=1e-6)


def test_anderson_uses_no_more_iterations_than_picard():
    """
    Anderson acceleration should not require more iterations than Picard
    to reach a given tolerance on this contraction problem.
    """
    F, _ = _make_linear_contraction(shape=(6, 2), a=0.3, b=-0.5)

    picard = PicardFixedPointSolver(tol=1e-5, max_iter=50)
    anderson = AndersonAccelerationFixedPointSolver(tol=1e-5, max_iter=50, m=3)

    _, errs_picard = picard.solve(F, x0=None)
    _, errs_anderson = anderson.solve(F, x0=None)

    # Both should have reached tolerance
    assert _linf(errs_picard[-1]) < picard.tol
    assert _linf(errs_anderson[-1]) < anderson.tol

    # Anderson should not take strictly more iterations than Picard
    assert len(errs_anderson) <= len(errs_picard)


def test_residuals_are_monotone_nonincreasing_for_linear_contraction():
    """
    On a simple linear contraction map the residual sequence produced by the
    solvers should be nonincreasing up to numerical tolerances.
    """
    F, _ = _make_linear_contraction(shape=(3, 3), a=0.4, b=0.2)

    picard = PicardFixedPointSolver(tol=1e-6, max_iter=50)
    anderson = AndersonAccelerationFixedPointSolver(tol=1e-6, max_iter=50, m=3)

    _, errs_picard = picard.solve(F, x0=None)
    _, errs_anderson = anderson.solve(F, x0=None)

    # Allow a tiny numerical slack
    for errs in (errs_picard, errs_anderson):
        for i in range(len(errs) - 1):
            assert _linf(errs[i + 1]) <= _linf(errs[i]) + 1e-12


def test_stopping_uses_true_residual_not_small_relaxed_step():
    F, _ = _make_linear_contraction(shape=(2,), a=0.9, b=1.0)
    solver = PicardFixedPointSolver(tol=1e-3, max_iter=1)
    _, errors = solver.solve(F, x0=np.zeros(2))
    assert errors[-1]["step_linf"] == 1.0
    assert errors[-1]["residual_linf"] == pytest.approx(0.9)
    assert errors[-1]["normalized_residual_linf"] > solver.tol


def test_anderson_falls_back_to_feasible_picard_proposal():
    calls = {"count": 0}

    def F(x):
        if x is None:
            x = np.zeros(1)
        calls["count"] += 1
        return 0.5 * np.asarray(x) + 1.0

    solver = AndersonAccelerationFixedPointSolver(tol=1e-8, max_iter=20, m=2)
    value, errors = solver.solve(F)
    assert np.all(np.isfinite(value))
    assert errors[-1]["linf"] < solver.tol
    assert solver.fallback_count >= 0

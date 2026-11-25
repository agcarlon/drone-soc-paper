"""
Unit tests for covariance_from_eigen using the original convention:
- lambda1: log of largest eigenvalue
- lambda2: log of smallest eigenvalue
- angle: angle of the largest-eigenvalue eigenvector.
"""

import numpy as np

from _drone_soc_internal import covariance_from_eigen, eigen_from_covariance


def test_scalar_inputs_log_eigs():
    """
    Test covariance_from_eigen with scalar log-eigenvalue inputs.
    """
    lambda1, lambda2, angle = np.log(2.0), np.log(1.0), np.pi / 4
    c, s = np.cos(angle), np.sin(angle)
    # Columns [v_max, v_min] with v_max at angle "angle"
    Q = np.array([[c, -s], [s, c]])
    lambdas = np.array([np.exp(lambda1), np.exp(lambda2)])
    expected = (Q * lambdas) @ Q.T

    result = covariance_from_eigen(lambda1, lambda2, angle)

    assert result.shape == (2, 2)
    assert np.allclose(result, expected)


def test_array_inputs_log_eigs():
    """
    Test covariance_from_eigen with array log-eigenvalue inputs.
    """
    lambda1 = np.log(np.array([2.0, 3.0]))
    lambda2 = np.log(np.array([1.0, 0.5]))
    angle = np.array([0.0, np.pi / 4])

    result = covariance_from_eigen(lambda1, lambda2, angle)
    assert result.shape == (2, 2, 2)
    for i in range(result.shape[0]):
        assert np.allclose(result[i], result[i].T)


def test_covariance_eigen_roundtrip_identity():
    """
    For an identity covariance, the eigen/parametrization roundtrip
    should return (approximately) the same matrix.
    """
    Sigma = np.eye(2)
    vals_log, ang = eigen_from_covariance(Sigma)
    # lambda1_log: largest eigenvalue, lambda2_log: smallest eigenvalue
    Sigma_rec = covariance_from_eigen(vals_log[0], vals_log[1], ang)
    assert Sigma_rec.shape == (2, 2)
    assert np.allclose(Sigma_rec, Sigma, atol=1e-6)


def test_covariance_eigen_roundtrip_random_spd():
    """
    For a random symmetric positive-definite 2x2 matrix, applying
    eigen_from_covariance and covariance_from_eigen should recover it
    up to numerical tolerance.
    """
    rng = np.random.default_rng(0)
    A = rng.normal(size=(2, 2))
    Sigma = A @ A.T + 0.1 * np.eye(2)

    vals_log, ang = eigen_from_covariance(Sigma)
    # eigen_from_covariance returns logs of (lambda_max, lambda_min) and angle of lambda_max eigenvector
    Sigma_rec = covariance_from_eigen(vals_log[0], vals_log[1], ang)

    assert Sigma_rec.shape == (2, 2)
    assert np.allclose(Sigma_rec, Sigma, atol=1e-6)

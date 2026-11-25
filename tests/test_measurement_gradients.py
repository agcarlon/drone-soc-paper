import numpy as np

from measurement import (
    GaussianPlume,
    PlanarAdvectionDiffusionDecayPlume,
    SlicedScreenedAdvectionDiffusionPlume,
)


def _make_plume() -> GaussianPlume:
    return GaussianPlume(
        source=np.array([0.2, -0.1], dtype=float),
        q=3.5,
        ds=0.8,
        tau_s=6.0,
        wind=np.array([0.3, -0.2], dtype=float),
        sigma_e=1e-3,
    )


def _make_planar_plume() -> PlanarAdvectionDiffusionDecayPlume:
    return PlanarAdvectionDiffusionDecayPlume(
        source=np.array([0.2, -0.1], dtype=float),
        q=3.5,
        ds=0.8,
        tau_s=6.0,
        wind=np.array([0.3, -0.2], dtype=float),
        sigma_e=1e-3,
        eps_c=1e-3,
    )


def _make_sliced_plume() -> SlicedScreenedAdvectionDiffusionPlume:
    return SlicedScreenedAdvectionDiffusionPlume(
        source=np.array([0.2, -0.1], dtype=float),
        q=3.5,
        ds=0.8,
        tau_s=6.0,
        wind=np.array([0.3, -0.2], dtype=float),
        sigma_e=1e-3,
        eps_c=1e-3,
    )


def test_gaussian_gradient_wrt_theta_matches_finite_difference():
    pol = _make_plume()
    x = np.array([[0.1, 0.0], [0.7, -0.2], [-0.3, 0.4]], dtype=float)
    theta = np.array([0.05, -0.15], dtype=float)
    t = 1.0
    q = 2.1

    g = pol.get_concentration_gradient_wrt_theta(x, t, theta, intensity=q)

    eps = 1e-6
    g_fd = np.zeros_like(g)
    for i in range(2):
        e = np.zeros(2, dtype=float)
        e[i] = eps
        c_plus = pol.get_pollutant_concentration(x, t, theta + e, intensity=q)
        c_minus = pol.get_pollutant_concentration(x, t, theta - e, intensity=q)
        g_fd[:, i] = (c_plus - c_minus) / (2.0 * eps)

    assert np.allclose(g, g_fd, rtol=5e-4, atol=1e-7)


def test_gaussian_derivative_wrt_logq_matches_concentration_and_fd():
    pol = _make_plume()
    x = np.array([[0.2, -0.1], [0.4, 0.6], [-0.5, -0.4]], dtype=float)
    theta = np.array([0.0, 0.0], dtype=float)
    t = 1.0
    q = 1.7

    d = pol.get_concentration_derivative_wrt_logq(x, t, theta, intensity=q)
    c = pol.get_pollutant_concentration(x, t, theta, intensity=q)

    eps = 1e-6
    c_plus = pol.get_pollutant_concentration(x, t, theta, intensity=q * np.exp(eps))
    c_minus = pol.get_pollutant_concentration(x, t, theta, intensity=q * np.exp(-eps))
    d_fd = (c_plus - c_minus) / (2.0 * eps)

    assert np.allclose(d, c, rtol=1e-10, atol=1e-12)
    assert np.allclose(d, d_fd, rtol=5e-4, atol=1e-8)


def _check_theta_gradient(pol) -> None:
    x = np.array([[0.1, 0.0], [0.7, -0.2], [-0.3, 0.4]], dtype=float)
    theta = np.array([0.05, -0.15], dtype=float)
    t = 1.0
    q = 2.1

    g = pol.get_concentration_gradient_wrt_theta(x, t, theta, intensity=q)

    eps = 1e-6
    g_fd = np.zeros_like(g)
    for i in range(2):
        e = np.zeros(2, dtype=float)
        e[i] = eps
        c_plus = pol.get_pollutant_concentration(x, t, theta + e, intensity=q)
        c_minus = pol.get_pollutant_concentration(x, t, theta - e, intensity=q)
        g_fd[:, i] = (c_plus - c_minus) / (2.0 * eps)

    assert np.allclose(g, g_fd, rtol=5e-4, atol=1e-7)


def _check_logq_derivative(pol) -> None:
    x = np.array([[0.2, -0.1], [0.4, 0.6], [-0.5, -0.4]], dtype=float)
    theta = np.array([0.0, 0.0], dtype=float)
    t = 1.0
    q = 1.7

    d = pol.get_concentration_derivative_wrt_logq(x, t, theta, intensity=q)
    c = pol.get_pollutant_concentration(x, t, theta, intensity=q)

    eps = 1e-6
    c_plus = pol.get_pollutant_concentration(x, t, theta, intensity=q * np.exp(eps))
    c_minus = pol.get_pollutant_concentration(x, t, theta, intensity=q * np.exp(-eps))
    d_fd = (c_plus - c_minus) / (2.0 * eps)

    assert np.allclose(d, c, rtol=1e-10, atol=1e-12)
    assert np.allclose(d, d_fd, rtol=5e-4, atol=1e-8)


def test_planar_plume_gradient_wrt_theta_matches_finite_difference():
    _check_theta_gradient(_make_planar_plume())


def test_planar_plume_derivative_wrt_logq_matches_concentration_and_fd():
    _check_logq_derivative(_make_planar_plume())


def test_sliced_plume_gradient_wrt_theta_matches_finite_difference():
    _check_theta_gradient(_make_sliced_plume())


def test_sliced_plume_derivative_wrt_logq_matches_concentration_and_fd():
    _check_logq_derivative(_make_sliced_plume())

# measurement.py ──────────────────────────────────────────────────────────────
"""Helpers for the drone source-seeking project: generate noisy concentration
readings at drone positions and visualise pollutant fields.

Implements continuous point-source plume surrogates with wind and dispersion.
For the instationary model,
    G(x, t; θ, s) = 1 / [4π(d_s/2)(t-s)] * exp(-‖x - θ - W(t-s)‖² / [4(d_s/2)(t-s)])
    C(x, t) = ∫₀ᵗ q · G(x, t; θ, s) ds
where the time integral is approximated by the trapezoidal rule.
Sensor noise is zero-mean Gaussian with standard deviation σₑ.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
import numpy as np
from scipy.special import k0, k1


def get_wind_angle(wind: np.ndarray) -> float:
    """
    Return wind direction angle (radians) for the *actual* blowing direction.
    """
    wind = np.asarray(wind, dtype=float)
    return float(np.arctan2(wind[1], wind[0]))


def get_plume_axis_angle(wind: np.ndarray) -> float:
    """
    Return the plume principal-axis angle used by our plume model convention.
    This is opposite to the stored wind vector.
    """
    return get_wind_angle(-np.asarray(wind, dtype=float))


def _broadcast_positions_and_thetas(
    x: np.ndarray,
    theta: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Broadcast positions x and source locations theta to matching (n, 2) arrays.
    """
    x_arr = np.atleast_2d(np.asarray(x, dtype=float))
    theta_arr = np.asarray(theta, dtype=float)
    if theta_arr.ndim == 1:
        theta_arr = np.broadcast_to(theta_arr.reshape(1, 2), x_arr.shape)
    else:
        theta_arr = np.atleast_2d(theta_arr)
        if theta_arr.shape[0] == 1 and x_arr.shape[0] != 1:
            theta_arr = np.broadcast_to(theta_arr, x_arr.shape)
        elif theta_arr.shape[0] != x_arr.shape[0]:
            raise ValueError(
                f"theta must have shape (2,) or ({x_arr.shape[0]}, 2); got {theta_arr.shape}"
            )
    return x_arr, theta_arr


@dataclass
class Pollutant(ABC):
    """
    Abstract base class for pollutant modeling.
    """
    source: np.ndarray              # True 2D source location, shape (2,)
    q: float                       # Emission rate
    wind: np.ndarray               # Constant wind vector W, shape (2,)
    sigma_e: float                 # Sensor noise std-dev ε ~ N(0, σₑ²)


    @abstractmethod
    def get_pollutant_concentration(self,
                                    x: np.ndarray,
                                    t: float,
                                    theta: np.ndarray,
                                    intensity: Optional[float] = None) -> np.ndarray:
        """
        Return the pollutant concentration C(x, t) at multiple positions x and time t.

        Args:
            x (np.ndarray): Positions (n, 2).
            t (float): Time.
            theta (np.ndarray): Source location (2,).
            intensity (float, optional): Emission rate q. Defaults to self.q.

        Returns:
            np.ndarray: Concentrations at each position.
        """
        pass
    
    def get_concentration_derivative_wrt_logq(self,
                                              x: np.ndarray,
                                              t: float,
                                              theta: np.ndarray,
                                              intensity: Optional[float] = None) -> np.ndarray:
        """
        Return ∂C / ∂(log q) at the given positions/time/source.
        Default implementation uses a central finite difference in log-intensity.
        """
        intensity_val = self.q if intensity is None else intensity
        eps = 1e-6
        q_plus = intensity_val * np.exp(eps)
        q_minus = intensity_val * np.exp(-eps)
        c_plus = self.get_pollutant_concentration(x, t, theta, intensity=q_plus)
        c_minus = self.get_pollutant_concentration(x, t, theta, intensity=q_minus)
        return (c_plus - c_minus) / (2.0 * eps)

    def get_concentration_gradient_wrt_theta(
        self,
        x: np.ndarray,
        t: float,
        theta: np.ndarray,
        intensity: Optional[float] = None,
    ) -> np.ndarray:
        """
        Return ∂C/∂theta as an (n, 2) array at given positions x.
        Default implementation uses central finite differences in theta.
        """
        x = np.atleast_2d(x)
        theta = np.asarray(theta, dtype=float)
        d_theta = 1e-6 * np.maximum(1.0, np.abs(theta))
        grads = []
        for i in range(2):
            e_i = np.zeros(2, dtype=float)
            e_i[i] = d_theta[i]
            c_minus = self.get_pollutant_concentration(x, t, theta - e_i, intensity=intensity)
            c_plus = self.get_pollutant_concentration(x, t, theta + e_i, intensity=intensity)
            grads.append((c_plus - c_minus) / (2.0 * d_theta[i]))
        return np.stack(grads, axis=1)

@dataclass
class InstationaryPlume(Pollutant):
    """
    Parameters for the pollutant source and environment.
    """
    ds: float                      # Molecular diffusivity d_s [m² s⁻¹]
    n_quad: int = 80               # Number of points for trapezoidal integration


    def get_pollutant_concentration(self,
                                    x: np.ndarray,
                                    t: float,
                                    theta: np.ndarray,
                                    intensity: Optional[float] = None) -> np.ndarray:
        """
        Return the pollutant concentration C(x, t) at multiple positions x and time t.

        Args:
            x (np.ndarray): Positions (n, 2).
            t (float): Time.
            theta (np.ndarray): Source location (2,).

        Returns:
            np.ndarray: Concentrations at each position.
        """
        x = np.atleast_2d(x)
        W = self.wind
        q = self.q if intensity is None else intensity
        ds = self.ds

        n = max(3, self.n_quad)
        tau = np.linspace(0.0, t, n, dtype=float)
        tau[-1] = max(0.0, t - 1e-12)

        delta = x[:, None, :] - theta[None, None, :] - (W[None, None, :] * (t - tau)[None, :, None])
        dist2 = np.linalg.norm(delta, axis=-1) ** 2
        denom = 4.0 * np.pi * (0.5 * ds) * (t - tau)
        denom = np.where(denom == 0.0, np.inf, denom)

        G_tau = (1.0 / denom) * np.exp(-dist2 / (4.0 * (0.5 * ds) * (t - tau)[None, :]))
        C_xt = q * np.trapezoid(G_tau, tau)
        return np.array(C_xt)



@dataclass
class GaussianPlume(Pollutant):
    """
    Parameters for the pollutant source and environment.
    """
    ds: float                      # Dispersion coefficient d_s [m^2/s]
    tau_s: float                   # Pollutant lifetime \tau_s [s]

    def get_pollutant_concentration(self,
                                    x: np.ndarray,
                                    t: float,
                                    theta: np.ndarray,
                                    intensity: Optional[float] = None) -> np.ndarray:
        """
        Return the pollutant concentration C(x, t) at multiple positions x and time t.

        Args:
            x (np.ndarray): Positions (n, 2).
            t (float): Time.
            theta (np.ndarray): Source location (2,).

        Returns:
            np.ndarray: Concentrations at each position.
        """
        x = np.atleast_2d(x)
        q = self.q if intensity is None else intensity                  # Emission rate
        ds = self.ds                # Dispersion coefficient
        tau_s = self.tau_s          # Pollutant lifetime
        W = -self.wind              # Wind vector [m/s] in plume-model convention
        theta = np.atleast_2d(theta)
        us = np.linalg.norm(W)
        phi = get_plume_axis_angle(self.wind)   # Wind direction angle in plume convention

        lambda_ = np.sqrt((ds * tau_s) / (1 + (us**2 * tau_s) / (4 * ds)))
        
        dx = x[:, 0] - theta[:,0]
        dy = x[:, 1] - theta[:,1]
        dist = np.sqrt(dx**2 + dy**2)
        dist = np.maximum(dist, 1e-8)

        C = (q / (4 * np.pi * ds * dist)) \
            * np.exp(-dist / lambda_) \
            * np.exp(-(dx * us * np.cos(phi)) / (2 * ds)) \
            * np.exp(-(dy * us * np.sin(phi)) / (2 * ds))
        return C

    def get_concentration_gradient_wrt_theta(
        self,
        x: np.ndarray,
        t: float,
        theta: np.ndarray,
        intensity: Optional[float] = None,
    ) -> np.ndarray:
        """
        Analytic gradient ∂C/∂theta for the Gaussian plume model.
        """
        x = np.atleast_2d(x)
        theta = np.asarray(theta, dtype=float).reshape(2,)
        ds = self.ds
        tau_s = self.tau_s
        W = -self.wind
        us = np.linalg.norm(W)
        lambda_ = np.sqrt((ds * tau_s) / (1 + (us**2 * tau_s) / (4 * ds)))

        d = x - theta[None, :]
        r = np.linalg.norm(d, axis=1)
        r = np.maximum(r, 1e-8)

        C = self.get_pollutant_concentration(x, t, theta, intensity=intensity)
        grad_log = d / (r[:, None] ** 2) + d / (lambda_ * r)[:, None] + W[None, :] / (2.0 * ds)
        return C[:, None] * grad_log

    def get_concentration_derivative_wrt_logq(
        self,
        x: np.ndarray,
        t: float,
        theta: np.ndarray,
        intensity: Optional[float] = None,
    ) -> np.ndarray:
        """
        For Gaussian plume, concentration is linear in q, so ∂C/∂log q = C.
        """
        return self.get_pollutant_concentration(x, t, theta, intensity=intensity)


@dataclass
class PlanarAdvectionDiffusionDecayPlume(Pollutant):
    """
    Steady 2-D advection-diffusion-decay plume with modified-Bessel kernel.
    """
    ds: float
    tau_s: float
    eps_c: float = 1e-8

    def _kappa(self) -> float:
        wind_norm = float(np.linalg.norm(self.wind))
        return float(np.sqrt((1.0 / (self.ds * self.tau_s)) + (wind_norm ** 2) / (4.0 * self.ds ** 2)))

    def get_pollutant_concentration(
        self,
        x: np.ndarray,
        t: float,
        theta: np.ndarray,
        intensity: Optional[float] = None,
    ) -> np.ndarray:
        x_arr, theta_arr = _broadcast_positions_and_thetas(x, theta)
        q = self.q if intensity is None else intensity
        d = x_arr - theta_arr
        r_c = np.sqrt(np.sum(d * d, axis=1) + self.eps_c ** 2)
        kappa = self._kappa()
        adv = np.exp(np.einsum("ij,j->i", d, self.wind) / (2.0 * self.ds))
        return (q / (2.0 * np.pi * self.ds)) * adv * k0(kappa * r_c)

    def get_concentration_gradient_wrt_theta(
        self,
        x: np.ndarray,
        t: float,
        theta: np.ndarray,
        intensity: Optional[float] = None,
    ) -> np.ndarray:
        x_arr, theta_arr = _broadcast_positions_and_thetas(x, theta)
        q = self.q if intensity is None else intensity
        d = x_arr - theta_arr
        r_c = np.sqrt(np.sum(d * d, axis=1) + self.eps_c ** 2)
        kappa = self._kappa()
        adv = np.exp(np.einsum("ij,j->i", d, self.wind) / (2.0 * self.ds))
        amplitude = (q / (2.0 * np.pi * self.ds)) * adv
        concentration = amplitude * k0(kappa * r_c)
        radial_term = amplitude[:, None] * (kappa * k1(kappa * r_c))[:, None] * (d / r_c[:, None])
        advection_term = concentration[:, None] * (-self.wind[None, :] / (2.0 * self.ds))
        return radial_term + advection_term

    def get_concentration_derivative_wrt_logq(
        self,
        x: np.ndarray,
        t: float,
        theta: np.ndarray,
        intensity: Optional[float] = None,
    ) -> np.ndarray:
        return self.get_pollutant_concentration(x, t, theta, intensity=intensity)


@dataclass
class SlicedScreenedAdvectionDiffusionPlume(Pollutant):
    """
    Steady 3-D screened Green's-function surrogate evaluated on the 2-D search plane.
    """
    ds: float
    tau_s: float
    eps_c: float = 1e-8

    def _kappa(self) -> float:
        wind_norm = float(np.linalg.norm(np.asarray(self.wind, dtype=float)))
        return float(np.sqrt((1.0 / (self.ds * self.tau_s)) + (wind_norm ** 2) / (4.0 * self.ds ** 2)))

    def get_pollutant_concentration(
        self,
        x: np.ndarray,
        t: float,
        theta: np.ndarray,
        intensity: Optional[float] = None,
    ) -> np.ndarray:
        x_arr, theta_arr = _broadcast_positions_and_thetas(x, theta)
        q = self.q if intensity is None else intensity
        wind_vec = np.asarray(self.wind, dtype=float)
        d = x_arr - theta_arr
        r_c = np.sqrt(np.sum(d * d, axis=1) + self.eps_c ** 2)
        kappa = self._kappa()
        adv = np.exp(np.einsum("ij,j->i", d, wind_vec) / (2.0 * self.ds))
        return (q / (4.0 * np.pi * self.ds * r_c)) * np.exp(-kappa * r_c) * adv

    def get_concentration_gradient_wrt_theta(
        self,
        x: np.ndarray,
        t: float,
        theta: np.ndarray,
        intensity: Optional[float] = None,
    ) -> np.ndarray:
        x_arr, theta_arr = _broadcast_positions_and_thetas(x, theta)
        wind_vec = np.asarray(self.wind, dtype=float)
        d = x_arr - theta_arr
        r_c = np.sqrt(np.sum(d * d, axis=1) + self.eps_c ** 2)
        kappa = self._kappa()
        concentration = self.get_pollutant_concentration(x_arr, t, theta_arr, intensity=intensity)
        grad_log = d / (r_c[:, None] ** 2) + (kappa * d / r_c[:, None]) - wind_vec[None, :] / (2.0 * self.ds)
        return concentration[:, None] * grad_log

    def get_concentration_derivative_wrt_logq(
        self,
        x: np.ndarray,
        t: float,
        theta: np.ndarray,
        intensity: Optional[float] = None,
    ) -> np.ndarray:
        return self.get_pollutant_concentration(x, t, theta, intensity=intensity)




def take_measurement(x: np.ndarray, t: float, pol: Pollutant) -> np.ndarray:
    """
    Return a noisy concentration reading Y = C(x, t) + ε.

    Args:
        x (np.ndarray): Drone positions (n, 2).
        t (float): Time.
        pol (Pollutant): Pollutant and environment parameters.

    Returns:
        np.ndarray: Noisy concentration readings.
    """
    C_xt = pol.get_pollutant_concentration(x, t, pol.source, intensity=pol.q)
    noise = np.random.normal(0.0, pol.sigma_e, size=C_xt.shape)
    return np.array(C_xt + noise)


if __name__ == "__main__":
    from plotting.plotting import plot_concentration_contours
    # pol = InstationaryPlume(
    #     source=np.array([0.0, 0.0]),
    #     q=1.0,
    #     ds=0.01,
    #     wind=np.array([-.10, 0.5]),
    #     sigma_e=0.05
    # )
    pol = GaussianPlume(
        source=np.array([0.0, 0.0]),
        q=10.0,
        ds=0.01,
        tau_s=400,
        wind=np.array([-.10, 0.5]),
        sigma_e=0.05
    )
    pos = np.array([5.0, 1.0])
    y_sample = take_measurement(pos, t=10, pol=pol)
    print("C(x,t) + noise =", y_sample)
    times = [5, 10, 20, 100]
    out = plot_concentration_contours(
        pol=pol,
        times=times,
        x_bounds=(-20, 20),
        y_bounds=(-20, 20),
        n_grid=120
    )
    for i, (fig, ax) in enumerate(out):
        fig.savefig(f"contour_{times[i]}.pdf")

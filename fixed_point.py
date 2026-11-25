from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple, List, Any
import numpy as np


FixedPointMap = Callable[[Optional[np.ndarray]], np.ndarray]


def _norms(value: np.ndarray) -> tuple[float, float]:
    return float(np.max(np.abs(value))), float(np.linalg.norm(value))


def _diagnostic(*, residual: np.ndarray, iterate: np.ndarray, step: np.ndarray, fallback: bool = False) -> dict:
    residual_inf, residual_l2 = _norms(residual)
    step_inf, step_l2 = _norms(step)
    scale = 1.0 + float(np.max(np.abs(iterate)))
    normalized = residual_inf / scale
    return {
        # ``linf`` remains the generic criterion consumed by existing callers.
        "linf": normalized,
        "l2": residual_l2 / scale,
        "residual_linf": residual_inf,
        "residual_l2": residual_l2,
        "normalized_residual_linf": normalized,
        "step_linf": step_inf,
        "step_l2": step_l2,
        "fallback": bool(fallback),
    }


def _initial_pair(F: FixedPointMap, x0: Optional[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    first = np.asarray(F(x0), dtype=float)
    current = np.zeros_like(first) if x0 is None else np.asarray(x0, dtype=float)
    return current, first


@dataclass
class BaseFixedPointSolver:
    tol: float = 1e-3
    max_iter: int = 25
    verbose: bool = False

    def solve(self, F: FixedPointMap, x0: Optional[np.ndarray] = None) -> Tuple[np.ndarray, List[dict]]:
        """
        Solve x = F(x) with a black-box Picard-style iteration policy.
        Returns:
            x: fixed point approximation
            errors: list of residual norms per iteration as dicts:
                ``linf`` is the normalized Bellman residual; absolute residual
                and successive-step norms are stored under explicit keys.
        """
        raise NotImplementedError


@dataclass
class PicardFixedPointSolver(BaseFixedPointSolver):
    def solve(self, F: FixedPointMap, x0: Optional[np.ndarray] = None) -> Tuple[np.ndarray, List[dict]]:
        errors: List[dict] = []
        x_prev, Fx_prev = _initial_pair(F, x0)
        for k in range(self.max_iter):
            x_new = Fx_prev
            Fx_new = np.asarray(F(x_new), dtype=float)
            item = _diagnostic(residual=Fx_new - x_new, iterate=x_new, step=x_new - x_prev)
            if self.verbose:
                print(f"Picard iteration {k}: normalized_residual_inf={item['linf']:.3e}")
            errors.append(item)
            if item["linf"] < self.tol:
                return x_new, errors
            x_prev, Fx_prev = x_new, Fx_new
        return x_new, errors


@dataclass
class UnderRelaxationFixedPointSolver(BaseFixedPointSolver):
    beta: float = 0.5  # relaxation factor in (0,1]

    def solve(self, F: FixedPointMap, x0: Optional[np.ndarray] = None) -> Tuple[np.ndarray, List[dict]]:
        errors: List[dict] = []
        x_prev, Fx_prev = _initial_pair(F, x0)
        for k in range(self.max_iter):
            x_new = (1.0 - self.beta) * x_prev + self.beta * Fx_prev
            Fx_new = np.asarray(F(x_new), dtype=float)
            item = _diagnostic(residual=Fx_new - x_new, iterate=x_new, step=x_new - x_prev)
            if self.verbose:
                print(f"Under relaxation iteration {k}: normalized_residual_inf={item['linf']:.3e}")
            errors.append(item)
            if item["linf"] < self.tol:
                return x_new, errors
            x_prev, Fx_prev = x_new, Fx_new
        return x_new, errors


@dataclass
class AndersonAccelerationFixedPointSolver(BaseFixedPointSolver):
    m: int = 3  # depth
    ridge: float = 1e-10  # regularization
    plateau_tol: float = 1e-3  # relative improvement threshold to flag stagnation
    plateau_iters: int = 4     # number of consecutive stagnant steps before fallback

    def solve(self, F: FixedPointMap, x0: Optional[np.ndarray] = None) -> Tuple[np.ndarray, List[dict]]:
        """
        Basic Anderson acceleration (Type-I) for fixed-point iteration.
        x_{k+1} = sum_{j=0}^m alpha_j F(x_{k-m+j}), with alpha minimizing residual in least-squares.
        """
        errors: List[dict] = []
        xs: List[np.ndarray] = []
        gs: List[np.ndarray] = []
        self.fallback_count = 0
        _, Fx_old = _initial_pair(F, x0)
        x = Fx_old
        Fx = np.asarray(F(x), dtype=float)

        for k in range(self.max_iter):
            g = Fx - x
            raw = Fx
            raw_F = np.asarray(F(raw), dtype=float)
            raw_item = _diagnostic(residual=raw_F - raw, iterate=raw, step=raw - x)
            xs.append(x)
            gs.append(g)
            m_use = min(self.m, len(xs))
            # Build G and ΔX matrices
            # Flatten residuals to 2D (N, m_use) for least-squares in coefficient space
            G = np.stack([gs[-j].ravel() for j in range(1, m_use + 1)], axis=1)  # (N, m_use)
            # Solve (G^T G + ridge I) alpha = 1 subject to sum alpha = 1
            GTG = G.T @ G + self.ridge * np.eye(m_use)
            ones = np.ones((m_use, 1))
            try:
                # Solve for coefficients minimizing ||G alpha|| with sum(alpha)=1 using Lagrange multipliers
                KKT = np.block([[GTG, ones], [ones.T, np.zeros((1, 1))]])
                rhs = np.vstack([np.zeros((m_use, 1)), np.ones((1, 1))])
                sol = np.linalg.lstsq(KKT, rhs, rcond=None)[0]
                alpha = sol[:m_use, 0]
            except np.linalg.LinAlgError:
                alpha = np.zeros((m_use,), dtype=float)
                alpha[-1] = 1.0  # fallback to last
            # Form accelerated proposal from recent Picard images.
            x_candidates = [xs[-j] for j in range(1, m_use + 1)]
            Fx_candidates = [x + g for (x, g) in zip(x_candidates, [gs[-j] for j in range(1, m_use + 1)])]
            proposal = np.zeros_like(xs[-1])
            for aj, Fj in zip(alpha, Fx_candidates):
                proposal += aj * Fj
            proposal_finite = bool(np.all(np.isfinite(proposal)))
            if proposal_finite:
                proposal_F = np.asarray(F(proposal), dtype=float)
                proposal_item = _diagnostic(
                    residual=proposal_F - proposal, iterate=proposal, step=proposal - x
                )
                proposal_finite = bool(np.all(np.isfinite(proposal_F)))
            else:
                proposal_F = raw_F
                proposal_item = raw_item

            # Accept acceleration only when it does not increase the true residual.
            fallback = (not proposal_finite) or proposal_item["linf"] > raw_item["linf"] * (1.0 + 1e-12)
            if fallback:
                x_new, Fx_new, item = raw, raw_F, raw_item
                item = dict(item)
                item["fallback"] = True
                self.fallback_count += 1
                xs.clear()
                gs.clear()
            else:
                x_new, Fx_new, item = proposal, proposal_F, proposal_item
            if self.verbose:
                print(
                    f"Anderson iteration {k}: normalized_residual_inf={item['linf']:.3e}"
                    f" fallback={fallback}"
                )
            errors.append(item)
            if item["linf"] < self.tol:
                return x_new, errors
            if len(xs) > self.m + 1:
                xs.pop(0)
                gs.pop(0)
            x, Fx = x_new, Fx_new
        return x, errors


@dataclass
class FixedPointSpec:
    name: str
    kwargs: dict


def make_solver(spec: FixedPointSpec) -> BaseFixedPointSolver:
    """
    Factory to instantiate a fixed-point solver by name.
    Names:
      - 'picard'
      - 'under_relaxation'
      - 'anderson'
    """
    name = (spec.name or "picard").lower()
    if name in ("picard", "plain", "default"):
        return PicardFixedPointSolver(**spec.kwargs)
    if name in ("under_relaxation", "under-relaxation", "relaxation", "relaxed"):
        return UnderRelaxationFixedPointSolver(**spec.kwargs)
    if name in ("anderson", "aa"):
        return AndersonAccelerationFixedPointSolver(**spec.kwargs)
    # Fallback
    return PicardFixedPointSolver(**spec.kwargs)

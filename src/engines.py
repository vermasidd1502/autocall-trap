"""
engines.py — Monte Carlo Simulation Engines
============================================
Implements two competing stochastic models for pricing path-dependent
structured products:

1. GBM (Geometric Brownian Motion) — the naive benchmark
2. Heston Stochastic Volatility — the fair-value model

The core thesis: GBM underprices tail risk in autocallable notes because
it cannot capture stochastic volatility, leverage effect, or vol clustering.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class HestonParams:
    """
    Heston model parameters.

    dS = r·S·dt + √v·S·dW_S
    dv = κ(θ − v)dt + ξ√v·dW_v
    corr(dW_S, dW_v) = ρ

    Parameters
    ----------
    v0 : float
        Initial instantaneous variance.
    kappa : float
        Mean-reversion speed of variance.
    theta : float
        Long-run variance level.
    xi : float
        Volatility of variance (vol-of-vol).
    rho : float
        Correlation between spot and variance Brownian motions.
    """
    v0: float = 0.065
    kappa: float = 2.0
    theta: float = 0.07
    xi: float = 0.5
    rho: float = -0.65

    @property
    def feller_ratio(self) -> float:
        """Feller condition ratio: 2κθ/ξ². Must be > 1 to avoid zero variance."""
        return 2 * self.kappa * self.theta / (self.xi ** 2)

    @property
    def feller_satisfied(self) -> bool:
        return self.feller_ratio > 1.0

    def summary(self) -> str:
        lines = [
            f"  v0 = {self.v0:.4f}  (√v0 = {np.sqrt(self.v0)*100:.1f}% implied vol)",
            f"  κ  = {self.kappa:.2f}   (mean-reversion speed)",
            f"  θ  = {self.theta:.4f}  (√θ = {np.sqrt(self.theta)*100:.1f}% long-run vol)",
            f"  ξ  = {self.xi:.2f}   (vol-of-vol)",
            f"  ρ  = {self.rho:.2f}  (spot-vol correlation)",
            f"  Feller ratio: {self.feller_ratio:.3f} ({'satisfied' if self.feller_satisfied else 'VIOLATED'})",
        ]
        return "\n".join(lines)


# ── Preset calibrations ───────────────────────────────────────────────

def orcl_heston() -> HestonParams:
    """Representative Heston calibration for ORCL."""
    return HestonParams(v0=0.065, kappa=2.0, theta=0.07, xi=0.5, rho=-0.65)


def stress_heston() -> HestonParams:
    """Stress regime: higher vol-of-vol, more negative correlation."""
    return HestonParams(v0=0.10, kappa=1.5, theta=0.12, xi=0.8, rho=-0.80)


# ── Simulation engines ────────────────────────────────────────────────

def simulate_gbm(
    S0: float,
    r: float,
    sigma: float,
    T: float,
    n_obs: int,
    n_paths: int,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Geometric Brownian Motion Monte Carlo.

    dS = r·S·dt + σ·S·dW

    Parameters
    ----------
    S0 : float
        Initial stock price.
    r : float
        Risk-free rate (continuous).
    sigma : float
        Constant volatility.
    T : float
        Time horizon in years.
    n_obs : int
        Number of observation dates.
    n_paths : int
        Number of simulation paths.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    S : np.ndarray, shape (n_paths, n_obs)
        Simulated stock prices at each observation date.
    """
    if seed is not None:
        np.random.seed(seed)

    dt = T / n_obs
    Z = np.random.standard_normal((n_paths, n_obs))
    log_returns = (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z

    S = np.zeros((n_paths, n_obs))
    S[:, 0] = S0 * np.exp(log_returns[:, 0])
    for t in range(1, n_obs):
        S[:, t] = S[:, t-1] * np.exp(log_returns[:, t])

    return S


def simulate_heston(
    S0: float,
    r: float,
    params: HestonParams,
    T: float,
    n_obs: int,
    n_paths: int,
    n_substeps: int = 20,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Heston Stochastic Volatility Monte Carlo (truncated Euler scheme).

    dS = r·S·dt + √v·S·dW_S
    dv = κ(θ − v)dt + ξ√v·dW_v
    corr(dW_S, dW_v) = ρ

    Parameters
    ----------
    S0 : float
        Initial stock price.
    r : float
        Risk-free rate (continuous).
    params : HestonParams
        Heston model parameters.
    T : float
        Time horizon in years.
    n_obs : int
        Number of observation dates.
    n_paths : int
        Number of simulation paths.
    n_substeps : int
        Sub-steps between each observation date (for accuracy).
    seed : int, optional
        Random seed.

    Returns
    -------
    S_obs : np.ndarray, shape (n_paths, n_obs)
        Simulated stock prices at each observation date.
    """
    if seed is not None:
        np.random.seed(seed)

    dt_obs = T / n_obs
    dt = dt_obs / n_substeps
    total_steps = n_obs * n_substeps

    S_obs = np.zeros((n_paths, n_obs))
    S = np.full(n_paths, S0, dtype=np.float64)
    v = np.full(n_paths, params.v0, dtype=np.float64)

    sqrt_dt = np.sqrt(dt)
    rho = params.rho
    sqrt_1_rho2 = np.sqrt(1.0 - rho**2)

    for step in range(total_steps):
        v = np.maximum(v, 1e-8)

        # Correlated Brownian increments
        Z1 = np.random.standard_normal(n_paths)
        Z2 = np.random.standard_normal(n_paths)
        W_v = Z1
        W_S = rho * Z1 + sqrt_1_rho2 * Z2

        # Variance process (truncated Euler)
        v_new = v + params.kappa * (params.theta - v) * dt + \
                params.xi * np.sqrt(v) * sqrt_dt * W_v
        v_new = np.maximum(v_new, 1e-8)

        # Stock process (log-Euler)
        S = S * np.exp((r - 0.5 * v) * dt + np.sqrt(v) * sqrt_dt * W_S)

        v = v_new

        # Record at observation dates
        if (step + 1) % n_substeps == 0:
            obs_idx = (step + 1) // n_substeps - 1
            S_obs[:, obs_idx] = S

    return S_obs

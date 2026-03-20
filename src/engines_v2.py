"""
engines_v2.py — Enhanced Monte Carlo Engines (Revision 1)
==========================================================
Addresses peer review critiques:

1. QE (Quadratic Exponential) Scheme — Andersen (2008)
   Replaces truncated Euler for the Heston variance process.
   Accurately samples from the non-central chi-square transition
   density, eliminating negative-variance flooring bias.

2. Discrete Dividend Support
   Both GBM and Heston engines now accept a dividend schedule
   (ex-dates and dollar amounts) or a continuous dividend yield.
   Dividends lower the risk-neutral drift, making autocall less
   likely and barrier breach more likely.

References:
    Andersen, L. (2008). "Efficient Simulation of the Heston
    Stochastic Volatility Model." J. Computational Finance, 11(3).
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Tuple

from .engines import HestonParams


# ══════════════════════════════════════════════════════════════════════
# SECTION 1: DIVIDEND SPECIFICATION
# ══════════════════════════════════════════════════════════════════════

@dataclass
class DividendSchedule:
    """
    Discrete or continuous dividend specification.

    For discrete dividends, provide ex_dates and amounts.
    For continuous yield, provide yield_pa.
    If both are provided, discrete dividends take precedence.

    Parameters
    ----------
    ex_dates : list of float
        Ex-dividend dates in years from pricing date.
    amounts : list of float
        Dollar dividend amounts at each ex-date.
    yield_pa : float
        Continuous dividend yield (annualized). Used if no
        discrete schedule is provided.
    """
    ex_dates: List[float] = field(default_factory=list)
    amounts: List[float] = field(default_factory=list)
    yield_pa: float = 0.0

    @property
    def is_discrete(self) -> bool:
        return len(self.ex_dates) > 0

    def summary(self) -> str:
        if self.is_discrete:
            total = sum(self.amounts)
            return (f"  Discrete: {len(self.ex_dates)} payments, "
                    f"total ${total:.2f}/share")
        elif self.yield_pa > 0:
            return f"  Continuous yield: {self.yield_pa*100:.2f}% p.a."
        else:
            return "  No dividends"


def orcl_dividends(S0: float = 140.0) -> DividendSchedule:
    """
    ORCL quarterly dividend schedule (representative).
    ORCL pays ~$0.40/quarter ($1.60/year, ~1.14% yield at $140).
    """
    # Quarterly ex-dates over 2 years
    ex_dates = [0.25 * i for i in range(1, 9)]  # Q1 through Q8
    amounts = [0.40] * 8  # $0.40 per quarter
    return DividendSchedule(ex_dates=ex_dates, amounts=amounts, yield_pa=0.0114)


# ══════════════════════════════════════════════════════════════════════
# SECTION 2: GBM WITH DIVIDENDS
# ══════════════════════════════════════════════════════════════════════

def simulate_gbm_v2(
    S0: float,
    r: float,
    sigma: float,
    T: float,
    n_obs: int,
    n_paths: int,
    dividends: Optional[DividendSchedule] = None,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    GBM Monte Carlo with discrete or continuous dividend support.

    For continuous yield q:
        dS = (r - q)S dt + σS dW

    For discrete dividends at time τ_k with amount D_k:
        S(τ_k+) = S(τ_k-) - D_k
    """
    if seed is not None:
        np.random.seed(seed)

    dt = T / n_obs
    obs_times = np.linspace(dt, T, n_obs)

    # Determine effective drift
    q = 0.0
    if dividends is not None and not dividends.is_discrete:
        q = dividends.yield_pa

    Z = np.random.standard_normal((n_paths, n_obs))
    drift = (r - q - 0.5 * sigma**2) * dt

    S = np.zeros((n_paths, n_obs))
    S_prev = np.full(n_paths, S0)

    for t_idx in range(n_obs):
        t_now = obs_times[t_idx]
        t_prev = obs_times[t_idx - 1] if t_idx > 0 else 0.0

        # Apply discrete dividends that fall in this interval
        if dividends is not None and dividends.is_discrete:
            for ex_date, amount in zip(dividends.ex_dates, dividends.amounts):
                if t_prev < ex_date <= t_now:
                    # Proportional drop (prevents negative prices)
                    drop_frac = amount / S_prev
                    drop_frac = np.minimum(drop_frac, 0.5)  # Safety cap
                    S_prev = S_prev * (1.0 - drop_frac)

        S_prev = S_prev * np.exp(drift + sigma * np.sqrt(dt) * Z[:, t_idx])
        S[:, t_idx] = S_prev

    return S


# ══════════════════════════════════════════════════════════════════════
# SECTION 3: HESTON QE SCHEME WITH DIVIDENDS
# ══════════════════════════════════════════════════════════════════════

def simulate_heston_qe(
    S0: float,
    r: float,
    params: HestonParams,
    T: float,
    n_obs: int,
    n_paths: int,
    n_substeps: int = 20,
    dividends: Optional[DividendSchedule] = None,
    seed: Optional[int] = None,
    psi_c: float = 1.5,
) -> np.ndarray:
    """
    Heston Monte Carlo using the Quadratic Exponential (QE) scheme.

    The QE scheme (Andersen 2008) handles the variance process by
    switching between two approximations based on the ratio ψ = s²/m²
    where m, s² are the conditional mean and variance of v(t+Δt):

    - If ψ ≤ ψ_c: Quadratic approximation (matches first two moments)
    - If ψ > ψ_c: Exponential approximation (handles heavy right tail)

    This avoids negative variances WITHOUT flooring bias.

    Parameters
    ----------
    S0 : float
        Initial stock price.
    r : float
        Risk-free rate.
    params : HestonParams
        Heston model parameters.
    T : float
        Maturity in years.
    n_obs : int
        Number of observation dates.
    n_paths : int
        Number of simulation paths.
    n_substeps : int
        Sub-steps per observation interval.
    dividends : DividendSchedule, optional
        Dividend specification.
    seed : int, optional
        Random seed.
    psi_c : float
        Critical ψ threshold for switching between quadratic and
        exponential approximations. Andersen recommends 1.0–2.0.

    Returns
    -------
    S_obs : np.ndarray, shape (n_paths, n_obs)
        Simulated stock prices at observation dates.
    """
    if seed is not None:
        np.random.seed(seed)

    dt_obs = T / n_obs
    dt = dt_obs / n_substeps
    total_steps = n_obs * n_substeps
    obs_times = np.linspace(dt_obs, T, n_obs)

    # Continuous dividend yield
    q = 0.0
    if dividends is not None and not dividends.is_discrete:
        q = dividends.yield_pa

    # Precompute Heston constants
    kappa = params.kappa
    theta = params.theta
    xi = params.xi
    rho = params.rho

    # QE moment-matching constants
    e_kdt = np.exp(-kappa * dt)
    m_coeff_v = theta * (1.0 - e_kdt)  # Contribution to mean from theta
    # m(v) = v * e^{-κΔt} + θ(1 - e^{-κΔt})

    s2_coeff_v = (xi**2 * e_kdt / kappa) * (1.0 - e_kdt)
    # s²(v) = v * [ξ²e^{-κΔt}(1-e^{-κΔt})/κ]
    #        + θ * [ξ²(1-e^{-κΔt})²/(2κ)]
    s2_coeff_theta = theta * xi**2 * (1.0 - e_kdt)**2 / (2.0 * kappa)

    # Correlation helpers for log-stock
    K0 = -(rho * kappa * theta / xi) * dt
    K1 = 0.5 * dt * (rho * kappa / xi - 0.5) - rho / xi
    K2 = 0.5 * dt * (rho * kappa / xi - 0.5) + rho / xi
    K3 = 0.5 * dt * (1.0 - rho**2)
    K4 = 0.5 * dt * (1.0 - rho**2)

    S_obs = np.zeros((n_paths, n_obs))
    log_S = np.full(n_paths, np.log(S0))
    v = np.full(n_paths, params.v0)

    step_count = 0

    for obs_idx in range(n_obs):
        for sub in range(n_substeps):
            # ── QE step for variance ──────────────────────────────
            m = v * e_kdt + m_coeff_v
            s2 = v * s2_coeff_v + s2_coeff_theta
            s2 = np.maximum(s2, 1e-12)

            psi = s2 / (m**2 + 1e-12)

            # Split paths: quadratic vs exponential
            quad_mask = psi <= psi_c
            exp_mask = ~quad_mask

            v_new = np.empty(n_paths)
            U_v = np.random.uniform(size=n_paths)

            # ── Quadratic approximation (ψ ≤ ψ_c) ────────────────
            if np.any(quad_mask):
                m_q = m[quad_mask]
                s2_q = s2[quad_mask]
                psi_q = psi[quad_mask]

                inv_psi = 1.0 / (psi_q + 1e-12)
                b2 = 2.0 * inv_psi - 1.0 + np.sqrt(2.0 * inv_psi) * \
                     np.sqrt(np.maximum(2.0 * inv_psi - 1.0, 0.0))
                b2 = np.maximum(b2, 0.0)
                a = m_q / (1.0 + b2)

                Z_v = np.random.standard_normal(np.sum(quad_mask))
                b = np.sqrt(b2)
                v_new[quad_mask] = a * (b + Z_v)**2

            # ── Exponential approximation (ψ > ψ_c) ──────────────
            if np.any(exp_mask):
                m_e = m[exp_mask]
                psi_e = psi[exp_mask]

                p = (psi_e - 1.0) / (psi_e + 1.0)
                p = np.clip(p, 0.0, 0.999)
                beta = (1.0 - p) / (m_e + 1e-12)

                U_e = U_v[exp_mask]
                v_exp = np.where(
                    U_e <= p,
                    0.0,
                    np.log(np.maximum((1.0 - p) / (1.0 - U_e + 1e-12), 1e-12)) / (beta + 1e-12)
                )
                v_new[exp_mask] = np.maximum(v_exp, 0.0)

            # ── Log-stock update (martingale-corrected) ───────────
            Z_S = np.random.standard_normal(n_paths)

            log_S += (r - q) * dt + K0 + K1 * v + K2 * v_new + \
                     np.sqrt(K3 * v + K4 * v_new + 1e-12) * Z_S

            v = v_new
            step_count += 1

        # Record at observation date
        S_current = np.exp(log_S)

        # Apply discrete dividends at this observation
        if dividends is not None and dividends.is_discrete:
            t_now = obs_times[obs_idx]
            t_prev = obs_times[obs_idx - 1] if obs_idx > 0 else 0.0
            for ex_date, amount in zip(dividends.ex_dates, dividends.amounts):
                if t_prev < ex_date <= t_now:
                    drop_frac = amount / S_current
                    drop_frac = np.minimum(drop_frac, 0.5)
                    S_current = S_current * (1.0 - drop_frac)
                    log_S = np.log(S_current)

        S_obs[:, obs_idx] = S_current

    return S_obs


# ══════════════════════════════════════════════════════════════════════
# SECTION 4: COMPARISON UTILITY
# ══════════════════════════════════════════════════════════════════════

def compare_euler_vs_qe(
    S0: float,
    r: float,
    params: HestonParams,
    T: float,
    n_obs: int,
    n_paths: int = 100_000,
    seed: int = 42,
) -> dict:
    """
    Run both Euler and QE Heston and compare terminal distributions.
    Useful for validating that the QE scheme matches Euler in-sample
    while eliminating negative variance bias.
    """
    from .engines import simulate_heston as euler_heston

    S_euler = euler_heston(S0, r, params, T, n_obs, n_paths, seed=seed)
    S_qe = simulate_heston_qe(S0, r, params, T, n_obs, n_paths, seed=seed)

    return {
        'euler_terminal_mean': np.mean(S_euler[:, -1]),
        'euler_terminal_std': np.std(S_euler[:, -1]),
        'qe_terminal_mean': np.mean(S_qe[:, -1]),
        'qe_terminal_std': np.std(S_qe[:, -1]),
        'euler_terminal_skew': _skewness(S_euler[:, -1]),
        'qe_terminal_skew': _skewness(S_qe[:, -1]),
        'euler_terminal_kurt': _kurtosis(S_euler[:, -1]),
        'qe_terminal_kurt': _kurtosis(S_qe[:, -1]),
    }


def _skewness(x):
    m = np.mean(x)
    s = np.std(x)
    return np.mean(((x - m) / s) ** 3)


def _kurtosis(x):
    m = np.mean(x)
    s = np.std(x)
    return np.mean(((x - m) / s) ** 4) - 3.0

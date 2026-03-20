"""
sensitivity.py — Regime Sensitivity Analysis
=============================================
Examines how the mispricing gap between GBM and Heston varies across:
- Vol-of-vol (ξ) regimes
- Spot-vol correlation (ρ) regimes
- Initial volatility levels
- Barrier level perturbations
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple

from .note import AutocallableNote
from .engines import HestonParams, simulate_gbm, simulate_heston
from .pricer import price_autocallable


@dataclass
class SensitivityResult:
    """Container for a single sensitivity point."""
    param_name: str
    param_value: float
    heston_fv: float
    gbm_fv: float
    gap: float
    ki_breach_heston: float
    ki_breach_gbm: float
    autocall_prob_heston: float


def sweep_vol_of_vol(
    note: AutocallableNote,
    base_params: HestonParams,
    gbm_sigma: float = 0.255,
    xi_values: List[float] = None,
    n_paths: int = 100_000,
    seed: int = 42,
) -> List[SensitivityResult]:
    """Sweep vol-of-vol (ξ) and measure mispricing gap."""
    if xi_values is None:
        xi_values = [0.15, 0.25, 0.35, 0.50, 0.65, 0.80, 1.00]

    np.random.seed(seed)
    S_gbm = simulate_gbm(note.S0, note.r, gbm_sigma, note.maturity, note.n_obs, n_paths, seed=seed)
    res_gbm = price_autocallable(S_gbm, note)

    results = []
    for xi in xi_values:
        params = HestonParams(
            v0=base_params.v0, kappa=base_params.kappa,
            theta=base_params.theta, xi=xi, rho=base_params.rho,
        )
        S_h = simulate_heston(note.S0, note.r, params, note.maturity, note.n_obs, n_paths, seed=seed)
        res_h = price_autocallable(S_h, note)
        results.append(SensitivityResult(
            param_name="xi", param_value=xi,
            heston_fv=res_h.fair_value, gbm_fv=res_gbm.fair_value,
            gap=res_gbm.fair_value - res_h.fair_value,
            ki_breach_heston=res_h.ki_breach_prob,
            ki_breach_gbm=res_gbm.ki_breach_prob,
            autocall_prob_heston=res_h.autocall_prob,
        ))
    return results


def sweep_correlation(
    note: AutocallableNote,
    base_params: HestonParams,
    gbm_sigma: float = 0.255,
    rho_values: List[float] = None,
    n_paths: int = 100_000,
    seed: int = 42,
) -> List[SensitivityResult]:
    """Sweep spot-vol correlation (ρ) and measure mispricing gap."""
    if rho_values is None:
        rho_values = [-0.90, -0.75, -0.60, -0.45, -0.30, -0.15, 0.0]

    np.random.seed(seed)
    S_gbm = simulate_gbm(note.S0, note.r, gbm_sigma, note.maturity, note.n_obs, n_paths, seed=seed)
    res_gbm = price_autocallable(S_gbm, note)

    results = []
    for rho in rho_values:
        params = HestonParams(
            v0=base_params.v0, kappa=base_params.kappa,
            theta=base_params.theta, xi=base_params.xi, rho=rho,
        )
        S_h = simulate_heston(note.S0, note.r, params, note.maturity, note.n_obs, n_paths, seed=seed)
        res_h = price_autocallable(S_h, note)
        results.append(SensitivityResult(
            param_name="rho", param_value=rho,
            heston_fv=res_h.fair_value, gbm_fv=res_gbm.fair_value,
            gap=res_gbm.fair_value - res_h.fair_value,
            ki_breach_heston=res_h.ki_breach_prob,
            ki_breach_gbm=res_gbm.ki_breach_prob,
            autocall_prob_heston=res_h.autocall_prob,
        ))
    return results


def sweep_initial_vol(
    note: AutocallableNote,
    base_params: HestonParams,
    v0_values: List[float] = None,
    n_paths: int = 100_000,
    seed: int = 42,
) -> List[SensitivityResult]:
    """Sweep initial variance (v0) — simulates different vol environments."""
    if v0_values is None:
        v0_values = [0.03, 0.05, 0.065, 0.09, 0.12, 0.16, 0.25]

    results = []
    for v0 in v0_values:
        gbm_sigma = np.sqrt(v0)
        S_gbm = simulate_gbm(note.S0, note.r, gbm_sigma, note.maturity, note.n_obs, n_paths, seed=seed)
        res_gbm = price_autocallable(S_gbm, note)

        params = HestonParams(
            v0=v0, kappa=base_params.kappa,
            theta=base_params.theta, xi=base_params.xi, rho=base_params.rho,
        )
        S_h = simulate_heston(note.S0, note.r, params, note.maturity, note.n_obs, n_paths, seed=seed)
        res_h = price_autocallable(S_h, note)

        results.append(SensitivityResult(
            param_name="v0", param_value=v0,
            heston_fv=res_h.fair_value, gbm_fv=res_gbm.fair_value,
            gap=res_gbm.fair_value - res_h.fair_value,
            ki_breach_heston=res_h.ki_breach_prob,
            ki_breach_gbm=res_gbm.ki_breach_prob,
            autocall_prob_heston=res_h.autocall_prob,
        ))
    return results


def sweep_ki_barrier(
    note: AutocallableNote,
    base_params: HestonParams,
    gbm_sigma: float = 0.255,
    ki_levels: List[float] = None,
    n_paths: int = 100_000,
    seed: int = 42,
) -> List[SensitivityResult]:
    """Sweep knock-in barrier level and measure mispricing gap."""
    if ki_levels is None:
        ki_levels = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75]

    results = []
    for ki in ki_levels:
        # Create modified note
        mod_note = AutocallableNote(
            S0=note.S0, par=note.par, maturity=note.maturity, n_obs=note.n_obs,
            coupon_rate=note.coupon_rate, autocall_trigger=note.autocall_trigger,
            coupon_barrier=note.coupon_barrier, ki_barrier=ki, memory=note.memory,
            r=note.r, first_autocall_obs=note.first_autocall_obs,
        )

        S_gbm = simulate_gbm(note.S0, note.r, gbm_sigma, note.maturity, note.n_obs, n_paths, seed=seed)
        res_gbm = price_autocallable(S_gbm, mod_note)

        S_h = simulate_heston(note.S0, note.r, base_params, note.maturity, note.n_obs, n_paths, seed=seed)
        res_h = price_autocallable(S_h, mod_note)

        results.append(SensitivityResult(
            param_name="ki_barrier", param_value=ki,
            heston_fv=res_h.fair_value, gbm_fv=res_gbm.fair_value,
            gap=res_gbm.fair_value - res_h.fair_value,
            ki_breach_heston=res_h.ki_breach_prob,
            ki_breach_gbm=res_gbm.ki_breach_prob,
            autocall_prob_heston=res_h.autocall_prob,
        ))
    return results

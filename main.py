#!/usr/bin/env python3
"""
main.py — Run the complete Autocall Trap analysis
==================================================
Executes all pricing, sensitivity analysis, and figure generation.

Usage:
    python main.py                    # Full analysis (200k paths)
    python main.py --quick            # Quick run (50k paths)
    python main.py --paths 500000     # Custom path count
"""

import argparse
import time
import os
import sys
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.note import AutocallableNote, orcl_hsbc_note
from src.engines import HestonParams, orcl_heston, simulate_gbm, simulate_heston
from src.pricer import price_autocallable, compute_embedded_margin
from src.sensitivity import (
    sweep_vol_of_vol, sweep_correlation,
    sweep_initial_vol, sweep_ki_barrier,
)
from src.visualizations import (
    fig_payoff_distribution, fig_tail_risk_cdf, fig_autocall_timing,
    fig_vol_of_vol_sensitivity, fig_correlation_sensitivity,
    fig_ki_barrier_sensitivity, fig_sample_paths, fig_dashboard,
)


GBM_SIGMA = 0.255  # ATM implied vol for GBM benchmark


def main(n_paths: int = 200_000, seed: int = 42):
    os.makedirs("figures", exist_ok=True)

    note = orcl_hsbc_note()
    heston_params = orcl_heston()

    print(note.summary())
    print(f"\nHeston Parameters:\n{heston_params.summary()}")
    print(f"\nSimulation: {n_paths:,} paths, seed={seed}")

    # ── Layer 1: GBM Pricing ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("LAYER 1: GBM MONTE CARLO (NAIVE BENCHMARK)")
    print("=" * 60)
    t0 = time.time()
    S_gbm = simulate_gbm(note.S0, note.r, GBM_SIGMA, note.maturity,
                          note.n_obs, n_paths, seed=seed)
    res_gbm = price_autocallable(S_gbm, note)
    print(f"  Completed in {time.time()-t0:.1f}s")
    print(res_gbm.summary())
    margin_gbm, margin_gbm_pct = compute_embedded_margin(res_gbm.fair_value, note.par)
    print(f"  Embedded Margin:    ${margin_gbm:.2f} ({margin_gbm_pct:.2f}%)")

    # ── Layer 2: Heston Pricing ───────────────────────────────────────
    print("\n" + "=" * 60)
    print("LAYER 2: HESTON STOCHASTIC VOLATILITY")
    print("=" * 60)
    t0 = time.time()
    S_heston = simulate_heston(note.S0, note.r, heston_params, note.maturity,
                                note.n_obs, n_paths, seed=seed)
    res_heston = price_autocallable(S_heston, note)
    print(f"  Completed in {time.time()-t0:.1f}s")
    print(res_heston.summary())
    margin_h, margin_h_pct = compute_embedded_margin(res_heston.fair_value, note.par)
    print(f"  Embedded Margin:    ${margin_h:.2f} ({margin_h_pct:.2f}%)")

    # ── Layer 3: Mispricing Analysis ──────────────────────────────────
    print("\n" + "=" * 60)
    print("LAYER 3: MISPRICING ANALYSIS")
    print("=" * 60)
    gap = res_gbm.fair_value - res_heston.fair_value
    print(f"  Valuation Gap:      ${gap:.2f} ({gap/note.par*100:.2f}% of par)")
    print(f"  KI Breach Gap:      {(res_heston.ki_breach_prob - res_gbm.ki_breach_prob)*100:+.1f}pp")
    print(f"  Autocall Prob Gap:  {(res_heston.autocall_prob - res_gbm.autocall_prob)*100:+.1f}pp")
    print(f"  5th Pct Gap:        ${res_gbm.pct_5 - res_heston.pct_5:.0f}")

    # ── Sensitivity Analysis ──────────────────────────────────────────
    n_sens = min(n_paths, 100_000)
    print(f"\nRunning sensitivity analysis ({n_sens:,} paths each)...")

    t0 = time.time()
    xi_results = sweep_vol_of_vol(note, heston_params, GBM_SIGMA, n_paths=n_sens, seed=seed)
    rho_results = sweep_correlation(note, heston_params, GBM_SIGMA, n_paths=n_sens, seed=seed)
    ki_results = sweep_ki_barrier(note, heston_params, GBM_SIGMA, n_paths=n_sens, seed=seed)
    print(f"  Sensitivity analysis completed in {time.time()-t0:.1f}s")

    # ── Generate Figures ──────────────────────────────────────────────
    print("\nGenerating figures...")

    # Extract autocall timing data from the simulation
    # Re-run to get timing arrays (the pricer doesn't return these directly,
    # so we compute them here)
    autocall_times_gbm, terminated_gbm = _extract_autocall_times(S_gbm, note)
    autocall_times_heston, terminated_heston = _extract_autocall_times(S_heston, note)

    fig_payoff_distribution(res_gbm.payoffs, res_heston.payoffs, note)
    print("  [1/8] Payoff distribution")

    fig_tail_risk_cdf(res_gbm.payoffs, res_heston.payoffs, note)
    print("  [2/8] Tail risk CDF")

    fig_autocall_timing(autocall_times_gbm, terminated_gbm,
                        autocall_times_heston, terminated_heston, note)
    print("  [3/8] Autocall timing")

    fig_vol_of_vol_sensitivity(xi_results)
    print("  [4/8] Vol-of-vol sensitivity")

    fig_correlation_sensitivity(rho_results)
    print("  [5/8] Correlation sensitivity")

    fig_ki_barrier_sensitivity(ki_results)
    print("  [6/8] KI barrier sensitivity")

    fig_sample_paths(S_gbm, S_heston, note, n_show=50)
    print("  [7/8] Sample paths")

    fig_dashboard(res_gbm, res_heston, note)
    print("  [8/8] Summary dashboard")

    print(f"\nAll figures saved to figures/")
    print("Done.")


def _extract_autocall_times(S_paths, note):
    """Helper to extract autocall timing from simulated paths."""
    n_paths = S_paths.shape[0]
    terminated = np.zeros(n_paths, dtype=bool)
    autocall_time = np.full(n_paths, np.inf)

    for obs in range(note.n_obs):
        if obs < (note.first_autocall_obs - 1):
            continue
        S_t = S_paths[:, obs]
        t = note.obs_times[obs]
        hit = (S_t >= note.autocall_level) & ~terminated
        autocall_time = np.where(hit & (autocall_time == np.inf), t, autocall_time)
        terminated = terminated | hit

    return autocall_time, terminated


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="The Autocall Trap — Mispricing Analysis")
    parser.add_argument("--paths", type=int, default=200_000, help="Number of MC paths")
    parser.add_argument("--quick", action="store_true", help="Quick run with 50k paths")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    n = 50_000 if args.quick else args.paths
    main(n_paths=n, seed=args.seed)

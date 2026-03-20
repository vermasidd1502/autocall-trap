#!/usr/bin/env python3
"""
main_v2.py — Run the complete Revised Analysis (Post-Peer Review)
=================================================================
Incorporates all four peer review revisions:
  1. Heston calibration via characteristic function + LM optimizer
  2. Discrete dividend support (ORCL quarterly dividends)
  3. QE scheme replacing truncated Euler
  4. Historical stress test (2020 COVID, 2022 Tech Sell-off)

Usage:
    python main_v2.py                 # Full revised analysis
    python main_v2.py --quick         # Quick run (50k paths)
"""

import argparse
import time
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.note import AutocallableNote, orcl_hsbc_note
from src.engines import HestonParams, orcl_heston
from src.engines_v2 import (
    simulate_gbm_v2, simulate_heston_qe, compare_euler_vs_qe,
    DividendSchedule, orcl_dividends,
)
from src.pricer import price_autocallable, compute_embedded_margin
from src.calibration import (
    build_orcl_synthetic_surface, calibrate_heston, plot_calibration_fit,
)
from src.stress_test import (
    run_stress_tests, plot_stress_test_comparison, plot_rolling_regimes,
)


GBM_SIGMA = 0.255


def main(n_paths: int = 200_000, seed: int = 42):
    os.makedirs("figures", exist_ok=True)
    note = orcl_hsbc_note()
    divs = orcl_dividends(note.S0)

    print("=" * 65)
    print("THE AUTOCALL TRAP — REVISED ANALYSIS (Post-Peer Review)")
    print("=" * 65)
    print(note.summary())
    print(f"\nDividends: {divs.summary()}")

    # ══════════════════════════════════════════════════════════════════
    # REVISION 1: HESTON CALIBRATION
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'REVISION 1: HESTON CALIBRATION':=^65}")
    print("  Building synthetic ORCL vol surface...")
    surface = build_orcl_synthetic_surface(S0=note.S0, r=note.r, q=divs.yield_pa)
    print(f"  Surface points: {len(surface)}")

    print("  Running Levenberg-Marquardt optimization...")
    t0 = time.time()
    cal_result = calibrate_heston(
        surface, S0=note.S0, r=note.r, q=divs.yield_pa,
        initial_guess=HestonParams(v0=0.065, kappa=2.0, theta=0.07, xi=0.5, rho=-0.65),
        bounds={
            'v0':    (0.01, 0.15),
            'kappa': (0.5, 8.0),
            'theta': (0.01, 0.15),
            'xi':    (0.1, 1.2),
            'rho':   (-0.95, -0.1),
        },
    )
    print(f"  Completed in {time.time()-t0:.1f}s")
    print(cal_result.summary())

    plot_calibration_fit(cal_result, S0=note.S0)

    # Use calibrated params if fit is good, otherwise fall back to representative
    if cal_result.ivrmse < 0.02:  # < 200 bps IVRMSE
        cal_params = cal_result.params
        print(f"  Using CALIBRATED params (IVRMSE = {cal_result.ivrmse*10000:.1f} bps)")
    else:
        cal_params = orcl_heston()
        print(f"  Calibration IVRMSE too high ({cal_result.ivrmse*10000:.0f} bps).")
        print(f"  Using REPRESENTATIVE params (requires real option chain for production).")
        print(f"  To improve: export ORCL option chain from Bloomberg OVDV to CSV,")
        print(f"  then call: load_surface_from_csv('orcl_chain.csv', S0, r, q)")
        print(f"  Params:\n{cal_params.summary()}")

    # ══════════════════════════════════════════════════════════════════
    # REVISION 2+3: QE SCHEME + DIVIDENDS — CORE REPRICING
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'REVISION 2+3: QE SCHEME + DIVIDENDS':=^65}")

    # Compare Euler vs QE on terminal distribution
    print("  Comparing Euler vs QE terminal distributions...")
    comp = compare_euler_vs_qe(note.S0, note.r, cal_params, note.maturity, note.n_obs,
                                n_paths=min(n_paths, 100_000), seed=seed)
    print(f"    Euler mean: ${comp['euler_terminal_mean']:.2f}, "
          f"std: ${comp['euler_terminal_std']:.2f}, "
          f"skew: {comp['euler_terminal_skew']:.3f}")
    print(f"    QE    mean: ${comp['qe_terminal_mean']:.2f}, "
          f"std: ${comp['qe_terminal_std']:.2f}, "
          f"skew: {comp['qe_terminal_skew']:.3f}")

    # GBM with dividends
    print(f"\n  GBM pricing (with dividends)...")
    t0 = time.time()
    S_gbm = simulate_gbm_v2(note.S0, note.r, GBM_SIGMA, note.maturity,
                              note.n_obs, n_paths, dividends=divs, seed=seed)
    res_gbm = price_autocallable(S_gbm, note)
    print(f"  Completed in {time.time()-t0:.1f}s")
    print(res_gbm.summary())

    # Heston QE with dividends + calibrated params
    print(f"\n  Heston QE pricing (calibrated params, with dividends)...")
    t0 = time.time()
    S_heston = simulate_heston_qe(note.S0, note.r, cal_params, note.maturity,
                                    note.n_obs, n_paths, dividends=divs, seed=seed)
    res_heston = price_autocallable(S_heston, note)
    print(f"  Completed in {time.time()-t0:.1f}s")
    print(res_heston.summary())

    # Mispricing with all revisions applied
    gap = res_gbm.fair_value - res_heston.fair_value
    print(f"\n  {'REVISED MISPRICING':=^55}")
    print(f"  GBM FV (with divs):     ${res_gbm.fair_value:.2f}")
    print(f"  Heston FV (cal+QE+div): ${res_heston.fair_value:.2f}")
    print(f"  Valuation Gap:          ${gap:.2f} ({gap/note.par*100:.2f}%)")
    print(f"  KI Gap:                 "
          f"{(res_heston.ki_breach_prob - res_gbm.ki_breach_prob)*100:+.1f}pp")

    # ══════════════════════════════════════════════════════════════════
    # REVISION 4: HISTORICAL STRESS TEST
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'REVISION 4: HISTORICAL STRESS TEST':=^65}")

    n_stress = min(n_paths, 100_000)
    stress_results = run_stress_tests(note, dividends=divs,
                                       n_paths=n_stress, seed=seed)

    plot_stress_test_comparison(stress_results)
    plot_rolling_regimes()

    # ══════════════════════════════════════════════════════════════════
    # SUMMARY
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'REVISION SUMMARY':=^65}")
    print(f"""
  Original Analysis (v1):
    - Truncated Euler, no dividends, representative params
    - Hidden margin: ~1.42%

  Revised Analysis (v2):
    - QE scheme, discrete dividends, calibrated params
    - Hidden margin: {gap/note.par*100:.2f}%
    - Calibration IVRMSE: {cal_result.ivrmse*10000:.1f} bps

  Historical Stress Test:""")
    for key, sr in stress_results.items():
        print(f"    {sr.regime.name}: gap = ${sr.valuation_gap:.1f} "
              f"({sr.gap_pct:.2f}%), KI gap = {sr.ki_breach_gap_pp:+.1f}pp")

    print(f"""
  All figures saved to figures/
  New figures: fig9 (calibration), fig10 (stress test), fig11 (rolling regimes)
""")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="The Autocall Trap — Revised Analysis")
    parser.add_argument("--paths", type=int, default=200_000)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    n = 50_000 if args.quick else args.paths
    main(n_paths=n, seed=args.seed)

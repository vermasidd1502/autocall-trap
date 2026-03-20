"""
test_core.py — Unit tests for core pricing components
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.note import AutocallableNote, orcl_hsbc_note
from src.engines import HestonParams, orcl_heston, simulate_gbm, simulate_heston
from src.pricer import price_autocallable, compute_embedded_margin


def test_note_construction():
    """Term sheet parameters are computed correctly."""
    note = orcl_hsbc_note()
    assert note.autocall_level == 140.0
    assert note.coupon_level == 98.0
    assert note.ki_level == 84.0
    assert note.n_obs == 8
    assert len(note.obs_times) == 8
    assert np.isclose(note.obs_times[-1], 2.0)
    assert np.isclose(note.coupon_dollar, 26.25)
    print("PASS: test_note_construction")


def test_heston_feller():
    """Feller condition check works."""
    params = orcl_heston()
    assert params.feller_ratio > 1.0
    assert params.feller_satisfied
    # Violated case
    bad = HestonParams(v0=0.04, kappa=0.5, theta=0.04, xi=1.0, rho=-0.5)
    assert not bad.feller_satisfied
    print("PASS: test_heston_feller")


def test_gbm_shapes():
    """GBM simulation returns correct shapes."""
    S = simulate_gbm(100.0, 0.05, 0.2, 1.0, 4, 1000, seed=42)
    assert S.shape == (1000, 4)
    assert np.all(S > 0)
    print("PASS: test_gbm_shapes")


def test_heston_shapes():
    """Heston simulation returns correct shapes."""
    params = orcl_heston()
    S = simulate_heston(100.0, 0.05, params, 1.0, 4, 1000, seed=42)
    assert S.shape == (1000, 4)
    assert np.all(S > 0)
    print("PASS: test_heston_shapes")


def test_gbm_drift():
    """GBM mean should approximate risk-neutral drift."""
    n_paths = 500_000
    S = simulate_gbm(100.0, 0.05, 0.2, 1.0, 1, n_paths, seed=42)
    expected = 100.0 * np.exp(0.05 * 1.0)
    actual = np.mean(S[:, 0])
    assert abs(actual - expected) / expected < 0.01, \
        f"GBM drift off: expected {expected:.2f}, got {actual:.2f}"
    print("PASS: test_gbm_drift")


def test_pricing_par_convergence():
    """For very low vol, note should be worth approximately par + coupons."""
    note = AutocallableNote(S0=140.0, par=1000.0, maturity=2.0, n_obs=8,
                            coupon_rate=0.02, autocall_trigger=0.5,
                            coupon_barrier=0.1, ki_barrier=0.01,
                            memory=False, r=0.0, first_autocall_obs=1)
    S = simulate_gbm(140.0, 0.0, 0.01, 2.0, 8, 50000, seed=42)
    res = price_autocallable(S, note)
    # Should autocall at first obs with very high prob (trigger is 50%)
    assert res.autocall_prob > 0.99
    print("PASS: test_pricing_par_convergence")


def test_embedded_margin():
    """Margin computation is correct."""
    margin, pct = compute_embedded_margin(980.0, 1000.0)
    assert np.isclose(margin, 20.0)
    assert np.isclose(pct, 2.0)
    print("PASS: test_embedded_margin")


def test_heston_heavier_tails():
    """Heston should produce heavier tails than GBM (core thesis)."""
    note = orcl_hsbc_note()
    params = orcl_heston()
    n = 100_000
    S_gbm = simulate_gbm(note.S0, note.r, 0.255, note.maturity, note.n_obs, n, seed=42)
    S_heston = simulate_heston(note.S0, note.r, params, note.maturity, note.n_obs, n, seed=42)

    res_gbm = price_autocallable(S_gbm, note)
    res_heston = price_autocallable(S_heston, note)

    # Heston should have lower 5th percentile (heavier left tail)
    assert res_heston.pct_5 < res_gbm.pct_5, \
        f"Expected Heston 5th pct < GBM: {res_heston.pct_5:.0f} vs {res_gbm.pct_5:.0f}"
    # Heston should have higher KI breach probability
    assert res_heston.ki_breach_prob > res_gbm.ki_breach_prob
    print("PASS: test_heston_heavier_tails")


if __name__ == "__main__":
    test_note_construction()
    test_heston_feller()
    test_gbm_shapes()
    test_heston_shapes()
    test_gbm_drift()
    test_pricing_par_convergence()
    test_embedded_margin()
    test_heston_heavier_tails()
    print("\n" + "=" * 40)
    print("ALL TESTS PASSED")
    print("=" * 40)

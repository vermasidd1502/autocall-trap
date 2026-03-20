"""
pricer.py — Autocallable Note Payoff Engine
============================================
Computes path-dependent payoffs for autocallable contingent income
barrier notes with memory coupon, given simulated stock paths.
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple

from .note import AutocallableNote


@dataclass
class PricingResult:
    """Container for pricing output."""
    fair_value: float
    std_error: float
    payoffs: np.ndarray
    autocall_prob: float
    avg_autocall_time: float
    ki_breach_prob: float
    avg_coupons_paid: float
    pct_1: float
    pct_5: float
    pct_10: float
    pct_25: float
    es_5: float       # Expected Shortfall (CVaR) at 5% level
    es_1: float       # Expected Shortfall (CVaR) at 1% level
    max_loss: float   # Worst single-path payoff

    def summary(self) -> str:
        lines = [
            f"  Fair Value:         ${self.fair_value:.2f} ± ${1.96*self.std_error:.2f} (95% CI)",
            f"  Autocall Prob:      {self.autocall_prob*100:.1f}%",
            f"  Avg Autocall Time:  {self.avg_autocall_time:.2f}y",
            f"  KI Breach Prob:     {self.ki_breach_prob*100:.1f}%",
            f"  Avg Coupons Paid:   {self.avg_coupons_paid:.1f}",
            f"  VaR(5%):            ${self.pct_5:.0f}  (5th percentile payoff)",
            f"  ES(5%)/CVaR:        ${self.es_5:.0f}  (avg of worst 5% paths)",
            f"  VaR(1%):            ${self.pct_1:.0f}  (1st percentile payoff)",
            f"  ES(1%)/CVaR:        ${self.es_1:.0f}  (avg of worst 1% paths)",
            f"  Max Loss:           ${self.max_loss:.0f}  (worst single path)",
        ]
        return "\n".join(lines)


def price_autocallable(
    S_paths: np.ndarray,
    note: AutocallableNote,
) -> PricingResult:
    """
    Price the autocallable note given simulated stock paths.

    Parameters
    ----------
    S_paths : np.ndarray, shape (n_paths, n_obs)
        Simulated stock prices at each observation date.
    note : AutocallableNote
        The note term sheet.

    Returns
    -------
    PricingResult
        Comprehensive pricing output.
    """
    n_paths = S_paths.shape[0]
    payoffs = np.zeros(n_paths)
    terminated = np.zeros(n_paths, dtype=bool)
    autocall_time = np.full(n_paths, np.inf)

    unpaid_coupons = np.zeros(n_paths)
    total_coupons_pv = np.zeros(n_paths)
    coupons_paid_count = np.zeros(n_paths)

    for obs in range(note.n_obs):
        S_t = S_paths[:, obs]
        t = note.obs_times[obs]
        df = np.exp(-note.r * t)
        alive = ~terminated

        # ── Coupon logic ──
        above_barrier = (S_t >= note.coupon_level) & alive

        if note.memory:
            coupon_payment = np.where(
                above_barrier,
                (1.0 + unpaid_coupons) * note.coupon_dollar,
                0.0,
            )
            coupons_this_period = np.where(above_barrier, 1.0 + unpaid_coupons, 0.0)
            unpaid_coupons = np.where(
                above_barrier, 0.0,
                np.where(alive, unpaid_coupons + 1.0, unpaid_coupons),
            )
        else:
            coupon_payment = np.where(above_barrier, note.coupon_dollar, 0.0)
            coupons_this_period = np.where(above_barrier, 1.0, 0.0)

        total_coupons_pv += coupon_payment * df
        coupons_paid_count += coupons_this_period

        # ── Autocall logic ──
        if obs >= (note.first_autocall_obs - 1):
            autocall_hit = (S_t >= note.autocall_level) & alive
            payoffs += np.where(autocall_hit, note.par * df, 0.0)
            autocall_time = np.where(
                autocall_hit & (autocall_time == np.inf), t, autocall_time
            )
            terminated = terminated | autocall_hit

    # ── Maturity logic ──
    still_alive = ~terminated
    S_final = S_paths[:, -1]
    df_T = np.exp(-note.r * note.maturity)

    # Knock-in: stock closed below KI barrier at ANY observation
    min_obs_prices = np.min(S_paths, axis=1)
    knock_in = min_obs_prices < note.ki_level

    maturity_payoff = np.where(
        still_alive & knock_in,
        note.par * np.minimum(S_final / note.S0, 1.0),  # Capped at par
        np.where(still_alive, note.par, 0.0),
    )

    payoffs += maturity_payoff * df_T
    payoffs += total_coupons_pv

    # ── Compute statistics ──
    fair_value = np.mean(payoffs)
    std_error = np.std(payoffs) / np.sqrt(n_paths)
    autocall_prob = np.mean(terminated)
    avg_autocall_time = (
        np.mean(autocall_time[terminated]) if np.any(terminated) else np.nan
    )
    ki_breach_prob = np.mean(min_obs_prices < note.ki_level)

    # ── Tail risk metrics ──
    pct_1 = np.percentile(payoffs, 1)
    pct_5 = np.percentile(payoffs, 5)

    # Expected Shortfall (CVaR): average of payoffs BELOW the VaR threshold
    # ES(α) = E[X | X ≤ VaR(α)]
    tail_5_mask = payoffs <= pct_5
    es_5 = np.mean(payoffs[tail_5_mask]) if np.any(tail_5_mask) else pct_5

    tail_1_mask = payoffs <= pct_1
    es_1 = np.mean(payoffs[tail_1_mask]) if np.any(tail_1_mask) else pct_1

    return PricingResult(
        fair_value=fair_value,
        std_error=std_error,
        payoffs=payoffs,
        autocall_prob=autocall_prob,
        avg_autocall_time=avg_autocall_time,
        ki_breach_prob=ki_breach_prob,
        avg_coupons_paid=np.mean(coupons_paid_count),
        pct_1=pct_1,
        pct_5=pct_5,
        pct_10=np.percentile(payoffs, 10),
        pct_25=np.percentile(payoffs, 25),
        es_5=es_5,
        es_1=es_1,
        max_loss=np.min(payoffs),
    )


def compute_embedded_margin(
    fair_value: float, par: float = 1000.0
) -> Tuple[float, float]:
    """
    Compute embedded margin in the note.

    Returns (margin_dollars, margin_pct).
    Positive = issuer profits, negative = note trades above par.
    """
    margin = par - fair_value
    return margin, margin / par * 100

"""
note.py — Autocallable Note Term Sheet Specification
=====================================================
Encapsulates the complete structure of an Autocallable Contingent Income
Barrier Note with Memory Coupon, as typically filed in SEC Form 424B2.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AutocallableNote:
    """
    Full term sheet specification for an autocallable contingent income
    barrier note with memory coupon.

    Parameters
    ----------
    S0 : float
        Initial stock price at pricing date.
    par : float
        Face/notional value of the note.
    maturity : float
        Time to maturity in years.
    n_obs : int
        Number of observation dates (typically quarterly).
    coupon_rate : float
        Per-period contingent coupon rate (as decimal).
    autocall_trigger : float
        Autocall trigger as fraction of S0 (e.g. 1.0 = 100%).
    coupon_barrier : float
        Coupon barrier as fraction of S0 (e.g. 0.70 = 70%).
    ki_barrier : float
        Knock-in put barrier as fraction of S0 (e.g. 0.60 = 60%).
    memory : bool
        Whether the note has a memory coupon feature.
    r : float
        Risk-free rate (continuous compounding).
    first_autocall_obs : int
        First observation eligible for autocall (1-indexed).
    issuer_spread : float
        Issuer credit spread over risk-free (for funding adjustment).
    """

    S0: float = 140.0
    par: float = 1000.0
    maturity: float = 2.0
    n_obs: int = 8
    coupon_rate: float = 0.02625
    autocall_trigger: float = 1.0
    coupon_barrier: float = 0.70
    ki_barrier: float = 0.60
    memory: bool = True
    r: float = 0.045
    first_autocall_obs: int = 2
    issuer_spread: float = 0.005

    # Derived quantities (computed post-init)
    autocall_level: float = field(init=False)
    coupon_level: float = field(init=False)
    ki_level: float = field(init=False)
    dt: float = field(init=False)
    obs_times: np.ndarray = field(init=False, repr=False)
    coupon_dollar: float = field(init=False)

    def __post_init__(self):
        self.autocall_level = self.autocall_trigger * self.S0
        self.coupon_level = self.coupon_barrier * self.S0
        self.ki_level = self.ki_barrier * self.S0
        self.dt = self.maturity / self.n_obs
        self.obs_times = np.linspace(self.dt, self.maturity, self.n_obs)
        self.coupon_dollar = self.coupon_rate * self.par

    def summary(self) -> str:
        """Return formatted term sheet summary."""
        lines = [
            "=" * 60,
            "AUTOCALLABLE NOTE — TERM SHEET",
            "=" * 60,
            f"  Underlying:         S0 = ${self.S0:.2f}",
            f"  Par Value:          ${self.par:,.0f}",
            f"  Maturity:           {self.maturity:.1f}y ({self.n_obs} obs, {self.dt:.2f}y apart)",
            f"  Coupon:             {self.coupon_rate*100:.3f}% per period "
            f"({self.coupon_rate*self.n_obs/self.maturity*100:.1f}% p.a.)",
            f"  Memory Coupon:      {'Yes' if self.memory else 'No'}",
            f"  Autocall Trigger:   {self.autocall_trigger*100:.0f}% "
            f"(${self.autocall_level:.2f})",
            f"  Coupon Barrier:     {self.coupon_barrier*100:.0f}% "
            f"(${self.coupon_level:.2f})",
            f"  Knock-In Barrier:   {self.ki_barrier*100:.0f}% "
            f"(${self.ki_level:.2f})",
            f"  Risk-Free Rate:     {self.r*100:.2f}%",
            f"  Issuer Spread:      {self.issuer_spread*100:.1f}bp",
            f"  1st Autocall Obs:   #{self.first_autocall_obs}",
            "=" * 60,
        ]
        return "\n".join(lines)


# ── Preset configurations ─────────────────────────────────────────────

def orcl_hsbc_note() -> AutocallableNote:
    """HSBC ORCL-linked note (representative of SEC filing)."""
    return AutocallableNote(
        S0=140.0, par=1000.0, maturity=2.0, n_obs=8,
        coupon_rate=0.02625, autocall_trigger=1.0,
        coupon_barrier=0.70, ki_barrier=0.60, memory=True,
        r=0.045, first_autocall_obs=2, issuer_spread=0.005,
    )


def gs_wmt_note() -> AutocallableNote:
    """Goldman Sachs WMT-linked note (representative)."""
    return AutocallableNote(
        S0=165.0, par=1000.0, maturity=1.5, n_obs=6,
        coupon_rate=0.02, autocall_trigger=1.0,
        coupon_barrier=0.75, ki_barrier=0.65, memory=True,
        r=0.045, first_autocall_obs=2, issuer_spread=0.006,
    )

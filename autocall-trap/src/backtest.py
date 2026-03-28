"""
backtest.py — SCP Factor Backtesting Engine
============================================
A production-ready framework for backtesting the Structural Complexity
Premium factor across a historical universe of autocallable notes.

WORKFLOW:
  1. Load a CSV of historical note issuances (term sheets + outcomes)
  2. For each note, compute GBM and Heston fair values at issuance
  3. Compute the SCP factor (the gap)
  4. Sort notes into quintiles by SCP
  5. Track realized outcomes by quintile
  6. Generate performance attribution and statistical tests

DATA REQUIREMENTS (see docstring at bottom of file):
  - notes_universe.csv: term sheets + realized outcomes
  - vol_surfaces/: implied vol at issuance (optional, can use ATM proxy)

Author: Siddharth Verma
"""

import numpy as np
import csv
import os
import sys
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.note import AutocallableNote
from src.engines import HestonParams
from src.engines_v2 import simulate_gbm_v2, simulate_heston_qe, DividendSchedule
from src.pricer import price_autocallable


# ══════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ══════════════════════════════════════════════════════════════════

@dataclass
class HistoricalNote:
    """A single historical autocallable note with term sheet and outcome."""
    # -- Identifiers --
    note_id: str
    issuer: str
    underlying: str
    issue_date: str            # YYYY-MM-DD

    # -- Term sheet --
    S0: float                  # Initial stock price
    par: float                 # Face value (typically 1000)
    maturity: float            # Years
    n_obs: int                 # Number of observation dates
    coupon_rate: float         # Per-period coupon (decimal)
    autocall_trigger: float    # As fraction of S0 (e.g. 1.0)
    coupon_barrier: float      # As fraction of S0 (e.g. 0.70)
    ki_barrier: float          # As fraction of S0 (e.g. 0.60)
    memory: bool               # Memory coupon
    first_autocall_obs: int    # 1-indexed

    # -- Market data at issuance --
    risk_free_rate: float      # At issuance
    atm_iv: float              # ATM implied vol at issuance
    div_yield: float           # Continuous dividend yield
    issuer_estimated_value: float  # SEC-mandated estimated value (if available)

    # -- Heston calibration at issuance (if available) --
    heston_v0: Optional[float] = None
    heston_kappa: Optional[float] = None
    heston_theta: Optional[float] = None
    heston_xi: Optional[float] = None
    heston_rho: Optional[float] = None

    # -- Realized outcome --
    outcome: Optional[str] = None     # "autocalled", "matured_par", "matured_loss", "still_live"
    autocall_date: Optional[str] = None
    realized_payoff: Optional[float] = None  # Total cash received per $1000 par
    realized_return: Optional[float] = None  # (payoff - par) / par
    holding_period_years: Optional[float] = None

    # -- Computed by engine --
    gbm_fair_value: Optional[float] = None
    heston_fair_value: Optional[float] = None
    scp: Optional[float] = None           # Structural Complexity Premium (%)
    gbm_es5: Optional[float] = None
    heston_es5: Optional[float] = None
    es_gap: Optional[float] = None
    quintile: Optional[int] = None        # 1 (lowest SCP) to 5 (highest SCP)


@dataclass
class BacktestResult:
    """Aggregate results of the backtest."""
    n_notes: int
    n_with_outcomes: int
    avg_scp: float
    median_scp: float

    # Quintile performance
    quintile_avg_return: Dict[int, float] = field(default_factory=dict)
    quintile_avg_scp: Dict[int, float] = field(default_factory=dict)
    quintile_ki_breach_rate: Dict[int, float] = field(default_factory=dict)
    quintile_autocall_rate: Dict[int, float] = field(default_factory=dict)
    quintile_count: Dict[int, int] = field(default_factory=dict)

    # Strategy 1: Long-Only Q1 vs Benchmark (replaces impractical long-short)
    q1_avg_return: Optional[float] = None        # Avg return of fairest-priced notes
    benchmark_return: Optional[float] = None       # S&P 500 buy-and-hold over same period
    q1_vs_benchmark: Optional[float] = None        # Excess return of Q1 over benchmark

    # Legacy: Long-short spread (kept for reference, but flagged as impractical)
    ls_return: Optional[float] = None  # Q1 return minus Q5 return

    # Strategy 2: Synthetic Short Replication (conceptual)
    q5_avg_scp_margin: Optional[float] = None      # Avg issuer margin in Q5 (capturable via options)

    # Statistical significance
    t_stat: Optional[float] = None
    p_value: Optional[float] = None

    # Fama-MacBeth regression results
    fm_beta: Optional[float] = None        # SCP factor loading
    fm_t_stat_nw: Optional[float] = None   # Newey-West corrected t-stat
    fm_p_value_nw: Optional[float] = None  # Newey-West corrected p-value

    notes: List[HistoricalNote] = field(default_factory=list, repr=False)


# ══════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════

def load_notes_from_csv(filepath: str) -> List[HistoricalNote]:
    """
    Load historical notes from CSV.

    Required columns:
        note_id, issuer, underlying, issue_date, S0, par, maturity,
        n_obs, coupon_rate, autocall_trigger, coupon_barrier, ki_barrier,
        memory, first_autocall_obs, risk_free_rate, atm_iv, div_yield

    Optional columns:
        issuer_estimated_value, heston_v0, heston_kappa, heston_theta,
        heston_xi, heston_rho, outcome, autocall_date, realized_payoff,
        realized_return, holding_period_years
    """
    notes = []
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            note = HistoricalNote(
                note_id=row['note_id'],
                issuer=row['issuer'],
                underlying=row['underlying'],
                issue_date=row['issue_date'],
                S0=float(row['S0']),
                par=float(row.get('par', 1000)),
                maturity=float(row['maturity']),
                n_obs=int(row['n_obs']),
                coupon_rate=float(row['coupon_rate']),
                autocall_trigger=float(row['autocall_trigger']),
                coupon_barrier=float(row['coupon_barrier']),
                ki_barrier=float(row['ki_barrier']),
                memory=row.get('memory', 'True').lower() in ('true', '1', 'yes'),
                first_autocall_obs=int(row.get('first_autocall_obs', 2)),
                risk_free_rate=float(row['risk_free_rate']),
                atm_iv=float(row['atm_iv']),
                div_yield=float(row.get('div_yield', 0)),
                issuer_estimated_value=_float_or_none(row.get('issuer_estimated_value')),
                heston_v0=_float_or_none(row.get('heston_v0')),
                heston_kappa=_float_or_none(row.get('heston_kappa')),
                heston_theta=_float_or_none(row.get('heston_theta')),
                heston_xi=_float_or_none(row.get('heston_xi')),
                heston_rho=_float_or_none(row.get('heston_rho')),
                outcome=row.get('outcome'),
                autocall_date=row.get('autocall_date'),
                realized_payoff=_float_or_none(row.get('realized_payoff')),
                realized_return=_float_or_none(row.get('realized_return')),
                holding_period_years=_float_or_none(row.get('holding_period_years')),
            )
            notes.append(note)
    return notes


def _float_or_none(val):
    if val is None or val == '' or val == 'NA':
        return None
    return float(val)


# ══════════════════════════════════════════════════════════════════
# PRICING ENGINE
# ══════════════════════════════════════════════════════════════════

def price_single_note(
    note: HistoricalNote,
    n_paths: int = 100_000,
    seed: Optional[int] = None,
) -> HistoricalNote:
    """
    Compute GBM and Heston fair values for a single historical note.
    Returns the note with computed fields populated.
    """
    # Build AutocallableNote from term sheet
    ac_note = AutocallableNote(
        S0=note.S0, par=note.par, maturity=note.maturity,
        n_obs=note.n_obs, coupon_rate=note.coupon_rate,
        autocall_trigger=note.autocall_trigger,
        coupon_barrier=note.coupon_barrier,
        ki_barrier=note.ki_barrier, memory=note.memory,
        r=note.risk_free_rate, first_autocall_obs=note.first_autocall_obs,
    )

    # Dividend schedule (continuous yield approximation)
    divs = DividendSchedule(yield_pa=note.div_yield) if note.div_yield > 0 else None

    # -- GBM pricing --
    S_gbm = simulate_gbm_v2(
        note.S0, note.risk_free_rate, note.atm_iv, note.maturity,
        note.n_obs, n_paths, dividends=divs, seed=seed,
    )
    res_gbm = price_autocallable(S_gbm, ac_note)
    note.gbm_fair_value = res_gbm.fair_value
    note.gbm_es5 = res_gbm.es_5

    # -- Heston pricing --
    # Use provided calibration if available, otherwise estimate from ATM IV
    if note.heston_v0 is not None:
        params = HestonParams(
            v0=note.heston_v0, kappa=note.heston_kappa,
            theta=note.heston_theta, xi=note.heston_xi, rho=note.heston_rho,
        )
    else:
        # Default mapping from ATM IV when no calibration is available
        v0 = note.atm_iv ** 2
        params = HestonParams(
            v0=v0, kappa=2.0, theta=v0 * 1.05,
            xi=max(0.3, min(note.atm_iv * 2.0, 1.0)),
            rho=-0.65,
        )

    S_heston = simulate_heston_qe(
        note.S0, note.risk_free_rate, params, note.maturity,
        note.n_obs, n_paths, dividends=divs, seed=seed,
    )
    res_heston = price_autocallable(S_heston, ac_note)
    note.heston_fair_value = res_heston.fair_value
    note.heston_es5 = res_heston.es_5

    # -- SCP computation --
    note.scp = (note.par - note.heston_fair_value) / note.par * 100
    note.es_gap = res_gbm.es_5 - res_heston.es_5

    return note


# ══════════════════════════════════════════════════════════════════
# BACKTEST RUNNER
# ══════════════════════════════════════════════════════════════════

def _run_fama_macbeth(
    priced_notes: List[HistoricalNote],
    result: 'BacktestResult',
    verbose: bool = True,
):
    """
    Run Fama-MacBeth cross-sectional regression to validate SCP as a
    predictive factor for realized note returns.

    For each issuance cohort (grouped by quarter), run:
        R_i = alpha + beta * SCP_i + epsilon_i

    Then compute the time-series average of beta and apply Newey-West
    standard errors to correct for serial correlation from overlapping
    return horizons.

    This is the standard academic methodology for validating cross-sectional
    return predictors (Fama & MacBeth, 1973).
    """
    # Group notes by issuance quarter
    notes_with_data = [n for n in priced_notes
                       if n.scp is not None and n.realized_return is not None]

    if len(notes_with_data) < 10:
        if verbose:
            print(f"\n  Fama-MacBeth: Insufficient data ({len(notes_with_data)} notes, need ≥10)")
        return

    # Parse issue dates into quarters
    quarters = {}
    for n in notes_with_data:
        try:
            dt = datetime.strptime(n.issue_date, '%Y-%m-%d')
            q_key = f"{dt.year}Q{(dt.month - 1) // 3 + 1}"
        except (ValueError, TypeError):
            q_key = "unknown"
        if q_key not in quarters:
            quarters[q_key] = []
        quarters[q_key].append(n)

    # Run cross-sectional OLS for each quarter
    betas = []
    for q_key in sorted(quarters.keys()):
        q_notes = quarters[q_key]
        if len(q_notes) < 3:
            continue  # Need at least 3 notes for meaningful regression

        y = np.array([n.realized_return for n in q_notes])
        x = np.array([n.scp for n in q_notes])

        # Simple OLS: y = alpha + beta * x
        x_dm = x - np.mean(x)
        beta = np.sum(x_dm * y) / (np.sum(x_dm ** 2) + 1e-12)
        betas.append(beta)

    if len(betas) < 2:
        if verbose:
            print(f"\n  Fama-MacBeth: Too few quarterly cohorts ({len(betas)})")
        return

    betas = np.array(betas)
    T = len(betas)
    beta_bar = np.mean(betas)

    # Newey-West standard error with automatic lag selection
    # Lag = floor(4 * (T/100)^(2/9))  — Andrews (1991) rule of thumb
    max_lag = max(1, int(np.floor(4 * (T / 100) ** (2.0 / 9.0))))
    max_lag = min(max_lag, T - 1)

    # Compute autocovariance-weighted variance
    demean = betas - beta_bar
    gamma_0 = np.sum(demean ** 2) / T

    nw_var = gamma_0
    for lag in range(1, max_lag + 1):
        # Bartlett kernel weight
        w = 1.0 - lag / (max_lag + 1.0)
        gamma_lag = np.sum(demean[lag:] * demean[:-lag]) / T
        nw_var += 2 * w * gamma_lag

    nw_se = np.sqrt(nw_var / T)

    if nw_se > 0:
        t_stat_nw = beta_bar / nw_se
        from scipy.stats import norm
        p_value_nw = 2 * (1 - norm.cdf(abs(t_stat_nw)))
    else:
        t_stat_nw = 0.0
        p_value_nw = 1.0

    result.fm_beta = beta_bar
    result.fm_t_stat_nw = t_stat_nw
    result.fm_p_value_nw = p_value_nw

    if verbose:
        print(f"\n  Fama-MacBeth: {T} quarterly cohorts, "
              f"beta={beta_bar:.4f}, NW t={t_stat_nw:.3f}, p={p_value_nw:.4f}")


def run_backtest(
    notes: List[HistoricalNote],
    n_paths: int = 100_000,
    seed_base: int = 42,
    verbose: bool = True,
) -> BacktestResult:
    """
    Run the full SCP factor backtest.

    Steps:
    1. Price each note under GBM and Heston
    2. Compute SCP for each note
    3. Sort into quintiles by SCP
    4. Compute realized performance by quintile
    5. Test statistical significance of Q1-Q5 spread
    """
    if verbose:
        print(f"\n{'SCP FACTOR BACKTEST':=^60}")
        print(f"  Notes: {len(notes)}")
        print(f"  Paths per note: {n_paths:,}")
        print()

    # -- Step 1: Price all notes --
    for i, note in enumerate(notes):
        seed = seed_base + i
        price_single_note(note, n_paths=n_paths, seed=seed)
        if verbose:
            status = f"SCP={note.scp:.2f}%"
            if note.realized_return is not None:
                status += f", realized={note.realized_return*100:.1f}%"
            print(f"  [{i+1}/{len(notes)}] {note.note_id} ({note.underlying}): {status}")

    # -- Step 2: Sort into quintiles --
    priced_notes = [n for n in notes if n.scp is not None]
    priced_notes.sort(key=lambda n: n.scp)

    quintile_size = len(priced_notes) // 5
    remainder = len(priced_notes) % 5
    idx = 0
    for q in range(1, 6):
        size = quintile_size + (1 if q <= remainder else 0)
        for j in range(size):
            if idx < len(priced_notes):
                priced_notes[idx].quintile = q
                idx += 1

    # -- Step 3: Compute quintile statistics --
    result = BacktestResult(
        n_notes=len(notes),
        n_with_outcomes=sum(1 for n in notes if n.realized_return is not None),
        avg_scp=np.mean([n.scp for n in priced_notes]),
        median_scp=np.median([n.scp for n in priced_notes]),
        notes=notes,
    )

    for q in range(1, 6):
        q_notes = [n for n in priced_notes if n.quintile == q]
        result.quintile_count[q] = len(q_notes)
        result.quintile_avg_scp[q] = np.mean([n.scp for n in q_notes])

        q_with_outcomes = [n for n in q_notes if n.realized_return is not None]
        if q_with_outcomes:
            result.quintile_avg_return[q] = np.mean([n.realized_return for n in q_with_outcomes])
            result.quintile_autocall_rate[q] = np.mean([1 if n.outcome == 'autocalled' else 0 for n in q_with_outcomes])
            result.quintile_ki_breach_rate[q] = np.mean([1 if n.outcome == 'matured_loss' else 0 for n in q_with_outcomes])

    # -- Step 4: Strategy 1 — Long-Only Q1 vs Benchmark --
    q1_returns = [n.realized_return for n in priced_notes if n.quintile == 1 and n.realized_return is not None]
    q5_returns = [n.realized_return for n in priced_notes if n.quintile == 5 and n.realized_return is not None]

    if q1_returns:
        result.q1_avg_return = np.mean(q1_returns)
        # Approximate S&P 500 benchmark: ~10% annualized, scaled to avg holding period
        avg_hold = np.mean([n.holding_period_years for n in priced_notes
                           if n.quintile == 1 and n.holding_period_years is not None] or [1.5])
        result.benchmark_return = (1.10 ** avg_hold - 1)  # Compound 10% p.a.
        result.q1_vs_benchmark = result.q1_avg_return - result.benchmark_return

    # Legacy long-short spread (flagged as impractical for OTC notes)
    if q1_returns and q5_returns:
        result.ls_return = np.mean(q1_returns) - np.mean(q5_returns)

        # Welch's t-test
        n1, n5 = len(q1_returns), len(q5_returns)
        m1, m5 = np.mean(q1_returns), np.mean(q5_returns)
        s1, s5 = np.std(q1_returns, ddof=1), np.std(q5_returns, ddof=1)
        if s1 > 0 and s5 > 0:
            se = np.sqrt(s1**2 / n1 + s5**2 / n5)
            result.t_stat = (m1 - m5) / se
            from scipy.stats import norm
            result.p_value = 2 * (1 - norm.cdf(abs(result.t_stat)))

    # -- Step 5: Strategy 2 — Synthetic Short (Capturable Margin) --
    q5_notes_with_scp = [n for n in priced_notes if n.quintile == 5 and n.scp is not None]
    if q5_notes_with_scp:
        result.q5_avg_scp_margin = np.mean([n.scp for n in q5_notes_with_scp])

    # -- Step 6: Fama-MacBeth cross-sectional regression --
    _run_fama_macbeth(priced_notes, result, verbose)

    # -- Print summary --
    if verbose:
        print(f"\n{'BACKTEST RESULTS':=^60}")
        print(f"  Notes priced:     {len(priced_notes)}")
        print(f"  With outcomes:    {result.n_with_outcomes}")
        print(f"  Avg SCP:          {result.avg_scp:.2f}%")
        print(f"  Median SCP:       {result.median_scp:.2f}%")
        print()
        print(f"  {'Q':>3} | {'Count':>5} | {'Avg SCP':>8} | {'Avg Return':>10} | {'Autocall':>8} | {'KI Breach':>9}")
        print(f"  {'-'*3}-+-{'-'*5}-+-{'-'*8}-+-{'-'*10}-+-{'-'*8}-+-{'-'*9}")
        for q in range(1, 6):
            cnt = result.quintile_count.get(q, 0)
            scp = result.quintile_avg_scp.get(q, 0)
            ret = result.quintile_avg_return.get(q, None)
            ac = result.quintile_autocall_rate.get(q, None)
            ki = result.quintile_ki_breach_rate.get(q, None)
            ret_str = f"{ret*100:.1f}%" if ret is not None else "N/A"
            ac_str = f"{ac*100:.0f}%" if ac is not None else "N/A"
            ki_str = f"{ki*100:.0f}%" if ki is not None else "N/A"
            print(f"  Q{q:1d} | {cnt:5d} | {scp:7.2f}% | {ret_str:>10s} | {ac_str:>8s} | {ki_str:>9s}")

        # Strategy 1: Long-Only Q1 vs Benchmark
        print(f"\n  {'STRATEGY 1: Long-Only Q1 vs Benchmark':-^55}")
        if result.q1_avg_return is not None:
            print(f"  Q1 (fairest notes) avg return:  {result.q1_avg_return*100:+.2f}%")
            print(f"  S&P 500 benchmark return:       {result.benchmark_return*100:+.2f}%")
            print(f"  Q1 excess return:               {result.q1_vs_benchmark*100:+.2f}%")
        else:
            print(f"  (No Q1 outcome data available)")

        # Legacy long-short (with caveat)
        if result.ls_return is not None:
            print(f"\n  {'REFERENCE: Long-Short Spread (impractical for OTC)':-^55}")
            print(f"  Q1 - Q5 spread: {result.ls_return*100:.2f}%")
            if result.t_stat is not None:
                print(f"  t-stat: {result.t_stat:.3f}, p-value: {result.p_value:.4f}")
                sig = "***" if result.p_value < 0.01 else "**" if result.p_value < 0.05 else "*" if result.p_value < 0.10 else ""
                print(f"  Significance: {sig if sig else 'Not significant'}")
            print(f"  NOTE: Shorting retail structured notes is not feasible.")
            print(f"  See Strategy 2 for synthetic replication.")

        # Strategy 2: Synthetic Short
        if result.q5_avg_scp_margin is not None:
            print(f"\n  {'STRATEGY 2: Synthetic Short Replication':-^55}")
            print(f"  Q5 avg SCP (capturable margin): {result.q5_avg_scp_margin:.2f}%")
            print(f"  A quant desk can replicate Q5 note downside using")
            print(f"  exchange-traded options (put spreads + digital coupons)")
            print(f"  and capture this margin by selling the replication at")
            print(f"  the issuer's inflated price.")

        # Fama-MacBeth results
        if result.fm_beta is not None:
            print(f"\n  {'FAMA-MACBETH REGRESSION':-^55}")
            print(f"  SCP factor beta:     {result.fm_beta:.4f}")
            print(f"  NW t-stat:           {result.fm_t_stat_nw:.3f}")
            print(f"  NW p-value:          {result.fm_p_value_nw:.4f}")
            sig = "***" if result.fm_p_value_nw < 0.01 else "**" if result.fm_p_value_nw < 0.05 else "*" if result.fm_p_value_nw < 0.10 else ""
            print(f"  Significance:        {sig if sig else 'Not significant'}")

    return result


# ══════════════════════════════════════════════════════════════════
# VISUALIZATION
# ══════════════════════════════════════════════════════════════════

def plot_backtest_results(
    result: BacktestResult,
    save_path: str = "figures/fig_backtest",
):
    """Generate backtest performance charts."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    has_returns = any(result.quintile_avg_return.get(q) is not None for q in range(1, 6))

    if has_returns:
        fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    else:
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes = list(axes) + [None]

    # Panel 1: SCP by quintile
    ax = axes[0]
    qs = range(1, 6)
    scps = [result.quintile_avg_scp.get(q, 0) for q in qs]
    colors = ['#4CAF50', '#8BC34A', '#FFC107', '#FF9800', '#F44336']
    ax.bar(qs, scps, color=colors, edgecolor='white', linewidth=1.5)
    ax.set_xlabel('SCP Quintile', fontsize=12, color='black')
    ax.set_ylabel('Average SCP (%)', fontsize=12, color='black')
    ax.set_title('Structural Complexity Premium by Quintile', fontsize=13, fontweight='bold')
    ax.set_xticks(list(qs))
    ax.set_xticklabels(['Q1\n(Fairest)', 'Q2', 'Q3', 'Q4', 'Q5\n(Most\nOverpriced)'])
    for i, v in enumerate(scps):
        ax.text(i+1, v + 0.05, f'{v:.1f}%', ha='center', fontsize=10, fontweight='bold')

    # Panel 2: Returns by quintile (if available)
    if has_returns:
        ax = axes[1]
        rets = [result.quintile_avg_return.get(q, 0) * 100 for q in qs]
        bar_colors = ['#2E7D32' if r > 0 else '#C62828' for r in rets]
        ax.bar(qs, rets, color=bar_colors, edgecolor='white', linewidth=1.5)
        ax.axhline(0, color='black', linewidth=0.8)
        ax.set_xlabel('SCP Quintile', fontsize=12, color='black')
        ax.set_ylabel('Avg Realized Return (%)', fontsize=12, color='black')
        ax.set_title('Realized Returns by Quintile', fontsize=13, fontweight='bold')
        ax.set_xticks(list(qs))
        for i, v in enumerate(rets):
            ax.text(i+1, v + 0.2 * np.sign(v), f'{v:.1f}%', ha='center', fontsize=10, fontweight='bold')

        # Panel 3: KI breach rate
        ax = axes[2]
        ki_rates = [result.quintile_ki_breach_rate.get(q, 0) * 100 for q in qs]
        ax.bar(qs, ki_rates, color=['#FFCDD2', '#EF9A9A', '#E57373', '#EF5350', '#F44336'],
               edgecolor='white', linewidth=1.5)
        ax.set_xlabel('SCP Quintile', fontsize=12, color='black')
        ax.set_ylabel('KI Breach Rate (%)', fontsize=12, color='black')
        ax.set_title('Knock-In Breach Rate by Quintile', fontsize=13, fontweight='bold')
        ax.set_xticks(list(qs))
    else:
        ax = axes[1]
        ax.text(0.5, 0.5, 'Realized outcomes not yet available.\n'
                'Load notes with outcome data to see\nperformance by quintile.',
                ha='center', va='center', fontsize=12, transform=ax.transAxes,
                bbox=dict(boxstyle='round', facecolor='#FFF3E0', edgecolor='#FF9800'))
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(f"{save_path}.png", dpi=150, bbox_inches='tight')
    plt.savefig(f"{save_path}.pdf", bbox_inches='tight')
    plt.close()
    print(f"  Backtest figure saved: {save_path}")


# ══════════════════════════════════════════════════════════════════
# SYNTHETIC DEMO (runs without real data)
# ══════════════════════════════════════════════════════════════════

def generate_synthetic_universe(n_notes: int = 30) -> List[HistoricalNote]:
    """
    Generate a synthetic universe of autocallable notes for demo purposes.
    Varies underlyings, barriers, vols, and issuers to simulate realistic
    cross-sectional dispersion in SCP.
    """
    np.random.seed(123)

    underlyings = [
        ('ORCL', 140, 0.255, 0.012),
        ('NVDA', 480, 0.42, 0.004),
        ('TSLA', 250, 0.50, 0.0),
        ('AAPL', 190, 0.22, 0.005),
        ('AMZN', 185, 0.30, 0.0),
        ('MSFT', 420, 0.22, 0.008),
        ('JPM', 200, 0.25, 0.025),
        ('GS', 480, 0.28, 0.022),
        ('META', 500, 0.35, 0.003),
        ('GOOG', 175, 0.27, 0.0),
    ]
    issuers = ['Goldman Sachs', 'HSBC', 'JP Morgan', 'Morgan Stanley', 'Barclays', 'Citigroup']
    ki_barriers = [0.55, 0.60, 0.65, 0.70]

    notes = []
    for i in range(n_notes):
        ticker, s0, vol, div_y = underlyings[i % len(underlyings)]
        issuer = issuers[i % len(issuers)]
        ki = ki_barriers[i % len(ki_barriers)]
        coupon = 0.015 + vol * 0.04 + np.random.uniform(-0.003, 0.003)

        # Simulate a realized outcome
        outcome_roll = np.random.random()
        if outcome_roll < 0.65:
            outcome = 'autocalled'
            realized_payoff = 1000 + coupon * 1000 * np.random.choice([2, 3, 4])
            holding = np.random.choice([0.5, 0.75, 1.0])
        elif outcome_roll < 0.85:
            outcome = 'matured_par'
            realized_payoff = 1000 + coupon * 1000 * 8
            holding = 2.0
        else:
            outcome = 'matured_loss'
            loss_frac = np.random.uniform(0.25, 0.55)
            realized_payoff = 1000 * loss_frac + coupon * 1000 * np.random.choice([1, 2, 3])
            holding = 2.0

        realized_return = (realized_payoff - 1000) / 1000

        notes.append(HistoricalNote(
            note_id=f"NOTE-{i+1:03d}",
            issuer=issuer,
            underlying=ticker,
            issue_date=f"202{np.random.choice([0,1,2,3])}-{np.random.randint(1,13):02d}-15",
            S0=s0 * (1 + np.random.uniform(-0.05, 0.05)),
            par=1000.0,
            maturity=2.0,
            n_obs=8,
            coupon_rate=round(coupon, 4),
            autocall_trigger=1.0,
            coupon_barrier=0.70,
            ki_barrier=ki,
            memory=True,
            first_autocall_obs=2,
            risk_free_rate=round(0.03 + np.random.uniform(0, 0.025), 4),
            atm_iv=round(vol * (1 + np.random.uniform(-0.1, 0.1)), 4),
            div_yield=round(div_y, 4),
            issuer_estimated_value=round(1000 * np.random.uniform(0.92, 0.98), 2),
            outcome=outcome,
            realized_payoff=round(realized_payoff, 2),
            realized_return=round(realized_return, 4),
            holding_period_years=holding,
        ))

    return notes


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SCP Factor Backtest")
    parser.add_argument("--csv", type=str, default=None, help="Path to notes CSV")
    parser.add_argument("--demo", action="store_true", help="Run with synthetic data")
    parser.add_argument("--paths", type=int, default=50_000, help="MC paths per note")
    args = parser.parse_args()

    if args.csv:
        notes = load_notes_from_csv(args.csv)
        print(f"Loaded {len(notes)} notes from {args.csv}")
    elif args.demo:
        notes = generate_synthetic_universe(n_notes=20)
        print(f"Generated {len(notes)} synthetic notes")
    else:
        print("Usage: python backtest.py --demo   OR   python backtest.py --csv notes.csv")
        print("\nRun --demo to see the engine work with synthetic data.")
        sys.exit(0)

    os.makedirs("figures", exist_ok=True)
    result = run_backtest(notes, n_paths=args.paths)
    plot_backtest_results(result)


# ══════════════════════════════════════════════════════════════════
# DATA REQUIREMENTS DOCUMENTATION
# ══════════════════════════════════════════════════════════════════

DATA_REQUIREMENTS = """
╔══════════════════════════════════════════════════════════════════╗
║           DATA SHOPPING LIST FOR THE SCP BACKTEST               ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  1. NOTES UNIVERSE (notes_universe.csv)                          ║
║     -------------------------------------                        ║
║     Source: SEC EDGAR (424B2 filings)                            ║
║     URL:    https://efts.sec.gov/LATEST/search-index             ║
║                                                                  ║
║     How to get it:                                               ║
║     • EDGAR full-text search: form type "424B2"                  ║
║       + keyword "autocall" or "contingent coupon"                ║
║     • Filter issuers: GS, HSBC, JPM, MS, Barclays, Citi         ║
║     • Each filing contains the full term sheet in the PDF        ║
║     • Extract: underlying, S0, barriers, coupon, maturity,       ║
║       observation dates, memory feature, estimated value         ║
║     • Tools: Python + BeautifulSoup/regex, or paid services      ║
║       like StructuredRetail.com or Halo Investing data feeds     ║
║                                                                  ║
║     Target: 200-500 notes, 2018-2024, across 6+ issuers         ║
║                                                                  ║
║  2. IMPLIED VOLATILITY AT ISSUANCE                               ║
║     ---------------------------------                            ║
║     Source: Bloomberg OVDV, CBOE LiveVol, OptionMetrics          ║
║                                                                  ║
║     What you need:                                               ║
║     • ATM implied vol for each underlying on the issue date      ║
║     • Ideally: full strike x maturity surface for Heston calib   ║
║     • Minimum viable: just ATM vol (the engine estimates         ║
║       Heston params from ATM vol if no surface is provided)      ║
║                                                                  ║
║     Bloomberg workflow:                                          ║
║     OVDV <GO> → select underlying → set date to issue date       ║
║     → export surface as CSV                                      ║
║                                                                  ║
║     Free alternative: use CBOE delayed data or Yahoo Finance     ║
║     options chain (ATM vol only, no surface)                     ║
║                                                                  ║
║  3. REALIZED OUTCOMES                                            ║
║     -----------------                                            ║
║     Source: Issuer websites, Bloomberg PORT, Halo Investing       ║
║                                                                  ║
║     For each note, you need to know:                             ║
║     • Did it autocall? If so, on which date?                     ║
║     • Did it mature at par (no KI breach)?                       ║
║     • Did it mature with losses (KI breach)?                     ║
║     • Total cash received (coupons + terminal payoff)            ║
║                                                                  ║
║     This is the hardest data to get. Options:                    ║
║     a) Track notes you identify from EDGAR going forward         ║
║     b) Use issuer post-maturity reports (some publish these)     ║
║     c) Reconstruct from historical stock prices:                 ║
║        given the term sheet + ORCL daily closes, you can         ║
║        determine exactly what happened on each obs date          ║
║                                                                  ║
║  4. RISK-FREE RATES (historical)                                 ║
║     ------------------------------                               ║
║     Source: FRED (Federal Reserve Economic Data)                 ║
║     Series: DGS2 (2-year Treasury yield)                         ║
║     URL:    https://fred.stlouisfed.org/series/DGS2              ║
║                                                                  ║
║  5. DIVIDEND DATA                                                ║
║     --------------                                               ║
║     Source: Bloomberg BDVD, Yahoo Finance, or Nasdaq.com          ║
║     Need: ex-dates and amounts for each underlying               ║
║                                                                  ║
║  6. ESG RATINGS (optional, for Strategy 3)                       ║
║     --------------------------------------                       ║
║     Source: MSCI ESG Ratings, Sustainalytics, Refinitiv          ║
║     Need: ESG rating for each underlying + each issuer           ║
║     Access: Bloomberg ESG <GO>, or MSCI ESG Manager              ║
║                                                                  ║
║  ------------------------------------------------------------    ║
║                                                                  ║
║  CSV FORMAT (notes_universe.csv):                                ║
║                                                                  ║
║  note_id,issuer,underlying,issue_date,S0,par,maturity,           ║
║  n_obs,coupon_rate,autocall_trigger,coupon_barrier,              ║
║  ki_barrier,memory,first_autocall_obs,risk_free_rate,            ║
║  atm_iv,div_yield,issuer_estimated_value,                        ║
║  outcome,realized_payoff,realized_return,holding_period_years    ║
║                                                                  ║
║  EXAMPLE ROW:                                                    ║
║  NOTE-001,HSBC,ORCL,2022-03-15,140.00,1000,2.0,8,              ║
║  0.02625,1.0,0.70,0.60,True,2,0.045,0.255,0.0114,              ║
║  962.50,autocalled,1052.50,0.0525,0.5                            ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
"""

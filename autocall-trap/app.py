"""
Autocall Trap — Interactive Note Evaluator & Backtest Dashboard
Run: streamlit run app.py
"""

import sys, os, io, time, tempfile
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Tuple
from datetime import datetime, timedelta

# ── Add project root to path ──
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from src.note import AutocallableNote
from src.engines import HestonParams, simulate_gbm, simulate_heston
from src.engines_v2 import simulate_gbm_v2, simulate_heston_qe, DividendSchedule
from src.pricer import price_autocallable, PricingResult, compute_embedded_margin
from src.backtest import (
    HistoricalNote, BacktestResult, load_notes_from_csv,
    price_single_note, run_backtest,
)

# ═══════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Autocall Trap | Note Evaluator",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ──
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #1a1a2e, #16213e, #0f3460);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1rem;
        color: #666;
        margin-top: -10px;
        margin-bottom: 20px;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 12px;
        color: white;
        text-align: center;
    }
    .metric-card h3 { color: white; margin: 0; font-size: 0.85rem; opacity: 0.85; }
    .metric-card h1 { color: white; margin: 5px 0 0 0; font-size: 1.8rem; }
    .good { background: linear-gradient(135deg, #11998e, #38ef7d); }
    .warn { background: linear-gradient(135deg, #f093fb, #f5576c); }
    .neutral { background: linear-gradient(135deg, #4facfe, #00f2fe); }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        border-radius: 8px 8px 0 0;
    }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════════════════════════
if "evaluated_notes" not in st.session_state:
    st.session_state.evaluated_notes = []  # List[dict] with pricing results
if "backtest_result" not in st.session_state:
    st.session_state.backtest_result = None
if "historical_notes" not in st.session_state:
    st.session_state.historical_notes = []


# ═══════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════

def _safe_float(val, default=None):
    """Safely extract float from pandas row (handles NaN, None, empty)."""
    if val is None:
        return default
    try:
        f = float(val)
        if np.isnan(f):
            return default
        return f
    except (ValueError, TypeError):
        return default


def _safe_str(val, default=None):
    """Safely extract string from pandas row (handles NaN)."""
    if val is None:
        return default
    if isinstance(val, float) and np.isnan(val):
        return default
    s = str(val).strip()
    return s if s else default


def calibrate_heston_from_iv(atm_iv: float, maturity: float) -> HestonParams:
    """Quick Heston calibration from ATM IV — maps vol surface shape."""
    v0 = atm_iv ** 2
    theta = v0 * 1.05  # slight upward mean-reversion
    kappa = 2.0
    xi = max(0.3, min(atm_iv * 1.2, 1.0))  # vol-of-vol scales with IV
    rho = -0.65 if atm_iv < 0.35 else -0.55  # less negative for high-vol names
    return HestonParams(v0=v0, kappa=kappa, theta=theta, xi=xi, rho=rho)


def evaluate_note(
    S0: float, par: float, maturity: float, n_obs: int,
    coupon_rate: float, autocall_trigger: float, coupon_barrier: float,
    ki_barrier: float, memory: bool, first_autocall_obs: int,
    risk_free_rate: float, atm_iv: float, div_yield: float,
    n_paths: int = 100_000, seed: int = 42,
    note_id: str = "", issuer: str = "", underlying: str = "",
    issuer_estimated_value: float = None,
    issue_date: str = "", outcome: str = None,
    realized_payoff: float = None, realized_return: float = None,
    holding_period_years: float = None,
) -> dict:
    """Run full GBM + Heston evaluation on a single note."""

    note = AutocallableNote(
        S0=S0, par=par, maturity=maturity, n_obs=n_obs,
        coupon_rate=coupon_rate, autocall_trigger=autocall_trigger,
        coupon_barrier=coupon_barrier, ki_barrier=ki_barrier,
        memory=memory, r=risk_free_rate, first_autocall_obs=first_autocall_obs,
    )

    heston_params = calibrate_heston_from_iv(atm_iv, maturity)
    divs = DividendSchedule(yield_pa=div_yield)

    # GBM simulation
    S_gbm = simulate_gbm_v2(
        S0, risk_free_rate, atm_iv, maturity, n_obs, n_paths,
        dividends=divs, seed=seed,
    )
    result_gbm = price_autocallable(S_gbm, note)

    # Heston QE simulation
    S_heston = simulate_heston_qe(
        S0, risk_free_rate, heston_params, maturity, n_obs, n_paths,
        dividends=divs, seed=seed,
    )
    result_heston = price_autocallable(S_heston, note)

    # SCP
    margin_gbm_dollar, margin_gbm_pct = compute_embedded_margin(result_gbm.fair_value, par)
    margin_heston_dollar, margin_heston_pct = compute_embedded_margin(result_heston.fair_value, par)

    scp = margin_heston_pct  # Heston-based SCP
    sec_margin = ((par - issuer_estimated_value) / par * 100) if issuer_estimated_value else None
    es_gap = result_gbm.es_5 - result_heston.es_5

    return {
        "note_id": note_id or f"NOTE-{int(time.time())}",
        "issuer": issuer,
        "underlying": underlying,
        "S0": S0, "par": par, "maturity": maturity, "n_obs": n_obs,
        "coupon_rate": coupon_rate,
        "autocall_trigger": autocall_trigger,
        "coupon_barrier": coupon_barrier,
        "ki_barrier": ki_barrier,
        "memory": memory,
        "risk_free_rate": risk_free_rate,
        "atm_iv": atm_iv,
        "div_yield": div_yield,
        "issuer_estimated_value": issuer_estimated_value,
        "first_autocall_obs": first_autocall_obs,
        # GBM results
        "gbm_fair_value": result_gbm.fair_value,
        "gbm_autocall_prob": result_gbm.autocall_prob,
        "gbm_ki_breach_prob": result_gbm.ki_breach_prob,
        "gbm_es5": result_gbm.es_5,
        "gbm_es1": result_gbm.es_1,
        "gbm_margin_pct": margin_gbm_pct,
        # Heston results
        "heston_fair_value": result_heston.fair_value,
        "heston_autocall_prob": result_heston.autocall_prob,
        "heston_ki_breach_prob": result_heston.ki_breach_prob,
        "heston_es5": result_heston.es_5,
        "heston_es1": result_heston.es_1,
        "heston_margin_pct": margin_heston_pct,
        # Derived
        "scp": scp,
        "sec_margin": sec_margin,
        "es_gap": es_gap,
        # Heston params used
        "heston_v0": heston_params.v0,
        "heston_kappa": heston_params.kappa,
        "heston_theta": heston_params.theta,
        "heston_xi": heston_params.xi,
        "heston_rho": heston_params.rho,
        # Realized outcome data (from historical CSV)
        "issue_date": issue_date or "2023-01-01",
        "outcome": outcome,
        "realized_payoff": realized_payoff,
        "realized_return": realized_return,
        "holding_period_years": holding_period_years,
        # Raw payoffs for plots
        "payoffs_gbm": result_gbm.payoffs,
        "payoffs_heston": result_heston.payoffs,
        "result_gbm": result_gbm,
        "result_heston": result_heston,
        "note_obj": note,
    }


def make_payoff_distribution_plot(result: dict) -> go.Figure:
    """Interactive payoff distribution histogram."""
    fig = make_subplots(rows=1, cols=1)
    payoffs_gbm = result["payoffs_gbm"]
    payoffs_heston = result["payoffs_heston"]

    fig.add_trace(go.Histogram(
        x=payoffs_gbm, nbinsx=100, name="GBM",
        marker_color="rgba(55, 128, 255, 0.6)",
        histnorm="probability density",
    ))
    fig.add_trace(go.Histogram(
        x=payoffs_heston, nbinsx=100, name="Heston",
        marker_color="rgba(255, 65, 54, 0.6)",
        histnorm="probability density",
    ))

    fig.add_vline(x=result["par"], line_dash="dash", line_color="green",
                  annotation_text="Par ($1,000)")
    fig.add_vline(x=result["heston_fair_value"], line_dash="dot", line_color="red",
                  annotation_text=f"Heston FV (${result['heston_fair_value']:.0f})")

    fig.update_layout(
        title="Payoff Distribution: GBM vs Heston",
        xaxis_title="Discounted Payoff ($)",
        yaxis_title="Density",
        barmode="overlay",
        template="plotly_white",
        height=400,
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
    )
    return fig


def make_tail_risk_cdf(result: dict) -> go.Figure:
    """CDF with Expected Shortfall annotations."""
    payoffs_gbm = np.sort(result["payoffs_gbm"])
    payoffs_heston = np.sort(result["payoffs_heston"])
    n = len(payoffs_gbm)
    cdf = np.arange(1, n + 1) / n

    fig = go.Figure()
    # Sample every 100th point for performance
    step = max(1, n // 2000)
    fig.add_trace(go.Scatter(
        x=payoffs_gbm[::step], y=cdf[::step],
        name="GBM", line=dict(color="blue", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=payoffs_heston[::step], y=cdf[::step],
        name="Heston", line=dict(color="red", width=2),
    ))

    # ES markers
    fig.add_hline(y=0.05, line_dash="dash", line_color="gray",
                  annotation_text="5th percentile")
    fig.add_vline(x=result["heston_es5"], line_dash="dot", line_color="red",
                  annotation_text=f"Heston ES5% ${result['heston_es5']:.0f}")
    fig.add_vline(x=result["gbm_es5"], line_dash="dot", line_color="blue",
                  annotation_text=f"GBM ES5% ${result['gbm_es5']:.0f}")

    fig.update_layout(
        title="Tail Risk: CDF with Expected Shortfall",
        xaxis_title="Discounted Payoff ($)",
        yaxis_title="Cumulative Probability",
        template="plotly_white",
        height=400,
    )
    return fig


def make_scp_waterfall(result: dict) -> go.Figure:
    """Waterfall chart showing value decomposition."""
    labels = ["Par Value", "Coupon Value", "Autocall Optionality",
              "KI Put Sold", "Issuer Credit", "Hidden Margin (SCP)"]

    heston_fv = result["heston_fair_value"]
    par = result["par"]
    scp_dollar = par - heston_fv

    # Approximate decomposition
    coupon_value = result["result_heston"].avg_coupons_paid * result["coupon_rate"] * par
    autocall_cost = (1 - result["result_heston"].autocall_prob) * par * 0.02
    ki_put_cost = result["result_heston"].ki_breach_prob * par * 0.15
    credit_cost = par * 0.005 * result["maturity"]

    values = [par, coupon_value, -autocall_cost, -ki_put_cost, -credit_cost, -scp_dollar]

    fig = go.Figure(go.Waterfall(
        name="Value", orientation="v",
        measure=["absolute", "relative", "relative", "relative", "relative", "relative"],
        x=labels, y=values,
        connector={"line": {"color": "rgb(63,63,63)"}},
        decreasing={"marker": {"color": "#ef553b"}},
        increasing={"marker": {"color": "#00cc96"}},
        totals={"marker": {"color": "#636efa"}},
        text=[f"${v:+.0f}" for v in values],
        textposition="outside",
    ))

    fig.update_layout(
        title="Note Value Decomposition (Heston Model)",
        yaxis_title="Value ($)",
        template="plotly_white",
        height=400,
        showlegend=False,
    )
    return fig


def make_comparison_radar(result: dict) -> go.Figure:
    """Radar chart comparing GBM vs Heston metrics."""
    categories = ["Fair Value", "Autocall Prob", "KI Breach Prob",
                   "ES 5%", "ES 1%", "Avg Coupons"]

    r_gbm = [
        result["gbm_fair_value"] / result["par"] * 100,
        result["gbm_autocall_prob"] * 100,
        (1 - result["gbm_ki_breach_prob"]) * 100,
        result["gbm_es5"] / result["par"] * 100,
        result["gbm_es1"] / result["par"] * 100,
        result["result_gbm"].avg_coupons_paid / result["n_obs"] * 100,
    ]
    r_heston = [
        result["heston_fair_value"] / result["par"] * 100,
        result["heston_autocall_prob"] * 100,
        (1 - result["heston_ki_breach_prob"]) * 100,
        result["heston_es5"] / result["par"] * 100,
        result["heston_es1"] / result["par"] * 100,
        result["result_heston"].avg_coupons_paid / result["n_obs"] * 100,
    ]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=r_gbm, theta=categories, fill="toself",
        name="GBM", line_color="blue", opacity=0.6,
    ))
    fig.add_trace(go.Scatterpolar(
        r=r_heston, theta=categories, fill="toself",
        name="Heston", line_color="red", opacity=0.6,
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True)),
        title="Model Comparison Radar",
        template="plotly_white",
        height=400,
    )
    return fig


def generate_report_pdf(results: list) -> bytes:
    """Generate a PDF report from evaluated notes using matplotlib."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        # ── Title Page ──
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis("off")
        ax.text(0.5, 0.7, "Autocall Trap", fontsize=32, fontweight="bold",
                ha="center", va="center", transform=ax.transAxes)
        ax.text(0.5, 0.62, "Note Evaluation Report", fontsize=18,
                ha="center", va="center", transform=ax.transAxes, color="#555")
        ax.text(0.5, 0.55, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                fontsize=12, ha="center", va="center", transform=ax.transAxes, color="#888")
        ax.text(0.5, 0.48, f"{len(results)} Notes Evaluated",
                fontsize=14, ha="center", va="center", transform=ax.transAxes, color="#333")
        avg_scp = np.mean([r["scp"] for r in results])
        ax.text(0.5, 0.40, f"Average SCP: {avg_scp:.2f}%",
                fontsize=16, ha="center", va="center", transform=ax.transAxes,
                color="#d63031", fontweight="bold")
        pdf.savefig(fig)
        plt.close()

        # ── Summary Table ──
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis("off")
        ax.set_title("Summary of All Evaluated Notes", fontsize=16, fontweight="bold", pad=20)

        headers = ["Note ID", "Underlying", "Maturity", "Coupon", "KI",
                    "GBM FV", "Heston FV", "SCP %", "ES Gap"]
        rows = []
        for r in results:
            rows.append([
                r["note_id"][:18],
                r["underlying"],
                f"{r['maturity']:.1f}y",
                f"{r['coupon_rate']*100:.2f}%",
                f"{r['ki_barrier']*100:.0f}%",
                f"${r['gbm_fair_value']:.0f}",
                f"${r['heston_fair_value']:.0f}",
                f"{r['scp']:.2f}%",
                f"${r['es_gap']:.0f}",
            ])

        colors = []
        for r in results:
            if r["scp"] > 5:
                colors.append(["#ffe0e0"] * len(headers))
            elif r["scp"] > 2:
                colors.append(["#fff3e0"] * len(headers))
            else:
                colors.append(["#e0ffe0"] * len(headers))

        table = ax.table(cellText=rows, colLabels=headers, loc="center",
                         cellLoc="center", cellColours=colors if colors else None)
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1.0, 1.4)
        for (row, col), cell in table.get_celld().items():
            if row == 0:
                cell.set_facecolor("#2c3e50")
                cell.set_text_props(color="white", fontweight="bold")
        pdf.savefig(fig)
        plt.close()

        # ── Individual Note Pages ──
        for r in results:
            fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
            fig.suptitle(f"Note: {r['note_id']} ({r['underlying']})",
                         fontsize=14, fontweight="bold")

            # Payoff histogram
            ax = axes[0, 0]
            ax.hist(r["payoffs_gbm"], bins=80, alpha=0.5, label="GBM",
                    color="blue", density=True)
            ax.hist(r["payoffs_heston"], bins=80, alpha=0.5, label="Heston",
                    color="red", density=True)
            ax.axvline(r["par"], color="green", linestyle="--", label="Par")
            ax.set_title("Payoff Distribution")
            ax.set_xlabel("Payoff ($)")
            ax.legend(fontsize=7)

            # CDF
            ax = axes[0, 1]
            sorted_gbm = np.sort(r["payoffs_gbm"])
            sorted_heston = np.sort(r["payoffs_heston"])
            cdf = np.arange(1, len(sorted_gbm) + 1) / len(sorted_gbm)
            step = max(1, len(cdf) // 500)
            ax.plot(sorted_gbm[::step], cdf[::step], label="GBM", color="blue")
            ax.plot(sorted_heston[::step], cdf[::step], label="Heston", color="red")
            ax.axhline(0.05, color="gray", linestyle="--", alpha=0.5)
            ax.set_title("Tail Risk CDF")
            ax.set_xlabel("Payoff ($)")
            ax.legend(fontsize=7)

            # Metrics table
            ax = axes[1, 0]
            ax.axis("off")
            metrics = [
                ["Metric", "GBM", "Heston"],
                ["Fair Value", f"${r['gbm_fair_value']:.2f}", f"${r['heston_fair_value']:.2f}"],
                ["Autocall Prob", f"{r['gbm_autocall_prob']:.1%}", f"{r['heston_autocall_prob']:.1%}"],
                ["KI Breach Prob", f"{r['gbm_ki_breach_prob']:.1%}", f"{r['heston_ki_breach_prob']:.1%}"],
                ["ES 5%", f"${r['gbm_es5']:.2f}", f"${r['heston_es5']:.2f}"],
                ["ES 1%", f"${r['gbm_es1']:.2f}", f"${r['heston_es1']:.2f}"],
                ["SCP", "", f"{r['scp']:.2f}%"],
                ["ES Gap", "", f"${r['es_gap']:.2f}"],
            ]
            t = ax.table(cellText=metrics[1:], colLabels=metrics[0],
                         loc="center", cellLoc="center")
            t.auto_set_font_size(False)
            t.set_fontsize(9)
            t.scale(1.0, 1.5)
            for (row, col), cell in t.get_celld().items():
                if row == 0:
                    cell.set_facecolor("#2c3e50")
                    cell.set_text_props(color="white", fontweight="bold")

            # SCP gauge
            ax = axes[1, 1]
            scp_val = r["scp"]
            color = "#27ae60" if scp_val < 2 else "#f39c12" if scp_val < 5 else "#e74c3c"
            ax.barh(0, scp_val, color=color, height=0.4, edgecolor="black")
            ax.set_xlim(0, max(15, scp_val + 2))
            ax.set_yticks([])
            ax.axvline(2, color="green", linestyle="--", alpha=0.5, label="Fair (<2%)")
            ax.axvline(5, color="orange", linestyle="--", alpha=0.5, label="Caution (5%)")
            ax.axvline(10, color="red", linestyle="--", alpha=0.5, label="Danger (>10%)")
            ax.set_title(f"SCP: {scp_val:.2f}%", fontsize=13, fontweight="bold")
            ax.legend(fontsize=7, loc="lower right")

            plt.tight_layout()
            pdf.savefig(fig)
            plt.close()

    buf.seek(0)
    return buf.read()


# ═══════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🎯 Autocall Trap")
    st.markdown("---")

    mode = st.radio(
        "Mode",
        ["📝 Manual Entry", "📁 Upload CSV", "📂 Load Existing Data"],
        index=0,
    )

    st.markdown("---")
    st.markdown("### ⚙️ Simulation Settings")
    n_paths = st.select_slider(
        "Monte Carlo Paths",
        options=[10_000, 50_000, 100_000, 250_000, 500_000],
        value=100_000,
    )
    seed = st.number_input("Random Seed", value=42, min_value=1)

    st.markdown("---")
    st.markdown(f"**Notes evaluated:** {len(st.session_state.evaluated_notes)}")

    if st.button("🗑️ Clear All Notes", use_container_width=True):
        st.session_state.evaluated_notes = []
        st.session_state.backtest_result = None
        st.session_state.historical_notes = []
        st.rerun()


# ═══════════════════════════════════════════════════════════════════
# MAIN CONTENT
# ═══════════════════════════════════════════════════════════════════

st.markdown('<p class="main-header">The Autocall Trap</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Evaluate autocallable barrier notes with GBM & Heston stochastic volatility pricing</p>', unsafe_allow_html=True)

tabs = st.tabs(["📝 Input & Evaluate", "📊 Results Dashboard", "🔬 Backtest", "📄 Report"])

# ═══════════════════════════════════════════════════════════════════
# TAB 1: INPUT & EVALUATE
# ═══════════════════════════════════════════════════════════════════
with tabs[0]:
    if mode == "📝 Manual Entry":
        st.markdown("### Enter Note Parameters")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Identifiers**")
            note_id = st.text_input("Note ID", value="MY-NOTE-001")
            issuer = st.text_input("Issuer", value="JPMorgan")
            underlying = st.text_input("Underlying Ticker", value="AAPL")

            st.markdown("**Structure**")
            S0 = st.number_input("Initial Price (S0)", value=150.0, min_value=0.01, format="%.2f")
            par = st.number_input("Par Value ($)", value=1000.0, min_value=100.0)
            maturity = st.number_input("Maturity (years)", value=2.0, min_value=0.25, max_value=10.0, step=0.25)
            n_obs = st.number_input("Observation Dates", value=8, min_value=1, max_value=60)

        with col2:
            st.markdown("**Coupon & Barriers**")
            coupon_rate = st.number_input("Coupon Rate (per period)", value=0.025, min_value=0.0, max_value=0.2, format="%.4f")
            autocall_trigger = st.number_input("Autocall Trigger (% of S0)", value=100.0, min_value=50.0, max_value=150.0) / 100
            coupon_barrier = st.number_input("Coupon Barrier (% of S0)", value=70.0, min_value=20.0, max_value=100.0) / 100
            ki_barrier = st.number_input("Knock-In Barrier (% of S0)", value=60.0, min_value=20.0, max_value=100.0) / 100
            memory = st.checkbox("Memory Coupon", value=True)
            first_autocall_obs = st.number_input("First Autocall Obs", value=1, min_value=1, max_value=20)

        with col3:
            st.markdown("**Market Data**")
            risk_free_rate = st.number_input("Risk-Free Rate", value=0.045, min_value=0.0, max_value=0.2, format="%.4f")
            atm_iv = st.number_input("ATM Implied Vol", value=0.30, min_value=0.05, max_value=2.0, format="%.4f")
            div_yield = st.number_input("Dividend Yield", value=0.01, min_value=0.0, max_value=0.15, format="%.4f")
            issuer_ev = st.number_input("Issuer Estimated Value ($)", value=970.0, min_value=0.0,
                                        help="SEC-disclosed estimated value per $1000 par")

        st.markdown("---")
        if st.button("🚀 Evaluate Note", type="primary", use_container_width=True):
            with st.spinner(f"Running {n_paths:,} Monte Carlo paths (GBM + Heston)..."):
                result = evaluate_note(
                    S0=S0, par=par, maturity=maturity, n_obs=n_obs,
                    coupon_rate=coupon_rate, autocall_trigger=autocall_trigger,
                    coupon_barrier=coupon_barrier, ki_barrier=ki_barrier,
                    memory=memory, first_autocall_obs=first_autocall_obs,
                    risk_free_rate=risk_free_rate, atm_iv=atm_iv, div_yield=div_yield,
                    n_paths=n_paths, seed=seed,
                    note_id=note_id, issuer=issuer, underlying=underlying,
                    issuer_estimated_value=issuer_ev,
                )
                st.session_state.evaluated_notes.append(result)
            st.success(f"Note {note_id} evaluated! SCP = {result['scp']:.2f}%")
            st.rerun()

    elif mode == "📁 Upload CSV":
        st.markdown("### Upload Notes CSV")
        st.markdown("""
        CSV must have columns: `note_id, issuer, underlying, S0, par, maturity, n_obs,
        coupon_rate, autocall_trigger, coupon_barrier, ki_barrier, memory, first_autocall_obs,
        risk_free_rate, atm_iv, div_yield, issuer_estimated_value`
        """)

        uploaded = st.file_uploader("Choose CSV file", type=["csv"])

        if uploaded:
            df = pd.read_csv(uploaded)
            st.dataframe(df, use_container_width=True)

            st.markdown(f"**{len(df)} notes found in CSV**")

            if st.button("🚀 Evaluate All Notes", type="primary", use_container_width=True):
                progress = st.progress(0, text="Evaluating notes...")
                for idx, row in df.iterrows():
                    progress.progress(
                        (idx + 1) / len(df),
                        text=f"Evaluating {row.get('note_id', idx+1)} ({idx+1}/{len(df)})..."
                    )
                    result = evaluate_note(
                        S0=row["S0"],
                        par=row.get("par", 1000.0),
                        maturity=row["maturity"],
                        n_obs=int(row["n_obs"]),
                        coupon_rate=row["coupon_rate"],
                        autocall_trigger=row.get("autocall_trigger", 1.0),
                        coupon_barrier=row["coupon_barrier"],
                        ki_barrier=row["ki_barrier"],
                        memory=bool(row.get("memory", True)),
                        first_autocall_obs=int(row.get("first_autocall_obs", 1)),
                        risk_free_rate=row.get("risk_free_rate", 0.045),
                        atm_iv=row.get("atm_iv", 0.30),
                        div_yield=row.get("div_yield", 0.01),
                        n_paths=n_paths, seed=seed + idx,
                        note_id=row.get("note_id", f"CSV-{idx+1}"),
                        issuer=row.get("issuer", ""),
                        underlying=row.get("underlying", ""),
                        issuer_estimated_value=_safe_float(row.get("issuer_estimated_value")),
                        issue_date=_safe_str(row.get("issue_date"), "2023-01-01"),
                        outcome=_safe_str(row.get("outcome")),
                        realized_payoff=_safe_float(row.get("realized_payoff")),
                        realized_return=_safe_float(row.get("realized_return")),
                        holding_period_years=_safe_float(row.get("holding_period_years")),
                    )
                    st.session_state.evaluated_notes.append(result)

                progress.empty()
                st.success(f"All {len(df)} notes evaluated!")
                st.rerun()

    elif mode == "📂 Load Existing Data":
        st.markdown("### Load Existing Enriched Data")
        data_path = os.path.join(ROOT, "data", "notes_enriched.csv")

        if os.path.exists(data_path):
            st.info(f"Found: `{data_path}`")
            df = pd.read_csv(data_path)
            st.dataframe(df, use_container_width=True)

            if st.button("🚀 Evaluate All 20 Notes", type="primary", use_container_width=True):
                progress = st.progress(0, text="Loading & evaluating...")
                for idx, row in df.iterrows():
                    progress.progress(
                        (idx + 1) / len(df),
                        text=f"Evaluating {row.get('note_id', idx+1)} ({idx+1}/{len(df)})..."
                    )
                    result = evaluate_note(
                        S0=row["S0"],
                        par=row.get("par", 1000.0),
                        maturity=row["maturity"],
                        n_obs=int(row["n_obs"]),
                        coupon_rate=row["coupon_rate"],
                        autocall_trigger=row.get("autocall_trigger", 1.0),
                        coupon_barrier=row.get("coupon_barrier", 0.7),
                        ki_barrier=row.get("ki_barrier", 0.6),
                        memory=bool(row.get("memory", True)),
                        first_autocall_obs=int(row.get("first_autocall_obs", 1)),
                        risk_free_rate=row.get("risk_free_rate", 0.045),
                        atm_iv=row.get("atm_iv", 0.30),
                        div_yield=row.get("div_yield", 0.01),
                        n_paths=n_paths, seed=seed + idx,
                        note_id=row.get("note_id", f"NOTE-{idx+1}"),
                        issuer=row.get("issuer", ""),
                        underlying=row.get("underlying", ""),
                        issuer_estimated_value=_safe_float(row.get("issuer_estimated_value")),
                        issue_date=_safe_str(row.get("issue_date"), "2023-01-01"),
                        outcome=_safe_str(row.get("outcome")),
                        realized_payoff=_safe_float(row.get("realized_payoff")),
                        realized_return=_safe_float(row.get("realized_return")),
                        holding_period_years=_safe_float(row.get("holding_period_years")),
                    )
                    st.session_state.evaluated_notes.append(result)

                progress.empty()
                st.success(f"All {len(df)} notes evaluated!")
                st.rerun()
        else:
            st.warning("No enriched data found. Run the data pipeline first or upload a CSV.")


# ═══════════════════════════════════════════════════════════════════
# TAB 2: RESULTS DASHBOARD
# ═══════════════════════════════════════════════════════════════════
with tabs[1]:
    notes = st.session_state.evaluated_notes

    if not notes:
        st.info("No notes evaluated yet. Go to the Input tab to add notes.")
    else:
        # ── Summary Metrics Row ──
        avg_scp = np.mean([n["scp"] for n in notes])
        avg_es_gap = np.mean([n["es_gap"] for n in notes])
        max_scp = max(n["scp"] for n in notes)
        min_scp = min(n["scp"] for n in notes)

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Notes Evaluated", len(notes))
        c2.metric("Avg SCP", f"{avg_scp:.2f}%")
        c3.metric("Avg ES Gap", f"${avg_es_gap:.0f}")
        c4.metric("Max SCP", f"{max_scp:.2f}%", delta=f"{notes[[n['scp'] for n in notes].index(max_scp)]['underlying']}")
        c5.metric("Min SCP", f"{min_scp:.2f}%", delta=f"{notes[[n['scp'] for n in notes].index(min_scp)]['underlying']}")

        st.markdown("---")

        # ── Note Selector ──
        note_ids = [n["note_id"] for n in notes]
        selected_id = st.selectbox("Select a note to inspect:", note_ids)
        selected = notes[note_ids.index(selected_id)]

        # ── Quick Metrics for Selected Note ──
        mc1, mc2, mc3, mc4, mc5, mc6 = st.columns(6)
        mc1.metric("GBM Fair Value", f"${selected['gbm_fair_value']:.2f}")
        mc2.metric("Heston Fair Value", f"${selected['heston_fair_value']:.2f}")
        mc3.metric("SCP", f"{selected['scp']:.2f}%",
                    delta="Overpriced" if selected['scp'] > 3 else "Fair",
                    delta_color="inverse")
        mc4.metric("Autocall Prob (H)", f"{selected['heston_autocall_prob']:.1%}")
        mc5.metric("KI Breach Prob (H)", f"{selected['heston_ki_breach_prob']:.1%}")
        mc6.metric("ES Gap (5%)", f"${selected['es_gap']:.0f}")

        # ── Plots ──
        col_left, col_right = st.columns(2)
        with col_left:
            st.plotly_chart(make_payoff_distribution_plot(selected), use_container_width=True)
        with col_right:
            st.plotly_chart(make_tail_risk_cdf(selected), use_container_width=True)

        col_left2, col_right2 = st.columns(2)
        with col_left2:
            st.plotly_chart(make_scp_waterfall(selected), use_container_width=True)
        with col_right2:
            st.plotly_chart(make_comparison_radar(selected), use_container_width=True)

        st.markdown("---")

        # ── All Notes Comparison ──
        st.markdown("### All Notes Comparison")

        summary_df = pd.DataFrame([{
            "Note ID": n["note_id"],
            "Underlying": n["underlying"],
            "Issuer": n["issuer"],
            "Maturity": f"{n['maturity']:.1f}y",
            "Coupon": f"{n['coupon_rate']*100:.2f}%",
            "KI Barrier": f"{n['ki_barrier']*100:.0f}%",
            "ATM IV": f"{n['atm_iv']*100:.1f}%",
            "GBM FV": f"${n['gbm_fair_value']:.0f}",
            "Heston FV": f"${n['heston_fair_value']:.0f}",
            "SCP %": round(n["scp"], 2),
            "ES Gap": f"${n['es_gap']:.0f}",
        } for n in notes])

        st.dataframe(
            summary_df.style.background_gradient(subset=["SCP %"], cmap="RdYlGn_r"),
            use_container_width=True,
            hide_index=True,
        )

        # SCP bar chart
        fig_scp = px.bar(
            summary_df, x="Note ID", y="SCP %", color="SCP %",
            color_continuous_scale="RdYlGn_r",
            title="SCP by Note (Higher = More Overpriced for Investor)",
        )
        fig_scp.update_layout(template="plotly_white", height=400)
        st.plotly_chart(fig_scp, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════
# TAB 3: BACKTEST
# ═══════════════════════════════════════════════════════════════════
with tabs[2]:
    notes = st.session_state.evaluated_notes

    if len(notes) < 5:
        st.info(f"Need at least 5 notes for backtesting. Currently have {len(notes)}.")
    else:
        st.markdown("### SCP Factor Backtest")
        st.markdown("Sort notes into quintiles by SCP. Test if low-SCP notes outperform high-SCP notes.")

        if st.button("🔬 Run Backtest", type="primary", use_container_width=True):
            with st.spinner("Running backtest with Fama-MacBeth regressions..."):
                # Convert evaluated notes to HistoricalNote format
                hist_notes = []
                for n in notes:
                    hn = HistoricalNote(
                        note_id=n["note_id"],
                        issuer=n["issuer"],
                        underlying=n["underlying"],
                        issue_date=n.get("issue_date", "2023-01-01"),
                        S0=n["S0"], par=n["par"],
                        maturity=n["maturity"], n_obs=n["n_obs"],
                        coupon_rate=n["coupon_rate"],
                        autocall_trigger=n["autocall_trigger"],
                        coupon_barrier=n["coupon_barrier"],
                        ki_barrier=n["ki_barrier"],
                        memory=n["memory"],
                        first_autocall_obs=n["first_autocall_obs"],
                        risk_free_rate=n["risk_free_rate"],
                        atm_iv=n["atm_iv"],
                        div_yield=n["div_yield"],
                        issuer_estimated_value=n.get("issuer_estimated_value"),
                        # Realized outcome data — critical for backtest!
                        outcome=n.get("outcome"),
                        realized_payoff=n.get("realized_payoff"),
                        realized_return=n.get("realized_return"),
                        holding_period_years=n.get("holding_period_years"),
                        # Pre-computed pricing (avoid redundant re-pricing)
                        gbm_fair_value=n["gbm_fair_value"],
                        heston_fair_value=n["heston_fair_value"],
                        scp=n["scp"],
                        gbm_es5=n["gbm_es5"],
                        heston_es5=n["heston_es5"],
                        es_gap=n["es_gap"],
                    )
                    hist_notes.append(hn)

                # Run backtest
                bt = run_backtest(hist_notes, n_paths=n_paths, seed_base=seed, verbose=False)
                st.session_state.backtest_result = bt

            st.success("Backtest complete!")

        bt = st.session_state.backtest_result
        if bt:
            # Summary metrics
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Avg SCP", f"{bt.avg_scp:.2f}%")
            c2.metric("Median SCP", f"{bt.median_scp:.2f}%")
            c3.metric("FM Beta", f"{bt.fm_beta:.4f}" if bt.fm_beta else "N/A")
            c4.metric("FM p-value (NW)", f"{bt.fm_p_value_nw:.3f}" if bt.fm_p_value_nw else "N/A")

            st.markdown("---")

            # Quintile table
            st.markdown("### Quintile Performance")
            q_data = []
            for q in sorted(bt.quintile_avg_scp.keys()):
                q_data.append({
                    "Quintile": f"Q{q}",
                    "Avg SCP %": round(bt.quintile_avg_scp.get(q, 0), 2),
                    "Avg Return %": round(bt.quintile_avg_return.get(q, 0) * 100, 2),
                    "Autocall Rate %": round(bt.quintile_autocall_rate.get(q, 0) * 100, 1),
                    "KI Breach Rate %": round(bt.quintile_ki_breach_rate.get(q, 0) * 100, 1),
                    "Count": bt.quintile_count.get(q, 0),
                })
            q_df = pd.DataFrame(q_data)
            st.dataframe(q_df, use_container_width=True, hide_index=True)

            # Quintile chart
            fig_q = make_subplots(specs=[[{"secondary_y": True}]])
            fig_q.add_trace(go.Bar(
                x=q_df["Quintile"], y=q_df["Avg SCP %"],
                name="Avg SCP %", marker_color="#636efa",
            ))
            fig_q.add_trace(go.Scatter(
                x=q_df["Quintile"], y=q_df["Avg Return %"],
                name="Avg Return %", line=dict(color="red", width=3),
                mode="lines+markers",
            ), secondary_y=True)
            fig_q.update_layout(
                title="Quintile Sort: SCP vs Realized Returns",
                template="plotly_white", height=400,
            )
            fig_q.update_yaxes(title_text="Avg SCP %", secondary_y=False)
            fig_q.update_yaxes(title_text="Avg Return %", secondary_y=True)
            st.plotly_chart(fig_q, use_container_width=True)

            # Strategy results
            st.markdown("### Strategy Results")
            strat_data = {
                "Strategy": [
                    "Long-Only Q1 vs S&P 500",
                    "Synthetic Short (Q5 margin)",
                    "Fama-MacBeth SCP beta",
                ],
                "Result": [
                    f"{bt.q1_vs_benchmark:.2f}%" if bt.q1_vs_benchmark else "N/A",
                    f"{bt.q5_avg_scp_margin:.2f}%" if bt.q5_avg_scp_margin else "N/A",
                    f"beta={bt.fm_beta:.4f}, p={bt.fm_p_value_nw:.3f}" if bt.fm_beta else "N/A",
                ],
                "Interpretation": [
                    "Q1 outperformance vs benchmark" if bt.q1_vs_benchmark and bt.q1_vs_benchmark > 0 else "Q1 underperformed",
                    "Capturable via options replication",
                    "Significant" if bt.fm_p_value_nw and bt.fm_p_value_nw < 0.05 else "Not significant (need more notes)",
                ],
            }
            st.dataframe(pd.DataFrame(strat_data), use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════
# TAB 4: REPORT
# ═══════════════════════════════════════════════════════════════════
with tabs[3]:
    notes = st.session_state.evaluated_notes

    if not notes:
        st.info("Evaluate some notes first, then come back to generate a report.")
    else:
        st.markdown("### Generate Report")
        st.markdown(f"Report will include **{len(notes)} evaluated notes** with full pricing analysis.")

        report_format = st.radio("Format", ["PDF Report", "CSV Export"], horizontal=True)

        if report_format == "CSV Export":
            if st.button("📥 Generate CSV", type="primary", use_container_width=True):
                export_rows = []
                for n in notes:
                    export_rows.append({
                        "note_id": n["note_id"],
                        "issuer": n["issuer"],
                        "underlying": n["underlying"],
                        "S0": n["S0"],
                        "par": n["par"],
                        "maturity": n["maturity"],
                        "n_obs": n["n_obs"],
                        "coupon_rate": n["coupon_rate"],
                        "autocall_trigger": n["autocall_trigger"],
                        "coupon_barrier": n["coupon_barrier"],
                        "ki_barrier": n["ki_barrier"],
                        "memory": n["memory"],
                        "risk_free_rate": n["risk_free_rate"],
                        "atm_iv": n["atm_iv"],
                        "div_yield": n["div_yield"],
                        "issuer_estimated_value": n.get("issuer_estimated_value"),
                        "gbm_fair_value": round(n["gbm_fair_value"], 2),
                        "heston_fair_value": round(n["heston_fair_value"], 2),
                        "scp_pct": round(n["scp"], 4),
                        "gbm_es5": round(n["gbm_es5"], 2),
                        "heston_es5": round(n["heston_es5"], 2),
                        "es_gap": round(n["es_gap"], 2),
                        "gbm_autocall_prob": round(n["gbm_autocall_prob"], 4),
                        "heston_autocall_prob": round(n["heston_autocall_prob"], 4),
                        "gbm_ki_breach_prob": round(n["gbm_ki_breach_prob"], 4),
                        "heston_ki_breach_prob": round(n["heston_ki_breach_prob"], 4),
                        "heston_v0": round(n["heston_v0"], 4),
                        "heston_kappa": n["heston_kappa"],
                        "heston_theta": round(n["heston_theta"], 4),
                        "heston_xi": round(n["heston_xi"], 4),
                        "heston_rho": n["heston_rho"],
                    })
                csv_df = pd.DataFrame(export_rows)
                csv_buf = csv_df.to_csv(index=False)
                st.download_button(
                    "📥 Download CSV",
                    csv_buf,
                    file_name=f"autocall_evaluation_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

        else:  # PDF
            if st.button("📄 Generate PDF Report", type="primary", use_container_width=True):
                with st.spinner("Generating PDF report..."):
                    pdf_bytes = generate_report_pdf(notes)
                st.download_button(
                    "📥 Download PDF Report",
                    pdf_bytes,
                    file_name=f"autocall_report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )

        # ── Quick preview ──
        st.markdown("---")
        st.markdown("### Quick Preview")
        preview_df = pd.DataFrame([{
            "Note": n["note_id"],
            "Underlying": n["underlying"],
            "SCP %": round(n["scp"], 2),
            "Heston FV": f"${n['heston_fair_value']:.0f}",
            "ES Gap": f"${n['es_gap']:.0f}",
            "Verdict": "🟢 Fair" if n["scp"] < 2 else "🟡 Caution" if n["scp"] < 5 else "🔴 Overpriced",
        } for n in notes])
        st.dataframe(preview_df, use_container_width=True, hide_index=True)

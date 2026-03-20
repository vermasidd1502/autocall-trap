"""
visualizations.py — Publication-Quality Figure Generation
=========================================================
Generates all figures for the paper and GitHub repo, saved as both
PNG (for README/GitHub) and PDF (for LaTeX report).
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import List, Optional

from .note import AutocallableNote
from .pricer import PricingResult
from .sensitivity import SensitivityResult


# ── Style configuration ───────────────────────────────────────────────

COLORS = {
    'gbm': '#2196F3',
    'heston': '#E53935',
    'accent1': '#7B1FA2',
    'accent2': '#00897B',
    'accent3': '#FF8F00',
    'par': '#212121',
    'grid': '#E0E0E0',
}

def set_style():
    """Apply consistent publication style."""
    plt.rcParams.update({
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'axes.edgecolor': '#666666',
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.color': COLORS['grid'],
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.titleweight': 'bold',
        'axes.labelsize': 10,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.dpi': 150,
    })


# ── Individual figure generators ──────────────────────────────────────

def fig_payoff_distribution(
    payoffs_gbm: np.ndarray,
    payoffs_heston: np.ndarray,
    note: AutocallableNote,
    save_path: str = "figures/fig1_payoff_distribution",
):
    """Fig 1: Payoff distribution histogram comparison."""
    set_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    bins = np.linspace(
        min(payoffs_gbm.min(), payoffs_heston.min()),
        max(payoffs_gbm.max(), payoffs_heston.max()),
        80,
    )
    ax.hist(payoffs_gbm, bins=bins, alpha=0.5, density=True,
            color=COLORS['gbm'], label='GBM (Naive)', edgecolor='white', linewidth=0.3)
    ax.hist(payoffs_heston, bins=bins, alpha=0.5, density=True,
            color=COLORS['heston'], label='Heston (Fair Value)', edgecolor='white', linewidth=0.3)

    fv_gbm = np.mean(payoffs_gbm)
    fv_heston = np.mean(payoffs_heston)
    ax.axvline(fv_gbm, color=COLORS['gbm'], linestyle='--', linewidth=2,
               label=f'GBM Mean: ${fv_gbm:.0f}')
    ax.axvline(fv_heston, color=COLORS['heston'], linestyle='--', linewidth=2,
               label=f'Heston Mean: ${fv_heston:.0f}')
    ax.axvline(note.par, color=COLORS['par'], linestyle=':', linewidth=1.5,
               label=f'Par (${note.par:,.0f})')

    ax.set_xlabel('Discounted Payoff ($)')
    ax.set_ylabel('Density')
    ax.set_title('Payoff Distribution: GBM vs Heston Stochastic Volatility')
    ax.legend(framealpha=0.9)

    plt.tight_layout()
    plt.savefig(f"{save_path}.png", dpi=150, bbox_inches='tight')
    plt.savefig(f"{save_path}.pdf", bbox_inches='tight')
    plt.close()


def fig_tail_risk_cdf(
    payoffs_gbm: np.ndarray,
    payoffs_heston: np.ndarray,
    note: AutocallableNote,
    save_path: str = "figures/fig2_tail_risk_cdf",
):
    """Fig 2: CDF comparison highlighting tail divergence."""
    set_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    sorted_gbm = np.sort(payoffs_gbm)
    sorted_heston = np.sort(payoffs_heston)
    n = len(sorted_gbm)
    cdf = np.arange(1, n + 1) / n

    ax.plot(sorted_gbm, cdf, color=COLORS['gbm'], linewidth=1.5, label='GBM', alpha=0.85)
    ax.plot(sorted_heston, cdf, color=COLORS['heston'], linewidth=1.5, label='Heston', alpha=0.85)

    # Reference lines
    for pct in [0.05, 0.10]:
        ax.axhline(pct, color='gray', linestyle=':', alpha=0.5, linewidth=0.8)
        ax.text(50, pct + 0.01, f'{pct*100:.0f}th pct', fontsize=7, color='gray')

    ax.axvline(note.par, color=COLORS['par'], linestyle=':', linewidth=1, alpha=0.4)

    # Annotate 5th percentile
    p5_gbm = np.percentile(payoffs_gbm, 5)
    p5_heston = np.percentile(payoffs_heston, 5)
    ax.annotate(f'GBM 5th: ${p5_gbm:.0f}', xy=(p5_gbm, 0.05),
                xytext=(p5_gbm + 50, 0.18), fontsize=8, color=COLORS['gbm'],
                arrowprops=dict(arrowstyle='->', color=COLORS['gbm'], lw=1.2))
    ax.annotate(f'Heston 5th: ${p5_heston:.0f}', xy=(p5_heston, 0.05),
                xytext=(p5_heston - 200, 0.25), fontsize=8, color=COLORS['heston'],
                arrowprops=dict(arrowstyle='->', color=COLORS['heston'], lw=1.2))

    # Shade the gap region
    ax.fill_betweenx([0, 0.05], p5_heston, p5_gbm, alpha=0.1, color='red',
                     label=f'Tail gap: ${p5_gbm - p5_heston:.0f}')

    ax.set_xlabel('Discounted Payoff ($)')
    ax.set_ylabel('Cumulative Probability')
    ax.set_title('Tail Risk: CDF Comparison')
    ax.legend(framealpha=0.9)

    plt.tight_layout()
    plt.savefig(f"{save_path}.png", dpi=150, bbox_inches='tight')
    plt.savefig(f"{save_path}.pdf", bbox_inches='tight')
    plt.close()


def fig_autocall_timing(
    autocall_times_gbm: np.ndarray,
    terminated_gbm: np.ndarray,
    autocall_times_heston: np.ndarray,
    terminated_heston: np.ndarray,
    note: AutocallableNote,
    save_path: str = "figures/fig3_autocall_timing",
):
    """Fig 3: Autocall timing distribution comparison."""
    set_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    start = note.first_autocall_obs - 1
    obs_labels = [f'Q{i+1}\n({note.obs_times[i]:.2f}y)' for i in range(start, note.n_obs)]

    gbm_times = autocall_times_gbm[terminated_gbm]
    heston_times = autocall_times_heston[terminated_heston]

    gbm_counts = [np.sum(np.isclose(gbm_times, t)) for t in note.obs_times[start:]]
    heston_counts = [np.sum(np.isclose(heston_times, t)) for t in note.obs_times[start:]]

    # Normalize to percentages
    n_gbm = len(autocall_times_gbm)
    gbm_pct = [c / n_gbm * 100 for c in gbm_counts]
    heston_pct = [c / n_gbm * 100 for c in heston_counts]

    x = np.arange(len(obs_labels))
    width = 0.35
    ax.bar(x - width/2, gbm_pct, width, color=COLORS['gbm'], alpha=0.8,
           label='GBM', edgecolor='white', linewidth=0.5)
    ax.bar(x + width/2, heston_pct, width, color=COLORS['heston'], alpha=0.8,
           label='Heston', edgecolor='white', linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(obs_labels)
    ax.set_ylabel('Paths Autocalled (%)')
    ax.set_title('Autocall Timing Distribution')
    ax.legend(framealpha=0.9)

    plt.tight_layout()
    plt.savefig(f"{save_path}.png", dpi=150, bbox_inches='tight')
    plt.savefig(f"{save_path}.pdf", bbox_inches='tight')
    plt.close()


def fig_vol_of_vol_sensitivity(
    results: List[SensitivityResult],
    save_path: str = "figures/fig4_xi_sensitivity",
):
    """Fig 4: Mispricing gap vs vol-of-vol."""
    set_style()
    fig, ax1 = plt.subplots(figsize=(8, 5))

    xi_vals = [r.param_value for r in results]
    gaps = [r.gap for r in results]
    ki_gaps = [(r.ki_breach_heston - r.ki_breach_gbm) * 100 for r in results]

    ax1.plot(xi_vals, gaps, 'o-', color=COLORS['accent1'], linewidth=2, markersize=8,
             label='Valuation Gap ($)')
    ax1.fill_between(xi_vals, 0, gaps, alpha=0.12, color=COLORS['accent1'])
    ax1.set_xlabel('Vol-of-Vol (ξ)')
    ax1.set_ylabel('Mispricing Gap ($)', color=COLORS['accent1'])
    ax1.tick_params(axis='y', labelcolor=COLORS['accent1'])

    ax2 = ax1.twinx()
    ax2.plot(xi_vals, ki_gaps, 's--', color=COLORS['accent3'], linewidth=1.5,
             markersize=6, label='KI Breach Gap (pp)')
    ax2.set_ylabel('KI Breach Probability Gap (pp)', color=COLORS['accent3'])
    ax2.tick_params(axis='y', labelcolor=COLORS['accent3'])

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', framealpha=0.9)

    ax1.set_title('Hidden Margin Increases with Vol-of-Vol')

    plt.tight_layout()
    plt.savefig(f"{save_path}.png", dpi=150, bbox_inches='tight')
    plt.savefig(f"{save_path}.pdf", bbox_inches='tight')
    plt.close()


def fig_correlation_sensitivity(
    results: List[SensitivityResult],
    base_rho: float = -0.65,
    save_path: str = "figures/fig5_rho_sensitivity",
):
    """Fig 5: Mispricing gap vs spot-vol correlation."""
    set_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    rho_vals = [r.param_value for r in results]
    gaps = [r.gap for r in results]

    ax.plot(rho_vals, gaps, 'o-', color=COLORS['accent2'], linewidth=2, markersize=8)
    ax.fill_between(rho_vals, 0, gaps, alpha=0.12, color=COLORS['accent2'])
    ax.axvline(base_rho, color='red', linestyle=':', alpha=0.6,
               label=f'Base case ρ = {base_rho}')

    ax.set_xlabel('Spot-Vol Correlation (ρ)')
    ax.set_ylabel('Mispricing Gap ($)')
    ax.set_title('Hidden Margin Increases with Negative Spot-Vol Correlation')
    ax.legend(framealpha=0.9)

    plt.tight_layout()
    plt.savefig(f"{save_path}.png", dpi=150, bbox_inches='tight')
    plt.savefig(f"{save_path}.pdf", bbox_inches='tight')
    plt.close()


def fig_ki_barrier_sensitivity(
    results: List[SensitivityResult],
    save_path: str = "figures/fig6_ki_sensitivity",
):
    """Fig 6: Mispricing gap vs knock-in barrier level."""
    set_style()
    fig, ax1 = plt.subplots(figsize=(8, 5))

    ki_vals = [r.param_value * 100 for r in results]
    gaps = [r.gap for r in results]
    ki_probs = [r.ki_breach_heston * 100 for r in results]

    ax1.plot(ki_vals, gaps, 'o-', color=COLORS['accent1'], linewidth=2, markersize=8,
             label='Valuation Gap ($)')
    ax1.fill_between(ki_vals, 0, gaps, alpha=0.12, color=COLORS['accent1'])
    ax1.set_xlabel('Knock-In Barrier (% of Initial)')
    ax1.set_ylabel('Mispricing Gap ($)', color=COLORS['accent1'])

    ax2 = ax1.twinx()
    ax2.plot(ki_vals, ki_probs, 's--', color=COLORS['accent3'], linewidth=1.5,
             markersize=6, label='KI Breach Prob (Heston)')
    ax2.set_ylabel('KI Breach Probability (%)', color=COLORS['accent3'])

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', framealpha=0.9)

    ax1.set_title('Hidden Margin vs Knock-In Barrier Level')

    plt.tight_layout()
    plt.savefig(f"{save_path}.png", dpi=150, bbox_inches='tight')
    plt.savefig(f"{save_path}.pdf", bbox_inches='tight')
    plt.close()


def fig_sample_paths(
    S_gbm: np.ndarray,
    S_heston: np.ndarray,
    note: AutocallableNote,
    n_show: int = 50,
    save_path: str = "figures/fig7_sample_paths",
):
    """Fig 7: Sample simulation paths showing vol clustering in Heston."""
    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    times = np.concatenate([[0], note.obs_times])

    for i in range(min(n_show, S_gbm.shape[0])):
        paths_gbm = np.concatenate([[note.S0], S_gbm[i, :]])
        paths_heston = np.concatenate([[note.S0], S_heston[i, :]])
        axes[0].plot(times, paths_gbm, alpha=0.15, linewidth=0.5, color=COLORS['gbm'])
        axes[1].plot(times, paths_heston, alpha=0.15, linewidth=0.5, color=COLORS['heston'])

    for ax, title in zip(axes, ['GBM Paths', 'Heston Paths']):
        ax.axhline(note.autocall_level, color='green', linestyle='--', alpha=0.7,
                   linewidth=1, label=f'Autocall ({note.autocall_trigger*100:.0f}%)')
        ax.axhline(note.coupon_level, color='orange', linestyle='--', alpha=0.7,
                   linewidth=1, label=f'Coupon Barrier ({note.coupon_barrier*100:.0f}%)')
        ax.axhline(note.ki_level, color='red', linestyle='--', alpha=0.7,
                   linewidth=1, label=f'KI Barrier ({note.ki_barrier*100:.0f}%)')
        ax.set_xlabel('Time (years)')
        ax.set_title(title)
        ax.legend(fontsize=7, loc='lower left', framealpha=0.9)

    axes[0].set_ylabel('Stock Price ($)')
    fig.suptitle('Simulated Stock Paths: GBM vs Heston', fontsize=13, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f"{save_path}.png", dpi=150, bbox_inches='tight')
    plt.savefig(f"{save_path}.pdf", bbox_inches='tight')
    plt.close()


def fig_dashboard(
    res_gbm: PricingResult,
    res_heston: PricingResult,
    note: AutocallableNote,
    save_path: str = "figures/fig8_dashboard",
):
    """Fig 8: Summary dashboard — single-panel key findings."""
    set_style()
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.axis('off')

    margin_gbm = note.par - res_gbm.fair_value
    margin_heston = note.par - res_heston.fair_value
    gap = res_gbm.fair_value - res_heston.fair_value

    text = (
        f"{'THE AUTOCALL TRAP — KEY FINDINGS':^50}\n"
        f"{'━' * 50}\n\n"
        f"{'MODEL COMPARISON':^50}\n"
        f"  GBM Fair Value:          ${res_gbm.fair_value:>10.2f}\n"
        f"  Heston Fair Value:       ${res_heston.fair_value:>10.2f}\n"
        f"  Issue Price (Par):       ${note.par:>10.0f}\n\n"
        f"{'EMBEDDED MARGIN':^50}\n"
        f"  GBM Margin:              {margin_gbm/note.par*100:>9.2f}%\n"
        f"  Heston Margin:           {margin_heston/note.par*100:>9.2f}%\n"
        f"  Hidden Margin:           {gap/note.par*100:>9.2f}%  (${gap:.2f})\n\n"
        f"{'RISK METRICS':^50}\n"
        f"  GBM KI Breach:           {res_gbm.ki_breach_prob*100:>9.1f}%\n"
        f"  Heston KI Breach:        {res_heston.ki_breach_prob*100:>9.1f}%\n\n"
        f"{'TAIL RISK — THE DISASTER SCENARIOS':^50}\n"
        f"  {'':30s} {'GBM':>8s}  {'Heston':>8s}  {'Gap':>8s}\n"
        f"  VaR (5th pct):           ${res_gbm.pct_5:>7.0f}  ${res_heston.pct_5:>7.0f}  ${res_gbm.pct_5-res_heston.pct_5:>7.0f}\n"
        f"  ES/CVaR (5%):            ${res_gbm.es_5:>7.0f}  ${res_heston.es_5:>7.0f}  ${res_gbm.es_5-res_heston.es_5:>7.0f}\n"
        f"  VaR (1st pct):           ${res_gbm.pct_1:>7.0f}  ${res_heston.pct_1:>7.0f}  ${res_gbm.pct_1-res_heston.pct_1:>7.0f}\n"
        f"  ES/CVaR (1%):            ${res_gbm.es_1:>7.0f}  ${res_heston.es_1:>7.0f}  ${res_gbm.es_1-res_heston.es_1:>7.0f}\n"
        f"  Worst Path:              ${res_gbm.max_loss:>7.0f}  ${res_heston.max_loss:>7.0f}\n"
    )

    ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.8', facecolor='#FFFDE7', alpha=0.9,
                      edgecolor='#FBC02D', linewidth=1.5))

    plt.tight_layout()
    plt.savefig(f"{save_path}.png", dpi=150, bbox_inches='tight')
    plt.savefig(f"{save_path}.pdf", bbox_inches='tight')
    plt.close()

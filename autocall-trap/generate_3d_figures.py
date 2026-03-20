"""
generate_3d_figures.py — Publication-quality 3D figures for the revised paper
=============================================================================
Generates:
  fig12: 3D implied volatility surface (strike × maturity × vol)
  fig13: 3D mispricing surface (xi × rho × gap)
  fig14: 3D ES surface (xi × rho × ES gap)
  fig15: Payoff waterfall diagram (3D bar chart by scenario)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.note import AutocallableNote, orcl_hsbc_note
from src.engines import HestonParams, orcl_heston
from src.engines_v2 import simulate_gbm_v2, simulate_heston_qe, orcl_dividends
from src.pricer import price_autocallable
from src.calibration import build_orcl_synthetic_surface


def fig_3d_vol_surface(save_path="figures/fig12_vol_surface_3d"):
    """3D implied volatility surface showing smile and term structure."""
    print("  Generating 3D vol surface...")
    surface = build_orcl_synthetic_surface(S0=140.0, r=0.045, q=0.0114)

    maturities = sorted(set(p.maturity for p in surface))
    moneyness_vals = sorted(set(round(p.strike / 140.0 * 100, 1) for p in surface))

    T_grid, M_grid = np.meshgrid(maturities, moneyness_vals)
    IV_grid = np.zeros_like(T_grid)

    for p in surface:
        m = round(p.strike / 140.0 * 100, 1)
        t = p.maturity
        i = moneyness_vals.index(m)
        j = maturities.index(t)
        IV_grid[i, j] = p.market_iv * 100

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(T_grid, M_grid, IV_grid,
                           cmap=cm.RdYlBu_r, edgecolor='gray',
                           linewidth=0.3, alpha=0.85, antialiased=True)

    ax.set_xlabel('Maturity (years)', fontsize=11, labelpad=12)
    ax.set_ylabel('Moneyness (%)', fontsize=11, labelpad=12)
    ax.set_zlabel('Implied Volatility (%)', fontsize=11, labelpad=10)
    ax.set_title('ORCL Implied Volatility Surface\n'
                 'The "Smile" That GBM Cannot See',
                 fontsize=13, fontweight='bold', pad=20)

    # Mark the key barrier levels
    ax.plot([0.25, 2.0], [70, 70], [20, 20], 'r--', linewidth=2, label='Coupon Barrier (70%)')
    ax.plot([0.25, 2.0], [60, 60], [20, 20], 'r-', linewidth=2.5, label='Knock-In Barrier (60%)')

    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='Implied Vol (%)')
    ax.legend(fontsize=8, loc='upper left')
    ax.view_init(elev=25, azim=-60)

    plt.tight_layout()
    plt.savefig(f"{save_path}.png", dpi=200, bbox_inches='tight')
    plt.savefig(f"{save_path}.pdf", bbox_inches='tight')
    plt.close()


def fig_3d_mispricing_surface(save_path="figures/fig13_mispricing_surface_3d"):
    """3D surface: mispricing gap as function of xi and rho."""
    print("  Generating 3D mispricing surface (this takes ~2 minutes)...")
    note = orcl_hsbc_note()
    divs = orcl_dividends(note.S0)
    base = orcl_heston()
    n_paths = 40_000

    xi_vals = np.array([0.15, 0.25, 0.35, 0.50, 0.65, 0.80, 1.00])
    rho_vals = np.array([-0.90, -0.75, -0.60, -0.45, -0.30, -0.15, 0.0])

    XI, RHO = np.meshgrid(xi_vals, rho_vals)
    GAP = np.zeros_like(XI)

    # GBM baseline (computed once)
    np.random.seed(42)
    S_gbm = simulate_gbm_v2(note.S0, note.r, 0.255, note.maturity,
                              note.n_obs, n_paths, dividends=divs, seed=42)
    res_gbm = price_autocallable(S_gbm, note)

    for i, rho in enumerate(rho_vals):
        for j, xi in enumerate(xi_vals):
            params = HestonParams(v0=base.v0, kappa=base.kappa,
                                  theta=base.theta, xi=xi, rho=rho)
            S_h = simulate_heston_qe(note.S0, note.r, params, note.maturity,
                                      note.n_obs, n_paths, dividends=divs, seed=42)
            res_h = price_autocallable(S_h, note)
            GAP[i, j] = res_gbm.fair_value - res_h.fair_value

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(XI, RHO, GAP, cmap=cm.magma_r,
                           edgecolor='gray', linewidth=0.3,
                           alpha=0.85, antialiased=True)

    # Mark the base case
    base_gap = GAP[np.argmin(np.abs(rho_vals - (-0.65))),
                   np.argmin(np.abs(xi_vals - 0.5))]
    ax.scatter([0.5], [-0.65], [base_gap], color='red', s=100, zorder=10,
               edgecolors='black', linewidths=1.5, label=f'Base Case (${base_gap:.0f})')

    ax.set_xlabel('Vol-of-Vol (ξ)', fontsize=11, labelpad=12)
    ax.set_ylabel('Spot-Vol Correlation (ρ)', fontsize=11, labelpad=12)
    ax.set_zlabel('Mispricing Gap ($)', fontsize=11, labelpad=10)
    ax.set_title('The Autocall Trap: Hidden Margin Surface\n'
                 'How Vol-of-Vol and Leverage Effect Drive Mispricing',
                 fontsize=13, fontweight='bold', pad=20)

    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='Gap ($)')
    ax.legend(fontsize=9, loc='upper left')
    ax.view_init(elev=30, azim=-50)

    plt.tight_layout()
    plt.savefig(f"{save_path}.png", dpi=200, bbox_inches='tight')
    plt.savefig(f"{save_path}.pdf", bbox_inches='tight')
    plt.close()

    return XI, RHO, GAP


def fig_3d_es_surface(save_path="figures/fig14_es_surface_3d"):
    """3D surface: ES(5%) gap as function of xi and rho."""
    print("  Generating 3D Expected Shortfall surface (~2 minutes)...")
    note = orcl_hsbc_note()
    divs = orcl_dividends(note.S0)
    base = orcl_heston()
    n_paths = 40_000

    xi_vals = np.array([0.15, 0.30, 0.50, 0.70, 1.00])
    rho_vals = np.array([-0.90, -0.70, -0.50, -0.30, -0.10])

    XI, RHO = np.meshgrid(xi_vals, rho_vals)
    ES_GAP = np.zeros_like(XI)

    np.random.seed(42)
    S_gbm = simulate_gbm_v2(note.S0, note.r, 0.255, note.maturity,
                              note.n_obs, n_paths, dividends=divs, seed=42)
    res_gbm = price_autocallable(S_gbm, note)

    for i, rho in enumerate(rho_vals):
        for j, xi in enumerate(xi_vals):
            params = HestonParams(v0=base.v0, kappa=base.kappa,
                                  theta=base.theta, xi=xi, rho=rho)
            S_h = simulate_heston_qe(note.S0, note.r, params, note.maturity,
                                      note.n_obs, n_paths, dividends=divs, seed=42)
            res_h = price_autocallable(S_h, note)
            ES_GAP[i, j] = res_gbm.es_5 - res_h.es_5

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(XI, RHO, ES_GAP, cmap=cm.inferno_r,
                           edgecolor='gray', linewidth=0.3,
                           alpha=0.85, antialiased=True)

    ax.set_xlabel('Vol-of-Vol (ξ)', fontsize=11, labelpad=12)
    ax.set_ylabel('Spot-Vol Correlation (ρ)', fontsize=11, labelpad=12)
    ax.set_zlabel('ES(5%) Gap ($)', fontsize=11, labelpad=10)
    ax.set_title('Expected Shortfall Gap Surface\n'
                 'Severity of Wipeout Underestimation Across Regimes',
                 fontsize=13, fontweight='bold', pad=20)

    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='ES Gap ($)')
    ax.view_init(elev=25, azim=-55)

    plt.tight_layout()
    plt.savefig(f"{save_path}.png", dpi=200, bbox_inches='tight')
    plt.savefig(f"{save_path}.pdf", bbox_inches='tight')
    plt.close()


def fig_payoff_scenarios_3d(save_path="figures/fig15_payoff_scenarios_3d"):
    """3D bar chart: payoff across scenarios and percentiles."""
    print("  Generating 3D payoff scenario comparison...")

    scenarios = ['Par\n($1,000)', 'Mean\nPayoff', 'VaR\n(5%)', 'ES/CVaR\n(5%)',
                 'VaR\n(1%)', 'ES/CVaR\n(1%)', 'Worst\nPath']
    gbm_vals = [1000, 1012, 624, 527, 464, 409, 263]
    heston_vals = [1000, 997, 561, 419, 328, 250, 27]

    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(111, projection='3d')

    n = len(scenarios)
    x_gbm = np.arange(n)
    x_heston = np.arange(n)
    y_gbm = np.zeros(n)
    y_heston = np.ones(n) * 0.6

    width = 0.4
    depth = 0.4

    # GBM bars
    bars1 = ax.bar3d(x_gbm, y_gbm, np.zeros(n), width, depth, gbm_vals,
                     color='#2196F3', alpha=0.8, edgecolor='white', linewidth=0.5)

    # Heston bars
    bars2 = ax.bar3d(x_heston, y_heston, np.zeros(n), width, depth, heston_vals,
                     color='#E53935', alpha=0.8, edgecolor='white', linewidth=0.5)

    # Add value labels on top
    for i in range(n):
        ax.text(x_gbm[i] + width/2, y_gbm[i] + depth/2, gbm_vals[i] + 15,
                f'${gbm_vals[i]}', fontsize=7, ha='center', color='#1565C0', fontweight='bold')
        ax.text(x_heston[i] + width/2, y_heston[i] + depth/2, heston_vals[i] + 15,
                f'${heston_vals[i]}', fontsize=7, ha='center', color='#C62828', fontweight='bold')

    ax.set_xticks(np.arange(n) + width/2)
    ax.set_xticklabels(scenarios, fontsize=8)
    ax.set_yticks([depth/2, 0.6 + depth/2])
    ax.set_yticklabels(['GBM', 'Heston'], fontsize=10, fontweight='bold')
    ax.set_zlabel('Payoff ($)', fontsize=11)
    ax.set_title('The Full Picture: From Mean to Catastrophe\n'
                 'How Much the Investor Actually Gets Back',
                 fontsize=13, fontweight='bold', pad=20)

    ax.set_zlim(0, 1100)
    ax.view_init(elev=20, azim=-35)

    plt.tight_layout()
    plt.savefig(f"{save_path}.png", dpi=200, bbox_inches='tight')
    plt.savefig(f"{save_path}.pdf", bbox_inches='tight')
    plt.close()


def fig_note_structure_diagram(save_path="figures/fig00_note_structure"):
    """Clean 2D diagram explaining the note payoff logic visually."""
    print("  Generating note structure diagram...")
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Title
    ax.text(5, 9.5, 'How an Autocallable Note Works', fontsize=16,
            fontweight='bold', ha='center', va='top')

    # Timeline
    ax.annotate('', xy=(9, 7), xytext=(1, 7),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))
    ax.text(0.5, 7, 'Time', fontsize=10, va='center')

    quarters = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7', 'Q8']
    for i, q in enumerate(quarters):
        x = 1.5 + i * 0.9
        ax.plot(x, 7, 'o', color='#1565C0', markersize=8, zorder=5)
        ax.text(x, 6.6, q, fontsize=7, ha='center', color='#1565C0')

    # Autocall zone
    box_props_green = dict(boxstyle='round,pad=0.4', facecolor='#E8F5E9', edgecolor='#4CAF50', linewidth=1.5)
    ax.text(5, 8.5, 'If stock >= 100% of initial: NOTE TERMINATES, investor gets par + coupons',
            fontsize=9, ha='center', bbox=box_props_green, color='#2E7D32')
    ax.annotate('', xy=(5, 7.3), xytext=(5, 8.1),
                arrowprops=dict(arrowstyle='->', color='#4CAF50', lw=1.5))

    # Coupon zone
    box_props_blue = dict(boxstyle='round,pad=0.4', facecolor='#E3F2FD', edgecolor='#2196F3', linewidth=1.5)
    ax.text(5, 5.5, 'If stock >= 70% of initial: Coupon PAID ($26.25) + any unpaid memory coupons',
            fontsize=9, ha='center', bbox=box_props_blue, color='#1565C0')

    # Danger zone
    box_props_orange = dict(boxstyle='round,pad=0.4', facecolor='#FFF3E0', edgecolor='#FF9800', linewidth=1.5)
    ax.text(5, 4.2, 'If stock < 70% of initial: Coupon DEFERRED (saved by memory feature)',
            fontsize=9, ha='center', bbox=box_props_orange, color='#E65100')

    # KI zone
    box_props_red = dict(boxstyle='round,pad=0.4', facecolor='#FFEBEE', edgecolor='#F44336', linewidth=1.5)
    ax.text(5, 2.8, 'If stock EVER < 60% of initial: KNOCK-IN TRIGGERED',
            fontsize=10, ha='center', bbox=box_props_red, color='#C62828', fontweight='bold')

    # Maturity outcomes
    ax.text(5, 1.6, 'At Maturity (Year 2):', fontsize=10, ha='center', fontweight='bold')
    ax.text(2.5, 0.9, 'No knock-in: Get $1,000 back', fontsize=9, ha='center', color='#2E7D32')
    ax.text(7.5, 0.9, 'Knock-in triggered: Get $1,000 × (final price / initial price)',
            fontsize=9, ha='center', color='#C62828')
    ax.text(7.5, 0.3, 'If stock fell 50%, you get only $500 back', fontsize=8,
            ha='center', color='#C62828', style='italic')

    plt.tight_layout()
    plt.savefig(f"{save_path}.png", dpi=200, bbox_inches='tight')
    plt.savefig(f"{save_path}.pdf", bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    os.makedirs("figures", exist_ok=True)

    fig_note_structure_diagram()
    print("  [1/5] Note structure diagram")

    fig_3d_vol_surface()
    print("  [2/5] 3D vol surface")

    fig_3d_mispricing_surface()
    print("  [3/5] 3D mispricing surface")

    fig_3d_es_surface()
    print("  [4/5] 3D ES surface")

    fig_payoff_scenarios_3d()
    print("  [5/5] 3D payoff scenarios")

    print("\nAll 3D figures generated.")

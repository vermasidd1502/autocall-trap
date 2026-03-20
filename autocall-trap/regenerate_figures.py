"""
regenerate_figures.py — Clean, publication-quality figures
Only fig13 (mispricing surface) and fig14 (ES surface) are truly 3D.
Everything else is large, clear 2D with proper labels.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.note import orcl_hsbc_note
from src.engines import HestonParams, orcl_heston
from src.engines_v2 import simulate_gbm_v2, simulate_heston_qe, orcl_dividends
from src.pricer import price_autocallable

plt.rcParams.update({
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'axes.edgecolor': '#444444', 'axes.grid': True, 'grid.alpha': 0.25,
    'font.size': 12, 'axes.titlesize': 14, 'axes.titleweight': 'bold',
    'axes.labelsize': 12, 'xtick.labelsize': 11, 'ytick.labelsize': 11,
    'legend.fontsize': 11, 'figure.dpi': 150,
})

note = orcl_hsbc_note()
divs = orcl_dividends(note.S0)
heston = orcl_heston()
N = 50_000

# Simulations
np.random.seed(42)
S_gbm = simulate_gbm_v2(note.S0, note.r, 0.255, note.maturity, note.n_obs, N, dividends=divs, seed=42)
res_gbm = price_autocallable(S_gbm, note)
S_hes = simulate_heston_qe(note.S0, note.r, heston, note.maturity, note.n_obs, N, dividends=divs, seed=42)
res_hes = price_autocallable(S_hes, note)

os.makedirs("figures", exist_ok=True)

# ── Fig 1: Payoff Distribution ────────────────────────────────────
print("[1] Payoff distribution")
fig, ax = plt.subplots(figsize=(10, 6))
bins = np.linspace(min(res_gbm.payoffs.min(), res_hes.payoffs.min()),
                   max(res_gbm.payoffs.max(), res_hes.payoffs.max()), 80)
ax.hist(res_gbm.payoffs, bins=bins, alpha=0.5, density=True, color='#2196F3', label='GBM (Naive)', edgecolor='white', linewidth=0.3)
ax.hist(res_hes.payoffs, bins=bins, alpha=0.5, density=True, color='#E53935', label='Heston (Fair Value)', edgecolor='white', linewidth=0.3)
ax.axvline(np.mean(res_gbm.payoffs), color='#1565C0', ls='--', lw=2, label=f'GBM Mean: ${np.mean(res_gbm.payoffs):.0f}')
ax.axvline(np.mean(res_hes.payoffs), color='#C62828', ls='--', lw=2, label=f'Heston Mean: ${np.mean(res_hes.payoffs):.0f}')
ax.axvline(1000, color='black', ls=':', lw=1.5, label='Par ($1,000)')
ax.set_xlabel('Discounted Payoff ($)')
ax.set_ylabel('Density')
ax.set_title('Payoff Distribution: GBM vs Heston Stochastic Volatility')
ax.legend(fontsize=10)
plt.tight_layout(); plt.savefig("figures/fig1_payoff_distribution.pdf", bbox_inches='tight'); plt.close()

# ── Fig 2: Tail Risk CDF ─────────────────────────────────────────
print("[2] Tail CDF")
fig, ax = plt.subplots(figsize=(10, 6))
sg = np.sort(res_gbm.payoffs); sh = np.sort(res_hes.payoffs)
cdf = np.arange(1, len(sg)+1)/len(sg)
ax.plot(sg, cdf, color='#2196F3', lw=1.8, label='GBM')
ax.plot(sh, cdf, color='#E53935', lw=1.8, label='Heston')
p5g, p5h = np.percentile(res_gbm.payoffs,5), np.percentile(res_hes.payoffs,5)
ax.axhline(0.05, color='gray', ls=':', alpha=0.6)
ax.fill_betweenx([0,0.05], p5h, p5g, alpha=0.15, color='red', label=f'5th pct gap: ${p5g-p5h:.0f}')
ax.annotate(f'GBM 5th: ${p5g:.0f}', xy=(p5g,0.05), xytext=(p5g+80,0.18), fontsize=10, color='#1565C0',
            arrowprops=dict(arrowstyle='->', color='#1565C0'))
ax.annotate(f'Heston 5th: ${p5h:.0f}', xy=(p5h,0.05), xytext=(p5h-250,0.25), fontsize=10, color='#C62828',
            arrowprops=dict(arrowstyle='->', color='#C62828'))
ax.set_xlabel('Discounted Payoff ($)'); ax.set_ylabel('Cumulative Probability')
ax.set_title('Tail Risk: CDF Comparison')
ax.legend(fontsize=10)
plt.tight_layout(); plt.savefig("figures/fig2_tail_risk_cdf.pdf", bbox_inches='tight'); plt.close()

# ── Fig 3: Autocall Timing ───────────────────────────────────────
print("[3] Autocall timing")
def get_autocall_times(S, note):
    n = S.shape[0]; t_arr = np.full(n, np.inf); done = np.zeros(n, dtype=bool)
    for obs in range(note.n_obs):
        if obs < note.first_autocall_obs - 1: continue
        hit = (S[:,obs] >= note.autocall_level) & ~done
        t_arr = np.where(hit & (t_arr==np.inf), note.obs_times[obs], t_arr)
        done |= hit
    return t_arr, done

tg, dg = get_autocall_times(S_gbm, note)
th, dh = get_autocall_times(S_hes, note)
start = note.first_autocall_obs - 1
obs_labels = [f'Q{i+1} ({note.obs_times[i]:.2f}y)' for i in range(start, note.n_obs)]
gc = [np.sum(np.isclose(tg[dg], t))/N*100 for t in note.obs_times[start:]]
hc = [np.sum(np.isclose(th[dh], t))/N*100 for t in note.obs_times[start:]]

fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(obs_labels)); w=0.35
ax.bar(x-w/2, gc, w, color='#2196F3', alpha=0.8, label='GBM', edgecolor='white')
ax.bar(x+w/2, hc, w, color='#E53935', alpha=0.8, label='Heston', edgecolor='white')
ax.set_xticks(x); ax.set_xticklabels(obs_labels)
ax.set_ylabel('Paths Autocalled (%)'); ax.set_title('Autocall Timing Distribution')
ax.legend()
plt.tight_layout(); plt.savefig("figures/fig3_autocall_timing.pdf", bbox_inches='tight'); plt.close()

# ── Fig 7: Sample Paths ──────────────────────────────────────────
print("[7] Sample paths")
fig, axes = plt.subplots(1,2, figsize=(14,6), sharey=True)
times = np.concatenate([[0], note.obs_times])
for i in range(40):
    axes[0].plot(times, np.concatenate([[note.S0], S_gbm[i,:]]), alpha=0.15, lw=0.5, color='#2196F3')
    axes[1].plot(times, np.concatenate([[note.S0], S_hes[i,:]]), alpha=0.15, lw=0.5, color='#E53935')
for ax, title in zip(axes, ['GBM Paths (Constant Volatility)', 'Heston Paths (Stochastic Volatility)']):
    ax.axhline(note.autocall_level, color='green', ls='--', alpha=0.7, lw=1.2, label=f'Autocall ({note.autocall_trigger*100:.0f}%)')
    ax.axhline(note.coupon_level, color='#FF9800', ls='--', alpha=0.7, lw=1.2, label=f'Coupon Barrier ({note.coupon_barrier*100:.0f}%)')
    ax.axhline(note.ki_level, color='red', ls='--', alpha=0.7, lw=1.5, label=f'Knock-In ({note.ki_barrier*100:.0f}%)')
    ax.set_xlabel('Time (years)'); ax.set_title(title); ax.legend(fontsize=8, loc='lower left')
axes[0].set_ylabel('Stock Price ($)')
fig.suptitle('Simulated Stock Price Paths: Note the Vol Clustering in Heston', fontsize=14, fontweight='bold')
plt.tight_layout(); plt.savefig("figures/fig7_sample_paths.pdf", bbox_inches='tight'); plt.close()

# ── Fig 4,5,6: Sensitivity (2D, clean) ───────────────────────────
print("[4-6] Sensitivity sweeps")
from src.sensitivity import sweep_vol_of_vol, sweep_correlation, sweep_ki_barrier

xi_res = sweep_vol_of_vol(note, heston, 0.255, n_paths=N, seed=42)
rho_res = sweep_correlation(note, heston, 0.255, n_paths=N, seed=42)
ki_res = sweep_ki_barrier(note, heston, 0.255, n_paths=N, seed=42)

# Fig 4
fig, ax = plt.subplots(figsize=(10,6))
xv = [r.param_value for r in xi_res]; gv = [r.gap for r in xi_res]
ax.plot(xv, gv, 'o-', color='#7B1FA2', lw=2.5, ms=9)
ax.fill_between(xv, 0, gv, alpha=0.12, color='#7B1FA2')
ax.set_xlabel('Vol-of-Vol (ξ)'); ax.set_ylabel('Mispricing Gap ($)')
ax.set_title('Hidden Margin Increases with Vol-of-Vol')
plt.tight_layout(); plt.savefig("figures/fig4_xi_sensitivity.pdf", bbox_inches='tight'); plt.close()

# Fig 5
fig, ax = plt.subplots(figsize=(10,6))
rv = [r.param_value for r in rho_res]; gr = [r.gap for r in rho_res]
ax.plot(rv, gr, 'o-', color='#00897B', lw=2.5, ms=9)
ax.fill_between(rv, 0, gr, alpha=0.12, color='#00897B')
ax.axvline(-0.65, color='red', ls=':', alpha=0.6, label='Base case ρ = -0.65')
ax.set_xlabel('Spot-Vol Correlation (ρ)'); ax.set_ylabel('Mispricing Gap ($)')
ax.set_title('Hidden Margin Increases with Negative Spot-Vol Correlation')
ax.legend()
plt.tight_layout(); plt.savefig("figures/fig5_rho_sensitivity.pdf", bbox_inches='tight'); plt.close()

# Fig 6
fig, ax = plt.subplots(figsize=(10,6))
kv = [r.param_value*100 for r in ki_res]; gk = [r.gap for r in ki_res]
ax.plot(kv, gk, 'o-', color='#7B1FA2', lw=2.5, ms=9)
ax.fill_between(kv, 0, gk, alpha=0.12, color='#7B1FA2')
ax.set_xlabel('Knock-In Barrier (% of Initial)'); ax.set_ylabel('Mispricing Gap ($)')
ax.set_title('Hidden Margin vs Knock-In Barrier Level')
plt.tight_layout(); plt.savefig("figures/fig6_ki_sensitivity.pdf", bbox_inches='tight'); plt.close()

# ── Fig 13: 3D Mispricing Surface (genuinely needs 3D) ───────────
print("[13] 3D mispricing surface")
xi_vals = np.array([0.15, 0.30, 0.50, 0.70, 1.00])
rho_vals = np.array([-0.90, -0.70, -0.50, -0.30, -0.10])
XI, RHO = np.meshgrid(xi_vals, rho_vals)
GAP = np.zeros_like(XI)
np.random.seed(42)
S_gbm_base = simulate_gbm_v2(note.S0, note.r, 0.255, note.maturity, note.n_obs, 30000, dividends=divs, seed=42)
r_gbm_base = price_autocallable(S_gbm_base, note)
for i, rho in enumerate(rho_vals):
    for j, xi in enumerate(xi_vals):
        p = HestonParams(v0=heston.v0, kappa=heston.kappa, theta=heston.theta, xi=xi, rho=rho)
        S_h = simulate_heston_qe(note.S0, note.r, p, note.maturity, note.n_obs, 30000, dividends=divs, seed=42)
        r_h = price_autocallable(S_h, note)
        GAP[i,j] = r_gbm_base.fair_value - r_h.fair_value

fig = plt.figure(figsize=(11, 8))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(XI, RHO, GAP, cmap=cm.magma_r, edgecolor='gray', linewidth=0.3, alpha=0.85)
ax.set_xlabel('Vol-of-Vol (ξ)', fontsize=12, labelpad=14)
ax.set_ylabel('Spot-Vol Corr (ρ)', fontsize=12, labelpad=14)
ax.set_zlabel('Mispricing Gap ($)', fontsize=12, labelpad=10)
ax.set_title('Hidden Margin Surface: ξ × ρ → Gap', fontsize=14, fontweight='bold', pad=18)
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='Gap ($)')
ax.view_init(elev=28, azim=-55)
plt.tight_layout(); plt.savefig("figures/fig13_mispricing_surface_3d.pdf", bbox_inches='tight'); plt.close()

# ── Fig 10,11: Stress test (reuse existing) ──────────────────────
print("[10-11] Stress test figures")
from src.stress_test import run_stress_tests, plot_stress_test_comparison, plot_rolling_regimes
stress_results = run_stress_tests(note, dividends=divs, n_paths=N, seed=42)
plot_stress_test_comparison(stress_results)
plot_rolling_regimes()

# ── Fig 8: Dashboard (updated with ES) ───────────────────────────
print("[8] Dashboard")
from src.visualizations import fig_dashboard
fig_dashboard(res_gbm, res_hes, note)

print("\nAll figures regenerated.")

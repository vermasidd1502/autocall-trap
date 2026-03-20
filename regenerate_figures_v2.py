"""
Regenerate all figures with:
- Large black text (font size 13-14 for labels, 15+ for titles)
- High contrast, no light colors for text
- Figures sized 11x7 for readability in PDF
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
from src.sensitivity import sweep_vol_of_vol, sweep_correlation, sweep_ki_barrier

# ── Global style: large, black, readable ──────────────────────────
plt.rcParams.update({
    'figure.facecolor': 'white', 'axes.facecolor': '#FAFAFA',
    'axes.edgecolor': 'black', 'axes.grid': True, 'grid.alpha': 0.2,
    'grid.color': '#CCCCCC',
    'font.size': 13, 'font.family': 'sans-serif',
    'axes.titlesize': 15, 'axes.titleweight': 'bold', 'axes.titlecolor': 'black',
    'axes.labelsize': 13, 'axes.labelcolor': 'black',
    'xtick.labelsize': 12, 'ytick.labelsize': 12,
    'xtick.color': 'black', 'ytick.color': 'black',
    'legend.fontsize': 11, 'legend.framealpha': 0.95,
    'text.color': 'black',
    'figure.dpi': 150,
})

note = orcl_hsbc_note()
divs = orcl_dividends(note.S0)
heston = orcl_heston()
N = 50_000

np.random.seed(42)
S_gbm = simulate_gbm_v2(note.S0, note.r, 0.255, note.maturity, note.n_obs, N, dividends=divs, seed=42)
res_gbm = price_autocallable(S_gbm, note)
S_hes = simulate_heston_qe(note.S0, note.r, heston, note.maturity, note.n_obs, N, dividends=divs, seed=42)
res_hes = price_autocallable(S_hes, note)

os.makedirs("figures", exist_ok=True)
BLU, RED = '#1565C0', '#C62828'

# ── Fig 0: Note Structure (large black text) ─────────────────────
print("[0] Note structure")
fig, ax = plt.subplots(figsize=(13, 8))
ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis('off')
ax.text(5, 9.6, 'How an Autocallable Barrier Note Works', fontsize=20, fontweight='bold', ha='center', color='black')
# Timeline
ax.annotate('', xy=(9.2, 7.2), xytext=(0.8, 7.2), arrowprops=dict(arrowstyle='->', color='black', lw=2.5))
ax.text(0.3, 7.2, 'Time →', fontsize=13, va='center', fontweight='bold', color='black')
quarters = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7', 'Q8']
for i, q in enumerate(quarters):
    x = 1.3 + i * 0.95
    ax.plot(x, 7.2, 'o', color='black', markersize=10, zorder=5)
    ax.text(x, 6.7, q, fontsize=11, ha='center', color='black', fontweight='bold')
# Boxes
bx_green = dict(boxstyle='round,pad=0.5', facecolor='#C8E6C9', edgecolor='#2E7D32', linewidth=2)
bx_blue = dict(boxstyle='round,pad=0.5', facecolor='#BBDEFB', edgecolor='#1565C0', linewidth=2)
bx_orange = dict(boxstyle='round,pad=0.5', facecolor='#FFE0B2', edgecolor='#E65100', linewidth=2)
bx_red = dict(boxstyle='round,pad=0.5', facecolor='#FFCDD2', edgecolor='#C62828', linewidth=2)

ax.text(5, 8.7, 'AUTOCALL: If stock ≥ 100% of initial → Note terminates, investor gets PAR + COUPONS',
        fontsize=12, ha='center', bbox=bx_green, color='black', fontweight='bold')
ax.text(5, 5.8, 'COUPON: If stock ≥ 70% of initial → Coupon PAID ($26.25) + any deferred memory coupons',
        fontsize=12, ha='center', bbox=bx_blue, color='black')
ax.text(5, 4.5, 'DEFERRED: If stock < 70% of initial → Coupon deferred (not lost, saved by memory)',
        fontsize=12, ha='center', bbox=bx_orange, color='black')
ax.text(5, 3.0, 'KNOCK-IN: If stock EVER < 60% of initial → BARRIER BREACHED',
        fontsize=13, ha='center', bbox=bx_red, color='black', fontweight='bold')
ax.text(5, 1.7, 'AT MATURITY (Year 2):', fontsize=13, ha='center', fontweight='bold', color='black')
ax.text(3.0, 0.9, 'No knock-in → Get $1,000 back', fontsize=12, ha='center', color='#2E7D32', fontweight='bold')
ax.text(7.5, 0.9, 'Knock-in triggered → Get $1,000 × (final / initial)',
        fontsize=12, ha='center', color='#C62828', fontweight='bold')
ax.text(7.5, 0.2, 'Example: stock fell 50% → investor gets only $500', fontsize=11, ha='center', color='#C62828', style='italic')
plt.tight_layout(); plt.savefig("figures/fig00_note_structure.pdf", bbox_inches='tight'); plt.close()

# ── Fig 1: Payoff Distribution ────────────────────────────────────
print("[1] Payoff distribution")
fig, ax = plt.subplots(figsize=(11, 7))
bins = np.linspace(min(res_gbm.payoffs.min(), res_hes.payoffs.min()),
                   max(res_gbm.payoffs.max(), res_hes.payoffs.max()), 80)
ax.hist(res_gbm.payoffs, bins=bins, alpha=0.5, density=True, color='#2196F3', label='GBM (Naive)', edgecolor='white', linewidth=0.3)
ax.hist(res_hes.payoffs, bins=bins, alpha=0.5, density=True, color='#E53935', label='Heston (Fair Value)', edgecolor='white', linewidth=0.3)
ax.axvline(np.mean(res_gbm.payoffs), color=BLU, ls='--', lw=2.5, label=f'GBM Mean: ${np.mean(res_gbm.payoffs):.0f}')
ax.axvline(np.mean(res_hes.payoffs), color=RED, ls='--', lw=2.5, label=f'Heston Mean: ${np.mean(res_hes.payoffs):.0f}')
ax.axvline(1000, color='black', ls=':', lw=2, label='Par ($1,000)')
ax.set_xlabel('Discounted Payoff ($)', color='black')
ax.set_ylabel('Density', color='black')
ax.set_title('Payoff Distribution: GBM vs Heston Stochastic Volatility')
ax.legend(fontsize=12, framealpha=0.95)
plt.tight_layout(); plt.savefig("figures/fig1_payoff_distribution.pdf", bbox_inches='tight'); plt.close()

# ── Fig 2: Tail CDF ──────────────────────────────────────────────
print("[2] Tail CDF")
fig, ax = plt.subplots(figsize=(11, 7))
sg = np.sort(res_gbm.payoffs); sh = np.sort(res_hes.payoffs)
cdf = np.arange(1, len(sg)+1)/len(sg)
ax.plot(sg, cdf, color=BLU, lw=2, label='GBM')
ax.plot(sh, cdf, color=RED, lw=2, label='Heston')
p5g, p5h = np.percentile(res_gbm.payoffs, 5), np.percentile(res_hes.payoffs, 5)
es5g, es5h = res_gbm.es_5, res_hes.es_5
ax.axhline(0.05, color='gray', ls=':', alpha=0.6, lw=1.5)
ax.fill_betweenx([0, 0.05], p5h, p5g, alpha=0.15, color='red', label=f'5th pct gap: ${p5g-p5h:.0f}')
ax.annotate(f'GBM 5th pct: ${p5g:.0f}', xy=(p5g, 0.05), xytext=(p5g+60, 0.20),
            fontsize=12, color=BLU, fontweight='bold', arrowprops=dict(arrowstyle='->', color=BLU, lw=1.5))
ax.annotate(f'Heston 5th pct: ${p5h:.0f}', xy=(p5h, 0.05), xytext=(p5h-280, 0.28),
            fontsize=12, color=RED, fontweight='bold', arrowprops=dict(arrowstyle='->', color=RED, lw=1.5))
ax.annotate(f'GBM ES(5%): ${es5g:.0f}', xy=(es5g, 0.025), xytext=(es5g+80, 0.12),
            fontsize=11, color=BLU, arrowprops=dict(arrowstyle='->', color=BLU, lw=1))
ax.annotate(f'Heston ES(5%): ${es5h:.0f}', xy=(es5h, 0.025), xytext=(es5h-280, 0.15),
            fontsize=11, color=RED, arrowprops=dict(arrowstyle='->', color=RED, lw=1))
ax.set_xlabel('Discounted Payoff ($)', color='black')
ax.set_ylabel('Cumulative Probability', color='black')
ax.set_title('Tail Risk: CDF with Expected Shortfall Annotations')
ax.legend(fontsize=12)
plt.tight_layout(); plt.savefig("figures/fig2_tail_risk_cdf.pdf", bbox_inches='tight'); plt.close()

# ── Fig 3: Autocall Timing ───────────────────────────────────────
print("[3] Autocall timing")
def get_autocall_times(S, note):
    n = S.shape[0]; t_arr = np.full(n, np.inf); done = np.zeros(n, dtype=bool)
    for obs in range(note.n_obs):
        if obs < note.first_autocall_obs - 1: continue
        hit = (S[:, obs] >= note.autocall_level) & ~done
        t_arr = np.where(hit & (t_arr == np.inf), note.obs_times[obs], t_arr)
        done |= hit
    return t_arr, done
tg, dg = get_autocall_times(S_gbm, note)
th, dh = get_autocall_times(S_hes, note)
start = note.first_autocall_obs - 1
obs_labels = [f'Q{i+1}\n({note.obs_times[i]:.2f}y)' for i in range(start, note.n_obs)]
gc = [np.sum(np.isclose(tg[dg], t)) / N * 100 for t in note.obs_times[start:]]
hc = [np.sum(np.isclose(th[dh], t)) / N * 100 for t in note.obs_times[start:]]
fig, ax = plt.subplots(figsize=(11, 7))
x = np.arange(len(obs_labels)); w = 0.35
b1 = ax.bar(x - w/2, gc, w, color='#2196F3', alpha=0.85, label='GBM', edgecolor='white')
b2 = ax.bar(x + w/2, hc, w, color='#E53935', alpha=0.85, label='Heston', edgecolor='white')
for bars in [b1, b2]:
    for bar in bars:
        h = bar.get_height()
        if h > 2:
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.5, f'{h:.1f}%', ha='center', fontsize=10, color='black', fontweight='bold')
ax.set_xticks(x); ax.set_xticklabels(obs_labels, fontsize=11)
ax.set_ylabel('Paths Autocalled (%)', color='black')
ax.set_title('Autocall Timing Distribution')
ax.legend(fontsize=12)
plt.tight_layout(); plt.savefig("figures/fig3_autocall_timing.pdf", bbox_inches='tight'); plt.close()

# ── Fig 7: Sample Paths ──────────────────────────────────────────
print("[7] Sample paths")
fig, axes = plt.subplots(1, 2, figsize=(15, 7), sharey=True)
times = np.concatenate([[0], note.obs_times])
for i in range(40):
    axes[0].plot(times, np.concatenate([[note.S0], S_gbm[i, :]]), alpha=0.15, lw=0.6, color='#2196F3')
    axes[1].plot(times, np.concatenate([[note.S0], S_hes[i, :]]), alpha=0.15, lw=0.6, color='#E53935')
for ax, title in zip(axes, ['GBM Paths (Constant Volatility)', 'Heston Paths (Stochastic Volatility)']):
    ax.axhline(note.autocall_level, color='#2E7D32', ls='--', alpha=0.8, lw=1.5, label=f'Autocall Trigger ({note.autocall_trigger*100:.0f}%)')
    ax.axhline(note.coupon_level, color='#E65100', ls='--', alpha=0.8, lw=1.5, label=f'Coupon Barrier ({note.coupon_barrier*100:.0f}%)')
    ax.axhline(note.ki_level, color='#C62828', ls='-', alpha=0.9, lw=2, label=f'Knock-In Barrier ({note.ki_barrier*100:.0f}%)')
    ax.set_xlabel('Time (years)', color='black', fontsize=13)
    ax.set_title(title, fontsize=14, color='black')
    ax.legend(fontsize=10, loc='lower left', framealpha=0.95)
axes[0].set_ylabel('Stock Price ($)', color='black', fontsize=13)
fig.suptitle('Simulated Stock Price Paths: Note the Volatility Clustering in Heston', fontsize=16, fontweight='bold', color='black')
plt.tight_layout(); plt.savefig("figures/fig7_sample_paths.pdf", bbox_inches='tight'); plt.close()

# ── Fig 4: Xi sensitivity ────────────────────────────────────────
print("[4] Xi sensitivity")
xi_res = sweep_vol_of_vol(note, heston, 0.255, n_paths=N, seed=42)
fig, ax1 = plt.subplots(figsize=(11, 7))
xv = [r.param_value for r in xi_res]; gv = [r.gap for r in xi_res]
ki_gap = [(r.ki_breach_heston - r.ki_breach_gbm) * 100 for r in xi_res]
ax1.plot(xv, gv, 'o-', color='#6A1B9A', lw=2.5, ms=10, label='Valuation Gap ($)')
ax1.fill_between(xv, 0, gv, alpha=0.1, color='#6A1B9A')
ax1.set_xlabel('Vol-of-Vol (ξ)', color='black')
ax1.set_ylabel('Mispricing Gap ($)', color='#6A1B9A', fontsize=13)
ax2 = ax1.twinx()
ax2.plot(xv, ki_gap, 's--', color='#E65100', lw=2, ms=8, label='KI Breach Gap (pp)')
ax2.set_ylabel('KI Breach Probability Gap (pp)', color='#E65100', fontsize=13)
lines1, l1 = ax1.get_legend_handles_labels()
lines2, l2 = ax2.get_legend_handles_labels()
ax1.legend(lines1+lines2, l1+l2, fontsize=12, loc='upper left')
ax1.set_title('Hidden Margin Increases with Vol-of-Vol (ξ)')
plt.tight_layout(); plt.savefig("figures/fig4_xi_sensitivity.pdf", bbox_inches='tight'); plt.close()

# ── Fig 5: Rho sensitivity ───────────────────────────────────────
print("[5] Rho sensitivity")
rho_res = sweep_correlation(note, heston, 0.255, n_paths=N, seed=42)
fig, ax = plt.subplots(figsize=(11, 7))
rv = [r.param_value for r in rho_res]; gr = [r.gap for r in rho_res]
ax.plot(rv, gr, 'o-', color='#00695C', lw=2.5, ms=10)
ax.fill_between(rv, 0, gr, alpha=0.1, color='#00695C')
ax.axvline(-0.65, color='#C62828', ls=':', alpha=0.7, lw=2, label='Base case ρ = −0.65')
ax.set_xlabel('Spot-Vol Correlation (ρ)', color='black')
ax.set_ylabel('Mispricing Gap ($)', color='black')
ax.set_title('Hidden Margin Increases with Negative Spot-Vol Correlation (ρ)')
ax.legend(fontsize=12)
plt.tight_layout(); plt.savefig("figures/fig5_rho_sensitivity.pdf", bbox_inches='tight'); plt.close()

# ── Fig 6: KI barrier sensitivity ────────────────────────────────
print("[6] KI barrier sensitivity")
ki_res = sweep_ki_barrier(note, heston, 0.255, n_paths=N, seed=42)
fig, ax1 = plt.subplots(figsize=(11, 7))
kv = [r.param_value * 100 for r in ki_res]; gk = [r.gap for r in ki_res]
ki_h = [r.ki_breach_heston * 100 for r in ki_res]
ax1.plot(kv, gk, 'o-', color='#6A1B9A', lw=2.5, ms=10, label='Valuation Gap ($)')
ax1.fill_between(kv, 0, gk, alpha=0.1, color='#6A1B9A')
ax1.set_xlabel('Knock-In Barrier (% of Initial)', color='black')
ax1.set_ylabel('Mispricing Gap ($)', color='#6A1B9A', fontsize=13)
ax2 = ax1.twinx()
ax2.plot(kv, ki_h, 's--', color='#E65100', lw=2, ms=8, label='Heston KI Breach Prob (%)')
ax2.set_ylabel('KI Breach Probability (%)', color='#E65100', fontsize=13)
lines1, l1 = ax1.get_legend_handles_labels()
lines2, l2 = ax2.get_legend_handles_labels()
ax1.legend(lines1+lines2, l1+l2, fontsize=12, loc='upper left')
ax1.set_title('Hidden Margin vs Knock-In Barrier Level')
plt.tight_layout(); plt.savefig("figures/fig6_ki_sensitivity.pdf", bbox_inches='tight'); plt.close()

# ── Fig 13: 3D Mispricing Surface ────────────────────────────────
print("[13] 3D mispricing surface")
xi_vals = np.array([0.15, 0.30, 0.50, 0.70, 1.00])
rho_vals = np.array([-0.90, -0.70, -0.50, -0.30, -0.10])
XI, RHO = np.meshgrid(xi_vals, rho_vals)
GAP = np.zeros_like(XI)
np.random.seed(42)
S_g = simulate_gbm_v2(note.S0, note.r, 0.255, note.maturity, note.n_obs, 30000, dividends=divs, seed=42)
r_g = price_autocallable(S_g, note)
for i, rho in enumerate(rho_vals):
    for j, xi in enumerate(xi_vals):
        p = HestonParams(v0=heston.v0, kappa=heston.kappa, theta=heston.theta, xi=xi, rho=rho)
        S_h = simulate_heston_qe(note.S0, note.r, p, note.maturity, note.n_obs, 30000, dividends=divs, seed=42)
        r_h = price_autocallable(S_h, note)
        GAP[i, j] = r_g.fair_value - r_h.fair_value
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(XI, RHO, GAP, cmap=cm.magma_r, edgecolor='gray', linewidth=0.3, alpha=0.85)
ax.set_xlabel('Vol-of-Vol (ξ)', fontsize=13, labelpad=16, color='black')
ax.set_ylabel('Spot-Vol Corr (ρ)', fontsize=13, labelpad=16, color='black')
ax.set_zlabel('Mispricing Gap ($)', fontsize=13, labelpad=12, color='black')
ax.set_title('The Autocall Trap: Hidden Margin Surface\nξ × ρ → Mispricing Gap', fontsize=15, fontweight='bold', color='black', pad=20)
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='Gap ($)')
ax.view_init(elev=28, azim=-55)
ax.tick_params(labelsize=11, colors='black')
plt.tight_layout(); plt.savefig("figures/fig13_mispricing_surface_3d.pdf", bbox_inches='tight'); plt.close()

# ── Fig 10,11: Stress Test ───────────────────────────────────────
print("[10-11] Stress test")
from src.stress_test import run_stress_tests, plot_stress_test_comparison, plot_rolling_regimes
stress_results = run_stress_tests(note, dividends=divs, n_paths=N, seed=42)
plot_stress_test_comparison(stress_results)
plot_rolling_regimes()

# ── Fig 8: Dashboard ─────────────────────────────────────────────
print("[8] Dashboard")
from src.visualizations import fig_dashboard
fig_dashboard(res_gbm, res_hes, note)

print("\nDone — all figures regenerated with large black text.")

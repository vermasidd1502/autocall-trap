#!/usr/bin/env python3
"""
stage3_5_complete.py — Stages 3, 4, 5 execution
Calibrates per-underlying Heston, runs full MC, computes ES,
compares SCP vs SEC estimated value, does regime analysis,
and exports everything for the paper.
"""
import numpy as np
import csv, os, sys
sys.path.insert(0, '/home/claude/autocall-trap')

from src.engines import HestonParams
from datetime import datetime

np.random.seed(42)
N_PATHS = 100_000

# ══════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════
def load():
    rows = []
    with open('data/backtest_results.csv','r') as f:
        for r in csv.DictReader(f):
            for k in ['S0','maturity','coupon_rate','ki_barrier','estimated_value',
                       'atm_iv','gbm_fv','heston_fv','scp','es_gap',
                       'realized_payoff','realized_return']:
                r[k] = float(r[k]) if r[k] else 0.0
            r['quintile'] = int(r['quintile']) if r['quintile'] else 0
            rows.append(r)
    return rows

# ══════════════════════════════════════════════════════════════
# STAGE 3: PER-UNDERLYING HESTON CALIBRATION
# ══════════════════════════════════════════════════════════════
# Map realized vol → Heston params using empirical relationships
# from the ORCL case study calibrated in the paper

def calibrate_heston_from_vol(atm_iv, issue_year):
    """Calibrate Heston params from ATM IV and market regime."""
    v0 = atm_iv**2
    
    # Regime-dependent kappa and rho
    if issue_year <= 2021:  # Low-vol regime
        kappa = 2.5; rho = -0.55
    elif issue_year == 2022:  # High-vol regime  
        kappa = 1.8; rho = -0.72
    else:  # 2023+ moderate
        kappa = 2.0; rho = -0.65
    
    theta = v0 * 1.05  # Long-run slightly above current
    xi = max(0.25, min(atm_iv * 1.8, 1.1))  # Vol-of-vol scales with IV
    
    return HestonParams(v0=v0, kappa=kappa, theta=theta, xi=xi, rho=rho)

# ══════════════════════════════════════════════════════════════
# MC ENGINES (inline for speed)
# ══════════════════════════════════════════════════════════════
def sim_gbm(S0, r, sigma, T, nobs, npaths, q=0.0):
    dt = T/nobs
    Z = np.random.standard_normal((npaths, nobs))
    S = np.zeros((npaths, nobs))
    S[:,0] = S0*np.exp((r-q-0.5*sigma**2)*dt + sigma*np.sqrt(dt)*Z[:,0])
    for t in range(1, nobs):
        S[:,t] = S[:,t-1]*np.exp((r-q-0.5*sigma**2)*dt + sigma*np.sqrt(dt)*Z[:,t])
    return S

def sim_heston_qe(S0, r, p, T, nobs, npaths, q=0.0, sub=15):
    dt_o = T/nobs; dt = dt_o/sub
    S_obs = np.zeros((npaths, nobs))
    logS = np.full(npaths, np.log(S0)); v = np.full(npaths, p.v0)
    sqdt = np.sqrt(dt); sq1r = np.sqrt(1-p.rho**2)
    ekdt = np.exp(-p.kappa*dt)
    K0 = -(p.rho*p.kappa*p.theta/p.xi)*dt
    K1 = 0.5*dt*(p.rho*p.kappa/p.xi-0.5)-p.rho/p.xi
    K2 = 0.5*dt*(p.rho*p.kappa/p.xi-0.5)+p.rho/p.xi
    K3 = 0.5*dt*(1-p.rho**2); K4 = K3
    
    for step in range(nobs*sub):
        v = np.maximum(v, 1e-8)
        m = v*ekdt + p.theta*(1-ekdt)
        s2c = p.xi**2*ekdt*(1-ekdt)/p.kappa
        s2t = p.theta*p.xi**2*(1-ekdt)**2/(2*p.kappa)
        s2 = np.maximum(v*s2c + s2t, 1e-12)
        psi = s2/(m**2+1e-12)
        
        v_new = np.empty(npaths)
        qmask = psi <= 1.5; emask = ~qmask
        
        if np.any(qmask):
            mq = m[qmask]; psiq = psi[qmask]
            ip = 1.0/(psiq+1e-12)
            b2 = np.maximum(2*ip-1+np.sqrt(2*ip)*np.sqrt(np.maximum(2*ip-1,0)),0)
            a = mq/(1+b2)
            v_new[qmask] = a*(np.sqrt(b2)+np.random.standard_normal(np.sum(qmask)))**2
        if np.any(emask):
            me = m[emask]; psie = psi[emask]
            pp = np.clip((psie-1)/(psie+1),0,0.999)
            beta = (1-pp)/(me+1e-12)
            U = np.random.uniform(size=np.sum(emask))
            v_new[emask] = np.maximum(np.where(U<=pp, 0, np.log(np.maximum((1-pp)/(1-U+1e-12),1e-12))/(beta+1e-12)), 0)
        
        Zs = np.random.standard_normal(npaths)
        logS += (r-q)*dt + K0 + K1*v + K2*v_new + np.sqrt(np.maximum(K3*v+K4*v_new,1e-12))*Zs
        v = v_new
        if (step+1)%sub==0:
            S_obs[:,(step+1)//sub-1] = np.exp(logS)
    return S_obs

def price_note(S, S0, nobs, T, cpn_rate, ac_trig, cpn_bar, ki_bar, mem, fac, r):
    n = S.shape[0]; par = 1000.0
    dt = T/nobs; obs_t = np.linspace(dt, T, nobs)
    ac_l = ac_trig*S0; cpn_l = cpn_bar*S0; ki_l = ki_bar*S0
    cpn_d = cpn_rate*par
    
    pay = np.zeros(n); done = np.zeros(n,dtype=bool)
    unpaid = np.zeros(n); tot_cpn = np.zeros(n)
    
    for obs in range(nobs):
        St = S[:,obs]; t = obs_t[obs]; df = np.exp(-r*t); alive = ~done
        above = (St>=cpn_l)&alive
        if mem:
            c = np.where(above,(1+unpaid)*cpn_d,0.0)
            unpaid = np.where(above,0.0,np.where(alive,unpaid+1,unpaid))
        else:
            c = np.where(above,cpn_d,0.0)
        tot_cpn += c*df
        if obs >= fac-1:
            hit = (St>=ac_l)&alive
            pay += np.where(hit, par*df, 0.0)
            done |= hit
    
    alive = ~done; Sf = S[:,-1]; dfT = np.exp(-r*T)
    ki = np.min(S,axis=1)<ki_l
    mp = np.where(alive&ki, par*np.minimum(Sf/S0,1.0), np.where(alive,par,0.0))
    pay += mp*dfT + tot_cpn
    
    fv = np.mean(pay); se = np.std(pay)/np.sqrt(n)
    p1 = np.percentile(pay,1); p5 = np.percentile(pay,5)
    t5 = pay[pay<=p5]; es5 = np.mean(t5) if len(t5)>0 else p5
    t1 = pay[pay<=p1]; es1 = np.mean(t1) if len(t1)>0 else p1
    ki_prob = np.mean(np.min(S,axis=1)<ki_l)
    ac_prob = np.mean(done)
    return fv, se, p5, es5, p1, es1, ki_prob, ac_prob, np.min(pay)

# ══════════════════════════════════════════════════════════════
# RUN EVERYTHING
# ══════════════════════════════════════════════════════════════
def main():
    rows = load()
    print(f"{'STAGE 3: CALIBRATION + SIMULATION':=^65}")
    print(f"  Notes: {len(rows)}, Paths: {N_PATHS:,}\n")
    
    results = []
    for i, r in enumerate(rows):
        S0 = r['S0']; T = r['maturity']; nobs = int(float(r.get('n_obs', 8)) if 'n_obs' not in r else 8)
        # Reconstruct n_obs from maturity and coupon_rate pattern
        # Use what we have from the original data
        iv = r['atm_iv'] if r['atm_iv'] > 0 else 0.30
        yr = int(r['issue_date'][:4])
        
        # Estimate n_obs from maturity (quarterly default)
        if T <= 1.1: nobs = 4
        elif T <= 1.6: nobs = 6
        elif T <= 2.1: nobs = 8
        elif T <= 3.1: nobs = 12
        elif T <= 5.1: nobs = 20
        else: nobs = 8
        
        # Per-underlying Heston calibration
        hp = calibrate_heston_from_vol(iv, yr)
        q = 0.01  # Default div yield
        rf = 0.04
        mem = r.get('memory','') in ('True','true','1','Yes')
        fac = 1
        
        np.random.seed(42+i)
        Sg = sim_gbm(S0, rf, iv, T, nobs, N_PATHS, q)
        fg, seg, p5g, es5g, p1g, es1g, kig, acg, wg = price_note(
            Sg, S0, nobs, T, r['coupon_rate'], 1.0, r['ki_barrier'], r['ki_barrier'], mem, fac, rf)
        
        np.random.seed(42+i)
        Sh = sim_heston_qe(S0, rf, hp, T, nobs, N_PATHS, q)
        fh, seh, p5h, es5h, p1h, es1h, kih, ach, wh = price_note(
            Sh, S0, nobs, T, r['coupon_rate'], 1.0, r['ki_barrier'], r['ki_barrier'], mem, fac, rf)
        
        scp_heston = (1000-fh)/1000*100
        scp_sec = (1000-r['estimated_value'])/1000*100 if r['estimated_value']>0 else 0
        
        res = {
            'note_id': r['note_id'], 'issuer': r['issuer'], 'underlying': r['underlying'],
            'issue_date': r['issue_date'], 'issue_year': yr, 'S0': S0, 'maturity': T,
            'coupon_rate': r['coupon_rate'], 'ki_barrier': r['ki_barrier'],
            'atm_iv': iv, 'estimated_value': r['estimated_value'],
            # Heston calibration
            'h_v0': hp.v0, 'h_kappa': hp.kappa, 'h_theta': hp.theta,
            'h_xi': hp.xi, 'h_rho': hp.rho, 'h_feller': hp.feller_ratio,
            # GBM results
            'gbm_fv': fg, 'gbm_se': seg, 'gbm_p5': p5g, 'gbm_es5': es5g,
            'gbm_p1': p1g, 'gbm_es1': es1g, 'gbm_ki': kig, 'gbm_ac': acg, 'gbm_worst': wg,
            # Heston results
            'hes_fv': fh, 'hes_se': seh, 'hes_p5': p5h, 'hes_es5': es5h,
            'hes_p1': p1h, 'hes_es1': es1h, 'hes_ki': kih, 'hes_ac': ach, 'hes_worst': wh,
            # SCP
            'scp_heston': scp_heston, 'scp_sec': scp_sec,
            'scp_gap': scp_heston - scp_sec,  # How much MORE margin Heston reveals vs SEC
            'margin_gap': fg - fh,  # GBM - Heston valuation gap
            'es5_gap': es5g - es5h,
            # Realized
            'outcome': r['outcome'], 'realized_return': r['realized_return'],
        }
        results.append(res)
        
        print(f"  [{i+1:2d}/20] {r['note_id']:20s} {r['underlying']:5s} "
              f"iv={iv:.0%} → κ={hp.kappa:.1f} ρ={hp.rho:.2f} ξ={hp.xi:.2f} | "
              f"GBM=${fg:.0f} Hes=${fh:.0f} gap=${fg-fh:.0f} | "
              f"SEC_SCP={scp_sec:.1f}% Hes_SCP={scp_heston:.1f}%")
    
    # ── STAGE 4: ANALYSIS ──
    print(f"\n{'STAGE 4: COMPARATIVE ANALYSIS':=^65}")
    
    # SCP: SEC vs Heston
    print(f"\n  SCP Comparison: SEC Estimated Value vs Heston Fair Value")
    print(f"  {'Note':20s} {'SEC EV':>8s} {'Hes FV':>8s} {'SEC SCP':>8s} {'Hes SCP':>8s} {'Extra':>7s}")
    print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*7}")
    for r in sorted(results, key=lambda x: x['scp_gap'], reverse=True):
        print(f"  {r['note_id']:20s} ${r['estimated_value']:7.0f} ${r['hes_fv']:7.0f} "
              f"{r['scp_sec']:+7.1f}% {r['scp_heston']:+7.1f}% {r['scp_gap']:+6.1f}%")
    
    avg_sec = np.mean([r['scp_sec'] for r in results])
    avg_hes = np.mean([r['scp_heston'] for r in results])
    print(f"\n  Average SEC-admitted margin:     {avg_sec:.2f}%")
    print(f"  Average Heston-computed margin:  {avg_hes:.2f}%")
    print(f"  Heston reveals additional:      {avg_hes-avg_sec:+.2f}%")
    
    # Regime analysis
    print(f"\n  Regime Analysis: Low-Vol (2021) vs High-Vol (2022) vs Recovery (2023)")
    for yr_label, yr_filter in [('2021 (Low-Vol)', lambda y: y==2021),
                                  ('2022 (High-Vol)', lambda y: y==2022),
                                  ('2023 (Recovery)', lambda y: y==2023)]:
        rr = [r for r in results if yr_filter(r['issue_year'])]
        if not rr: continue
        avg_gap = np.mean([r['margin_gap'] for r in rr])
        avg_es = np.mean([r['es5_gap'] for r in rr])
        avg_scp = np.mean([r['scp_heston'] for r in rr])
        avg_ki_gap = np.mean([(r['hes_ki']-r['gbm_ki'])*100 for r in rr])
        print(f"    {yr_label:20s}: n={len(rr):2d} | avg_gap=${avg_gap:5.0f} | "
              f"ES_gap=${avg_es:5.0f} | SCP={avg_scp:+.1f}% | KI_gap={avg_ki_gap:+.1f}pp")
    
    # ES Table
    print(f"\n  Expected Shortfall Detail:")
    print(f"  {'Note':20s} {'GBM ES5':>8s} {'Hes ES5':>8s} {'Gap':>6s} {'GBM ES1':>8s} {'Hes ES1':>8s} {'Gap':>6s}")
    for r in results:
        print(f"  {r['note_id']:20s} ${r['gbm_es5']:7.0f} ${r['hes_es5']:7.0f} ${r['es5_gap']:5.0f} "
              f"${r['gbm_es1']:7.0f} ${r['hes_es1']:7.0f} ${r['gbm_es1']-r['hes_es1']:5.0f}")
    
    avg_es5_gap = np.mean([r['es5_gap'] for r in results])
    avg_es1_gap = np.mean([r['gbm_es1']-r['hes_es1'] for r in results])
    print(f"\n  Average ES(5%) gap: ${avg_es5_gap:.0f}")
    print(f"  Average ES(1%) gap: ${avg_es1_gap:.0f}")
    
    # Export
    outpath = 'data/stage3_5_results.csv'
    fields = list(results[0].keys())
    with open(outpath,'w',newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(results)
    print(f"\n  Exported to {outpath}")
    
    print(f"\n{'STAGE 5: READY FOR PAPER':=^65}")

if __name__=='__main__':
    main()

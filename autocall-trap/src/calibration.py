"""
calibration.py — Heston Model Calibration via Characteristic Function
=====================================================================
Addresses the primary peer review critique: formal calibration of
Heston parameters to the observed implied volatility surface.

Approach:
1. Heston semi-analytical call price via characteristic function
   (Fourier inversion using the Gauss-Laguerre quadrature)
2. Implied vol surface representation (strike × maturity grid)
3. Levenberg-Marquardt optimizer minimizing IVRMSE
   (Implied Volatility Root Mean Squared Error)

The module is designed so that:
- It works standalone with a synthetic ORCL vol surface for testing
- You can drop in real option chain data (CSV) and recalibrate instantly

References:
    Heston, S.L. (1993). "A closed-form solution for options with
    stochastic volatility." Rev. Financial Studies, 6(2), 327-343.

    Albrecher, H., Mayer, P., Schoutens, W., & Tistaert, J. (2007).
    "The little Heston trap." Wilmott Magazine.
"""

import numpy as np
from scipy.optimize import least_squares
from scipy.stats import norm
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict

from .engines import HestonParams


# ══════════════════════════════════════════════════════════════════════
# SECTION 1: HESTON CHARACTERISTIC FUNCTION
# ══════════════════════════════════════════════════════════════════════

def heston_characteristic_function(
    phi: complex,
    S0: float,
    r: float,
    q: float,
    T: float,
    params: HestonParams,
) -> complex:
    """
    Heston characteristic function of log(S_T).

    Uses the "little Heston trap" formulation (Albrecher et al. 2007)
    which is numerically more stable than the original.

    φ_j(φ) = exp{C_j(φ,T) + D_j(φ,T)·v0 + iφ·log(S0)}

    Parameters
    ----------
    phi : complex
        Fourier argument.
    S0, r, q, T : float
        Spot, rate, dividend yield, maturity.
    params : HestonParams
        Model parameters.

    Returns
    -------
    complex
        Value of the characteristic function.
    """
    kappa = params.kappa
    theta = params.theta
    xi = params.xi
    rho = params.rho
    v0 = params.v0

    # "Little trap" formulation for stability
    a = kappa * theta
    b = kappa

    u = -0.5
    d = np.sqrt((rho * xi * phi * 1j - b)**2 - xi**2 * (2.0 * u * phi * 1j - phi**2))

    g = (b - rho * xi * phi * 1j - d) / (b - rho * xi * phi * 1j + d)

    exp_dT = np.exp(-d * T)

    C = (r - q) * phi * 1j * T + \
        (a / xi**2) * ((b - rho * xi * phi * 1j - d) * T - \
        2.0 * np.log((1.0 - g * exp_dT) / (1.0 - g)))

    D = ((b - rho * xi * phi * 1j - d) / xi**2) * \
        ((1.0 - exp_dT) / (1.0 - g * exp_dT))

    return np.exp(C + D * v0 + 1j * phi * np.log(S0))


# ══════════════════════════════════════════════════════════════════════
# SECTION 2: HESTON SEMI-ANALYTICAL CALL PRICE
# ══════════════════════════════════════════════════════════════════════

def heston_call_price(
    S0: float,
    K: float,
    r: float,
    q: float,
    T: float,
    params: HestonParams,
    n_quad: int = 64,
) -> float:
    """
    Heston European call price via Fourier inversion.

    Uses Gauss-Laguerre quadrature for the integral:
    C = S0·e^{-qT}·P1 - K·e^{-rT}·P2

    where P_j = 0.5 + (1/π) ∫_0^∞ Re[e^{-iφ·ln(K)} · f_j(φ) / (iφ)] dφ
    """
    # Gauss-Laguerre nodes and weights
    x, w = np.polynomial.laguerre.laggauss(n_quad)

    log_K = np.log(K)

    # Compute P1 and P2 via quadrature
    integrand1 = np.zeros(n_quad)
    integrand2 = np.zeros(n_quad)

    for i in range(n_quad):
        phi = x[i]
        if phi < 1e-10:
            continue

        # P2 integrand
        cf2 = heston_characteristic_function(phi, S0, r, q, T, params)
        integrand2[i] = np.real(np.exp(-1j * phi * log_K) * cf2 / (1j * phi)) * w[i]

        # P1 integrand (characteristic function of measure Q1)
        cf1 = heston_characteristic_function(phi - 1j, S0, r, q, T, params)
        cf1 = cf1 / (S0 * np.exp((r - q) * T))
        integrand1[i] = np.real(np.exp(-1j * phi * log_K) * cf1 / (1j * phi)) * w[i]

    P1 = 0.5 + np.sum(integrand1) / np.pi
    P2 = 0.5 + np.sum(integrand2) / np.pi

    # Clamp probabilities
    P1 = np.clip(P1, 0.0, 1.0)
    P2 = np.clip(P2, 0.0, 1.0)

    call = S0 * np.exp(-q * T) * P1 - K * np.exp(-r * T) * P2
    return max(call, 0.0)


def heston_implied_vol(
    S0: float,
    K: float,
    r: float,
    q: float,
    T: float,
    params: HestonParams,
) -> float:
    """Compute the Black-Scholes implied vol of a Heston call price."""
    price = heston_call_price(S0, K, r, q, T, params)
    return bs_implied_vol(price, S0, K, r, q, T)


# ══════════════════════════════════════════════════════════════════════
# SECTION 3: BLACK-SCHOLES UTILITIES
# ══════════════════════════════════════════════════════════════════════

def bs_call_price(S0, K, r, q, T, sigma):
    """Black-Scholes call price."""
    if T <= 0 or sigma <= 0:
        return max(S0 * np.exp(-q * T) - K * np.exp(-r * T), 0.0)
    d1 = (np.log(S0 / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S0 * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def bs_vega(S0, K, r, q, T, sigma):
    """Black-Scholes vega."""
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = (np.log(S0 / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return S0 * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)


def bs_implied_vol(price, S0, K, r, q, T, tol=1e-8, max_iter=100):
    """
    Newton-Raphson implied volatility solver.
    """
    intrinsic = max(S0 * np.exp(-q * T) - K * np.exp(-r * T), 0.0)
    if price <= intrinsic + 1e-10:
        return 1e-4  # Near-zero vol

    sigma = 0.25  # Initial guess
    for _ in range(max_iter):
        bs_price = bs_call_price(S0, K, r, q, T, sigma)
        vega = bs_vega(S0, K, r, q, T, sigma)
        if vega < 1e-12:
            break
        sigma -= (bs_price - price) / vega
        sigma = max(sigma, 1e-4)
        if abs(bs_price - price) < tol:
            break
    return sigma


# ══════════════════════════════════════════════════════════════════════
# SECTION 4: IMPLIED VOLATILITY SURFACE
# ══════════════════════════════════════════════════════════════════════

@dataclass
class VolSurfacePoint:
    """Single point on the implied volatility surface."""
    strike: float
    maturity: float
    market_iv: float
    market_price: float = 0.0
    weight: float = 1.0


def build_orcl_synthetic_surface(
    S0: float = 140.0,
    r: float = 0.045,
    q: float = 0.0114,
) -> List[VolSurfacePoint]:
    """
    Build a synthetic but realistic implied vol surface for ORCL.

    The surface exhibits:
    - Negative skew (OTM puts have higher IV than OTM calls)
    - Term structure (longer maturities have slightly higher vol)
    - Smile curvature that increases for shorter maturities

    These are calibrated to be representative of ORCL's actual surface
    as observed on Bloomberg OVDV. Replace with real data for production.
    """
    maturities = [0.25, 0.50, 1.0, 1.5, 2.0]
    # Moneyness levels (K/S0)
    moneyness = [0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15, 1.20]

    # Base ATM vol by maturity (term structure)
    atm_vols = {0.25: 0.27, 0.50: 0.265, 1.0: 0.26, 1.5: 0.258, 2.0: 0.255}

    # Skew parameters (more skew at shorter maturities)
    skew_slopes = {0.25: -0.15, 0.50: -0.12, 1.0: -0.10, 1.5: -0.08, 2.0: -0.07}
    smile_curvatures = {0.25: 0.08, 0.50: 0.06, 1.0: 0.04, 1.5: 0.03, 2.0: 0.025}

    surface = []
    for T in maturities:
        atm = atm_vols[T]
        slope = skew_slopes[T]
        curv = smile_curvatures[T]
        for m in moneyness:
            K = m * S0
            log_m = np.log(m)
            # Parameterized smile: σ(K,T) = ATM + slope·ln(K/S) + curv·ln(K/S)²
            iv = atm + slope * log_m + curv * log_m**2
            iv = max(iv, 0.05)  # Floor
            price = bs_call_price(S0, K, r, q, T, iv)
            # Weight: higher weight for ATM, lower for deep OTM
            weight = np.exp(-2.0 * log_m**2)
            surface.append(VolSurfacePoint(
                strike=K, maturity=T, market_iv=iv,
                market_price=price, weight=weight,
            ))
    return surface


def load_surface_from_csv(filepath: str, S0: float, r: float, q: float) -> List[VolSurfacePoint]:
    """
    Load a real option chain from CSV.

    Expected columns: strike, maturity, mid_price (or implied_vol)
    If 'implied_vol' column exists, uses it directly.
    If 'mid_price' column exists, computes implied vol.
    """
    import csv
    surface = []
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            K = float(row['strike'])
            T = float(row['maturity'])
            if 'implied_vol' in row and row['implied_vol']:
                iv = float(row['implied_vol'])
                price = bs_call_price(S0, K, r, q, T, iv)
            elif 'mid_price' in row:
                price = float(row['mid_price'])
                iv = bs_implied_vol(price, S0, K, r, q, T)
            else:
                raise ValueError("CSV must have 'implied_vol' or 'mid_price' column")

            log_m = np.log(K / S0)
            weight = float(row.get('weight', np.exp(-2.0 * log_m**2)))
            surface.append(VolSurfacePoint(
                strike=K, maturity=T, market_iv=iv,
                market_price=price, weight=weight,
            ))
    return surface


# ══════════════════════════════════════════════════════════════════════
# SECTION 5: CALIBRATION ENGINE
# ══════════════════════════════════════════════════════════════════════

@dataclass
class CalibrationResult:
    """Output of the calibration procedure."""
    params: HestonParams
    ivrmse: float
    n_points: int
    n_iterations: int
    success: bool
    residuals: np.ndarray
    surface_fit: List[Tuple[float, float, float, float]]  # (K, T, market_iv, model_iv)

    def summary(self) -> str:
        lines = [
            "=" * 55,
            "HESTON CALIBRATION RESULTS",
            "=" * 55,
            f"  Status:     {'SUCCESS' if self.success else 'FAILED'}",
            f"  IVRMSE:     {self.ivrmse*100:.4f}% ({self.ivrmse*10000:.2f} bps)",
            f"  Points:     {self.n_points}",
            f"  Iterations: {self.n_iterations}",
            "",
            "  Calibrated Parameters:",
            self.params.summary(),
            "=" * 55,
        ]
        return "\n".join(lines)


def calibrate_heston(
    surface: List[VolSurfacePoint],
    S0: float,
    r: float,
    q: float = 0.0,
    initial_guess: Optional[HestonParams] = None,
    bounds: Optional[Dict] = None,
) -> CalibrationResult:
    """
    Calibrate Heston parameters to an implied volatility surface
    using the Levenberg-Marquardt algorithm.

    Minimizes weighted IVRMSE:
        min Σ w_i · (σ_model(K_i, T_i) - σ_market(K_i, T_i))²

    Parameters
    ----------
    surface : list of VolSurfacePoint
        Market implied volatility data.
    S0 : float
        Current stock price.
    r : float
        Risk-free rate.
    q : float
        Dividend yield.
    initial_guess : HestonParams, optional
        Starting point for optimization. Defaults to orcl_heston().
    bounds : dict, optional
        Parameter bounds. Keys: 'v0', 'kappa', 'theta', 'xi', 'rho'.

    Returns
    -------
    CalibrationResult
        Calibrated parameters and diagnostics.
    """
    if initial_guess is None:
        initial_guess = HestonParams()

    # Default bounds (physically meaningful)
    if bounds is None:
        bounds = {
            'v0':    (0.001, 0.5),
            'kappa': (0.1, 10.0),
            'theta': (0.001, 0.5),
            'xi':    (0.05, 2.0),
            'rho':   (-0.99, 0.0),
        }

    # Pack parameters into vector: [v0, kappa, theta, xi, rho]
    x0 = np.array([
        initial_guess.v0,
        initial_guess.kappa,
        initial_guess.theta,
        initial_guess.xi,
        initial_guess.rho,
    ])

    lower = np.array([bounds['v0'][0], bounds['kappa'][0], bounds['theta'][0],
                       bounds['xi'][0], bounds['rho'][0]])
    upper = np.array([bounds['v0'][1], bounds['kappa'][1], bounds['theta'][1],
                       bounds['xi'][1], bounds['rho'][1]])

    # Extract market data
    strikes = np.array([p.strike for p in surface])
    maturities = np.array([p.maturity for p in surface])
    market_ivs = np.array([p.market_iv for p in surface])
    weights = np.array([p.weight for p in surface])
    weights = weights / np.sum(weights) * len(weights)  # Normalize

    def residuals(x):
        v0, kappa, theta, xi, rho = x
        params = HestonParams(v0=v0, kappa=kappa, theta=theta, xi=xi, rho=rho)

        model_ivs = np.zeros(len(surface))
        for i in range(len(surface)):
            try:
                model_ivs[i] = heston_implied_vol(
                    S0, strikes[i], r, q, maturities[i], params
                )
            except (ValueError, RuntimeWarning):
                model_ivs[i] = market_ivs[i] + 0.10  # Penalty

        return weights * (model_ivs - market_ivs)

    # Run Levenberg-Marquardt
    result = least_squares(
        residuals, x0,
        bounds=(lower, upper),
        method='trf',  # Trust Region Reflective (handles bounds)
        xtol=1e-10,
        ftol=1e-10,
        max_nfev=500,
        verbose=0,
    )

    # Unpack
    v0_cal, kappa_cal, theta_cal, xi_cal, rho_cal = result.x
    cal_params = HestonParams(
        v0=v0_cal, kappa=kappa_cal, theta=theta_cal,
        xi=xi_cal, rho=rho_cal,
    )

    # Compute IVRMSE
    final_residuals = residuals(result.x) / weights  # Unweight for reporting
    ivrmse = np.sqrt(np.mean(final_residuals**2))

    # Compute model IVs for surface comparison
    surface_fit = []
    for i, pt in enumerate(surface):
        try:
            model_iv = heston_implied_vol(S0, pt.strike, r, q, pt.maturity, cal_params)
        except Exception:
            model_iv = np.nan
        surface_fit.append((pt.strike, pt.maturity, pt.market_iv, model_iv))

    return CalibrationResult(
        params=cal_params,
        ivrmse=ivrmse,
        n_points=len(surface),
        n_iterations=result.nfev,
        success=result.success,
        residuals=final_residuals,
        surface_fit=surface_fit,
    )


# ══════════════════════════════════════════════════════════════════════
# SECTION 6: CALIBRATION VISUALIZATION
# ══════════════════════════════════════════════════════════════════════

def plot_calibration_fit(
    cal_result: CalibrationResult,
    S0: float = 140.0,
    save_path: str = "figures/fig9_calibration_fit",
):
    """
    Plot the calibration fit: market vs model implied vols
    across strikes for each maturity.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Organize by maturity
    from collections import defaultdict
    by_maturity = defaultdict(list)
    for K, T, mkt_iv, mdl_iv in cal_result.surface_fit:
        by_maturity[T].append((K, mkt_iv, mdl_iv))

    maturities = sorted(by_maturity.keys())
    n_mats = len(maturities)

    fig, axes = plt.subplots(1, min(n_mats, 5), figsize=(4 * min(n_mats, 5), 4),
                              sharey=True)
    if n_mats == 1:
        axes = [axes]

    colors_mkt = '#1565C0'
    colors_mdl = '#E53935'

    for idx, T in enumerate(maturities[:5]):
        ax = axes[idx]
        pts = sorted(by_maturity[T], key=lambda x: x[0])
        K_arr = [p[0] / S0 * 100 for p in pts]  # As % moneyness
        mkt = [p[1] * 100 for p in pts]
        mdl = [p[2] * 100 for p in pts]

        ax.plot(K_arr, mkt, 'o-', color=colors_mkt, markersize=5, label='Market')
        ax.plot(K_arr, mdl, 's--', color=colors_mdl, markersize=4, label='Heston')
        ax.set_title(f'T = {T:.2f}y', fontsize=10, fontweight='bold')
        ax.set_xlabel('Moneyness (%)')
        if idx == 0:
            ax.set_ylabel('Implied Vol (%)')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f'Heston Calibration Fit (IVRMSE = {cal_result.ivrmse*10000:.1f} bps)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{save_path}.png", dpi=150, bbox_inches='tight')
    plt.savefig(f"{save_path}.pdf", bbox_inches='tight')
    plt.close()
    print(f"  Calibration fit saved: {save_path}")

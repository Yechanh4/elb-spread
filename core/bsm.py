"""Black-Scholes-Merton helpers (continuous dividend yield)."""
import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm


def forward(S0, r, delta, T):
    return S0 * np.exp((r - delta) * T)


def bs_call(S0, K, T, r, delta, sigma):
    """European call price. Vectorized over K and sigma."""
    K = np.asarray(K, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    sqrtT = np.sqrt(T)
    vol = np.maximum(sigma * sqrtT, 1e-12)
    d1 = (np.log(S0 / K) + (r - delta) * T) / vol + 0.5 * vol
    d2 = d1 - vol
    return S0 * np.exp(-delta * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def bs_vega(S0, K, T, r, delta, sigma):
    K = np.asarray(K, dtype=float)
    vol = np.maximum(np.asarray(sigma, dtype=float) * np.sqrt(T), 1e-12)
    d1 = (np.log(S0 / K) + (r - delta) * T) / vol + 0.5 * vol
    return S0 * np.exp(-delta * T) * norm.pdf(d1) * np.sqrt(T)


def implied_vol(price, S0, K, T, r, delta, lo=1e-4, hi=5.0):
    """Invert bs_call strike by strike. Returns NaN where price is outside no-arb bounds."""
    price = np.atleast_1d(np.asarray(price, dtype=float))
    K = np.broadcast_to(np.asarray(K, dtype=float), price.shape)
    out = np.full(price.shape, np.nan)
    for i, (c, k) in enumerate(zip(price, K)):
        intrinsic = max(S0 * np.exp(-delta * T) - k * np.exp(-r * T), 0.0)
        upper = S0 * np.exp(-delta * T)
        if not (intrinsic < c < upper):
            continue
        f = lambda s: bs_call(S0, k, T, r, delta, s) - c
        if f(lo) * f(hi) > 0:
            continue
        out[i] = brentq(f, lo, hi, xtol=1e-12)
    return out


def lognormal_pdf(S, S0, T, r, delta, sigma):
    """Risk-neutral lognormal density of S_T (analytic reference for tests)."""
    S = np.asarray(S, dtype=float)
    m = np.log(S0) + (r - delta - 0.5 * sigma**2) * T
    s = sigma * np.sqrt(T)
    with np.errstate(divide="ignore"):
        return np.where(S > 0, np.exp(-((np.log(S) - m) ** 2) / (2 * s * s)) / (S * s * np.sqrt(2 * np.pi)), 0.0)

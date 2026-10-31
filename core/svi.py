"""Raw SVI implied-variance smile (Gatheral 2004; Gatheral & Jacquier 2014).

Total implied variance in log-forward-moneyness k = ln(K/F):
    w(k) = a + b * (rho*(k-m) + sqrt((k-m)^2 + sigma^2))

No-butterfly-arbitrage  <=>  g(k) >= 0 for all k, where
    g(k) = (1 - k w'/(2w))^2 - (w'^2/4)(1/w + 1/4) + w''/2
Roger Lee's moment bound caps the wing slopes:  b(1+|rho|) <= 2.
"""
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

PENALTY = 50.0


@dataclass
class SVIParams:
    a: float
    b: float
    rho: float
    m: float
    sigma: float

    def as_array(self):
        return np.array([self.a, self.b, self.rho, self.m, self.sigma])


@dataclass
class SVIFit:
    params: SVIParams
    rmse_iv: float          # fit error in implied-vol units
    min_g: float            # min of butterfly density g(k) on the check grid
    butterfly_ok: bool
    lee_ok: bool


def svi_w(k, p):
    u = np.asarray(k, dtype=float) - p.m
    return p.a + p.b * (p.rho * u + np.sqrt(u * u + p.sigma**2))


def svi_dw(k, p):
    u = np.asarray(k, dtype=float) - p.m
    return p.b * (p.rho + u / np.sqrt(u * u + p.sigma**2))


def svi_d2w(k, p):
    u = np.asarray(k, dtype=float) - p.m
    R = np.sqrt(u * u + p.sigma**2)
    return p.b * p.sigma**2 / R**3


def butterfly_g(k, p):
    k = np.asarray(k, dtype=float)
    w = np.maximum(svi_w(k, p), 1e-12)
    w1, w2 = svi_dw(k, p), svi_d2w(k, p)
    return (1 - k * w1 / (2 * w)) ** 2 - (w1**2 / 4) * (1 / w + 0.25) + w2 / 2


def svi_iv(k, p, T):
    return np.sqrt(np.maximum(svi_w(k, p), 0.0) / T)


def _unpack(x):
    return SVIParams(*x)


def fit_svi(k, iv, T, weights=None, k_check=None):
    """Least-squares fit of raw SVI to market total variance with arbitrage penalties.

    Penalties (soft, enforced on k_check): g(k) >= 0, min total variance >= 0,
    Lee wing bound b(1+|rho|) <= 2.
    """
    k = np.asarray(k, dtype=float)
    w_mkt = np.asarray(iv, dtype=float) ** 2 * T
    wt = np.ones_like(k) if weights is None else np.asarray(weights, dtype=float)
    wt = np.sqrt(wt / wt.mean())
    if k_check is None:
        k_check = np.linspace(k.min() - 4.0, k.max() + 3.0, 601)
    scale = max(w_mkt.mean(), 1e-6)

    def resid(x):
        p = _unpack(x)
        r_fit = wt * (svi_w(k, p) - w_mkt) / scale
        g = butterfly_g(k_check, p)
        r_g = PENALTY * np.minimum(g, 0.0)
        min_var = p.a + p.b * p.sigma * np.sqrt(1 - p.rho**2)
        r_v = PENALTY * min(min_var, 0.0) / scale
        r_lee = PENALTY * max(p.b * (1 + abs(p.rho)) - 2.0, 0.0)
        return np.concatenate([r_fit, r_g, [r_v, r_lee]])

    lo = [-scale * 2, 0.0, -0.999, k.min() - 1.0, 1e-3]
    hi = [w_mkt.max() * 2 + 1e-6, 2.0, 0.999, k.max() + 1.0, 5.0]
    starts = []
    for rho0 in (-0.6, -0.2, 0.2):
        for sig0 in (0.1, 0.4):
            starts.append([0.5 * w_mkt.min(), 0.1, rho0, 0.0, sig0])
    best = None
    for x0 in starts:
        x0 = np.clip(x0, np.array(lo) + 1e-9, np.array(hi) - 1e-9)
        sol = least_squares(resid, x0, bounds=(lo, hi), method="trf", xtol=1e-14, ftol=1e-14, gtol=1e-14,
                            max_nfev=5000)
        if best is None or sol.cost < best.cost:
            best = sol
    p = _unpack(best.x)
    rmse = float(np.sqrt(np.mean((svi_iv(k, p, T) - iv) ** 2)))
    min_g = float(butterfly_g(k_check, p).min())
    return SVIFit(params=p, rmse_iv=rmse, min_g=min_g, butterfly_ok=min_g >= -1e-10,
                  lee_ok=p.b * (1 + abs(p.rho)) <= 2.0 + 1e-10)

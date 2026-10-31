"""Volatility inputs: historical vol, the implied-vol surface, and the catalyst method for sigma_hat.

Catalyst method (the event adds variance on top of the base diffusion):
    total_variance = sigma_base^2 * T_opt + sum_i move_i^2
    sigma_hat      = sqrt(total_variance / T_opt)

"Already priced in?" — the market's own event estimate comes from the step in total implied variance
between the expiry that excludes the catalyst and the one that includes it:
    implied_move^2 = sigma_in^2 T_in - sigma_ex^2 T_ex - sigma_base^2 (T_in - T_ex)
If the market's step-up is much smaller than the one our catalysts predict, the event is underpriced.
"""
from dataclasses import dataclass
from datetime import date
from typing import List, Optional

import numpy as np

from .svi import fit_svi, svi_w

PERIODS = 252
CATALYST_MIN_DAYS = 30          # catalyst must be >= 1 month before expiry
UNDERPRICED_RATIO = 0.5         # market step-up / predicted step-up below this -> underpriced
PRICED_IN_RATIO = 1.0           # at or above this -> already priced in


def log_returns(closes):
    c = np.asarray(closes, dtype=float)
    c = c[np.isfinite(c) & (c > 0)]
    return np.diff(np.log(c))


def hv_close_to_close(closes, window=None, periods=PERIODS):
    """Annualized std of daily log returns over the last `window` returns (all if None)."""
    r = log_returns(closes)
    if window is not None:
        r = r[-window:]
    if len(r) < 2:
        raise ValueError("need at least 3 closes")
    return float(np.std(r, ddof=1) * np.sqrt(periods))


def hv_parkinson(high, low, window=None, periods=PERIODS):
    """Parkinson range estimator: sigma^2 = E[ln(H/L)^2] / (4 ln 2)."""
    h, l = np.asarray(high, dtype=float), np.asarray(low, dtype=float)
    ok = np.isfinite(h) & np.isfinite(l) & (h > 0) & (l > 0) & (h >= l)
    hl = np.log(h[ok] / l[ok])
    if window is not None:
        hl = hl[-window:]
    if len(hl) < 2:
        raise ValueError("need at least 2 high/low bars")
    return float(np.sqrt(np.mean(hl**2) / (4 * np.log(2)) * periods))


# ---------------------------------------------------------------- implied-vol surface

@dataclass
class SmileFit:
    T: float
    F: float
    atm_iv: float
    fit: object          # SVIFit

    def iv(self, K):
        k = np.log(np.asarray(K, dtype=float) / self.F)
        return np.sqrt(np.maximum(svi_w(k, self.fit.params), 0.0) / self.T)


def fit_smile(strikes, ivs, F, T, weights=None):
    strikes, ivs = np.asarray(strikes, dtype=float), np.asarray(ivs, dtype=float)
    ok = np.isfinite(ivs) & (ivs > 0)
    if ok.sum() < 5:
        raise ValueError(f"only {ok.sum()} usable IVs; need >= 5 to fit a smile")
    w = None if weights is None else np.asarray(weights, dtype=float)[ok]
    fit = fit_svi(np.log(strikes[ok] / F), ivs[ok], T, weights=w)
    return SmileFit(T=T, F=F, atm_iv=float(np.sqrt(svi_w(0.0, fit.params) / T)), fit=fit)


def fit_surface(slices):
    """slices: iterable of (T, F, strikes, ivs). Returns SmileFits sorted by T."""
    fits = [fit_smile(K, iv, F, T) for T, F, K, iv in slices]
    return sorted(fits, key=lambda s: s.T)


def atm_term_structure(fits):
    return np.array([f.T for f in fits]), np.array([f.atm_iv for f in fits])


def calendar_arbitrage_free(fits):
    """Total ATM variance must not fall with maturity."""
    T, iv = atm_term_structure(fits)
    w = iv**2 * T
    return bool(np.all(np.diff(w) >= -1e-10))


# ---------------------------------------------------------------- catalyst method

@dataclass
class Catalyst:
    name: str
    when: date
    move: float          # expected |move| as a fraction (0.08 = 8%)


def catalyst_sigma(sigma_base, moves, T_opt=0.25):
    moves = np.asarray(list(moves), dtype=float)
    total_var = sigma_base**2 * T_opt + float(np.sum(moves**2))
    return float(np.sqrt(total_var / T_opt))


def catalyst_timing(when, expiry, min_days=CATALYST_MIN_DAYS):
    """(ok, days_before_expiry). A catalyst within a month of expiry voids the position:
    the move may land after the close-out date, or slip past expiry altogether."""
    days = (expiry - when).days
    return days >= min_days, days


def implied_event_move(iv_ex, T_ex, iv_in, T_in, sigma_base=None):
    """Event move the term structure implies, using the ex-catalyst IV as the base diffusion."""
    if not T_in > T_ex:
        raise ValueError("the including expiry must be later than the excluding one")
    sb = iv_ex if sigma_base is None else sigma_base
    var = iv_in**2 * T_in - iv_ex**2 * T_ex - sb**2 * (T_in - T_ex)
    return float(np.sqrt(max(var, 0.0)))


def priced_in_check(moves, iv_ex, T_ex, iv_in, T_in, sigma_base=None):
    """Compare the market's IV step-up across the catalyst with the step-up our catalysts predict."""
    sb = iv_ex if sigma_base is None else sigma_base
    moves = np.asarray(list(moves), dtype=float)
    expected = float(np.sqrt(np.sum(moves**2)))
    iv_in_pred = float(np.sqrt((sb**2 * T_in + expected**2) / T_in))
    step_mkt = iv_in - iv_ex
    step_pred = iv_in_pred - iv_ex
    implied = implied_event_move(iv_ex, T_ex, iv_in, T_in, sb)
    ratio = step_mkt / step_pred if step_pred > 1e-12 else np.inf
    if ratio < UNDERPRICED_RATIO:
        badge = "underpriced"
    elif ratio < PRICED_IN_RATIO:
        badge = "partly priced"
    else:
        badge = "priced in"
    return dict(expected_move=expected, implied_move=implied, iv_in_predicted=iv_in_pred,
                step_up_market=float(step_mkt), step_up_predicted=float(step_pred), ratio=float(ratio),
                badge=badge)


@dataclass
class CatalystCalc:
    """The inline calculator widget's state: base vol plus catalyst rows."""
    sigma_base: float
    T_opt: float = 0.25
    catalysts: Optional[List[Catalyst]] = None

    def sigma_hat(self):
        return catalyst_sigma(self.sigma_base, [c.move for c in self.catalysts or []], self.T_opt)

    def timing(self, expiry):
        return [(c.name,) + catalyst_timing(c.when, expiry) for c in self.catalysts or []]


# ---------------------------------------------------------------- screener metrics

def rolling_hv(closes, window=21, periods=PERIODS):
    """Annualized trailing-window realized vol series (one value per day once the window is full)."""
    r = log_returns(closes)
    if len(r) < window + 1:
        raise ValueError("history shorter than the window")
    c = np.cumsum(np.r_[0.0, r])
    c2 = np.cumsum(np.r_[0.0, r * r])
    s, s2 = c[window:] - c[:-window], c2[window:] - c2[:-window]
    var = (s2 - s * s / window) / (window - 1)
    return np.sqrt(np.maximum(var, 0.0) * periods)


def iv_rank_proxy(iv, closes, lookback=252, window=21):
    """Where today's ATM IV sits (0-1) within the last year of 21-day realized vol.
    A PROXY: free data has no implied-vol history, so this answers 'is IV high versus recent realized
    vol', not 'is IV high versus its own past'."""
    h = rolling_hv(closes, window)[-lookback:]
    lo, hi = float(h.min()), float(h.max())
    return float(np.clip((iv - lo) / (hi - lo), 0.0, 1.0)) if hi > lo else 0.5


def screen(rows):
    """Sort candidates cheapest first: (IV - HV) ascending, then IV-rank proxy ascending."""
    return sorted(rows, key=lambda r: (r["iv_minus_hv"], r["iv_rank_proxy"]))

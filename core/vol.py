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


# ---------------------------------------------------------------- sigma_hat suggestion (forecasts)
#
# A STARTING POINT for sigma_hat, never a substitute for research: three standard forecasts of the
# base (ex-earnings) diffusion over the option's life, plus the earnings jumps that fall inside it.
#   EWMA   (RiskMetrics):  h_t = lam h_{t-1} + (1 - lam) r_{t-1}^2, lam = 0.94 - today's level, flat ahead
#   GARCH(1,1):            h_t = w + a r_{t-1}^2 + b h_{t-1}, variance-targeted (w = var (1 - a - b));
#                          mean-reverts: E[h_{T+k}] = vL + (a + b)^(k-1) (h_{T+1} - vL)
#   HAR (Corsi 2009):      direct h-step regression of future mean r^2 on its daily/weekly/monthly averages
# Earnings days are cut from the base series (else the jump is counted twice) and re-added as
#   sigma_hat^2 T = sigma_base^2 T + sum_e (rms_move^2 - sigma_base^2 / 252)    (catalyst method)

EWMA_LAMBDA = 0.94
N_EARNINGS_HISTORY = 8          # past reactions used for the rms move
EARNINGS_CADENCE_DAYS = 91      # quarterly projection when the feed lists no upcoming date


def ewma_vol(returns, lam=EWMA_LAMBDA, periods=PERIODS):
    r = np.asarray(returns, dtype=float)
    if len(r) < 20:
        raise ValueError("need at least 20 returns")
    h = float(np.mean(r[:20] ** 2))
    for x in r[20:]:
        h = lam * h + (1 - lam) * x * x
    return float(np.sqrt(h * periods))


def _garch_filter(r2, a, b, v):
    from scipy.signal import lfilter
    x = (v * (1 - a - b)) + a * np.r_[v, r2[:-1]]
    h, _ = lfilter([1.0], [1.0, -b], x, zi=[b * v])
    return h


def garch11(returns):
    """Gaussian quasi-MLE of a variance-targeted GARCH(1,1). -> dict(alpha, beta, persistence,
    long_run_vol, h_next) with h_next the one-day-ahead variance (daily units)."""
    from scipy.optimize import minimize
    r = np.asarray(returns, dtype=float)
    if len(r) < 120:
        raise ValueError("need at least 120 returns for GARCH")
    r = r - r.mean()
    r2, v = r * r, float(np.mean(r * r))

    def nll(p):
        a, b = p
        if a <= 0 or b < 0 or a + b >= 0.999:
            return 1e12
        h = _garch_filter(r2, a, b, v)
        return 0.5 * float(np.sum(np.log(h) + r2 / h))

    best = min((minimize(nll, x0, method="Nelder-Mead", options=dict(xatol=1e-5, fatol=1e-6, maxiter=400))
                for x0 in ((0.05, 0.90), (0.10, 0.80), (0.03, 0.95))), key=lambda o: o.fun)
    a, b = map(float, best.x)
    if not (a > 0 and b >= 0 and a + b < 0.999):
        raise ValueError("GARCH fit did not converge to a stationary model")
    h = _garch_filter(r2, a, b, v)
    h_next = v * (1 - a - b) + a * r2[-1] + b * h[-1]
    return dict(alpha=a, beta=b, persistence=a + b, long_run_vol=float(np.sqrt(v * PERIODS)), h_next=float(h_next),
                var_target=v)


def garch_term_vol(fit, n_days, periods=PERIODS):
    """Annualized average vol over the next n_days trading days."""
    p, vL, h1 = fit["persistence"], fit["var_target"], fit["h_next"]
    n = max(int(n_days), 1)
    avg = vL + (h1 - vL) * (1 - p ** n) / ((1 - p) * n)
    return float(np.sqrt(max(avg, 0.0) * periods))


def har_vol(returns, n_days, periods=PERIODS):
    """Direct HAR forecast of the mean squared return over the next n_days, annualized."""
    r2 = np.asarray(returns, dtype=float) ** 2
    h = max(int(n_days), 1)
    c = np.r_[0.0, np.cumsum(r2)]
    t = np.arange(21, len(r2))                     # regressors need 22 days of history
    d, w, m = r2[t], (c[t + 1] - c[t - 4]) / 5, (c[t + 1] - c[t - 21]) / 22
    fit_t = t[t + h < len(r2)]
    if len(fit_t) < 60:
        raise ValueError("history too short for the HAR regression")
    k = fit_t - 21
    X = np.c_[np.ones(len(k)), d[k], w[k], m[k]]
    y = (c[fit_t + 1 + h] - c[fit_t + 1]) / h
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    f = float(np.r_[1.0, d[-1], w[-1], m[-1]] @ beta)
    if not f > 0:
        raise ValueError("HAR forecast is not positive")
    return float(np.sqrt(f * periods))


def earnings_reactions(dates, closes, past):
    """Close-to-close reaction to each past report. past: [{date, timing: 'bmo'|'amc'|None}].
    bmo -> return into the report day; amc -> return into the next session; unknown -> the larger of the
    two. -> [dict(date, timing, move, idx)] with idx the index (into the return series) of that day."""
    ds = [str(x) for x in dates]
    r = log_returns(closes)
    out = []
    for e in past:
        i = int(np.searchsorted(ds, e["date"]))            # first session on/after the report date
        if i >= len(ds):
            continue
        same = ds[i] == e["date"]
        bmo_idx = i - 1                                    # return index: r[j] = ln(C[j+1] / C[j])
        amc_idx = i if same else i - 1                     # report on a non-session day reacts at session i
        tm = e.get("timing")
        cands = [j for j in ((bmo_idx,) if tm == "bmo" else (amc_idx,) if tm == "amc" else (bmo_idx, amc_idx))
                 if 0 <= j < len(r)]
        if not cands:
            continue
        j = max(cands, key=lambda j: abs(r[j]))
        out.append(dict(date=e["date"], timing=tm, move=float(abs(r[j])), idx=j))
    return out


def sigma_suggestion(dates, closes, past_earnings, upcoming_earnings, asof, expiry, iv=None):
    """Suggested sigma_hat for the option expiring `expiry`, with every piece shown. Never raises for a
    single failed model: what fails is reported and left out of the blend."""
    T = max((expiry - asof).days / 365, 1 / 365)
    n_days = max(int(round(T * PERIODS)), 1)
    r = log_returns(closes)
    reacts = earnings_reactions(dates, closes, past_earnings or [])
    cut = {x["idx"] for x in reacts}
    base_r = np.array([x for j, x in enumerate(r) if j not in cut])
    models, errors, garch_info = {}, {}, None
    for name, fn in (("ewma", lambda: ewma_vol(base_r)), ("garch", lambda: garch11(base_r)),
                     ("har", lambda: har_vol(base_r, n_days))):
        try:
            models[name] = fn()
        except (ValueError, np.linalg.LinAlgError) as e:
            errors[name] = str(e)
    if "garch" in models:
        g = models["garch"]
        models["garch"] = garch_term_vol(g, n_days)
        garch_info = dict(persistence=g["persistence"], long_run_vol=g["long_run_vol"],
                          half_life_days=float(np.log(0.5) / np.log(g["persistence"])))
    blend = [models[k] for k in ("garch", "har") if k in models]
    base = float(np.mean(blend)) if blend else models.get("ewma", hv_close_to_close(closes, 252))
    base_src = " + ".join(k.upper() for k in ("garch", "har") if k in models) or ("EWMA" if "ewma" in models else "HV 1y")
    recent = sorted(reacts, key=lambda x: x["date"])[-N_EARNINGS_HISTORY:]
    rms = float(np.sqrt(np.mean([x["move"] ** 2 for x in recent]))) if recent else None
    mean_abs = float(np.mean([x["move"] for x in recent])) if recent else None
    # earnings inside (asof, expiry]: the feed first, else project the quarterly cadence
    window = []
    for e in upcoming_earnings or []:
        d = date.fromisoformat(e["date"])
        if asof < d <= expiry:
            window.append(dict(date=e["date"], source="feed"))
    if not window and recent:
        d = date.fromisoformat(recent[-1]["date"])
        while d <= expiry:
            if d > asof:
                window.append(dict(date=d.isoformat(), source="projected (91-day cadence)"))
            d = date.fromordinal(d.toordinal() + EARNINGS_CADENCE_DAYS)
    ev_var = max(rms ** 2 - base ** 2 / PERIODS, 0.0) if rms else 0.0
    total = base ** 2 * T + ev_var * len(window)
    sug = float(np.sqrt(total / T))
    return dict(
        suggested=sug, base=base, base_source=base_src, T=T, n_days=n_days, models=models, model_errors=errors,
        garch=garch_info, hv1y=hv_close_to_close(closes, 252),
        earnings=dict(n_history=len(recent), rms_move=rms, mean_abs_move=mean_abs,
                      history=[dict(date=x["date"], timing=x["timing"], move=x["move"]) for x in recent],
                      in_window=window, variance_added=ev_var * len(window)),
        iv=iv, edge_vs_iv=(sug - iv) if iv is not None else None)

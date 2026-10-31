"""3-month call spreads against the real listed strikes, and the spanning results that derive them.

Why a spread: the goal objective makes payoff above G worthless, so the target payoff is truncated,
f_T = min(f*, G). Spanning f_T (Carr-Madan) gives f_T'' = +delta(K - K1) - delta(K - K2): buy K1,
sell K2. The short leg is covered by the long leg, so net receipt is always in [0, K2 - K1] and the
max loss is the premium paid.

Strike rule (sigma-normalized, not fixed multiples):
    K1_target = S0 (1 + z sigma_hat sqrt(T_opt)),  z = 1.56
    K1 = lowest listed strike >= K1_target;  K2 = narrowest listed strike above K1 that still meets
    the truncation condition n (K2 - K1) >= target
    n  = B_alloc / ((c1 - c2)(1 + round_trip_cost))
"""
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

Z_STRIKE = 1.56
ROUND_TRIP = 0.04
MULTIPLIER = 100
MIN_OI = 100
MAX_SPREAD_PCT = 0.10


@dataclass
class ChainSlice:
    """One expiry of a call chain. Arrays are aligned by strike."""
    expiry: object
    T: float
    strike: np.ndarray
    bid: np.ndarray
    ask: np.ndarray
    oi: np.ndarray
    iv: Optional[np.ndarray] = None
    last: Optional[np.ndarray] = None
    volume: Optional[np.ndarray] = None

    def __post_init__(self):
        for name in ("strike", "bid", "ask", "oi", "iv", "last", "volume"):
            v = getattr(self, name)
            if v is not None:
                setattr(self, name, np.asarray(v, dtype=float))
        order = np.argsort(self.strike)
        for name in ("strike", "bid", "ask", "oi", "iv", "last", "volume"):
            v = getattr(self, name)
            if v is not None:
                setattr(self, name, v[order])

    @property
    def mid(self):
        return 0.5 * (self.bid + self.ask)

    def quoted(self):
        """Strikes with a two-sided market (bid > 0 and ask >= bid)."""
        return (self.bid > 0) & (self.ask >= self.bid) & np.isfinite(self.bid) & np.isfinite(self.ask)


def spread_payoff(S_T, K1, K2):
    """Per-share payoff at expiry of long K1 / short K2: clip(S_T - K1, 0, K2 - K1)."""
    return np.clip(np.asarray(S_T, dtype=float) - K1, 0.0, K2 - K1)


def k1_target(S0, sigma_hat, T_opt=0.25, z=Z_STRIKE):
    return S0 * (1 + z * sigma_hat * np.sqrt(T_opt))


def chain_round_trip_cost(bid1, ask1, bid2, ask2):
    """Round-trip cost as a fraction of the mid premium: cross half the spread on both legs to enter
    and again to exit (exit spreads assumed equal to entry)."""
    mid1, mid2 = 0.5 * (bid1 + ask1), 0.5 * (bid2 + ask2)
    net = mid1 - mid2
    if net <= 0:
        return float("nan")
    return float(2 * ((ask1 - mid1) + (mid2 - bid2)) / net)


@dataclass
class Spread:
    expiry: object
    T: float
    S0: float
    sigma_hat: float
    K1_target: float
    K1: float
    K2: float
    c1: dict                 # bid / ask / mid / oi / spread_pct
    c2: dict
    net_mid: float           # c1 - c2 at mid, per share
    rt_cost: float           # round-trip cost applied (fraction of premium)
    rt_cost_chain: float     # chain-implied round-trip cost, for comparison
    net_cost: float          # per share incl. round-trip cost = net_mid (1 + rt_cost)
    budget_krw: float
    fx: float
    n: float                 # contracts (fractional; the simulator scales capital continuously)
    n_int: int               # contracts you can actually buy
    max_payoff_usd: float
    max_payoff_krw: float
    payoff_multiple: float   # max payoff / budget = (K2 - K1) / net_cost
    breakeven: float
    breakeven_pct: float
    exercise_cash_usd: float  # pin risk: n K1 multiplier if only the long leg is exercised
    exercise_cash_krw: float
    exercise_to_premium: float
    target_krw: Optional[float]
    target_ratio: float      # n (K2 - K1) in KRW / target (nan if no target)
    target_ok: bool
    liquidity: dict
    sigma_distance: float    # (K1/S0 - 1) / (sigma_hat sqrt(T_chain)): where K1 landed vs the real expiry
    notes: list = field(default_factory=list)
    k2_rule: str = "narrowest"   # narrowest (spec) | cost-aware | candidate

    @property
    def k1(self):
        return self.K1 / self.S0

    @property
    def k2(self):
        return self.K2 / self.S0

    @property
    def cost_rel(self):
        return self.net_cost / self.S0

    @property
    def liquidity_ok(self):
        return all(self.liquidity[k] for k in ("oi_k1", "oi_k2", "spread_k1", "spread_k2"))


def _leg(chain, i):
    bid, ask = float(chain.bid[i]), float(chain.ask[i])
    mid = 0.5 * (bid + ask)
    return dict(strike=float(chain.strike[i]), bid=bid, ask=ask, mid=mid, oi=float(chain.oi[i]),
                spread_pct=(ask - bid) / mid if mid > 0 else float("inf"),
                iv=None if chain.iv is None else float(chain.iv[i]))


def _effective_rt(leg1, leg2, rt_cost, cost_mode, leg_cross):
    chain_rt = chain_round_trip_cost(leg1["bid"], leg1["ask"], leg2["bid"], leg2["ask"])
    if cost_mode == "flat":
        return rt_cost, chain_rt
    if cost_mode == "chain":
        if not np.isfinite(chain_rt):
            raise ValueError("chain cost undefined (non-positive net premium)")
        return leg_cross * chain_rt, chain_rt
    raise ValueError(f"unknown cost_mode {cost_mode!r}")


def _assemble(chain, S0, sigma_hat, Kt, leg1, leg2, budget_krw, fx, rt, chain_rt, target_krw, multiplier,
              min_oi, max_spread_pct, rule, notes):
    net_mid = leg1["mid"] - leg2["mid"]
    net_cost = net_mid * (1 + rt)
    n = budget_krw / (net_cost * multiplier * fx)
    K1, K2 = leg1["strike"], leg2["strike"]
    maxpay_usd = n * (K2 - K1) * multiplier
    ratio = maxpay_usd * fx / target_krw if target_krw else float("nan")
    be = K1 + net_cost
    ex_usd = n * K1 * multiplier
    liq = dict(oi_k1=leg1["oi"] >= min_oi, oi_k2=leg2["oi"] >= min_oi,
               spread_k1=leg1["spread_pct"] <= max_spread_pct, spread_k2=leg2["spread_pct"] <= max_spread_pct,
               min_oi=min_oi, max_spread_pct=max_spread_pct)
    return Spread(expiry=chain.expiry, T=chain.T, S0=S0, sigma_hat=sigma_hat, K1_target=float(Kt), K1=K1, K2=K2,
                  c1=leg1, c2=leg2, net_mid=net_mid, rt_cost=rt, rt_cost_chain=chain_rt,
                  net_cost=net_cost, budget_krw=budget_krw, fx=fx, n=n, n_int=int(np.floor(n)),
                  max_payoff_usd=maxpay_usd, max_payoff_krw=maxpay_usd * fx,
                  payoff_multiple=(K2 - K1) / net_cost, breakeven=be, breakeven_pct=be / S0 - 1,
                  exercise_cash_usd=ex_usd, exercise_cash_krw=ex_usd * fx, exercise_to_premium=K1 / net_cost,
                  target_krw=target_krw, target_ratio=ratio, target_ok=(not target_krw) or ratio >= 1.0,
                  liquidity=liq, sigma_distance=(K1 / S0 - 1) / (sigma_hat * np.sqrt(chain.T)), notes=notes,
                  k2_rule=rule)


def _k1_leg(chain, S0, sigma_hat, T_opt, z):
    ok = chain.quoted()
    idx = np.flatnonzero(ok)
    if len(idx) < 2:
        raise ValueError("fewer than two quoted strikes in the chain")
    Kt = k1_target(S0, sigma_hat, T_opt, z)
    above = idx[chain.strike[idx] >= Kt - 1e-9]
    if len(above) < 2:
        raise ValueError(f"no listed strike pair at or above K1_target={Kt:.2f}")
    return Kt, above


def build_spread(chain, S0, sigma_hat, budget_krw, fx, rt_cost=ROUND_TRIP, T_opt=0.25, z=Z_STRIKE,
                 target_krw=None, multiplier=MULTIPLIER, min_oi=MIN_OI, max_spread_pct=MAX_SPREAD_PCT,
                 cost_mode="flat", leg_cross=1.0):
    """Spec rule: K1 = lowest listed strike >= K1_target, K2 = narrowest listed strike above it that meets
    the truncation condition. K1_target uses the spec's T_opt = 0.25 even though the listed monthly is
    rarely exactly 3 months out; sigma_distance reports where K1 landed against the ACTUAL expiry.
    cost_mode 'flat' charges rt_cost on the premium; 'chain' charges leg_cross x the chain-implied cost."""
    Kt, above = _k1_leg(chain, S0, sigma_hat, T_opt, z)
    leg1 = _leg(chain, above[0])
    notes = []
    chosen = None
    for i2 in above[1:]:
        leg2 = _leg(chain, i2)
        if leg1["mid"] - leg2["mid"] <= 0:
            notes.append(f"K2={chain.strike[i2]:g}: non-positive net premium, skipped")
            continue
        rt, crt = _effective_rt(leg1, leg2, rt_cost, cost_mode, leg_cross)
        sp = _assemble(chain, S0, sigma_hat, Kt, leg1, leg2, budget_krw, fx, rt, crt, target_krw, multiplier,
                       min_oi, max_spread_pct, "narrowest", notes)
        if chosen is None:
            chosen = sp         # narrowest valid spread, kept if nothing meets the target
        if target_krw is None or sp.target_ratio >= 1.0:
            chosen = sp
            break
    if chosen is None:
        raise ValueError("no strike above K1 gives a positive net premium")
    if target_krw and chosen.target_ratio < 1.0:
        notes.append("no listed K2 meets the truncation condition n(K2-K1) >= target; narrowest kept")
    return chosen


def candidate_spreads(chain, S0, sigma_hat, budget_krw, fx, rt_cost=ROUND_TRIP, T_opt=0.25, z=Z_STRIKE,
                      target_krw=None, multiplier=MULTIPLIER, min_oi=MIN_OI, max_spread_pct=MAX_SPREAD_PCT,
                      cost_mode="chain", leg_cross=1.0, max_width_pct=0.60):
    """Every listed K2 above the spec K1 (up to K1 (1 + max_width_pct)), each priced with its own cost."""
    Kt, above = _k1_leg(chain, S0, sigma_hat, T_opt, z)
    leg1 = _leg(chain, above[0])
    out = []
    for i2 in above[1:]:
        leg2 = _leg(chain, i2)
        if leg2["strike"] > leg1["strike"] * (1 + max_width_pct):
            break
        if leg1["mid"] - leg2["mid"] <= 0:
            continue
        rt, crt = _effective_rt(leg1, leg2, rt_cost, cost_mode, leg_cross)
        out.append(_assemble(chain, S0, sigma_hat, Kt, leg1, leg2, budget_krw, fx, rt, crt, target_krw,
                             multiplier, min_oi, max_spread_pct, "candidate", []))
    if not out:
        raise ValueError("no strike above K1 gives a positive net premium")
    return out


# ---------------------------------------------------------------- expected return and its sources

def expected_multiple(S, dens, w, K1, K2, net_cost):
    """E[payoff] / premium paid under the density `dens` on grid S (trapezoid weights w)."""
    return float(np.sum(w * dens * spread_payoff(S, K1, K2)) / net_cost)


def return_decomposition(md, sigma_hat, mu_hat, conf, K1, K2, net_cost):
    """Split E^p[multiple] - E^q[multiple] into the volatility view and the drift view.

    vol   : p tilted to sigma_used with the market's mean kept       minus q
    drift : p tilted to sigma_used AND mu_hat                        minus the vol-only p
    Drift does not move the option price (it cancels in Black-Scholes) but it moves the payoff,
    so it matters for P&L without being separately observable in the quote.
    """
    from .density import subjective_density
    S, w = md.S, md.weights
    e_q = expected_multiple(S, md.q, w, K1, K2, net_cost)
    p_vol = subjective_density(md, sigma_hat, conf, mu=None)
    p_all = subjective_density(md, sigma_hat, conf, mu=mu_hat)
    e_vol = expected_multiple(S, p_vol.p, w, K1, K2, net_cost)
    e_all = expected_multiple(S, p_all.p, w, K1, K2, net_cost)
    return dict(e_q=e_q, e_vol_only=e_vol, e_total=e_all, vol_contribution=e_vol - e_q,
                drift_contribution=e_all - e_vol, kl=p_all.kl, sigma_used=p_all.sigma_used)


# ---------------------------------------------------------------- spanning

def span_weights(f, strikes):
    """Discrete Carr-Madan with expansion point 0:  f(S) ~ f(K_0) + f'(K_0)(S - K_0) + sum_i w_i (S - K_i)^+,
    w_i = slope change of the piecewise-linear interpolant of f at K_i (the bucketed f''(K) dK).

    Returns (cash, slope0, weights) with weights aligned to strikes. Replication is exact on [K_0, K_m]
    for any payoff that is linear between listed strikes, and the error on smooth payoffs is O(dK^2).
    """
    K = np.asarray(strikes, dtype=float)
    fv = np.array([f(k) for k in K]) if callable(f) else np.asarray(f, dtype=float)
    slopes = np.diff(fv) / np.diff(K)
    w = np.zeros_like(K)
    w[1:-1] = np.diff(slopes)
    return float(fv[0]), float(slopes[0]), w


def span_eval(S, K0, cash, slope0, strikes, weights):
    S = np.asarray(S, dtype=float)
    calls = np.maximum(S[:, None] - np.asarray(strikes)[None, :], 0.0)
    return cash + slope0 * (S - K0) + calls @ weights


def project_payoff(f_vals, S_grid, strikes, include_bond=True):
    """Orthogonal projection of a target payoff onto the span of the listed calls (and a bond).

    A_ij = max(S_i - K_j, 0);  w* = argmin ||f - A w||  <=>  A^T (f - A w*) = 0.
    Returns (weights, bond, residual_rms) where residual = ||(I - P) f|| / sqrt(len(S)).
    """
    S = np.asarray(S_grid, dtype=float)
    f = np.asarray(f_vals, dtype=float)
    A = np.maximum(S[:, None] - np.asarray(strikes, dtype=float)[None, :], 0.0)
    if include_bond:
        A = np.hstack([np.ones((len(S), 1)), A])
    w, *_ = np.linalg.lstsq(A, f, rcond=None)
    resid = f - A @ w
    if include_bond:
        return w[1:], float(w[0]), float(np.linalg.norm(resid) / np.sqrt(len(S)))
    return w, 0.0, float(np.linalg.norm(resid) / np.sqrt(len(S)))

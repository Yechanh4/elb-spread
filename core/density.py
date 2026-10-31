"""Market density q via Breeden-Litzenberger, subjective density p via min-relative-entropy tilting.

    q(K) = e^{rT} d2C/dK2      on C(K) rebuilt from an arbitrage-free SVI smile on a dense grid
    p(S) = q(S) exp(lam . phi(S)) / Z      min KL(p||q) s.t. the analyst's moments
    sigma_used = conf * sigma_hat + (1 - conf) * sigma_IV_atm      (shrinkage before tilting)

All densities live on one uniform price grid S; integrals use trapezoid weights.
Ported from ~/callopt-sim (validated there on live yfinance chains); see README for the extra checks.
"""
from dataclasses import dataclass, field
from typing import List

import numpy as np
from scipy.special import logsumexp

from .bsm import bs_call, bs_vega, forward, implied_vol
from .svi import SVIFit, butterfly_g, fit_svi, svi_iv, svi_w

# Sanity checks on q. The spec's mandatory one is MEAN; the others close its blind spots:
#  - On a grid symmetric in log-moneyness around F, a lognormal truncated at +-a and renormalized has
#    mean EXACTLY F (the S-weighting just mirrors the window), so MEAN cannot see tail truncation.  MASS can.
#  - C(K) is rebuilt with the r and delta you pass, so a wrong dividend input is absorbed into the smile
#    and MEAN stays ~0.  Repricing the chain (REPRICE) and counting quotes that admit no IV (BAD_QUOTES)
#    expose it. A single stale quote has no IV and is excluded from the fit and from REPRICE.
MEAN_TOL = 0.01      # |E^q[S]/F - 1|
MASS_TOL = 0.01      # |integral of e^{rT} C'' before renormalizing - 1|
NEG_TOL = 0.01       # mass removed by clipping q < 0 (butterfly arbitrage)
REPRICE_TOL = 0.005  # no spreads given: max |C_fit - C_mkt| over valid quotes, as a fraction of S0
BAND_TOL = 1.0       # spreads given: 90th pct of |C_fit - mid| / half-spread  (<= 1: fit inside the quote)
BAD_QUOTE_TOL = 0.10  # share of listed quotes outside no-arbitrage bounds (no IV exists): stale or wrong inputs
TAIL_TOL = 1e-5      # grid is widened until each tail outside it holds less than this


class DensityExtractionError(RuntimeError):
    pass


def trap_weights(S):
    d = np.diff(S)
    w = np.empty_like(S)
    w[0], w[-1] = d[0] / 2, d[-1] / 2
    w[1:-1] = (d[:-1] + d[1:]) / 2
    return w


# ---------------------------------------------------------------- Step 3: q(S)

@dataclass
class MarketDensity:
    S: np.ndarray            # dense price grid
    q: np.ndarray            # risk-neutral density on S (>= 0, integrates to 1)
    C: np.ndarray            # call prices rebuilt from the fitted smile
    neg_dC: np.ndarray       # -dC/dK  (= e^{-rT} Q(S_T > K))
    d2C: np.ndarray          # d2C/dK2 before clipping
    S0: float
    r: float
    delta: float
    T: float
    F: float
    svi: SVIFit
    iv_atm: float            # ATM-forward implied vol from the fit
    mean: float              # E^q[S_T]
    mean_rel_err: float      # E^q[S_T]/F - 1
    raw_mass: float          # integral of e^{rT} d2C/dK2 before clip/renormalize
    neg_mass: float          # mass removed by clipping negatives
    reprice_err: float       # max |C_fit - C_mkt| / S0 over listed strikes
    n_bad_iv: int            # listed quotes whose IV could not be inverted (outside no-arb bounds)
    left_atom: float         # Q(S_T < grid start) implied by the smile, kept as mass on the first node
    min_g_grid: float        # butterfly g(k) minimum over the whole grid
    checks: dict             # name -> (value, tolerance, ok)
    passed: bool
    weights: np.ndarray = field(repr=False, default=None)

    def assert_sane(self):
        if not self.passed:
            bad = ", ".join(f"{k}={v:.3g} (tol {tol:.3g})" for k, (v, tol, ok) in self.checks.items() if not ok)
            raise DensityExtractionError(f"q extraction failed sanity check: {bad}. Do not proceed.")
        return self

    def interp_call(self, K):
        return np.interp(K, self.S, self.C)


def clean_chain(K, bid, ask, oi, r, T, min_oi=1):
    """Drop quotes that should not shape the smile, before fitting.

    - open interest below min_oi: an indicative quote nobody trades (often stale)
    - adjacent call spreads that break no-arbitrage even at the most favourable side of the quotes:
        ask(K1) < bid(K2)                    (call price rising with strike)
        bid(K1) - ask(K2) > (K2-K1) e^{-rT}  (falling faster than a bond paying the strike gap)
      greedily removing the quote in the most violations (ties: lower OI, then deeper ITM)
    Returns (keep mask, {strike: reason}).
    """
    K, bid, ask, oi = (np.asarray(v, dtype=float) for v in (K, bid, ask, oi))
    keep = np.ones(len(K), dtype=bool)
    why = {}
    for i in np.flatnonzero(oi < min_oi):
        keep[i] = False
        why[float(K[i])] = "open interest 0"
    disc = np.exp(-r * T)
    while True:
        idx = np.flatnonzero(keep)
        if len(idx) < 3:
            break
        a, b = idx[:-1], idx[1:]
        bad = (ask[a] < bid[b]) | (bid[a] - ask[b] > (K[b] - K[a]) * disc + 1e-9)
        if not bad.any():
            break
        count = np.zeros(len(K))
        np.add.at(count, a[bad], 1)
        np.add.at(count, b[bad], 1)
        cand = np.flatnonzero(count == count.max())
        j = min(cand, key=lambda i: (oi[i], K[i]))
        keep[j] = False
        why[float(K[j])] = "call-spread no-arbitrage violation"
    return keep, why


def _tail_masses(S0, r, delta, T, F, params, lo, hi):
    """Q(S_T < lo) and Q(S_T > hi) from the smile: e^{rT} * (-dC/dK) at the edges."""
    def surv(K):
        h = 1e-4 * K
        Ks = np.array([K - h, K + h])
        c = bs_call(S0, Ks, T, r, delta, svi_iv(np.log(Ks / F), params, T))
        return np.exp(r * T) * -(c[1] - c[0]) / (2 * h)
    return 1 - surv(lo), surv(hi)


def extract_q(strikes, S0, r, delta, T, call_prices=None, ivs=None, width=6.0, h_rel=0.002,
              max_grid=40001, half_spreads=None, max_width=12.0, left_atom=True):
    """Breeden-Litzenberger: q(K) = e^{rT} d2C/dK2 on an arbitrage-free SVI-smoothed C(K).

    SVI is fitted in IV space with weights (vega / price uncertainty)^2: a deep-ITM call has almost no
    vega, so a cent of quote noise there is tens of vol points and must not steer the smile.
    Price uncertainty = half the bid-ask when given, else uniform.
    Grid: uniform in K with spacing h_rel*F, starting at +-width ATM standard deviations and widened
    on each side (up to max_width) until the tail beyond it holds < TAIL_TOL.
    left_atom: whatever left-tail mass the smile still puts below the grid (a steep put wing implies a
    real chance of S_T ~ 0) is kept as mass on the first node instead of being renormalized away.
    """
    strikes = np.asarray(strikes, dtype=float)
    if len(strikes) < 6:
        raise DensityExtractionError(f"only {len(strikes)} usable quotes; need >= 6 to fit a smile")
    F = forward(S0, r, delta, T)
    if ivs is None:
        ivs = implied_vol(call_prices, S0, strikes, T, r, delta)
    ivs = np.asarray(ivs, dtype=float)
    ok = np.isfinite(ivs)
    k_mkt = np.log(strikes[ok] / F)

    # 1. smooth arbitrage-free smile
    vega = bs_vega(S0, strikes[ok], T, r, delta, ivs[ok])
    unc = np.ones(ok.sum()) if half_spreads is None else np.maximum(np.asarray(half_spreads, float)[ok], 1e-4)
    fit = fit_svi(k_mkt, ivs[ok], T, weights=(vega / unc) ** 2)
    s_atm = float(np.sqrt(svi_w(0.0, fit.params)))

    # 2. dense uniform strike grid, rebuild C(K)
    wl = wr = width
    while True:
        lo_m, hi_m = _tail_masses(S0, r, delta, T, F, fit.params, F * np.exp(-wl * s_atm), F * np.exp(wr * s_atm))
        grow_l, grow_r = lo_m >= TAIL_TOL and wl < max_width, hi_m >= TAIL_TOL and wr < max_width
        if not (grow_l or grow_r):
            break
        wl += 1.0 if grow_l else 0.0
        wr += 1.0 if grow_r else 0.0
    lo, hi = F * np.exp(-wl * s_atm), F * np.exp(wr * s_atm)
    n_grid = int(min(max_grid, np.ceil((hi - lo) / (h_rel * F)) + 1))
    S = np.linspace(lo, hi, n_grid)
    h = S[1] - S[0]
    k_grid = np.log(S / F)
    C = bs_call(S0, S, T, r, delta, svi_iv(k_grid, fit.params, T))

    # 3. central differences
    d2C = np.zeros_like(S)
    d2C[1:-1] = (C[2:] - 2 * C[1:-1] + C[:-2]) / (h * h)
    neg_dC = np.empty_like(S)
    neg_dC[1:-1] = -(C[2:] - C[:-2]) / (2 * h)
    neg_dC[0], neg_dC[-1] = -(C[1] - C[0]) / h, -(C[-1] - C[-2]) / h

    w = trap_weights(S)
    q_raw = np.exp(r * T) * d2C
    atom = 0.0
    if left_atom:
        atom = max(float(_tail_masses(S0, r, delta, T, F, fit.params, S[1], hi)[0]), 0.0)
        q_raw[0] += atom / w[0]
    raw_mass = float(np.sum(w * q_raw))
    neg_mass = float(-np.sum(w * np.minimum(q_raw, 0.0)))

    # 4. clip and renormalize
    q = np.maximum(q_raw, 0.0)
    q /= np.sum(w * q)

    mean = float(np.sum(w * S * q))
    rel = mean / F - 1
    c_mkt = (np.asarray(call_prices, dtype=float) if call_prices is not None
             else bs_call(S0, strikes, T, r, delta, ivs))
    c_fit = bs_call(S0, strikes, T, r, delta, svi_iv(np.log(strikes / F), fit.params, T))
    bad_share = float((~ok).mean())
    if half_spreads is not None:
        hs = np.maximum(np.asarray(half_spreads, dtype=float), np.maximum(0.005 * c_mkt, 0.01))
        reprice = float(np.percentile((np.abs(c_fit - c_mkt) / hs)[ok], 90))
        reprice_tol = BAND_TOL
    else:
        reprice = float(np.max(np.abs(c_fit - c_mkt)[ok]) / S0)
        reprice_tol = REPRICE_TOL
    checks = {
        "mean": (abs(rel), MEAN_TOL, abs(rel) < MEAN_TOL),
        "mass": (abs(raw_mass - 1), MASS_TOL, abs(raw_mass - 1) < MASS_TOL),
        "neg": (neg_mass, NEG_TOL, neg_mass < NEG_TOL),
        "reprice": (reprice, reprice_tol, reprice < reprice_tol),
        "bad_quotes": (bad_share, BAD_QUOTE_TOL, bad_share < BAD_QUOTE_TOL),
    }
    return MarketDensity(S=S, q=q, C=C, neg_dC=neg_dC, d2C=d2C, S0=S0, r=r, delta=delta, T=T, F=F,
                         svi=fit, iv_atm=s_atm / np.sqrt(T), mean=mean, mean_rel_err=rel,
                         raw_mass=raw_mass, neg_mass=neg_mass, reprice_err=reprice,
                         n_bad_iv=int((~ok).sum()), left_atom=atom, min_g_grid=float(butterfly_g(k_grid, fit.params).min()),
                         checks=checks, passed=all(v[2] for v in checks.values()), weights=w)


# ---------------------------------------------------------------- Step 4: p(S)

@dataclass
class Tilt:
    """p = q * exp(lam . phi(S) - logZ), the min-KL density subject to E^p[phi] = targets."""
    p: np.ndarray
    ell: np.ndarray          # p/q, exact (no division): exp(lam.phi - logZ)
    lam: np.ndarray
    logZ: float
    targets: np.ndarray
    achieved: np.ndarray
    kl: float
    converged: bool
    iters: int
    sigma_used: float
    mean_x: float            # E^p[ln(S/S0)] (reported; the constraints are on R = S/S0)
    edge_mass: float         # p-mass in the outer 1% of the grid on either side


def exp_tilt(q, w, Phi, c, tol=1e-11, max_iter=100):
    """Minimize KL(p||q) s.t. E^p[Phi] = c via Newton on the convex dual
        L(lam) = log E^q[exp(lam.Phi)] - lam.c,   grad = E^lam[Phi] - c,  Hess = Cov^lam[Phi].
    Phi: (n_features, n_grid). Returns (lam, logZ, z) with log(p/q) = z - logZ.
    """
    Phi = np.atleast_2d(Phi)
    c = np.atleast_1d(np.asarray(c, dtype=float))
    support = q > 0
    logqw = np.full(q.shape, -np.inf)
    logqw[support] = np.log(q[support] * w[support])
    lam = np.zeros(Phi.shape[0])

    def state(lam):
        z = lam @ Phi
        lz = logsumexp(logqw + z)
        pi = np.exp(logqw + z - lz)
        return z, lz, pi

    z, lz, pi = state(lam)
    it = 0
    for it in range(1, max_iter + 1):
        m = Phi @ pi
        g = m - c
        if np.max(np.abs(g)) < tol * max(1.0, np.max(np.abs(c))):
            break
        H = (Phi * pi) @ Phi.T - np.outer(m, m)
        step = np.linalg.solve(H + 1e-14 * np.eye(len(lam)), g)
        L0 = lz - lam @ c
        t = 1.0
        while True:
            lam_new = lam - t * step
            z_n, lz_n, pi_n = state(lam_new)
            if lz_n - lam_new @ c <= L0 - 1e-4 * t * (g @ step) + 1e-13 * (1 + abs(L0)) or t < 1e-12:
                break
            t *= 0.5
        if t < 1e-12:      # no further decrease possible: at the optimum to float precision
            break
        lam, z, lz, pi = lam_new, z_n, lz_n, pi_n
    m = Phi @ pi
    return lam, lz, z, m, bool(np.max(np.abs(m - c)) < 1e-8 * max(1.0, np.max(np.abs(c)))), it


def shrink_sigma(sigma_hat, conf, sigma_iv_atm):
    return conf * sigma_hat + (1 - conf) * sigma_iv_atm


def subjective_density(md, sigma_hat, conf=1.0, mu=None):
    """Min-KL tilt of q to the (shrunk) analyst view, with moments in PRICE space, R = S_T/S0:

        p = q exp(lam1 R + lam2 R^2) / Z       over the continuous part of q
        E^p[R]   = E^q[R]                       no drift view: the expected price stays at the forward
                 = e^{mu T}                     mu = ex-dividend price drift view
        Var^p[R] = Var^q[R] * (e^{sigma_used^2 T} - 1) / (e^{sigma_IV^2 T} - 1)
                   (lognormal variance ratio; scales the market's OWN variance, so no view <=> p = q)

    Why price space and not log returns: the market's log-variance is dominated by the crash tail (on
    live AMD, sd of ln R is ~1.8x ATM IV with skew -4), so "more variance" via exp(lam x^2) is cheapest
    to deliver by fattening the crash tail — a volatility view turned into a crash view, and the right
    tail actually thinned. In price space the left tail is bounded (R >= 0), so added variance goes to
    both tails and E[S_T] is held exactly (no hidden drift).
    The left atom (market-implied mass near S_T = 0, kept on the first node) is NOT tilted: sigma_hat is
    a view on ordinary volatility, not on default, so the atom keeps the market's mass (ell = 1 there).
    """
    S, q, w, T = md.S, md.q, md.weights, md.T
    sigma_used = shrink_sigma(sigma_hat, conf, md.iv_atm)
    R = S / md.S0
    frozen = (getattr(md, "left_atom", 0.0) or 0.0) > 1e-12
    qc = q.copy()
    atom = 0.0
    if frozen:
        atom = float(w[0] * q[0])
        qc[0] = 0.0
    qc = qc / np.sum(w * qc)
    mean_q = float(np.sum(w * qc * R))
    var_q = float(np.sum(w * qc * (R - mean_q) ** 2))
    var_t = var_q * np.expm1(sigma_used**2 * T) / np.expm1(md.iv_atm**2 * T)
    if mu is None:
        mean_t = mean_q
    else:
        mean_t = (np.exp(mu * T) - atom * R[0]) / (1 - atom)      # total E^p[R] = e^{mu T}
    Phi = np.vstack([R, R * R])
    c = np.array([mean_t, var_t + mean_t**2])
    lam, logZ, z, m, conv, it = exp_tilt(qc, w, Phi, c)
    ell = np.exp(z - logZ)
    if frozen:
        ell[0] = 1.0
    p = q * ell
    pos = (p > 0) & (ell != 1.0) if frozen else p > 0
    kl = float(np.sum(w[pos] * p[pos] * np.log(ell[pos])))
    n_edge = max(1, len(S) // 100)
    edge = float(np.sum((w * p)[1 if frozen else 0:n_edge]) + np.sum((w * p)[-n_edge:]))
    mean_x = float(np.sum(w * p * np.log(np.maximum(R, 1e-300))))
    return Tilt(p=p, ell=ell, lam=lam, logZ=float(logZ), targets=c, achieved=m, kl=kl, converged=conv,
                iters=it, sigma_used=float(sigma_used), mean_x=mean_x, edge_mass=edge)


@dataclass
class Mixture:
    S: np.ndarray
    q: np.ndarray
    p: np.ndarray
    ell: np.ndarray          # p_mix / q = sum_s pi_s ell_s  (exact)
    probs: np.ndarray
    components: List[Tilt]
    kl: float                # KL(p_mix || q)
    weights: np.ndarray = field(repr=False, default=None)


def mixture(md, tilts, probs):
    probs = np.asarray(probs, dtype=float)
    if np.any(probs < 0) or abs(probs.sum() - 1) > 1e-9:
        raise ValueError("scenario probabilities must be >= 0 and sum to 1")
    ell = sum(pi * t.ell for pi, t in zip(probs, tilts))
    p = md.q * ell
    w = md.weights
    pos = p > 0
    kl = float(np.sum(w[pos] * p[pos] * np.log(ell[pos])))
    return Mixture(S=md.S, q=md.q, p=p, ell=ell, probs=probs, components=list(tilts), kl=kl, weights=w)


def kl_level(kl):
    """Overfitting gauge: <0.1 humble, 0.1-0.3 ok, 0.3-0.5 amber, >0.5 alarm."""
    if kl < 0.1:
        return "humble"
    if kl < 0.3:
        return "ok"
    if kl < 0.5:
        return "amber"
    return "alarm"

"""Likelihood ratio, edge, and the Neyman-Pearson buy region.

    ell(S) = p(S) / q(S)
    max_A P^p(A)  s.t.  e^{-rT} * H * Q(A) <= B      (pay H on A)
    =>  A* = {S : ell(S) > c},  c found by bisection so the budget binds.

The lower bound of A* (restricted to the call side, S >= F) is the region-implied K1. The spread
module uses the sigma-normalized rule for K1 by default and reports this one beside it.
"""
from dataclasses import dataclass

import numpy as np


def edge(sigma_hat, sigma_iv_atm):
    """Edge = sigma_hat - sigma_IV_atm. Only Edge > 0 is eligible: in Black-Scholes the drift cancels
    out of the price, so cheapness can only be argued through volatility."""
    return float(sigma_hat - sigma_iv_atm)


def eligible(sigma_hat, sigma_iv_atm):
    return edge(sigma_hat, sigma_iv_atm) > 0


def likelihood_ratio(tilt):
    return tilt.ell


@dataclass
class BuyRegion:
    c: float
    mask: np.ndarray
    K_lo: float              # lower bound of the region (region-implied K1)
    P_p: float               # P^p(A)
    Q_q: float               # Q(A)
    cost: float              # e^{-rT} * H * Q(A)
    binding: bool            # False if even the whole allowed region costs less than the budget
    boundary_fraction: float = 0.0   # share of the next-best grid node bought so the budget binds exactly
    boundary_S: float = float("nan")


def call_region(S, F):
    """Calls can only buy S >= F. A variance view makes ell U-shaped (both tails up); the left tail is
    not purchasable with calls, so restrict A* to the right wing."""
    return np.asarray(S) >= F


def buy_region(S, q, ell, w, budget, payout, r, T, allowed=None, tol=1e-12, max_iter=200):
    """Bisection on c so that e^{-rT} * payout * Q(ell > c, allowed) = budget."""
    S, q, ell, w = (np.asarray(v, dtype=float) for v in (S, q, ell, w))
    allowed = np.ones_like(S, dtype=bool) if allowed is None else np.asarray(allowed, dtype=bool)
    disc = np.exp(-r * T)

    def cost_at(c):
        m = allowed & (ell > c)
        return disc * payout * float(np.sum(w[m] * q[m])), m

    full_cost, full = cost_at(-np.inf)
    if full_cost <= budget:
        c, m, binding = float(ell[allowed].min()) - 1e-12, full, False
    else:
        lo, hi = float(ell[allowed].min()) - 1e-12, float(ell[allowed].max())
        for _ in range(max_iter):
            mid = 0.5 * (lo + hi)
            cost, _ = cost_at(mid)
            if cost > budget:
                lo = mid
            else:
                hi = mid
            if hi - lo < tol * max(1.0, abs(hi)):
                break
        c = hi
        _, m = cost_at(c)
        binding = True
    cost, _ = cost_at(c)
    P_p, Q_q = float(np.sum(w[m] * q[m] * ell[m])), float(np.sum(w[m] * q[m]))
    frac, S_b = 0.0, float("nan")
    rest = np.flatnonzero(allowed & ~m & (q > 0))
    if binding and len(rest):
        # Neyman-Pearson on a grid: buy the boundary node fractionally so the budget binds exactly
        j = rest[np.argmax(ell[rest])]
        node_cost = disc * payout * w[j] * q[j]
        frac = float(min(max((budget - cost) / node_cost, 0.0), 1.0))
        P_p += frac * w[j] * q[j] * ell[j]
        Q_q += frac * w[j] * q[j]
        cost += frac * node_cost
        S_b = float(S[j])
    K_lo = float(S[m].min()) if m.any() else float("nan")
    return BuyRegion(c=c, mask=m, K_lo=K_lo, P_p=P_p, Q_q=Q_q, cost=cost, binding=binding,
                     boundary_fraction=frac, boundary_S=S_b)


def optimal_payoff(ell, gamma=1.0):
    """Power-utility optimal payoff shape f*(S) proportional to ell^(1/gamma). gamma=1: growth-optimal."""
    if gamma <= 0:
        raise ValueError("gamma must be > 0")
    return np.asarray(ell, dtype=float) ** (1.0 / gamma)


def truncate(f, cap):
    """f_T = min(f, G): payoff above the goal adds nothing to P(goal), so give it up for premium."""
    return np.minimum(np.asarray(f, dtype=float), cap)

import numpy as np
import pytest

from conftest import make_chain
from core.bsm import bs_call
from core.density import extract_q
from core.spread import (build_spread, chain_round_trip_cost, k1_target, project_payoff, span_eval,
                         span_weights, spread_payoff)

FX = 1400.0
B = 79.7e6


def test_strike_rule_uses_listed_strikes(flat_chain):
    s = build_spread(flat_chain, 100.0, 0.40, B, FX)
    Kt = k1_target(100.0, 0.40, 0.25)
    assert Kt == pytest.approx(100 * (1 + 1.56 * 0.40 * 0.5))
    listed = flat_chain.strike
    assert s.K1 in listed and s.K2 in listed
    assert s.K1 >= Kt and listed[listed >= Kt].min() == s.K1      # nearest listed at or above
    assert s.K2 == listed[listed > s.K1].min()                     # next listed strike
    assert s.sigma_distance >= 1.56


def test_unquoted_strike_is_skipped():
    ch = make_chain()
    Kt = k1_target(100.0, 0.40, 0.25)
    first = np.flatnonzero(ch.strike >= Kt)[0]
    ch.bid[first] = 0.0
    s = build_spread(ch, 100.0, 0.40, B, FX)
    assert s.K1 == ch.strike[first + 1]


def test_net_cost_max_payoff_breakeven(flat_chain):
    rt = 0.04
    s = build_spread(flat_chain, 100.0, 0.40, B, FX, rt_cost=rt)
    assert s.net_mid == pytest.approx(s.c1["mid"] - s.c2["mid"])
    assert s.net_cost == pytest.approx((s.c1["mid"] - s.c2["mid"]) * (1 + rt))
    assert s.n == pytest.approx(B / (s.net_cost * 100 * FX))
    assert s.n * s.net_cost * 100 * FX == pytest.approx(B)          # budget fully spent incl. costs
    assert s.max_payoff_usd == pytest.approx(s.n * (s.K2 - s.K1) * 100)
    assert s.max_payoff_krw == pytest.approx(s.max_payoff_usd * FX)
    assert s.payoff_multiple == pytest.approx(s.max_payoff_krw / B)
    assert s.breakeven == pytest.approx(s.K1 + s.net_cost)
    assert s.breakeven_pct == pytest.approx(s.breakeven / 100 - 1)
    assert s.exercise_cash_usd == pytest.approx(s.n * s.K1 * 100)
    assert s.exercise_to_premium == pytest.approx(s.exercise_cash_krw / B)
    assert s.n_int == int(np.floor(s.n))


def test_chain_implied_round_trip_cost(flat_chain):
    s = build_spread(flat_chain, 100.0, 0.40, B, FX)
    b1, a1, b2, a2 = s.c1["bid"], s.c1["ask"], s.c2["bid"], s.c2["ask"]
    expect = 2 * ((a1 - b1) / 2 + (a2 - b2) / 2) / ((a1 + b1) / 2 - (a2 + b2) / 2)
    assert s.rt_cost_chain == pytest.approx(expect)
    assert chain_round_trip_cost(1.0, 1.0, 0.5, 0.5) == 0.0


def test_truncation_condition_widens_only_when_needed(flat_chain):
    narrow = build_spread(flat_chain, 100.0, 0.40, B, FX)
    listed = flat_chain.strike
    wider_K2 = listed[listed > narrow.K2].min()
    # ratio target sits between the narrow and the next-wider spread's max payoff
    ch = flat_chain
    i1, i3 = np.flatnonzero(listed == narrow.K1)[0], np.flatnonzero(listed == wider_K2)[0]
    net3 = (ch.mid[i1] - ch.mid[i3]) * 1.04
    maxpay3 = B / (net3 * 100 * FX) * (wider_K2 - narrow.K1) * 100 * FX
    assert maxpay3 > narrow.max_payoff_krw
    target = 0.5 * (narrow.max_payoff_krw + maxpay3)
    s = build_spread(flat_chain, 100.0, 0.40, B, FX, target_krw=target)
    assert s.K2 == wider_K2 and s.target_ok and s.target_ratio >= 1
    easy = build_spread(flat_chain, 100.0, 0.40, B, FX, target_krw=0.5 * narrow.max_payoff_krw)
    assert easy.K2 == narrow.K2 and easy.target_ratio == pytest.approx(2.0)
    hard = build_spread(flat_chain, 100.0, 0.40, B, FX, target_krw=1e15)
    assert not hard.target_ok and hard.K2 == narrow.K2 and hard.notes


def test_liquidity_flags():
    ch = make_chain(oi=50, half_spread_frac=0.2)
    s = build_spread(ch, 100.0, 0.40, B, FX)
    assert not s.liquidity["oi_k1"] and not s.liquidity["spread_k1"] and not s.liquidity_ok
    ok = build_spread(make_chain(oi=500, half_spread_frac=0.02, min_half=0.0), 100.0, 0.40, B, FX)
    assert ok.liquidity_ok


def test_spread_payoff_bounded():
    S = np.linspace(0, 300, 3001)
    p = spread_payoff(S, 120.0, 125.0)
    assert p.min() == 0.0 and p.max() == 5.0
    assert np.all(np.diff(p) >= 0)


# ---------------------------------------------------------------- spanning

def test_spanning_truncated_payoff_is_a_call_spread():
    """f_T = min((S - K1)^+, K2 - K1) spans to +1 call at K1 and -1 call at K2: the spread is derived."""
    K = np.arange(80.0, 160.0 + 1e-9, 5.0)
    K1, K2 = 120.0, 125.0
    cash, slope0, w = span_weights(lambda s: min(max(s - K1, 0.0), K2 - K1), K)
    expected = np.zeros_like(K)
    expected[K == K1] = 1.0
    expected[K == K2] = -1.0
    assert cash == 0.0 and slope0 == 0.0
    assert np.allclose(w, expected)


def test_spanning_truncation_creates_short_leg():
    """Any convex payoff capped at G gets a negative weight (a short call) at the cap."""
    K = np.arange(100.0, 200.0 + 1e-9, 2.0)
    G = 30.0
    f = lambda s: min(max(s - 110.0, 0.0) ** 1.5 / 3.0, G)
    _, _, w = span_weights(f, K)
    cap_strike = 110.0 + (3 * G) ** (2 / 3)
    near = np.argmin(np.abs(K - cap_strike))
    assert w[max(near - 1, 0):near + 2].sum() < 0
    assert np.all(w[K < 110.0] == 0)


def test_spanning_error_decreases_with_grid_density():
    f = lambda s: np.sqrt(s) * 10.0
    S = np.linspace(60, 160, 4001)
    errs = []
    for dK in (10.0, 5.0, 2.5, 1.25):
        K = np.arange(60.0, 160.0 + 1e-9, dK)
        cash, slope0, w = span_weights(f, K)
        errs.append(np.max(np.abs(span_eval(S, K[0], cash, slope0, K, w) - f(S))))
    assert all(a > b for a, b in zip(errs, errs[1:]))
    assert errs[0] / errs[-1] > 40          # ~O(dK^2): 8x denser -> ~64x smaller


def test_projection_error_decreases_with_strike_density():
    S = np.linspace(60, 180, 2001)
    target = np.minimum(np.maximum(S - 100, 0) ** 2 / 40, 30.0)
    errs = []
    for n in (4, 8, 16, 32):
        K = np.linspace(60, 180, n + 1)[:-1]
        _, _, e = project_payoff(target, S, K)
        errs.append(e)
    assert all(a > b for a, b in zip(errs, errs[1:]))


def test_projection_residual_orthogonal():
    S = np.linspace(60, 180, 801)
    K = np.array([80.0, 100.0, 120.0, 140.0])
    f = np.minimum(np.maximum(S - 100, 0) ** 2 / 40, 30.0)
    w, bond, _ = project_payoff(f, S, K)
    A = np.hstack([np.ones((len(S), 1)), np.maximum(S[:, None] - K[None, :], 0)])
    resid = f - A @ np.r_[bond, w]
    assert np.allclose(A.T @ resid, 0, atol=1e-6 * np.abs(A.T @ f).max())


def test_return_decomposition_separates_vol_and_drift():
    from core.spread import return_decomposition
    S0, T, r = 100.0, 0.25, 0.04
    K = np.linspace(55, 190, 55)
    md = extract_q(K, S0, r, 0.0, T, call_prices=bs_call(S0, K, T, r, 0.0, 0.35))
    K1, K2 = 125.0, 127.5
    net_mid = float(bs_call(S0, K1, T, r, 0.0, 0.35) - bs_call(S0, K2, T, r, 0.0, 0.35))
    net = net_mid * 1.04
    no_view = return_decomposition(md, md.iv_atm, None, 1.0, K1, K2, net)
    assert no_view["e_q"] == pytest.approx(np.exp(r * T) / 1.04, rel=0.01)   # fair price minus costs
    assert no_view["vol_contribution"] == pytest.approx(0, abs=1e-6)
    view = return_decomposition(md, 0.45, 0.20, 1.0, K1, K2, net)
    assert view["vol_contribution"] > 0
    assert view["drift_contribution"] > 0
    assert view["e_total"] == pytest.approx(view["e_q"] + view["vol_contribution"] + view["drift_contribution"])

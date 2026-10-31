import numpy as np
import pytest

from core.bsm import bs_call
from core.density import extract_q, subjective_density
from core.selection import (buy_region, call_region, edge, eligible, optimal_payoff, truncate)

S0, T, r = 100.0, 0.25, 0.04


@pytest.fixture(scope="module")
def dens():
    K = np.linspace(55, 190, 55)
    md = extract_q(K, S0, r, 0.0, T, call_prices=bs_call(S0, K, T, r, 0.0, 0.35))
    return md, subjective_density(md, 0.45, conf=1.0, mu=0.10)


def test_edge_sign_rule():
    assert edge(0.45, 0.35) == pytest.approx(0.10)
    assert eligible(0.45, 0.35)
    assert not eligible(0.35, 0.35)
    assert not eligible(0.30, 0.35)


def test_threshold_makes_budget_bind(dens):
    md, t = dens
    payout, budget = 1.0, 0.08
    reg = buy_region(md.S, md.q, t.ell, md.weights, budget, payout, r, T, allowed=call_region(md.S, md.F))
    assert reg.binding
    assert reg.cost == pytest.approx(budget, rel=1e-9)
    assert 0.0 <= reg.boundary_fraction <= 1.0
    # a region is "cheap" in the view's eyes: P^p(A) > Q(A), and everything inside beats c
    assert reg.P_p > reg.Q_q
    assert np.all(t.ell[reg.mask] > reg.c)
    assert reg.K_lo >= md.F


def test_bigger_budget_lowers_threshold(dens):
    md, t = dens
    allowed = call_region(md.S, md.F)
    cs = [buy_region(md.S, md.q, t.ell, md.weights, b, 1.0, r, T, allowed=allowed).c for b in (0.03, 0.06, 0.12)]
    assert cs[0] > cs[1] > cs[2]


def test_non_binding_budget(dens):
    md, t = dens
    reg = buy_region(md.S, md.q, t.ell, md.weights, 10.0, 1.0, r, T, allowed=call_region(md.S, md.F))
    assert not reg.binding
    assert reg.mask.sum() == call_region(md.S, md.F).sum()


def test_power_utility_and_truncation():
    ell = np.array([0.5, 1.0, 2.0, 4.0])
    assert np.allclose(optimal_payoff(ell, 1.0), ell)
    assert np.allclose(optimal_payoff(ell, 2.0), np.sqrt(ell))
    assert np.allclose(truncate(ell * 10, 25), [5, 10, 20, 25])
    with pytest.raises(ValueError):
        optimal_payoff(ell, 0)

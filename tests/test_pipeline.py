"""End to end on a synthetic chain: chain -> q -> p -> listed spread -> 32-roll Monte Carlo."""
import numpy as np
import pytest

from conftest import make_chain
from core import budget as bud
from core.density import extract_q, subjective_density
from core.selection import edge
from core.simulate import Marginal, SimConfig, TickerSpec, pin_prob, simulate

S0, T, r = 100.0, 0.25, 0.04
FX = 1450.0


def smile(K):
    k = np.log(K / S0)
    return 0.36 - 0.10 * k + 0.25 * k * k


@pytest.fixture(scope="module")
def chain_and_density():
    ch = make_chain(S0=S0, T=T, r=r, iv_fn=smile, strikes=np.arange(60.0, 180.0 + 1e-9, 2.5))
    keep = ch.quoted()
    md = extract_q(ch.strike[keep], S0, r, 0.0, T, call_prices=ch.mid[keep],
                   half_spreads=0.5 * (ch.ask - ch.bid)[keep])
    return ch, md


def test_pipeline_end_to_end(chain_and_density):
    from core.spread import build_spread
    ch, md = chain_and_density
    md.assert_sane()
    sigma_hat = 0.48
    assert edge(sigma_hat, md.iv_atm) > 0
    tilt = subjective_density(md, sigma_hat, conf=0.8, mu=0.08)
    B = bud.budget()
    sp = build_spread(ch, S0, sigma_hat, B, FX, target_krw=bud.GOAL / bud.N_CYCLES)
    marg = Marginal.from_density(md.S, tilt.p, md.weights, S0)
    tk = TickerSpec.from_spread("SYN", sp, {"*": marg}, sector="Tech")
    assert tk.k1 == pytest.approx(sp.K1 / S0) and tk.cost_rel == pytest.approx(sp.net_cost / S0)
    res = simulate([tk], [1.0], np.eye(1), SimConfig(n_paths=20_000, n_store=50, seed=1))
    m = res.metrics()
    # MC pin frequency matches the density's own probability of finishing in [K1, K2)
    p_pin = pin_prob(marg, tk.k1, tk.k2)
    for run in res.runs:
        assert run.pin_by_ticker[0] == pytest.approx(p_pin, abs=0.002)
    # roll multiple under p matches the closed-form expectation on the grid
    from core.spread import expected_multiple
    e = expected_multiple(md.S, tilt.p, md.weights, sp.K1, sp.K2, sp.net_cost)
    assert m["roll_multiple_mean"] == pytest.approx(e, rel=0.03)
    assert 0 <= m["p_goal"] <= 1 and 0 <= m["ruin_per_cycle"] <= 1
    assert m["budget_total"] > 0 and m["quantiles"][0.99] >= m["median"]

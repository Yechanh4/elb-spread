import numpy as np
import pytest

from conftest import make_chain
from core import budget as bud
from core.simulate import Marginal, SimConfig
from core.spread import build_spread, candidate_spreads
from core.structure import expected_roll_multiple, select_k2

S0, T, B, FX = 100.0, 0.25, 79.7e6, 1400.0


@pytest.fixture(scope="module")
def chain():
    # an absolute minimum half-spread of $0.02 per leg, as on real chains: dominates cheap OTM legs
    return make_chain(S0=S0, sigma=0.45, T=T, half_spread_frac=0.02, min_half=0.02,
                      strikes=np.arange(60.0, 220.0 + 1e-9, 2.5))


def view():
    return Marginal.lognormal(0.10, 0.55, T)


def test_chain_cost_falls_with_width(chain):
    cands = candidate_spreads(chain, S0, 0.45, B, FX)
    rts = [c.rt_cost for c in cands]
    assert rts[0] > 0.4                       # narrowest: the bid-ask is a large share of a tiny premium
    assert rts[-1] < rts[0] / 3
    assert all(c.K1 == cands[0].K1 for c in cands)        # K1 stays on the sigma rule
    assert all(c.rt_cost == pytest.approx(c.rt_cost_chain) for c in cands)


def test_cost_aware_beats_narrowest_on_expected_multiple(chain):
    res = select_k2(chain, S0, 0.45, B, FX, view(), "expected")
    ch, nar = res["chosen"], res["narrowest"]
    assert ch.k2_rule == "cost-aware" and nar.k2_rule == "narrowest"
    assert nar.K2 == build_spread(chain, S0, 0.45, B, FX, cost_mode="chain").K2
    assert ch.K2 > nar.K2 and ch.rt_cost < nar.rt_cost
    scores = [r["score"] for r in res["table"] if r["liquidity_ok"] and r["target_ok"]]
    chosen_row = next(r for r in res["table"] if r["chosen"])
    assert chosen_row["score"] == pytest.approx(max(scores))
    nar_score = expected_roll_multiple(view(), nar.k1, nar.k2, nar.cost_rel)
    assert chosen_row["score"] > nar_score


def test_flat_cost_with_no_view_is_indifferent_to_width():
    """With a flat % cost and p = the pricing measure, every width returns e^{rT}/(1+rt): only real,
    width-dependent costs (or a view) make width matter — which is why the narrowest rule looked harmless."""
    r = 0.04
    ch = make_chain(S0=S0, sigma=0.45, T=T, r=r, half_spread_frac=0.0, min_half=0.0)
    q = Marginal.lognormal(r, 0.45, T)
    vals = [expected_roll_multiple(q, c.k1, c.k2, c.cost_rel)
            for c in candidate_spreads(ch, S0, 0.45, B, FX, cost_mode="flat", rt_cost=0.04)]
    assert np.allclose(vals, np.exp(r * T) / 1.04, rtol=5e-3)


def test_illiquid_candidates_are_not_chosen(chain):
    ch = make_chain(S0=S0, sigma=0.45, T=T, half_spread_frac=0.02, min_half=0.02,
                    strikes=np.arange(60.0, 220.0 + 1e-9, 2.5))
    cands = candidate_spreads(ch, S0, 0.45, B, FX)
    # make every wide candidate's short leg illiquid
    for c in cands[3:]:
        ch.oi[ch.strike == c.K2] = 10
    res = select_k2(ch, S0, 0.45, B, FX, view(), "expected")
    assert res["chosen"].liquidity_ok and res["eligible_found"]
    assert res["chosen"].K2 <= cands[2].K2


def test_goal_criterion_picks_max_goal_probability(chain):
    cfg = SimConfig(n_paths=1500, n_store=0, seed=3, scenarios=(bud.RateScenario("n", 0.0, 1.0),))
    res = select_k2(chain, S0, 0.45, B, FX, view(), "goal", cfg=cfg, max_width_pct=0.2)
    ok = [r for r in res["table"] if r["liquidity_ok"] and r["target_ok"]]
    assert next(r for r in res["table"] if r["chosen"])["p_goal"] == pytest.approx(max(r["p_goal"] for r in ok))
    with pytest.raises(ValueError):
        select_k2(chain, S0, 0.45, B, FX, view(), "median")

"""Page 3 and share links on the Black-Scholes fake provider (no network)."""
import json

import numpy as np
import pytest

from app.page3 import project_ex_divs, run_page3
from app.share import load_share, save_share, share_id
from data.cache import Cache
from data.service import MarketDataService, RefreshSchedule
from test_data import Clock, FakeMacro, NOW
from test_page1 import BSProvider
from datetime import date


@pytest.fixture(scope="module")
def svc():
    return MarketDataService(Cache(":memory:"), BSProvider(), FakeMacro(), RefreshSchedule(), Clock(NOW))


@pytest.fixture(scope="module")
def out(svc):
    st = dict(tickers=[dict(ticker="AAA"), dict(ticker="BBB"), dict(ticker="CCC")],
              weights=[[0.7, 0.3, 0.0], [0.7, 0.3, 0.0], [0.2, 0.8, 0.0], [0.5, 0.5, 0.0]],
              f=[1.0, 0.5, 0.5, 0.25], globals=dict(n_paths=3000))
    return run_page3(st, svc)


def test_page3_structure(out):
    json.dumps(out, allow_nan=False)
    assert out["included"] == ["AAA", "BBB"]                      # CCC illiquid -> excluded, weights renormalized
    W = np.array(out["weights"])
    assert W.shape == (4, 2) and np.allclose(W.sum(axis=1), 1) and np.allclose(W[2], [0.2, 0.8])
    assert len(out["schedule"]) == 32 and [r["f"] for r in out["schedule"][::8]] == [1.0, 0.5, 0.5, 0.25]
    r0 = out["schedule"][0]["legs"][0]
    assert r0["K1"] > 0 and r0["n_contracts"] >= 0 and r0["actions"] and {x["code"] for x in r0["rules"]} == {
        "R-1", "R-2", "R-3", "R-4", "R-5", "R-6"}
    later = out["schedule"][20]["legs"]
    assert [l["ticker"] for l in later] == ["AAA", "BBB"] and later[0]["weight"] == pytest.approx(0.2)


def test_page3_scenarios_and_bond_lane(out):
    S = out["scenarios_out"]
    assert set(S) == {"up", "neutral", "down"}
    assert S["down"]["budgets"][3] < S["neutral"]["budgets"][3] < S["up"]["budgets"][3]
    for v in S.values():
        for c, b in enumerate(v["bond"]):
            assert b["bond_cost"] + b["B"] == pytest.approx(b["matures_to"])          # bond + option budget = V - V_own
        band = v["band"]
        assert all(a <= m <= c for a, m, c in zip(band["p10"], band["p50"], band["p90"]))
        for rep in v["reps"].values():
            K1 = np.array(rep["strikes"]["AAA"]["K1"])
            assert len(K1) == 32 and np.all(K1 > 0)
            assert rep["terminal"] == pytest.approx(rep["cumulative"][-1])
    p = sum(S[k]["prob"] * S[k]["metrics_net"]["p_goal"] for k in S)
    assert out["mixture"]["metrics_net"]["p_goal"] == pytest.approx(p, abs=1e-9)


def test_ex_div_projection_over_program():
    d = project_ex_divs(["2026-01-06", "2026-04-06", "2026-07-06", "2026-10-06"], date(2026, 12, 1), date(2034, 12, 15))
    assert 30 <= len(d) <= 34 and all(date(2026, 12, 1) <= x <= date(2034, 12, 15) for x in d)
    assert project_ex_divs([], date(2026, 12, 1), date(2034, 12, 15)) == []


def test_share_links_roundtrip():
    c = Cache(":memory:")
    st = dict(tickers=[dict(ticker="AMD")], globals=dict(G=500e6))
    sid = save_share(c, "timeline", st)
    assert len(sid) == 10 and save_share(c, "timeline", dict(st)) == sid          # content-addressed
    assert load_share(c, sid) == dict(page="timeline", state=st)
    assert share_id("sim", st)[0] != sid
    assert load_share(c, "../etc") is None and load_share(c, "0123456789") is None

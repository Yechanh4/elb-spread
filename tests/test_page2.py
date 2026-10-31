"""Page 2 end to end on the Black-Scholes fake provider (no network)."""
import json

import numpy as np
import pytest

from app.page2 import run_page2, run_screener
from data.cache import Cache
from data.service import MarketDataService, RefreshSchedule
from test_data import Clock, FakeMacro, NOW
from test_page1 import SPEC, BSProvider

SPEC.update({"EEE": ("Energy", 0.62, 0.45, 500), "FFF": ("Information Technology", 0.58, 0.42, 500)})


@pytest.fixture(scope="module")
def svc():
    return MarketDataService(Cache(":memory:"), BSProvider(), FakeMacro(), RefreshSchedule(), Clock(NOW))


def state(**kw):
    s = dict(candidates=[dict(ticker=t) for t in ("AAA", "BBB", "CCC", "DDD", "EEE", "FFF")],
             globals=dict(n_paths=800, n_check=1500))
    s.update(kw)
    return s


@pytest.fixture(scope="module")
def out(svc):
    return run_page2(state(), svc)


def test_page2_json_and_eligibility(out):
    json.dumps(out, allow_nan=False)
    by = {c["ticker"]: c for c in out["candidates"]}
    assert not by["CCC"]["eligible"] and "liquidity" in by["CCC"]["reason"]
    assert not by["DDD"]["eligible"] and "Edge" in by["DDD"]["reason"]
    assert set(out["universe"]) == {"AAA", "BBB", "EEE", "FFF"}
    assert set(by["AAA"]["sigma"]) == {"up", "neutral", "down"}


def test_slider_picks_trade_off_and_constraints(out):
    picks = out["picks"]
    assert [p["lam"] for p in picks] == [round(x, 1) for x in np.linspace(0, 1, 11)]
    best_p = max(pt["p_goal"] for pt in out["points"])
    best_m = max(pt["mean"] for pt in out["points"])
    assert picks[0]["p_goal"] == pytest.approx(best_p) and picks[-1]["mean"] == pytest.approx(best_m)
    for p in picks:
        assert 3 <= len(p["names"]) <= 4 and abs(sum(p["weights"]) - 1) < 1e-9 and min(p["weights"]) >= 0
        secs = p["sectors"]
        assert max(secs.count(x) for x in secs) <= 2
        assert out["points"][p["point"]]["names"] == p["names"]
        assert "check" in p and set(p["sensitivity"]) >= {"sigma", "rt_cost", "f", "count"}
    assert any(pt["frontier"] for pt in out["points"])


def test_scenario_table_and_budget_coupling(out):
    t = out["table"]
    assert set(t) == {"up", "neutral", "down", "mixture", "minmax"} and out["recommended"] == "mixture"
    b = out["budgets"]
    assert b["down"]["budgets"][3] < b["neutral"]["budgets"][3] < b["up"]["budgets"][3]
    assert b["down"]["budgets"][0] == pytest.approx(b["up"]["budgets"][0])


def test_band_warning_and_screener(svc):
    o = run_page2(state(candidates=[dict(ticker="AAA")], scenarios=[
        dict(name="up", shift_bp=20, prob=0.25), dict(name="neutral", shift_bp=0, prob=0.5),
        dict(name="down", shift_bp=-75, prob=0.25)]), svc)
    assert any("outside its band" in w for w in o["warnings"])
    rows, err = run_screener(svc, ["AAA", "BBB", "DDD"], 0.04)
    assert [r["iv_minus_hv"] for r in rows] == sorted(r["iv_minus_hv"] for r in rows)
    assert all(0 <= r["iv_rank_proxy"] <= 1 for r in rows) and not err

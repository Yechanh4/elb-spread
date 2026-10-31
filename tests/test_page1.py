"""Page 1 end to end on a Black-Scholes fake provider (no network)."""
import json
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pytest

from conftest import make_chain
from core import budget as bud
from data.cache import Cache
from data.providers import MarketDataProvider
from data.service import MarketDataService, RefreshSchedule
from test_data import Clock, FakeMacro, NOW

ASOF = date(2026, 10, 29)
EXPIRIES = ["2026-12-18", "2027-01-15", "2027-02-19", "2027-03-19"]
SPEC = {  # ticker: (sector, realized vol of the fake history, chain IV, open interest)
    "AAA": ("Information Technology", 0.60, 0.45, 500),
    "BBB": ("Health Care", 0.55, 0.40, 500),
    "CCC": ("Energy", 0.60, 0.45, 10),          # illiquid
    "DDD": ("Financials", 0.20, 0.45, 500),     # HV < IV -> Edge <= 0 with the HV placeholder
}


class BSProvider(MarketDataProvider):
    name = "bs-fake"

    def info(self, t, asof):
        return dict(name=t + " Corp", spot=100.0, currency="USD", sector=SPEC[t][0], sector_source="fake",
                    div_yield=0.0, quote_time=None)

    def history(self, t, days=504):
        rng = np.random.default_rng(sum(map(ord, t)))   # str hash is salted per process
        common = rng.normal(0, 0.01, 504)
        r = 0.5 * common + rng.normal(0, SPEC[t][1] / np.sqrt(252), 504)
        r = r / r.std() * SPEC[t][1] / np.sqrt(252)
        c = list(100 * np.exp(np.cumsum(r)))
        d0 = date(2024, 11, 1)
        return dict(dates=[(d0 + timedelta(days=i)).isoformat() for i in range(504)], open=c,
                    high=[x * 1.01 for x in c], low=[x * 0.99 for x in c], close=c, volume=[1e6] * 504)

    def expiries(self, t):
        return EXPIRIES

    def call_chain(self, t, e):
        T = (date.fromisoformat(e) - ASOF).days / 365
        ch = make_chain(S0=100.0, sigma=SPEC[t][2], T=T, r=0.04, half_spread_frac=0.015, min_half=0.005,
                        strikes=np.arange(50.0, 250.0 + 1e-9, 2.5), oi=SPEC[t][3])
        rows = [dict(strike=float(k), bid=float(b), ask=float(a), mid=float((a + b) / 2), last=float((a + b) / 2),
                     iv=SPEC[t][2], oi=float(o), volume=10.0, last_trade=None)
                for k, b, a, o in zip(ch.strike, ch.bid, ch.ask, ch.oi)]
        return dict(expiry=e, rows=rows)

    def events(self, t, asof):
        return dict(earnings=[dict(date="2027-01-05", estimated=True)], ex_dividends=[],
                    earnings_past=[dict(date="2025-11-04", timing="amc"), dict(date="2026-02-03", timing="bmo")],
                    dividends=dict(dates=[], amounts=[]))


@pytest.fixture(scope="module")
def svc():
    return MarketDataService(Cache(":memory:"), BSProvider(), FakeMacro(), RefreshSchedule(), Clock(NOW))


def state(tickers, **g):
    return dict(tickers=[dict(ticker=t) if isinstance(t, str) else t for t in tickers],
                globals=dict(n_paths=2000, **g))


def test_page1_runs_and_is_json(svc):
    from app.page1 import run_page1
    out = run_page1(state(["AAA", "BBB", "CCC", "DDD"]), svc)
    json.dumps(out, allow_nan=False)                      # no NaN/inf leaks to the browser
    by = {b["ticker"]: b for b in out["tickers"]}
    assert by["AAA"]["in_simulation"] and by["BBB"]["in_simulation"]
    assert not by["CCC"]["in_simulation"] and "liquidity" in by["CCC"]["excluded_reason"]
    assert not by["DDD"]["in_simulation"] and "Edge" in by["DDD"]["excluded_reason"]
    P = out["portfolio"]
    assert P["included"] == ["AAA", "BBB"] and abs(sum(P["weights"]) - 1) < 1e-9
    assert out["budget"]["budgets"][0] == pytest.approx(bud.budget(r=0.031))   # r comes from the macro feed
    for b in by.values():
        assert set(b) >= {"market", "model"}              # every figure is grouped by origin
    m = by["AAA"]["model"]
    assert m["spread"]["k2_rule"] == "cost-aware" and m["narrowest"]["k2_rule"] == "narrowest"
    assert m["spread"]["K1"] == m["narrowest"]["K1"] and m["q_pass"]
    assert m["sigma_hat_source"].startswith("placeholder")
    assert 0 <= P["metrics_net"]["p_goal"] <= 1 and len(P["paths"]["cumulative"]) == 60
    assert [r["f"] for r in P["f_frontier"]] == [0.1, 0.25, 0.5, 0.75, 1.0]


def test_overrides_and_user_inputs(svc):
    from app.page1 import run_page1
    out = run_page1(state([dict(ticker="CCC", use_illiquid=True), dict(ticker="AAA", sigma_hat=0.70, mu_hat=0.10)],
                          r=0.02, k2_rule="narrowest", cost_mode="flat"), svc)
    by = {b["ticker"]: b for b in out["tickers"]}
    assert by["CCC"]["in_simulation"] and by["CCC"]["overrides"]["liquidity"]
    a = by["AAA"]["model"]
    assert a["sigma_hat"] == 0.70 and a["sigma_hat_source"] == "user" and a["mu_hat"] == 0.10
    assert a["spread"]["k2_rule"] == "narrowest" and a["spread"]["rt_cost"] == pytest.approx(0.04)
    assert out["budget"]["budgets"][0] == pytest.approx(bud.budget(r=0.02)) and out["budget"]["r_origin"] == "user"
    assert a["decomposition"]["drift_contribution"] > 0


def test_sigma_suggestion_on_card(svc):
    from app.page1 import run_page1
    out = run_page1(state(["AAA", "DDD"]), svc)
    for b in out["tickers"]:
        g = b["model"]["sigma_suggest"]
        assert set(g["models"]) == {"ewma", "garch", "har"}
        assert g["earnings"]["n_history"] == 2 and g["earnings"]["in_window"][0]["date"] == "2027-01-05"
        assert g["suggested"] >= g["base"] > 0     # an earnings day adds only what it moves beyond a normal day
        assert g["edge_vs_iv"] == pytest.approx(g["suggested"] - b["model"]["iv_atm"])
    by = {b["ticker"]: b["model"]["sigma_suggest"] for b in out["tickers"]}
    assert by["AAA"]["base"] > by["DDD"]["base"]                   # 60% vs 20% realized history
    assert by["DDD"]["edge_vs_iv"] < 0                             # 20% history cannot justify a 45% IV

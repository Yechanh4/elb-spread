"""Data layer, offline: a fake provider and a controllable clock. Live checks are in test_data_live.py."""
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from data import providers as pv
from data.cache import Cache
from data.providers import (MacroProvider, MarketDataProvider, ProviderError, parse_ecos, project_ex_dividends,
                            select_monthly_expiries, ttm_dividend_yield)
from data.scheduler import JOB_ID, build_scheduler, reschedule
from data.service import DataUnavailable, MarketDataService, RefreshSchedule, chain_slice

UTC = timezone.utc


class Clock:
    def __init__(self, t):
        self.t = t

    def __call__(self):
        return self.t


class FakeProvider(MarketDataProvider):
    name = "fake"

    def __init__(self):
        self.calls = 0
        self.fail = set()       # kinds that raise
        self.spot = 100.0

    def _hit(self, kind):
        self.calls += 1
        if kind in self.fail:
            raise ProviderError(f"{kind} down")

    def info(self, ticker, asof):
        self._hit("info")
        return dict(name=ticker, spot=self.spot, currency="USD", sector="Information Technology",
                    sector_source="fake", div_yield=0.01, quote_time=None)

    def history(self, ticker, days=504):
        self._hit("history")
        n = 300
        c = list(100 * np.exp(np.cumsum(np.full(n, 0.001))))
        return dict(dates=[(date(2025, 1, 1) + timedelta(days=i)).isoformat() for i in range(n)],
                    open=c, high=[x * 1.01 for x in c], low=[x * 0.99 for x in c], close=c, volume=[1e6] * n)

    def expiries(self, ticker):
        self._hit("expiries")
        return ["2026-11-06", "2026-12-18", "2027-01-15", "2027-01-22", "2027-02-19", "2027-03-19", "2027-04-16"]

    def call_chain(self, ticker, expiry):
        self._hit("chain")
        K = np.arange(80, 160, 5.0)
        mid = np.maximum(100 - K, 0) + 3 * np.exp(-((K - 100) / 25) ** 2) + 0.2
        return dict(expiry=expiry, rows=[dict(strike=float(k), bid=float(m - 0.05), ask=float(m + 0.05), mid=float(m),
                                              last=float(m), iv=0.4, oi=500.0, volume=10.0, last_trade=None)
                                         for k, m in zip(K, mid)])

    def events(self, ticker, asof):
        self._hit("events")
        return dict(earnings=[dict(date="2027-01-28", estimated=True)], ex_dividends=[], dividends=dict(dates=[], amounts=[]))


class FakeMacro(MacroProvider):
    name = "fakemacro"

    def __init__(self):
        self.fail = set()

    def get(self, key):
        if key in self.fail:
            raise ProviderError(f"{key} down")
        return dict(value={"usdkrw": 1400.0, "krw_2y": 0.031, "ust_2y": 0.045, "ust_3m": 0.040}[key], observed="2026-10-29",
                    source=f"fake {key}")


# Thursday 2026-10-29 14:00 New York = 18:00 UTC; scheduled refresh 15:30 NY
NOW = datetime(2026, 10, 29, 18, 0, tzinfo=UTC)


@pytest.fixture
def svc():
    return MarketDataService(Cache(":memory:"), FakeProvider(), FakeMacro(), RefreshSchedule(), Clock(NOW))


# ---------------------------------------------------------------- pure helpers

def test_monthly_expiries_beyond_60_days():
    ex = ["2026-11-20", "2026-12-18", "2027-01-08", "2027-01-15", "2027-01-22", "2027-02-19", "2027-03-19", "2027-04-16"]
    assert select_monthly_expiries(ex, date(2026, 10, 30)) == ["2027-01-15", "2027-02-19", "2027-03-19"]
    assert select_monthly_expiries(ex, date(2026, 10, 30), n=2) == ["2027-01-15", "2027-02-19"]
    # 2026-12-18 is 49 days out: excluded; exactly 60 days is excluded too (strictly beyond)
    assert select_monthly_expiries(["2026-12-29"], date(2026, 10, 30)) == []


def test_holiday_shifted_thursday_monthly():
    # 2027-06-18 (third Friday) is Juneteenth: the monthly trades Thursday 06-17 — seen live on JPM
    assert select_monthly_expiries(["2027-06-10", "2027-06-17", "2027-07-16"], date(2027, 3, 1)) == \
        ["2027-06-17", "2027-07-16"]
    # a Thursday weekly that sits next to a listed Friday is not the monthly
    assert select_monthly_expiries(["2027-07-15", "2027-07-16"], date(2027, 3, 1)) == ["2027-07-16"]


def test_ttm_dividend_yield():
    dd = ["2025-10-06", "2026-01-06", "2026-04-06", "2026-07-06", "2026-10-06"]
    amt = [1.4, 1.5, 1.5, 1.5, 1.65]
    assert ttm_dividend_yield(dd, amt, 300.0, date(2026, 10, 29)) == pytest.approx(6.15 / 300)
    assert ttm_dividend_yield([], [], 300.0, date(2026, 10, 29)) == 0.0


def test_ex_dividend_projection():
    dd = ["2025-10-06", "2026-01-06", "2026-04-06", "2026-07-06", "2026-10-06"]
    out = project_ex_dividends(dd, date(2026, 10, 29), horizon_days=200)
    assert [o["projected"] for o in out] == [True, True]
    assert out[0]["date"] == "2027-01-05"           # last + median gap (91 days)
    conf = project_ex_dividends(dd, date(2026, 10, 29), confirmed=date(2027, 1, 6))
    assert conf[0] == dict(date="2027-01-06", projected=False)
    assert all(abs((date.fromisoformat(o["date"]) - date(2027, 1, 6)).days) > 20 for o in conf[1:])
    # a stale "confirmed" date in the past (AMD shows 1995) is ignored
    assert project_ex_dividends([], date(2026, 10, 29), confirmed=date(1995, 4, 27)) == []


def test_parse_ecos():
    payload = {"StatisticSearch": {"row": [
        {"TIME": "20261028", "DATA_VALUE": "3.512", "ITEM_NAME1": "국고채(2년)"},
        {"TIME": "20261029", "DATA_VALUE": "3.498", "ITEM_NAME1": "국고채(2년)"}]}}
    v = parse_ecos(payload)
    assert v["value"] == pytest.approx(0.03498) and v["observed"] == "2026-10-29"
    with pytest.raises(ProviderError):
        parse_ecos({"RESULT": {"MESSAGE": "인증키가 유효하지 않습니다"}})


def test_us_session():
    assert pv.us_session(datetime(2026, 10, 29, 18, 0, tzinfo=UTC)) == "regular"     # 14:00 NY Thu
    assert pv.us_session(datetime(2026, 10, 29, 22, 0, tzinfo=UTC)) == "off-hours"   # 18:00 NY
    assert pv.us_session(datetime(2026, 10, 31, 16, 0, tzinfo=UTC)) == "off-hours"   # Saturday


# ---------------------------------------------------------------- schedule

def test_last_scheduled_run_handles_weekends_and_time_of_day():
    s = RefreshSchedule(15, 30, "America/New_York")
    # Thursday 14:00 NY -> Wednesday 15:30 NY (EDT: 19:30 UTC)
    assert s.last_run(NOW) == datetime(2026, 10, 28, 19, 30, tzinfo=UTC)
    # Sunday -> Friday 15:30 NY
    assert s.last_run(datetime(2026, 11, 1, 15, 0, tzinfo=UTC)) == datetime(2026, 10, 30, 19, 30, tzinfo=UTC)
    # after the DST switch (Nov 2) 15:30 NY is 20:30 UTC
    assert s.last_run(datetime(2026, 11, 3, 21, 0, tzinfo=UTC)) == datetime(2026, 11, 3, 20, 30, tzinfo=UTC)
    assert "KST" in s.describe()


def test_scheduler_registers_daily_job(svc):
    sch = build_scheduler(svc)
    job = sch.get_job(JOB_ID)
    fields = {f.name: str(f) for f in job.trigger.fields}
    assert fields["hour"] == "15" and fields["minute"] == "30" and fields["day_of_week"] == "mon,tue,wed,thu,fri"
    assert str(job.trigger.timezone) == "America/New_York"
    job = reschedule(sch, svc, 9, 45, "Asia/Seoul")
    fields = {f.name: str(f) for f in job.trigger.fields}
    assert fields["hour"] == "9" and fields["minute"] == "45" and str(job.trigger.timezone) == "Asia/Seoul"
    assert svc.cache.get_setting("schedule") == dict(hour=9, minute=45, tz="Asia/Seoul")


# ---------------------------------------------------------------- cache

def test_cache_roundtrip_and_prune():
    c = Cache(":memory:")
    t0 = datetime(2026, 9, 1, tzinfo=UTC)
    c.put("info", "AAA", {"v": 1}, "src", fetched_at=t0)
    c.put("info", "AAA", {"v": 2}, "src", fetched_at=t0 + timedelta(days=1))
    assert c.latest("info", "AAA")["payload"] == {"v": 2}
    assert c.latest("info", "BBB") is None
    c.put("info", "OLD", {"v": 9}, "src", fetched_at=t0)
    c.prune(keep_days=10, now=t0 + timedelta(days=40))
    assert c.latest("info", "OLD")["payload"] == {"v": 9}        # newest per key always kept
    assert c.latest("info", "AAA")["payload"] == {"v": 2}
    n = c.db.execute("SELECT COUNT(*) FROM snapshots WHERE key='AAA'").fetchone()[0]
    assert n == 1
    c.log("info", "AAA", False, "boom")
    assert c.last_attempt("info", "AAA")["error"] == "boom"
    c.watch("amd")
    assert c.watchlist() == ["AMD"]


def test_nan_survives_cache():
    c = Cache(":memory:")
    c.put("chain", "X|2027-01-15", {"rows": [{"mid": float("nan")}]}, "s")
    assert np.isnan(c.latest("chain", "X|2027-01-15")["payload"]["rows"][0]["mid"])


# ---------------------------------------------------------------- service freshness

def test_fetch_then_serve_from_cache(svc):
    b = svc.ticker_bundle("aaa")
    calls = svc.provider.calls
    assert calls == 7                                    # info, history, expiries, events + 3 chains
    assert b.selected_expiries == ["2027-01-15", "2027-02-19", "2027-03-19"]
    assert not b.stale and not b.errors and all(not i.from_cache for i in b.items())
    b2 = svc.ticker_bundle("AAA")
    assert svc.provider.calls == calls                  # nothing refetched
    assert all(i.from_cache and not i.stale for i in b2.items())
    assert b2.data_asof == NOW
    svc.ticker_bundle("AAA", refresh=True)
    assert svc.provider.calls == 2 * calls              # manual refresh ignores the cache


def test_failure_serves_cache_with_stale_badge(svc):
    svc.ticker_bundle("AAA")
    svc.provider.fail = {"chain", "info"}
    svc.provider.spot = 120.0
    b = svc.ticker_bundle("AAA", refresh=True)
    assert b.stale and not b.errors
    assert b.info.stale and b.info.from_cache and "refresh failed" in b.info.reason
    assert b.info.value["spot"] == 100.0                 # the cached value, clearly labeled
    assert all(c.stale for c in b.chains.values())
    assert not b.history.stale                           # parts that did refresh are fresh
    assert svc.cache.last_attempt("info", "AAA")["ok"] is False


def test_failure_without_cache_is_reported_not_hidden(svc):
    svc.provider.fail = {"chain"}
    b = svc.ticker_bundle("BBB")
    assert b.chains == {} and b.stale
    assert set(b.errors) == {"BBB|2027-01-15", "BBB|2027-02-19", "BBB|2027-03-19"}
    with pytest.raises(DataUnavailable):
        svc.get("info", "ZZZ", lambda: (_ for _ in ()).throw(ProviderError("x")))


def test_old_cache_refetched_after_next_scheduled_run(svc):
    svc.ticker_bundle("AAA")
    calls = svc.provider.calls
    svc.clock.t = NOW + timedelta(hours=1)               # still before today's 15:30 run
    svc.ticker_bundle("AAA")
    assert svc.provider.calls == calls
    svc.clock.t = NOW + timedelta(hours=3)               # 17:00 NY: 14:00 fetch is within the 2h grace of 15:30
    svc.ticker_bundle("AAA")
    assert svc.provider.calls == calls
    svc.clock.t = NOW + timedelta(days=1, hours=3)       # Friday 17:00 NY: Friday's run has passed
    b = svc.ticker_bundle("AAA")
    assert svc.provider.calls == 2 * calls and not b.stale
    svc.clock.t = NOW + timedelta(days=2, hours=3)       # Saturday: Friday's data is still current
    svc.ticker_bundle("AAA")
    assert svc.provider.calls == 2 * calls
    # Monday after the run, the refetch fails: the old data is shown as stale, never as current
    svc.clock.t = NOW + timedelta(days=4, hours=4)
    svc.provider.fail = {"info"}
    b = svc.ticker_bundle("AAA")
    assert b.info.stale and b.info.fetched_at == NOW + timedelta(days=1, hours=3)


# ---------------------------------------------------------------- macro

def test_macro_origin_and_override(svc):
    m = svc.macros()
    assert m["krw_2y"].origin == "market" and m["krw_2y"].value["value"] == 0.031
    svc.set_override("krw_2y", 0.0376)
    it = svc.macro_item("krw_2y")
    assert it.origin == "user" and it.value["value"] == 0.0376
    svc.clear_override("krw_2y")
    assert svc.macro_item("krw_2y").origin == "market"
    with pytest.raises(KeyError):
        svc.set_override("nope", 1.0)


def test_krw_2y_fallback_is_labeled_assumption():
    s = MarketDataService(Cache(":memory:"), FakeProvider(), FakeMacro(), RefreshSchedule(), Clock(NOW))
    s.macro.fail = {"krw_2y"}
    it = s.macro_item("krw_2y")
    assert it.origin == "fallback" and it.value["value"] == pytest.approx(0.0376) and it.reason
    s.macro.fail = {"usdkrw"}
    m = s.macros()
    assert m["usdkrw"] is None and "usdkrw_error" in m


def test_refresh_all_covers_watchlist(svc):
    svc.cache.watch("AAA")
    svc.cache.watch("BBB")
    rep = svc.refresh_all()
    assert set(rep["tickers"]) == {"AAA", "BBB"} and all(v["ok"] for v in rep["tickers"].values())
    assert rep["macro"]["krw_2y"]["origin"] == "market"
    assert svc.status()["last_refresh"]["finished"]


# ---------------------------------------------------------------- bridges

def test_chain_slice_bridge(svc):
    b = svc.ticker_bundle("AAA")
    cs = chain_slice(b.chains["2027-01-15"], b.asof_date)
    assert cs.T == pytest.approx((date(2027, 1, 15) - date(2026, 10, 29)).days / 365)
    assert np.all(np.diff(cs.strike) > 0) and cs.quoted().all()
    from core.spread import build_spread
    sp = build_spread(cs, 100.0, 0.40, 79.7e6, 1400.0)
    assert sp.K1 >= 100 * (1 + 1.56 * 0.40 * 0.5)


# ---------------------------------------------------------------- yfinance adapter (no network)

class _FakeYfTicker:
    def __init__(self, sym, book):
        self.sym, self.book = sym, book

    @property
    def options(self):
        return self.book.get("options", ())

    def option_chain(self, e):
        self.book["chain_calls"] = self.book.get("chain_calls", 0) + 1
        calls = self.book["chains"].pop(0)
        return type("OC", (), {"calls": calls})()

    @property
    def info(self):
        return self.book["info"]

    @property
    def fast_info(self):
        return {"lastPrice": 50.0}

    @property
    def dividends(self):
        return self.book["divs"]

    calendar = {}


def _yf(book):
    return type("YF", (), {"Ticker": staticmethod(lambda s: _FakeYfTicker(s, book))})


def test_yfinance_adapter_parsing_and_retry(monkeypatch):
    monkeypatch.setattr(pv.time, "sleep", lambda s: None)
    chain = pd.DataFrame(dict(strike=[100.0, 105.0], bid=[2.0, np.nan], ask=[2.2, 0.5], lastPrice=[2.1, 0.4],
                              impliedVolatility=[0.4, 0.5], openInterest=[100, np.nan], volume=[5, np.nan],
                              lastTradeDate=[pd.Timestamp("2026-10-29", tz="UTC"), pd.NaT]))
    book = dict(options=("2027-01-15",), chains=[pd.DataFrame(), chain],
                info=dict(currentPrice=101.0, sector="Healthcare", longName="Fake Co", currency="USD",
                          exDividendDate=None),
                divs=pd.Series([0.5, 0.5], index=pd.to_datetime(["2026-03-01", "2026-09-01"])))
    p = pv.YFinanceProvider.__new__(pv.YFinanceProvider)
    p.yf, p.pause, p.retries = _yf(book), 0, 2
    c = p.call_chain("FAKE", "2027-01-15")
    assert book["chain_calls"] == 2                      # empty answer retried
    r0, r1 = c["rows"]
    assert r0["mid"] == pytest.approx(2.1) and r0["last_trade"].startswith("2026-10-29")
    assert r1["bid"] == 0.0 and np.isnan(r1["mid"]) and r1["oi"] == 0.0 and r1["last_trade"] is None
    inf = p.info("FAKE", date(2026, 10, 29))
    assert inf["sector"] == "Health Care" and inf["div_yield"] == pytest.approx(1.0 / 101.0)
    book["options"] = ()
    with pytest.raises(ProviderError):
        p.expiries("FAKE")


def test_off_hours_empty_chain_is_refused_and_cache_kept(svc):
    svc.ticker_bundle("AAA")                                  # regular session (14:00 NY)
    good = svc.cache.latest("chain", "AAA|2027-01-15")
    svc.clock.t = NOW + timedelta(days=1, hours=8)            # Friday 22:00 NY, after the next run
    orig = svc.provider.call_chain
    def empty(t, e):
        c = orig(t, e)
        for r in c["rows"]:
            r["bid"] = 0.0
        return c
    svc.provider.call_chain = empty
    b = svc.ticker_bundle("AAA")
    it = b.chains["2027-01-15"]
    assert it.stale and "off-hours" in it.reason
    assert svc.cache.latest("chain", "AAA|2027-01-15")["fetched_at"] == good["fetched_at"]   # nothing cached
    nb = svc.ticker_bundle("NEW")                             # no regular-session cache: reported, not hidden
    assert nb.chains == {} and any("off-hours" in v for v in nb.errors.values())

"""Market-data service: provider -> SQLite cache -> app, with honest freshness.

Rules:
  * Every item carries fetched_at, source and a stale flag with a reason. Nothing old is shown as current.
  * Fresh = fetched at or after the most recent scheduled daily refresh (minus a grace window). Older
    cached data triggers a fetch on access; if that fetch fails, the cache is served with stale=True.
  * A provider failure with no cache raises DataUnavailable for that item only; the bundle reports it.
  * The KRW 2y yield has no free keyless feed: user override > ECOS (if keyed) > cache > 3.76% fallback,
    and the origin ('user' / 'market' / 'fallback') is always shown.
"""
import os
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, time as dtime, timedelta, timezone
from typing import Dict, List, Optional
from zoneinfo import ZoneInfo

import numpy as np

from .cache import Cache, iso, utcnow
from .providers import (DELAY_NOTE, KRW_2Y_FALLBACK, NY, MacroProvider, MarketDataProvider, ProviderError,
                        YahooFredMacro,
                        YFinanceProvider, select_monthly_expiries, us_session)

GRACE = timedelta(hours=2)
MACRO_KEYS = ("usdkrw", "krw_2y", "ust_2y", "ust_3m")


class DataUnavailable(RuntimeError):
    pass


@dataclass
class RefreshSchedule:
    """Daily refresh time. Default 15:30 New York, Mon-Fri: inside the session, so option quotes are live
    (after the close Yahoo often shows zero bids). That is 04:30 KST while New York is on EDT, 05:30 KST on EST."""
    hour: int = 15
    minute: int = 30
    tz: str = "America/New_York"
    weekdays: tuple = (0, 1, 2, 3, 4)

    def last_run(self, now=None):
        """Most recent scheduled run at or before now (UTC)."""
        z = ZoneInfo(self.tz)
        now_l = (now or utcnow()).astimezone(z)
        d = now_l.date()
        for _ in range(10):
            cand = datetime.combine(d, dtime(self.hour, self.minute), z)
            if d.weekday() in self.weekdays and cand <= now_l:
                return cand.astimezone(timezone.utc)
            d -= timedelta(days=1)
        raise RuntimeError("no scheduled weekday in the last 10 days")

    def describe(self):
        kst = datetime.combine(date.today(), dtime(self.hour, self.minute), ZoneInfo(self.tz)).astimezone(
            ZoneInfo("Asia/Seoul"))
        return f"{self.hour:02d}:{self.minute:02d} {self.tz} Mon-Fri (= {kst:%H:%M} KST at today's UTC offset)"


@dataclass
class Item:
    kind: str
    key: str
    value: object
    fetched_at: datetime
    source: str
    stale: bool = False
    reason: Optional[str] = None
    from_cache: bool = False
    origin: str = "market"          # market | user | fallback

    def meta(self):
        return dict(kind=self.kind, key=self.key, fetched_at=iso(self.fetched_at), source=self.source,
                    stale=self.stale, reason=self.reason, from_cache=self.from_cache, origin=self.origin)


@dataclass
class TickerBundle:
    ticker: str
    asof_date: date
    info: Optional[Item]
    history: Optional[Item]
    expiries: Optional[Item]
    selected_expiries: List[str]
    chains: Dict[str, Item]
    events: Optional[Item]
    errors: Dict[str, str] = field(default_factory=dict)

    def items(self):
        out = [x for x in (self.info, self.history, self.expiries, self.events) if x is not None]
        return out + list(self.chains.values())

    @property
    def stale(self):
        return bool(self.errors) or any(i.stale for i in self.items())

    @property
    def data_asof(self):
        """Oldest fetch time among the parts: the honest 'data as of'."""
        ts = [i.fetched_at for i in self.items()]
        return min(ts) if ts else None


class MarketDataService:
    def __init__(self, cache=None, provider=None, macro=None, schedule=None, clock=None):
        self.cache = cache or Cache()
        self._provider = provider
        self._macro = macro
        self.schedule = schedule or RefreshSchedule(**(self.cache.get_setting("schedule") or {}))
        self.clock = clock or utcnow

    @property
    def provider(self) -> MarketDataProvider:
        if self._provider is None:
            self._provider = YFinanceProvider()
        return self._provider

    @property
    def macro(self) -> MacroProvider:
        if self._macro is None:
            self._macro = YahooFredMacro(ecos_key=os.environ.get("ECOS_API_KEY"))
        return self._macro

    def set_schedule(self, hour, minute, tz="America/New_York"):
        self.schedule = RefreshSchedule(hour, minute, tz)
        self.cache.set_setting("schedule", dict(hour=hour, minute=minute, tz=tz))

    # ------------------------------------------------------------ core get

    def is_current(self, fetched_at, now=None):
        return fetched_at >= self.schedule.last_run(now or self.clock()) - GRACE

    def get(self, kind, key, fetch, refresh=False, source=None):
        now = self.clock()
        cached = self.cache.latest(kind, key)
        if cached and not refresh and self.is_current(cached["fetched_at"], now):
            return Item(kind, key, cached["payload"], cached["fetched_at"], cached["source"], from_cache=True)
        try:
            payload = fetch()
        except Exception as e:  # noqa: BLE001 - any provider failure degrades to cache
            self.cache.log(kind, key, False, f"{type(e).__name__}: {e}", at=now)
            if cached:
                return Item(kind, key, cached["payload"], cached["fetched_at"], cached["source"], stale=True,
                            reason=f"refresh failed ({type(e).__name__}: {e}); showing cache from "
                                   f"{iso(cached['fetched_at'])}", from_cache=True)
            raise DataUnavailable(f"{kind} {key}: {e}") from e
        src = source or self.provider.name
        self.cache.put(kind, key, payload, src, fetched_at=now)
        self.cache.log(kind, key, True, at=now)
        return Item(kind, key, payload, now, src)

    # ------------------------------------------------------------ tickers

    def asof_date(self):
        return self.clock().astimezone(NY).date()

    def ticker_bundle(self, ticker, refresh=False, n_expiries=3, min_days=60):
        T = ticker.upper()
        asof = self.asof_date()
        p = self.provider
        errors = {}

        def safe(kind, key, fn):
            try:
                return self.get(kind, key, fn, refresh)
            except DataUnavailable as e:
                errors[kind if kind != "chain" else key] = str(e)
                return None

        info = safe("info", T, lambda: p.info(T, asof))
        hist = safe("history", T, lambda: p.history(T))
        exps = safe("expiries", T, lambda: p.expiries(T))
        sel = select_monthly_expiries(exps.value, asof, min_days, n_expiries) if exps else []
        if exps and len(sel) < n_expiries:
            errors["expiries"] = f"only {len(sel)} monthly expiries more than {min_days} days out"

        def chain_fetch(e):
            return self._checked_chain(T, e)

        chains = {}
        for e in sel:
            it = safe("chain", f"{T}|{e}", lambda e=e: chain_fetch(e))
            if it is not None:
                chains[e] = it
        events = safe("events", T, lambda: p.events(T, asof))
        return TickerBundle(T, asof, info, hist, exps, sel, chains, events, errors)

    def _checked_chain(self, T, expiry):
        """Off-hours Yahoo chains come back with zero bids. Such a chain is refused (never cached), so the
        last regular-session chain is served as stale instead — or the item is reported unavailable."""
        c = self.provider.call_chain(T, expiry)
        c["session"] = us_session(self.clock())
        rows = c["rows"]
        two_sided = sum(1 for r in rows if r["bid"] > 0 and r["ask"] >= r["bid"])
        if c["session"] == "off-hours" and two_sided < max(5, 0.2 * len(rows)):
            raise ProviderError(f"off-hours quotes are empty ({two_sided}/{len(rows)} two-sided); "
                                f"refresh during the US session (22:30-05:00 KST)")
        return c

    def chain_item(self, ticker, expiry, refresh=False):
        """One extra expiry on demand (e.g. the one just before a catalyst, for the priced-in check)."""
        T = ticker.upper()
        return self.get("chain", f"{T}|{expiry}", lambda: self._checked_chain(T, expiry), refresh)

    def search(self, query, limit=8):
        """Ticker autocomplete: watchlist and cached tickers first, then Yahoo search (equities only)."""
        q = query.strip().upper()
        local = sorted(set(self.cache.watchlist()) | set(self.cache.keys("info")))
        out = [dict(symbol=t, name=(self.cache.latest("info", t) or {}).get("payload", {}).get("name", ""),
                    source="cache") for t in local if t.startswith(q)]
        if len(q) >= 1 and len(out) < limit:
            try:
                import yfinance as yf
                for r in yf.Search(query, max_results=limit, news_count=0).quotes:
                    sym = r.get("symbol", "")
                    if r.get("quoteType") == "EQUITY" and "." not in sym and sym not in {o["symbol"] for o in out}:
                        out.append(dict(symbol=sym, name=r.get("shortname") or r.get("longname") or "", source="yahoo"))
            except Exception:  # noqa: BLE001 - autocomplete is best-effort
                pass
        return out[:limit]

    # ------------------------------------------------------------ macro

    def set_override(self, key, value):
        """User-edited value (e.g. the KRW 2y yield). Takes precedence until cleared."""
        if key not in MACRO_KEYS:
            raise KeyError(key)
        self.cache.set_setting(f"override:{key}", value)

    def clear_override(self, key):
        self.cache.delete_setting(f"override:{key}")

    def macro_item(self, key, refresh=False):
        ov = self.cache.get_setting(f"override:{key}")
        if ov is not None:
            return Item("macro", key, dict(value=float(ov), observed=None, source="user"),
                        self.cache.setting_time(f"override:{key}"), "user", origin="user")
        try:
            it = self.get("macro", key, lambda: self.macro.get(key), refresh, source=self.macro.name)
            if it.value.get("source"):
                it.source = it.value["source"]
            return it
        except DataUnavailable as e:
            if key == "krw_2y":
                return Item("macro", key, dict(value=KRW_2Y_FALLBACK, observed=None, source="fallback"),
                            self.clock(), "fallback: spec default 3.76% (edit it)", origin="fallback",
                            reason=str(e))
            raise

    def macros(self, refresh=False):
        out = {}
        for k in MACRO_KEYS:
            try:
                out[k] = self.macro_item(k, refresh)
            except DataUnavailable as e:
                out[k] = None
                out[f"{k}_error"] = str(e)
        return out

    # ------------------------------------------------------------ scheduled job

    def refresh_all(self, tickers=None):
        tickers = tickers or self.cache.watchlist()
        report = dict(started=iso(self.clock()), tickers={}, macro={})
        for t in tickers:
            b = self.ticker_bundle(t, refresh=True)
            report["tickers"][t] = dict(ok=not b.errors, stale=b.stale, errors=b.errors)
        for k, v in self.macros(refresh=True).items():
            if not k.endswith("_error"):
                report["macro"][k] = None if v is None else dict(origin=v.origin, stale=v.stale)
        self.cache.prune()
        report["finished"] = iso(self.clock())
        self.cache.set_setting("last_refresh_report", report)
        return report

    def status(self):
        return dict(delay_note=DELAY_NOTE, schedule=self.schedule.describe(),
                    last_scheduled_run=iso(self.schedule.last_run(self.clock())),
                    last_refresh=self.cache.get_setting("last_refresh_report"), watchlist=self.cache.watchlist())


# ---------------------------------------------------------------- bridges to core

def chain_slice(item, asof_date):
    """Cached chain payload -> core.spread.ChainSlice, T in years (ACT/365) from asof_date."""
    from core.spread import ChainSlice
    rows = item.value["rows"]
    e = date.fromisoformat(item.value["expiry"])
    col = lambda k: np.array([r[k] for r in rows], dtype=float)
    return ChainSlice(expiry=e.isoformat(), T=(e - asof_date).days / 365.0, strike=col("strike"), bid=col("bid"),
                      ask=col("ask"), oi=col("oi"), iv=col("iv"), last=col("last"), volume=col("volume"))


def closes(item):
    return np.array(item.value["close"], dtype=float)

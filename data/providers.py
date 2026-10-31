"""Pluggable market-data providers. yfinance is the default; swap in IBKR/Polygon by implementing
MarketDataProvider (and MacroProvider for rates/FX) — nothing else in the app talks to a data source.

Every payload is plain JSON-serializable dicts/lists so the cache can store it as-is.
Free Yahoo data is delayed ~15-20 minutes; the app refreshes once a day, which is adequate for a
3-month-maturity strategy but not for execution. Verify quotes at the broker before trading.
"""
import math
import time
import warnings
from abc import ABC, abstractmethod
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np

NY = ZoneInfo("America/New_York")
DELAY_NOTE = ("Free source (Yahoo Finance): quotes delayed ~15-20 min, refreshed once a day. Adequate for "
              "3-month options; not real-time — verify rates, IVs and quotes at execution.")
KRW_2Y_FALLBACK = 0.0376
YF_TO_GICS = {
    "Technology": "Information Technology", "Communication Services": "Communication Services",
    "Consumer Cyclical": "Consumer Discretionary", "Consumer Defensive": "Consumer Staples",
    "Healthcare": "Health Care", "Financial Services": "Financials", "Industrials": "Industrials",
    "Energy": "Energy", "Basic Materials": "Materials", "Utilities": "Utilities", "Real Estate": "Real Estate",
}


class ProviderError(RuntimeError):
    pass


def us_session(now=None):
    """'regular' during NYSE hours (Mon-Fri 9:30-16:00 New York), else 'off-hours'. Off-hours option
    quotes are frozen or pulled; spreads look far wider than what trades."""
    now = (now or datetime.now(timezone.utc)).astimezone(NY)
    regular = now.weekday() < 5 and (9, 30) <= (now.hour, now.minute) <= (16, 0)
    return "regular" if regular else "off-hours"


def select_monthly_expiries(expiries, asof, min_days=60, n=3):
    """The n nearest monthly expiries more than min_days after asof. A Thursday counts only when the
    Friday right after it is absent from the listing (holiday-shifted monthly, e.g. Good Friday)."""
    ds = sorted(date.fromisoformat(e) if isinstance(e, str) else e for e in expiries)
    listed = set(ds)
    out = []
    for d in ds:
        if (d - asof).days <= min_days:
            continue
        fri = 15 <= d.day <= 21 and d.weekday() == 4
        thu = 15 <= d.day <= 20 and d.weekday() == 3 and (d + timedelta(days=1)) not in listed
        if fri or thu:
            out.append(d.isoformat())
        if len(out) == n:
            break
    return out


def ttm_dividend_yield(div_dates, div_amounts, spot, asof):
    """Trailing-12-month cash dividends / spot (unambiguous units, unlike Yahoo's dividendYield)."""
    cut = asof - timedelta(days=365)
    tot = sum(a for d, a in zip(div_dates, div_amounts) if cut < date.fromisoformat(d) <= asof)
    return float(tot / spot) if spot > 0 else 0.0


def project_ex_dividends(div_dates, asof, horizon_days=200, confirmed=None):
    """Next ex-dividend dates: the confirmed one if Yahoo has it, then the last date stepped forward by the
    median gap between recent payments. Projected dates are flagged — they are estimates."""
    out = []
    if confirmed and confirmed >= asof:
        out.append(dict(date=confirmed.isoformat(), projected=False))
    ds = sorted(date.fromisoformat(d) for d in div_dates)
    if len(ds) >= 3:
        gaps = np.diff([d.toordinal() for d in ds[-6:]])
        step = int(np.median(gaps))
        if 20 <= step <= 400:
            nxt = ds[-1]
            while nxt <= asof + timedelta(days=horizon_days):
                nxt = nxt + timedelta(days=step)
                if nxt >= asof and all(abs((date.fromisoformat(o["date"]) - nxt).days) > 20 for o in out):
                    out.append(dict(date=nxt.isoformat(), projected=True))
    return [o for o in out if date.fromisoformat(o["date"]) <= asof + timedelta(days=horizon_days)]


def _f(x, default=float("nan")):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return default
    return v if math.isfinite(v) else default


# ---------------------------------------------------------------- interfaces

class MarketDataProvider(ABC):
    name = "abstract"

    @abstractmethod
    def info(self, ticker, asof):
        """-> {name, spot, currency, sector (GICS name when mappable), sector_source, div_yield,
              quote_time (ISO or None)}"""

    @abstractmethod
    def history(self, ticker, days=504):
        """-> {dates, open, high, low, close, volume} oldest first, split/dividend adjusted."""

    @abstractmethod
    def expiries(self, ticker):
        """-> list of 'YYYY-MM-DD'."""

    @abstractmethod
    def call_chain(self, ticker, expiry):
        """-> {expiry, rows: [{strike, bid, ask, mid, last, iv, oi, volume, last_trade}]}"""

    @abstractmethod
    def events(self, ticker, asof):
        """-> {earnings: [{date, estimated}], ex_dividends: [{date, projected}], dividends: {dates, amounts}}"""


class MacroProvider(ABC):
    name = "abstract"

    @abstractmethod
    def get(self, key):
        """key in {'usdkrw', 'krw_2y', 'ust_2y', 'ust_3m'} -> {value (decimal for rates), observed (date ISO), source}"""


# ---------------------------------------------------------------- yfinance

class YFinanceProvider(MarketDataProvider):
    name = "yfinance"

    def __init__(self, pause=0.25, retries=3):
        import yfinance as yf
        self.yf = yf
        self.pause = pause
        self.retries = retries
        warnings.filterwarnings("ignore", module="yfinance")

    def _t(self, ticker):
        return self.yf.Ticker(ticker)

    def info(self, ticker, asof):
        t = self._t(ticker)
        i = t.info or {}
        spot = _f(i.get("currentPrice")) if i.get("currentPrice") else _f(i.get("regularMarketPrice"))
        if not spot > 0:
            spot = _f(t.fast_info.get("lastPrice"))
        if not spot > 0:
            raise ProviderError(f"{ticker}: no spot price")
        divs = t.dividends
        dd = [d.date().isoformat() for d in divs.index]
        sector = i.get("sector") or "Unknown"
        qt = i.get("regularMarketTime")
        return dict(name=i.get("longName") or i.get("shortName") or ticker, spot=spot,
                    currency=i.get("currency") or "USD", sector=YF_TO_GICS.get(sector, sector),
                    sector_source=f"Yahoo '{sector}' mapped to GICS" if sector in YF_TO_GICS else "Yahoo (unmapped)",
                    div_yield=ttm_dividend_yield(dd, divs.tolist(), spot, asof),
                    quote_time=datetime.fromtimestamp(qt, timezone.utc).isoformat() if qt else None)

    def history(self, ticker, days=504):
        h = self._t(ticker).history(period="2y" if days <= 504 else "3y", auto_adjust=True)
        h = h.dropna(subset=["Close"]).tail(days)
        if len(h) < 30:
            raise ProviderError(f"{ticker}: only {len(h)} daily bars")
        return dict(dates=[d.date().isoformat() for d in h.index],
                    open=h["Open"].astype(float).tolist(), high=h["High"].astype(float).tolist(),
                    low=h["Low"].astype(float).tolist(), close=h["Close"].astype(float).tolist(),
                    volume=h["Volume"].astype(float).tolist())

    def expiries(self, ticker):
        # Yahoo answers an empty list when it rate-limits: back off before concluding "no options"
        for attempt in range(self.retries + 1):
            e = list(self._t(ticker).options)
            if e:
                return e
            time.sleep(1.5 * (attempt + 1))
        raise ProviderError(f"{ticker}: no listed option expiries (or rate-limited)")

    def call_chain(self, ticker, expiry):
        c = None
        for attempt in range(self.retries + 1):
            try:
                c = self._t(ticker).option_chain(expiry).calls
                if len(c):
                    break
            except Exception:  # noqa: BLE001 - transient Yahoo errors
                if attempt == self.retries:
                    raise
            time.sleep(2.0 * (attempt + 1))
        if c is None or not len(c):
            raise ProviderError(f"{ticker} {expiry}: empty call chain")
        rows = []
        for r in c.itertuples(index=False):
            bid, ask = _f(r.bid, 0.0), _f(r.ask, 0.0)
            lt = getattr(r, "lastTradeDate", None)
            rows.append(dict(strike=_f(r.strike), bid=bid, ask=ask,
                             mid=0.5 * (bid + ask) if bid > 0 and ask >= bid else float("nan"),
                             last=_f(r.lastPrice), iv=_f(r.impliedVolatility), oi=_f(r.openInterest, 0.0),
                             volume=_f(r.volume, 0.0), last_trade=lt.isoformat() if lt is not None and lt == lt else None))
        time.sleep(self.pause)
        return dict(expiry=expiry, rows=rows)

    def events(self, ticker, asof):
        t = self._t(ticker)
        earnings = []
        try:
            cal = t.calendar or {}
            for d in cal.get("Earnings Date") or []:
                d = d if isinstance(d, date) else d.date()
                if d >= asof:
                    earnings.append(dict(date=d.isoformat(), estimated=True))
        except Exception:  # noqa: BLE001 - calendar is best-effort
            pass
        divs = t.dividends
        dd = [d.date().isoformat() for d in divs.index]
        confirmed = None
        try:
            ts = (t.info or {}).get("exDividendDate")
            if ts:
                confirmed = datetime.fromtimestamp(ts, timezone.utc).date()
        except Exception:  # noqa: BLE001
            pass
        return dict(earnings=earnings, ex_dividends=project_ex_dividends(dd, asof, confirmed=confirmed),
                    dividends=dict(dates=dd[-12:], amounts=[float(a) for a in divs.tolist()[-12:]]))


# ---------------------------------------------------------------- macro

class YahooFredMacro(MacroProvider):
    """USD/KRW from Yahoo (KRW=X), US 2y / 3m from FRED DGS2 / DTB3 (no key; 3m prices the options). The KRW 2y government bond yield has no
    free keyless source: use ECOS (Bank of Korea, needs ECOS_API_KEY) or the user-editable fallback."""
    name = "yahoo+fred"

    def __init__(self, ecos_key=None):
        self.ecos_key = ecos_key

    def get(self, key):
        if key == "usdkrw":
            import yfinance as yf
            warnings.filterwarnings("ignore", module="yfinance")
            h = yf.Ticker("KRW=X").history(period="5d")["Close"].dropna()
            if not len(h):
                raise ProviderError("USD/KRW: no data")
            return dict(value=float(h.iloc[-1]), observed=h.index[-1].date().isoformat(), source="Yahoo KRW=X")
        if key == "ust_2y":
            import pandas as pd
            df = pd.read_csv("https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS2")
            df = df[pd.to_numeric(df["DGS2"], errors="coerce").notna()]
            return dict(value=float(df["DGS2"].iloc[-1]) / 100, observed=str(df.iloc[-1, 0]), source="FRED DGS2")
        if key == "ust_3m":
            import pandas as pd
            df = pd.read_csv("https://fred.stlouisfed.org/graph/fredgraph.csv?id=DTB3")
            df = df[pd.to_numeric(df["DTB3"], errors="coerce").notna()]
            return dict(value=float(df["DTB3"].iloc[-1]) / 100, observed=str(df.iloc[-1, 0]), source="FRED DTB3")
        if key == "krw_2y":
            if not self.ecos_key:
                raise ProviderError("KRW 2y: no ECOS_API_KEY; using the editable fallback")
            return ecos_krw_2y(self.ecos_key)
        raise KeyError(key)


ECOS_TABLE, ECOS_ITEM = "817Y002", "010195000"   # 시장금리(일별) / 국고채(2년) — check the item code in ECOS


def ecos_krw_2y(key, today=None):
    import json
    import urllib.request
    today = today or date.today()
    start = (today - timedelta(days=20)).strftime("%Y%m%d")
    url = (f"https://ecos.bok.or.kr/api/StatisticSearch/{key}/json/kr/1/30/{ECOS_TABLE}/D/"
           f"{start}/{today.strftime('%Y%m%d')}/{ECOS_ITEM}")
    with urllib.request.urlopen(url, timeout=15) as r:
        return parse_ecos(json.load(r))


def parse_ecos(payload):
    rows = (payload.get("StatisticSearch") or {}).get("row") or []
    if not rows:
        raise ProviderError(f"ECOS: no rows ({payload.get('RESULT', {}).get('MESSAGE', 'unknown')})")
    last = rows[-1]
    t = last["TIME"]
    return dict(value=float(last["DATA_VALUE"]) / 100, observed=f"{t[:4]}-{t[4:6]}-{t[6:8]}",
                source=f"ECOS {ECOS_TABLE}/{ECOS_ITEM} ({last.get('ITEM_NAME1', '')})")

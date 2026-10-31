"""The API for the static (GitHub Pages) build: the same page functions as app/main.py, called from a Web
Worker running Pyodide instead of over HTTP. Market data comes only from the snapshot SQLite file that the
GitHub Actions workflow refreshes on US trading days; nothing is fetched from the browser.
"""
import json
from urllib.parse import parse_qs, urlparse

from data.cache import Cache
from data.providers import MacroProvider, MarketDataProvider, ProviderError
from data.service import MarketDataService

from .page1 import jsonable, run_page1
from .page2 import run_page2, run_screener
from .page3 import run_page3

REFRESH_NOTE = ("정적 사이트: 데이터는 평일 미국 장중에 GitHub Actions가 갱신합니다. "
                "다음 갱신 전까지는 마지막 스냅샷을 보여줍니다.")


class SnapshotProvider(MarketDataProvider):
    """Never fetches. Known tickers fall back to their cached snapshot (marked stale once a scheduled
    refresh has been missed); unknown tickers are reported, not invented."""
    name = "snapshot"

    def __init__(self, known):
        self.known = set(known)

    def _fail(self, ticker):
        if ticker.upper() not in self.known:
            raise ProviderError(f"'{ticker.upper()}'는 이 사이트의 데이터 목록(tickers.txt)에 없음 — "
                                f"추가하려면 저장소의 tickers.txt에 넣으면 다음 갱신 때 받음")
        raise ProviderError(REFRESH_NOTE)

    def info(self, ticker, asof):
        self._fail(ticker)

    def history(self, ticker, days=504):
        self._fail(ticker)

    def expiries(self, ticker):
        self._fail(ticker)

    def call_chain(self, ticker, expiry):
        self._fail(ticker)

    def events(self, ticker, asof):
        self._fail(ticker)


class SnapshotMacro(MacroProvider):
    name = "snapshot"

    def get(self, key):
        raise ProviderError(REFRESH_NOTE)


_svc = None


def service(db_path="/data/market.sqlite", reset=False):
    global _svc
    if _svc is None or reset:
        cache = Cache(db_path)
        _svc = MarketDataService(cache, SnapshotProvider(cache.keys("info")), SnapshotMacro())
    return _svc


def status(svc):
    st = svc.status()
    row = svc.cache.db.execute("SELECT MAX(fetched_at) FROM snapshots WHERE kind='chain'").fetchone()
    st.update(static=True, snapshot_asof=row[0] if row else None, tickers=svc.cache.keys("info"),
              refresh_note=REFRESH_NOTE, scheduler_running=False, next_refresh=None)
    return st


def dispatch(path, body_json):
    """dispatch('/api/page1', '{...}') -> JSON text. Errors come back as {"error": ...}."""
    svc = service()
    body = json.loads(body_json) if body_json else {}
    u = urlparse(path)
    p = u.path
    try:
        if p == "/api/status":
            out = status(svc)
        elif p == "/api/search":
            q = (parse_qs(u.query).get("q") or [""])[0].strip().upper()
            out = [dict(symbol=t, name=(svc.cache.latest("info", t) or {}).get("payload", {}).get("name", ""),
                        source="snapshot") for t in svc.cache.keys("info") if t.startswith(q)][:10] if q else []
        elif p == "/api/page1":
            out = run_page1(body, svc)
        elif p == "/api/page2":
            out = run_page2(body, svc)
        elif p == "/api/page3":
            out = run_page3(body, svc)
        elif p == "/api/screener":
            mac = svc.macros()
            r_usd = mac["ust_3m"].value["value"] if mac.get("ust_3m") else 0.04
            rows, errors = run_screener(svc, [t.upper() for t in body.get("tickers") or []] or svc.cache.keys("info"), r_usd)
            out = dict(rows=rows, errors=errors, note="IV rank is a proxy: today's ATM IV within the last year of "
                                                      "21-day realized vol (free data has no implied-vol history)")
        elif p == "/api/refresh":
            out = dict(error=REFRESH_NOTE)
        else:
            out = dict(error=f"unknown path {p}")
    except Exception as e:  # noqa: BLE001 - surfaced to the page like an HTTP error
        out = dict(error=f"{type(e).__name__}: {e}")
    return json.dumps(jsonable(out))

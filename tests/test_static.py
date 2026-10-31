"""Static (GitHub Pages) build: site assembly, snapshot compaction, and the in-browser API on a snapshot."""
import json
import sqlite3
import zipfile
from datetime import timedelta

import pytest

from app import static_api
from data.cache import Cache
from data.service import MarketDataService, RefreshSchedule
from test_data import Clock, FakeMacro, NOW
from test_page1 import BSProvider
from tools.build_site import build, read_tickers


@pytest.fixture(scope="module")
def snapshot(tmp_path_factory):
    d = tmp_path_factory.mktemp("snap")
    db = d / "market.sqlite"
    svc = MarketDataService(Cache(db), BSProvider(), FakeMacro(), RefreshSchedule(), Clock(NOW))
    for t in ("AAA", "BBB"):
        svc.ticker_bundle(t)
        svc.clock.t = NOW + timedelta(minutes=5)
        svc.ticker_bundle(t, refresh=True)            # two snapshots per key: compaction keeps the newest
    svc.macros()
    svc.cache.set_setting("share:abc1234567", dict(page="sim", state={}))
    return d, db


def test_build_site(snapshot, tmp_path):
    d, db = snapshot
    out = tmp_path / "site"
    build(db, out, log=lambda *a: None)
    assert (out / "index.html").read_text().count('data-static="1"') == 1
    assert (out / "worker.js").exists() and (out / ".nojekyll").exists()
    names = set(zipfile.ZipFile(out / "py.zip").namelist())
    assert {"app/static_api.py", "app/page1.py", "core/simulate.py", "data/service.py"} <= names
    assert not any(n.startswith("app/main") or "scheduler" in n for n in names)   # no server-only code
    c = sqlite3.connect(out / "data" / "market.sqlite")
    dup = c.execute("SELECT COUNT(*) FROM (SELECT kind, key FROM snapshots GROUP BY kind, key HAVING COUNT(*) > 1)").fetchone()[0]
    assert dup == 0
    assert c.execute("SELECT COUNT(*) FROM settings WHERE name LIKE 'share:%'").fetchone()[0] == 0
    assert c.execute("SELECT COUNT(*) FROM fetch_log").fetchone()[0] == 0


def test_static_api_on_snapshot(snapshot, tmp_path):
    d, db = snapshot
    out = tmp_path / "site"
    build(db, out, log=lambda *a: None)
    static_api.service(str(out / "data" / "market.sqlite"), reset=True)
    try:
        st = json.loads(static_api.dispatch("/api/status", ""))
        assert st["static"] and set(st["tickers"]) == {"AAA", "BBB"} and st["snapshot_asof"]
        assert json.loads(static_api.dispatch("/api/search?q=A", ""))[0]["symbol"] == "AAA"
        r = json.loads(static_api.dispatch("/api/page1", json.dumps(dict(tickers=[dict(ticker="AAA"), dict(ticker="ZZZ")],
                                                                          globals=dict(n_paths=1000)))))
        by = {b["ticker"]: b for b in r["tickers"]}
        assert by["AAA"]["in_simulation"] and by["AAA"]["stale"]           # snapshot is older than the last scheduled run
        assert "tickers.txt" in by["ZZZ"]["fatal"]                         # unknown ticker: reported, not invented
        assert "error" in json.loads(static_api.dispatch("/api/refresh", "{}"))
        assert "unknown path" in json.loads(static_api.dispatch("/api/nope", "{}"))["error"]
    finally:
        static_api._svc = None


def test_ticker_list():
    t = read_tickers("tickers.txt")
    assert len(t) == len(set(t)) and {"AMD", "NVDA", "JPM", "LLY"} <= set(t) and all(x.isupper() for x in t)

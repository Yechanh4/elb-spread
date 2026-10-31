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
from tools.build_site import export, merge, read_tickers, shard_of


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


def build(db, out):
    export(db, out, log=lambda *a: None, screener=True)


def test_build_site(snapshot, tmp_path):
    d, db = snapshot
    out = tmp_path / "site"
    build(db, out)
    assert (out / "index.html").read_text().count('data-static="1"') == 1
    assert (out / "worker.js").exists() and (out / ".nojekyll").exists()
    names = set(zipfile.ZipFile(out / "py.zip").namelist())
    assert {"app/static_api.py", "app/page1.py", "core/simulate.py", "data/service.py"} <= names
    assert not any(n.startswith("app/main") or "scheduler" in n for n in names)   # no server-only code
    core = sqlite3.connect(out / "data" / "core.sqlite")
    assert {r[0] for r in core.execute("SELECT DISTINCT kind FROM snapshots")} == {"macro"}
    assert core.execute("SELECT COUNT(*) FROM settings WHERE name LIKE 'share:%'").fetchone()[0] == 0
    idx = json.loads((out / "data" / "index.json").read_text())
    assert set(idx["tickers"]) == {"AAA", "BBB"} and idx["tickers"]["AAA"]["chains"] == 3
    assert [r["ticker"] for r in idx["screener"]["rows"]] and idx["screener"]["asof"]
    t = json.loads((out / "data" / "t" / "AAA.json").read_text())
    keys = [(r[0], r[1]) for r in t["rows"]]
    assert len(keys) == len(set(keys)) == 7                 # info, history, expiries, events + 3 chains, newest only
    assert (out / "data" / "full.sqlite.gz").exists()


def test_static_api_lazy_loading(snapshot, tmp_path):
    d, db = snapshot
    out = tmp_path / "site"
    build(db, out)
    static_api._svc = None
    static_api.set_index((out / "data" / "index.json").read_text())
    static_api.service(str(out / "data" / "core.sqlite"), reset=True)
    try:
        body = json.dumps(dict(tickers=[dict(ticker="AAA"), dict(ticker="ZZZ")], globals=dict(n_paths=1000)))
        assert json.loads(static_api.needed_tickers("/api/page1", body)) == ["AAA"]   # ZZZ is not in the universe
        r = json.loads(static_api.dispatch("/api/page1", body))
        assert r["tickers"][0]["fatal"]                       # AAA not loaded yet: reported, not invented
        static_api.load_ticker((out / "data" / "t" / "AAA.json").read_text())
        r = json.loads(static_api.dispatch("/api/page1", body))
        by = {b["ticker"]: b for b in r["tickers"]}
        assert by["AAA"]["in_simulation"] and by["AAA"]["stale"]
        assert "tickers.txt" in by["ZZZ"]["fatal"]
        st = json.loads(static_api.dispatch("/api/status", ""))
        assert st["static"] and st["tickers"] == ["AAA", "BBB"]
        assert [x["symbol"] for x in json.loads(static_api.dispatch("/api/search?q=B", ""))] == ["BBB"]
        scr = json.loads(static_api.dispatch("/api/screener", "{}"))
        assert scr["total"] == len(scr["rows"]) and "build time" in scr["note"]
        assert "error" in json.loads(static_api.dispatch("/api/refresh", "{}"))
    finally:
        static_api._svc = None


def test_merge_keeps_old_data_when_a_shard_is_missing(snapshot, tmp_path):
    d, db = snapshot
    shard = tmp_path / "s0.sqlite"
    c = sqlite3.connect(db)
    s = sqlite3.connect(shard)
    from data.cache import SCHEMA
    s.executescript(SCHEMA)
    row = c.execute("SELECT * FROM snapshots WHERE kind='info' AND key='AAA' ORDER BY fetched_at DESC").fetchone()
    s.execute("INSERT INTO snapshots VALUES (?,?,?,?,?)", (row[0], row[1], "2099-01-01T00:00:00+00:00", row[3], row[4]))
    s.commit()
    out = tmp_path / "full.sqlite"
    merge(db, [shard, tmp_path / "missing.sqlite"], out, log=lambda *a: None)
    m = sqlite3.connect(out)
    assert m.execute("SELECT fetched_at FROM snapshots WHERE kind='info' AND key='AAA'").fetchall() == [("2099-01-01T00:00:00+00:00",)]
    assert m.execute("SELECT COUNT(*) FROM snapshots WHERE key LIKE 'BBB%'").fetchone()[0] == 7     # untouched


def test_ticker_list_and_shards():
    t = read_tickers("tickers.txt")
    assert len(t) == len(set(t)) > 500 and {"AMD", "NVDA", "JPM", "LLY", "BRK-B"} <= set(t)
    assert all(x == x.upper() and "." not in x for x in t)
    parts = [shard_of(t, f"{i}/8") for i in range(8)]
    assert sorted(sum(parts, [])) == sorted(t) and max(map(len, parts)) - min(map(len, parts)) <= 1

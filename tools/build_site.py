"""Build the static GitHub Pages site.

    python -m tools.build_site site [--db data/market.sqlite] [--out site]     # default command
    python -m tools.build_site refresh --db shard.sqlite --shard 3/8           # Yahoo/FRED -> one shard
    python -m tools.build_site merge --base prev.sqlite --shards s0.sqlite s1.sqlite --out full.sqlite

Layout of site/data/ (the browser never downloads the whole universe):
    core.sqlite        macro series + settings (small, loaded at start)
    index.json         ticker -> name/sector/data time, plus the screener precomputed over every ticker
    t/<TICKER>.json    newest snapshot rows for one ticker, fetched only when that ticker is used
    full.sqlite.gz     everything, so the next workflow run starts from it (keeps data if Yahoo refuses)
"""
import argparse
import gzip
import json
import shutil
import sqlite3
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY_FILES = (["core/*.py"] + [f"app/{n}.py" for n in ("__init__", "page1", "page2", "page3", "share", "static_api")]
            + [f"data/{n}.py" for n in ("__init__", "cache", "service", "providers")])
TICKER_KINDS = ("info", "history", "expiries", "events")


def read_tickers(path):
    out = []
    for ln in Path(path).read_text().splitlines():
        t = ln.split("#")[0].strip().upper()
        if t and t not in out:
            out.append(t)
    return out


def shard_of(tickers, spec):
    i, n = (int(x) for x in spec.split("/"))
    return tickers[i::n]


def refresh(db, tickers, log=print, macros=True):
    from data.cache import Cache
    from data.service import MarketDataService
    svc = MarketDataService(Cache(db))
    bad = {}
    for k, t in enumerate(tickers):
        try:
            b = svc.ticker_bundle(t, refresh=True)
            if b.errors:
                bad[t] = list(b.errors.values())[0][:120]
        except Exception as e:  # noqa: BLE001 - one bad ticker never stops the shard
            bad[t] = f"{type(e).__name__}: {e}"[:120]
        if (k + 1) % 20 == 0:
            log(f"  {k + 1}/{len(tickers)}")
    if macros:
        svc.macros(refresh=True)
    log(f"refreshed {len(tickers) - len(bad)}/{len(tickers)} cleanly; problems: "
        + "; ".join(f"{t}: {e}" for t, e in list(bad.items())[:30]))
    return bad


def compact(src, dst):
    """Newest snapshot per (kind, key), settings and watchlist only — no history, logs or share ids."""
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    from data.cache import SCHEMA
    out = sqlite3.connect(dst)
    out.executescript(SCHEMA)
    out.execute("ATTACH DATABASE ? AS src", (str(src),))
    out.execute("""INSERT INTO snapshots SELECT s.* FROM src.snapshots s
                   JOIN (SELECT kind, key, MAX(fetched_at) m FROM src.snapshots GROUP BY kind, key) t
                   ON s.kind = t.kind AND s.key = t.key AND s.fetched_at = t.m""")
    out.execute("INSERT INTO settings SELECT * FROM src.settings WHERE name NOT LIKE 'share:%'")
    out.execute("INSERT OR IGNORE INTO watchlist SELECT * FROM src.watchlist")
    out.commit()
    out.execute("DETACH DATABASE src")
    out.execute("VACUUM")
    out.close()


def merge(base, shards, out, log=print):
    """Start from the previous snapshot, add every shard's rows, keep the newest per key."""
    work = Path(str(out) + ".work")
    shutil.copy(base, work)
    c = sqlite3.connect(work)
    for s in shards:
        if not Path(s).exists():
            log(f"  shard missing, keeping previous data: {s}")
            continue
        c.execute("ATTACH DATABASE ? AS sh", (str(s),))
        n = c.execute("SELECT COUNT(*) FROM sh.snapshots").fetchone()[0]
        c.execute("INSERT OR REPLACE INTO snapshots SELECT * FROM sh.snapshots")
        c.commit()
        c.execute("DETACH DATABASE sh")
        log(f"  merged {n} rows from {Path(s).name}")
    c.close()
    compact(work, out)
    work.unlink()


def _rows_by_ticker(db):
    c = sqlite3.connect(db)
    by = {}
    for kind, key, fa, src, payload in c.execute("SELECT kind, key, fetched_at, source, payload FROM snapshots "
                                                 "WHERE kind != 'macro'"):
        t = key.split("|")[0]
        by.setdefault(t, []).append([kind, key, fa, src, payload])
    return by


def export(db, out, log=print, screener=True):
    """site/: web files, py.zip, data/core.sqlite, data/t/*.json, data/index.json, data/full.sqlite.gz."""
    out = Path(out)
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(ROOT / "web", out)
    data = out / "data"
    (data / "t").mkdir(parents=True)
    full = data / "full.sqlite"
    compact(db, full)
    # core.sqlite: macro + settings only
    core = data / "core.sqlite"
    shutil.copy(full, core)
    c = sqlite3.connect(core)
    c.execute("DELETE FROM snapshots WHERE kind != 'macro'")
    c.execute("DELETE FROM watchlist")
    c.commit()
    c.execute("VACUUM")
    c.close()
    index = {}
    for t, rows in sorted(_rows_by_ticker(full).items()):
        info = next((json.loads(r[4]) for r in rows if r[0] == "info"), None)
        if info is None:
            continue
        (data / "t" / f"{t}.json").write_text(json.dumps(dict(ticker=t, rows=rows), separators=(",", ":")))
        index[t] = dict(name=info.get("name", t), sector=info.get("sector", ""),
                        asof=min(r[2] for r in rows), chains=sum(1 for r in rows if r[0] == "chain"))
    scr = dict(rows=[], errors={}, asof=None)
    if screener:
        scr = precompute_screener(full, list(index), log)
    (data / "index.json").write_text(json.dumps(dict(generated=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                                     tickers=index, screener=scr), separators=(",", ":")))
    with open(full, "rb") as f, gzip.open(str(full) + ".gz", "wb", compresslevel=9) as g:
        shutil.copyfileobj(f, g)
    full.unlink()
    with zipfile.ZipFile(out / "py.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for pat in PY_FILES:
            for f in sorted(ROOT.glob(pat)):
                z.write(f, f.relative_to(ROOT).as_posix())
    idx = out / "index.html"
    idx.write_text(idx.read_text().replace('<html lang="ko">', '<html lang="ko" data-static="1">', 1))
    (out / ".nojekyll").write_text("")
    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    log(f"site built in {out}: {len(index)} tickers, {size / 1e6:.1f} MB on disk")


def _screen_chunk(args):
    db, tickers, known = args
    from app.page2 import run_screener
    from app.static_api import SnapshotMacro, SnapshotProvider
    from data.cache import Cache
    from data.service import MarketDataService
    svc = MarketDataService(Cache(db), SnapshotProvider(known), SnapshotMacro())
    mac = svc.macros()
    r_usd = mac["ust_3m"].value["value"] if mac.get("ust_3m") else 0.04
    return run_screener(svc, tickers, r_usd)


def precompute_screener(db, tickers, log=print, workers=None):
    """Screener metrics for every ticker, computed once at build time (in parallel) so the static site can
    list the cheapest options across the whole universe without downloading it."""
    import os
    from concurrent.futures import ProcessPoolExecutor
    from app.page1 import jsonable
    from core.vol import screen
    workers = workers or min(4, os.cpu_count() or 1)
    chunks = [(str(db), tickers[i::workers], tickers) for i in range(workers)]
    rows, errors = [], {}
    if workers == 1:
        parts = [_screen_chunk(chunks[0])]
    else:
        with ProcessPoolExecutor(workers) as ex:
            parts = list(ex.map(_screen_chunk, chunks))
    for r, e in parts:
        rows += r
        errors.update(e)
    rows = screen(rows)
    log(f"screener: {len(rows)} ranked, {len(errors)} without a usable chain")
    return jsonable(dict(rows=rows, errors=errors, asof=datetime.now(timezone.utc).isoformat(timespec="seconds")))


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")
    s = sub.add_parser("site")
    s.add_argument("--db", default=str(ROOT / "data" / "market.sqlite"))
    s.add_argument("--out", default=str(ROOT / "site"))
    s.add_argument("--no-screener", action="store_true")
    r = sub.add_parser("refresh")
    r.add_argument("--db", required=True)
    r.add_argument("--tickers", default=str(ROOT / "tickers.txt"))
    r.add_argument("--shard", default="0/1")
    m = sub.add_parser("merge")
    m.add_argument("--base", required=True)
    m.add_argument("--shards", nargs="*", default=[])
    m.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "refresh":
        tickers = shard_of(read_tickers(a.tickers), a.shard)
        print(f"shard {a.shard}: {len(tickers)} tickers")
        refresh(a.db, tickers, macros=a.shard.startswith("0/"))
    elif a.cmd == "merge":
        merge(a.base, a.shards, a.out)
    else:
        export(getattr(a, "db", str(ROOT / "data" / "market.sqlite")), getattr(a, "out", str(ROOT / "site")),
               screener=not getattr(a, "no_screener", False))


if __name__ == "__main__":
    main()

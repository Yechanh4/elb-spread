"""Build the static GitHub Pages site into site/.

    python -m tools.build_site                      # snapshot = current data/market.sqlite
    python -m tools.build_site --refresh            # first refresh tickers.txt from Yahoo/FRED (Actions does this)
    python -m tools.build_site --db prev.sqlite     # start from another snapshot (e.g. the deployed one)

The browser runs core/ + app/ + data/ in Pyodide (web/worker.js) against site/data/market.sqlite.
"""
import argparse
import shutil
import sqlite3
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY_FILES = (["core/*.py"] + [f"app/{n}.py" for n in ("__init__", "page1", "page2", "page3", "share", "static_api")]
            + [f"data/{n}.py" for n in ("__init__", "cache", "service", "providers")])


def read_tickers(path):
    return [ln.strip().upper() for ln in Path(path).read_text().splitlines() if ln.strip() and not ln.startswith("#")]


def refresh(db, tickers, log=print):
    from data.cache import Cache
    from data.service import MarketDataService
    svc = MarketDataService(Cache(db))
    for t in tickers:
        svc.cache.watch(t)
    rep = svc.refresh_all(tickers)
    ok = [t for t, v in rep["tickers"].items() if v["ok"]]
    log(f"refreshed {len(ok)}/{len(tickers)} tickers cleanly; problems: "
        + "; ".join(f"{t}: {list(v['errors'].values())[:1]}" for t, v in rep["tickers"].items() if not v["ok"]))
    return rep


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
    out.execute("INSERT INTO watchlist SELECT * FROM src.watchlist")
    out.commit()
    out.execute("DETACH DATABASE src")
    out.execute("VACUUM")
    out.close()


def build(db, out, tickers=None, do_refresh=False, log=print):
    out = Path(out)
    if out.exists():
        shutil.rmtree(out)
    work = out.parent / (out.name + "-work.sqlite")
    shutil.copy(db, work)
    if do_refresh:
        refresh(str(work), tickers, log)
    shutil.copytree(ROOT / "web", out)
    compact(work, out / "data" / "market.sqlite")
    work.unlink()
    with zipfile.ZipFile(out / "py.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for pat in PY_FILES:
            for f in sorted(ROOT.glob(pat)):
                z.write(f, f.relative_to(ROOT).as_posix())
    idx = out / "index.html"
    idx.write_text(idx.read_text().replace('<html lang="ko">', '<html lang="ko" data-static="1">', 1))
    (out / ".nojekyll").write_text("")
    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    log(f"site built in {out} ({size / 1e6:.1f} MB)")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(ROOT / "data" / "market.sqlite"))
    ap.add_argument("--out", default=str(ROOT / "site"))
    ap.add_argument("--tickers", default=str(ROOT / "tickers.txt"))
    ap.add_argument("--refresh", action="store_true")
    a = ap.parse_args(argv)
    build(a.db, a.out, read_tickers(a.tickers), a.refresh)


if __name__ == "__main__":
    main()

"""FastAPI app: one process, one URL. API under /api, the static SPA at /.

    .venv/bin/uvicorn app.main:app --port 8780
Env: ELB_DB (SQLite path, default data/market.sqlite), ELB_SCHEDULER=0 to skip the daily refresh job,
ECOS_API_KEY for the KRW 2y yield, ELB_BASIC_AUTH="user:password" to put the whole app behind HTTP Basic
auth when it is hosted on a public URL (the health check /api/status stays open), ELB_MAX_PATHS to cap
Monte Carlo path counts (memory scales with paths; Render Starter has 512 MB).
"""
import base64
import hmac
import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from data.cache import Cache
from data.service import MACRO_KEYS, DataUnavailable, MarketDataService

from .page1 import jsonable, run_page1
from .page2 import run_page2, run_screener
from .page3 import run_page3
from .share import load_share, save_share

ROOT = Path(__file__).resolve().parents[1]
svc = MarketDataService(Cache(os.environ.get("ELB_DB", str(ROOT / "data" / "market.sqlite"))))
scheduler = None


@asynccontextmanager
async def lifespan(app):
    global scheduler
    if os.environ.get("ELB_SCHEDULER", "1") != "0":
        from data.scheduler import build_scheduler
        scheduler = build_scheduler(svc)
        scheduler.start()
    yield
    if scheduler:
        scheduler.shutdown(wait=False)


app = FastAPI(title="ELB call-spread simulator", lifespan=lifespan)


_AUTH = os.environ.get("ELB_BASIC_AUTH", "")


@app.middleware("http")
async def _basic_auth(request, call_next):
    """Optional: one shared login for a hosted copy. Off unless ELB_BASIC_AUTH is set."""
    if _AUTH and request.url.path != "/api/status":
        want = "Basic " + base64.b64encode(_AUTH.encode()).decode()
        if not hmac.compare_digest(request.headers.get("authorization", ""), want):
            return JSONResponse(status_code=401, content=dict(error="authentication required"),
                                headers={"WWW-Authenticate": 'Basic realm="elb-spread"'})
    return await call_next(request)


@app.middleware("http")
async def _no_stale_assets(request, call_next):
    """Static files revalidate on every load (ETag), so a new deploy is never masked by a cached page."""
    resp = await call_next(request)
    if not request.url.path.startswith("/api"):
        resp.headers["Cache-Control"] = "no-cache"
    return resp


@app.exception_handler(DataUnavailable)
def _unavailable(request, exc):
    return JSONResponse(status_code=503, content=dict(error=str(exc)))


@app.get("/api/status")
def status():
    st = svc.status()
    job = scheduler.get_job("daily-refresh") if scheduler else None
    st["next_refresh"] = job.next_run_time.isoformat() if job and job.next_run_time else None
    st["scheduler_running"] = bool(scheduler)
    return jsonable(st)


@app.get("/api/search")
def search(q: str = ""):
    return svc.search(q) if q.strip() else []


@app.get("/api/ticker/{ticker}")
def ticker(ticker: str, refresh: bool = False):
    b = svc.ticker_bundle(ticker, refresh=refresh)
    if b.info is None:
        raise HTTPException(404, detail=dict(error="; ".join(b.errors.values()) or "not found"))
    return jsonable(dict(ticker=b.ticker, info=b.info.value, expiries=b.selected_expiries, errors=b.errors,
                         stale=b.stale, data_asof=b.data_asof, freshness=[i.meta() for i in b.items()]))


@app.post("/api/refresh")
def refresh(body: dict = Body(default={})):
    """Manual refresh of the given tickers (default: watchlist) and the macro series."""
    tickers = [t.upper() for t in body.get("tickers") or []]
    for t in tickers:
        svc.cache.watch(t)
    return jsonable(svc.refresh_all(tickers or None))


@app.post("/api/macro/{key}")
def set_macro(key: str, body: dict = Body(...)):
    if key not in MACRO_KEYS:
        raise HTTPException(404, detail="unknown macro key")
    if body.get("value") is None:
        svc.clear_override(key)
    else:
        svc.set_override(key, float(body["value"]))
    return jsonable(svc.macro_item(key).meta() | dict(value=svc.macro_item(key).value["value"]))


# One simulation at a time: requests queue instead of stacking their peak memory.
_COMPUTE = threading.Lock()
_MAX_PATHS = int(os.environ.get("ELB_MAX_PATHS", "0") or 0)


def _capped(state, keys=("n_paths", "n_check")):
    if _MAX_PATHS:
        g = dict(state.get("globals") or {})
        for k in keys:
            if g.get(k) is not None:
                g[k] = min(int(g[k]), _MAX_PATHS)
        state = {**state, "globals": g}
    return state


@app.post("/api/page1")
def page1(state: dict = Body(...)):
    for t in state.get("tickers") or []:
        svc.cache.watch(t["ticker"])
    with _COMPUTE:
        return run_page1(_capped(state), svc)


@app.post("/api/page2")
def page2(state: dict = Body(...)):
    for c in state.get("candidates") or []:
        svc.cache.watch(c["ticker"])
    with _COMPUTE:
        return run_page2(_capped(state), svc)


@app.post("/api/screener")
def screener(body: dict = Body(default={})):
    """Screen the given tickers (default: watchlist), cheapest options first: (IV - HV), then IV-rank proxy."""
    tickers = [t.upper() for t in body.get("tickers") or []] or svc.cache.watchlist()
    mac = svc.macros()
    r_usd = mac["ust_3m"].value["value"] if mac.get("ust_3m") else 0.04
    rows, errors = run_screener(svc, tickers, r_usd)
    return jsonable(dict(rows=rows, errors=errors, note="IV rank is a proxy: today's ATM IV within the last "
                         "year of 21-day realized vol (free data has no implied-vol history)"))


@app.post("/api/page3")
def page3(state: dict = Body(...)):
    st = _capped(state)
    if _MAX_PATHS and "n_paths" not in (st.get("globals") or {}):
        st = {**st, "globals": {**(st.get("globals") or {}), "n_paths": min(20_000, _MAX_PATHS)}}
    with _COMPUTE:
        return run_page3(st, svc)


@app.post("/api/share")
def share(body: dict = Body(...)):
    """Store a page state under a short content-addressed id: the same state always gets the same link."""
    if body.get("page") not in ("sim", "optimizer", "timeline") or not isinstance(body.get("state"), dict):
        raise HTTPException(400, detail="need page in {sim, optimizer, timeline} and a state object")
    return dict(id=save_share(svc.cache, body["page"], body["state"]))


@app.get("/api/share/{sid}")
def get_share(sid: str):
    got = load_share(svc.cache, sid)
    if got is None:
        raise HTTPException(404, detail="unknown share id")
    return got


app.mount("/", StaticFiles(directory=str(ROOT / "web"), html=True), name="web")

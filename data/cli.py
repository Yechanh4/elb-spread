"""Command line for the data layer.

    .venv/bin/python -m data.cli fetch AMD                 # spot, events, 3 monthly chains (>60d), strike check
    .venv/bin/python -m data.cli fetch AMD --full          # every listed strike
    .venv/bin/python -m data.cli fetch AMD --sigma-hat 0.62
    .venv/bin/python -m data.cli watch AMD NVDA LLY        # add to the daily refresh list
    .venv/bin/python -m data.cli refresh-all
    .venv/bin/python -m data.cli set-krw2y 0.0376          # user-edited KRW 2y yield (until cleared)
    .venv/bin/python -m data.cli schedule 15:30 America/New_York
    .venv/bin/python -m data.cli status
    .venv/bin/python -m data.cli run-scheduler             # blocking; refreshes the watchlist daily
"""
import argparse
import json
import sys

import numpy as np

from core.bsm import implied_vol
from core.spread import build_spread, k1_target
from core.vol import hv_close_to_close, hv_parkinson

from .cache import iso
from .service import DataUnavailable, MarketDataService, chain_slice


def badge(item):
    if item is None:
        return "[MISSING]"
    if item.origin == "fallback":
        return "[ASSUMPTION]"
    if item.origin == "user":
        return "[USER]"
    return "[STALE]" if item.stale else "[fresh]"


def atm_iv(rows, F):
    K = np.array([r["strike"] for r in rows])
    iv = np.array([r["iv"] for r in rows])
    ok = np.isfinite(iv) & (iv > 0.01)
    if ok.sum() < 2:
        return float("nan")
    return float(np.interp(F, K[ok], iv[ok]))


def print_fetch(svc, ticker, refresh=False, full=False, sigma_hat=None, budget_krw=79.7e6):
    b = svc.ticker_bundle(ticker, refresh=refresh)
    mac = svc.macros()
    print(f"\n=== {b.ticker} === data as of {iso(b.data_asof) if b.data_asof else '-'} (UTC)"
          f"{'   ** STALE / PARTIAL **' if b.stale else ''}")
    print(svc.status()["delay_note"])
    for k, v in b.errors.items():
        print(f"  ! {k}: {v}")
    if b.info is None:
        return 1
    inf = b.info.value
    S0 = inf["spot"]
    print(f"\n[market] {inf['name']}  spot {S0:,.2f} {inf['currency']}  sector {inf['sector']} ({inf['sector_source']})"
          f"  div yield {inf['div_yield']:.2%} (TTM)  {badge(b.info)}")
    for k in ("usdkrw", "krw_2y", "ust_2y", "ust_3m"):
        it = mac.get(k)
        if it is None:
            print(f"[macro]  {k}: unavailable ({mac.get(k + '_error')})")
            continue
        v = it.value["value"]
        txt = f"{v:,.2f}" if k == "usdkrw" else f"{v:.2%}"
        print(f"[macro]  {k}: {txt}  ({it.source}, observed {it.value.get('observed') or '-'}) {badge(it)}")
    fx = mac["usdkrw"].value["value"] if mac.get("usdkrw") else 1400.0
    rf = mac["ust_3m"].value["value"] if mac.get("ust_3m") else 0.04

    if b.history is not None:
        h = b.history.value
        c = np.array(h["close"])
        print(f"\n[market] history {h['dates'][0]} .. {h['dates'][-1]} ({len(c)} bars) {badge(b.history)}")
        print(f"[model]  HV close-to-close 1y {hv_close_to_close(c, 252):.1%}  2y {hv_close_to_close(c):.1%}"
              f"   Parkinson 1y {hv_parkinson(h['high'], h['low'], 252):.1%}")
    if b.events is not None:
        ev = b.events.value
        er = ", ".join(e["date"] + (" (est.)" if e["estimated"] else "") for e in ev["earnings"]) or "none listed"
        xd = ", ".join(e["date"] + (" (projected)" if e["projected"] else "") for e in ev["ex_dividends"]) or "none"
        print(f"[market] earnings: {er}\n[market] ex-dividend: {xd}  {badge(b.events)}")

    print(f"\n[market] {len(b.expiries.value) if b.expiries else 0} listed expiries; "
          f"monthly > 60 days: {', '.join(b.selected_expiries) or 'none'}")
    for e, it in b.chains.items():
        cs = chain_slice(it, b.asof_date)
        rows = it.value["rows"]
        F = S0 * np.exp((rf - inf["div_yield"]) * cs.T)
        iv_atm = atm_iv(rows, F)
        sh = sigma_hat or iv_atm
        Kt = k1_target(S0, sh, 0.25)
        quoted = cs.quoted()
        print(f"\n--- {e}  T={cs.T:.3f}y  session at fetch: {it.value.get('session')}  {badge(it)}"
              f"  fetched {iso(it.fetched_at)}")
        print(f"    {len(rows)} strikes, {quoted.sum()} two-sided.  ATM IV (Yahoo, at forward {F:,.2f}) = {iv_atm:.1%}")
        print(f"    [model] K1_target = S0(1 + 1.56 sigma_hat sqrt(0.25)) with sigma_hat = {sh:.1%}"
              f"{' (placeholder = ATM IV: no research input, edge = 0)' if sigma_hat is None else ''} -> {Kt:,.2f}")
        try:
            sp = build_spread(cs, S0, sh, budget_krw, fx, T_opt=0.25)
            print(f"    [model] listed K1 = {sp.K1:g}, K2 = {sp.K2:g} (width {sp.K2 - sp.K1:g}), net mid {sp.net_mid:.2f}, "
                  f"chain round-trip {sp.rt_cost_chain:.1%} vs default 4%, liquidity "
                  f"{'PASS' if sp.liquidity_ok else 'FAIL'} {json.dumps({k: v for k, v in sp.liquidity.items() if k.startswith(('oi', 'spread'))})}")
        except ValueError as ex:
            sp = None
            print(f"    [model] no spread: {ex}")
        mid_iv = implied_vol(cs.mid, S0, cs.strike, cs.T, rf, inf["div_yield"])
        lo, hi = (0, len(rows)) if full else (
            int(np.searchsorted(cs.strike, S0 * 0.95)), int(np.searchsorted(cs.strike, (sp.K2 if sp else Kt) * 1.04)) + 1)
        print(f"    {'strike':>8} {'bid':>8} {'ask':>8} {'mid':>8} {'last':>8} {'IV(Y)':>7} {'IV(mid)':>7} "
              f"{'OI':>7} {'vol':>6} {'spr%':>6}")
        for i in range(max(lo, 0), min(hi, len(rows))):
            m = cs.mid[i]
            spr = (cs.ask[i] - cs.bid[i]) / m if quoted[i] and m > 0 else float("nan")
            tag = " <K1" if sp and cs.strike[i] == sp.K1 else (" <K2" if sp and cs.strike[i] == sp.K2 else "")
            print(f"    {cs.strike[i]:>8g} {cs.bid[i]:>8.2f} {cs.ask[i]:>8.2f} {m:>8.2f} {cs.last[i]:>8.2f} "
                  f"{cs.iv[i]:>7.1%} {mid_iv[i]:>7.1%} {cs.oi[i]:>7.0f} {cs.volume[i]:>6.0f} {spr:>6.1%}{tag}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="data.cli")
    ap.add_argument("--db", default="data/market.sqlite")
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("ticker")
    f.add_argument("--refresh", action="store_true")
    f.add_argument("--full", action="store_true")
    f.add_argument("--sigma-hat", type=float)
    w = sub.add_parser("watch")
    w.add_argument("tickers", nargs="+")
    sub.add_parser("refresh-all")
    k = sub.add_parser("set-krw2y")
    k.add_argument("value", type=float)
    sub.add_parser("clear-krw2y")
    s = sub.add_parser("schedule")
    s.add_argument("hhmm")
    s.add_argument("tz", nargs="?", default="America/New_York")
    sub.add_parser("status")
    sub.add_parser("run-scheduler")
    a = ap.parse_args(argv)

    from .cache import Cache
    svc = MarketDataService(Cache(a.db))
    if a.cmd == "fetch":
        return print_fetch(svc, a.ticker, a.refresh, a.full, a.sigma_hat)
    if a.cmd == "watch":
        for t in a.tickers:
            svc.cache.watch(t)
        print("watchlist:", svc.cache.watchlist())
    elif a.cmd == "refresh-all":
        print(json.dumps(svc.refresh_all(), indent=2))
    elif a.cmd == "set-krw2y":
        if not 0 < a.value < 0.2:
            sys.exit("give the yield as a decimal, e.g. 0.0376")
        svc.set_override("krw_2y", a.value)
        print("KRW 2y set to", a.value)
    elif a.cmd == "clear-krw2y":
        svc.clear_override("krw_2y")
    elif a.cmd == "schedule":
        hh, mm = (int(x) for x in a.hhmm.split(":"))
        svc.set_schedule(hh, mm, a.tz)
        print("refresh:", svc.schedule.describe())
    elif a.cmd == "status":
        print(json.dumps(svc.status(), indent=2, default=str))
    elif a.cmd == "run-scheduler":
        import time
        from .scheduler import build_scheduler
        sch = build_scheduler(svc)
        sch.start()
        print("scheduler running:", svc.schedule.describe(), "| next:", sch.get_job("daily-refresh").next_run_time)
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            sch.shutdown()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except DataUnavailable as e:
        sys.exit(f"data unavailable: {e}")

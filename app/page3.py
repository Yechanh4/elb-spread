"""Page 3 — 8-year execution timeline. run_page3(state, service) -> JSON-able dict.

All three rate scenarios run on one set of uniforms, so switching scenarios on the page changes only what
the scenario changes (rates -> budgets, and sigma/mu views), never the dice. Representative paths are
rebuilt down to each ticker's spot path from the same uniforms, which gives every future roll concrete
strikes (K = k x spot at that roll's entry) instead of only moneyness.
"""
from datetime import date, timedelta

import numpy as np

from core import budget as bud
from core.rules import STRUCTURAL_NOTES, Position, action_list, close_by_date, evaluate
from core.simulate import (SimConfig, SimResult, correlation_from_prices, draw_uniforms, pin_prob, summarize)
from core.simulate import simulate as run_sim
from data.service import DataUnavailable

from .page1 import aligned_closes, jsonable
from .page2 import DEFAULT as P2DEFAULT, prepare

DEFAULT_GLOBALS = dict(V_deposit=bud.V_DEPOSIT, V_own_rental=bud.V_OWN_RENTAL, r=None, T=bud.T_CYCLE, G=bud.GOAL,
                       tax=True, copula="gaussian", seed=0, start="2026-12-01", n_paths=20_000, leg_cross=1.0,
                       rt_cost=0.04, broker_confirmed=None, mortgage_rate=0.04, remaining_months=264)
N_BAND = 2000          # paths kept for per-roll bands and representative paths


def project_ex_divs(div_dates, start, end):
    """Quarterly-style projection over the whole program from the recent payment cadence."""
    ds = sorted(date.fromisoformat(d) for d in div_dates)
    if len(ds) < 3:
        return []
    step = int(np.median(np.diff([d.toordinal() for d in ds[-6:]])))
    if not 20 <= step <= 400:
        return []
    out, d = [], ds[-1]
    while d <= end:
        d = d + timedelta(days=step)
        if start <= d <= end:
            out.append(d)
    return out


def _sub_result(res, run):
    return SimResult(runs=[type(run)(**{**run.__dict__, "prob": 1.0})], cfg=res.cfg, tickers=res.tickers,
                     weights=res.weights, calendar=res.calendar)


def _rep_indices(x, goal):
    order = np.argsort(x)
    med = int(order[len(order) // 2])
    low = int(order[max(0, int(len(order) * 0.10))])
    hits = np.flatnonzero(x >= goal)
    hit = int(hits[np.argmin(np.abs(x[hits] - np.median(x[hits])))]) if len(hits) else None
    out = dict(median=med, low=low)
    if hit is not None:
        out["goal"] = hit
    return out


def run_page3(state, svc):
    g = {**DEFAULT_GLOBALS, **(state.get("globals") or {})}
    scen = state.get("scenarios") or P2DEFAULT["scenarios"]
    names = tuple(s["name"] for s in scen)
    probs = bud.check_scenarios([bud.RateScenario(s["name"], s["shift_bp"] / 1e4, float(s["prob"])) for s in scen])
    mac = svc.macros()
    if g["r"] is None:
        g["r"] = mac["krw_2y"].value["value"]
    if not mac.get("usdkrw"):
        raise DataUnavailable("USD/KRW unavailable")
    fx = mac["usdkrw"].value["value"]
    r_usd = mac["ust_3m"].value["value"] if mac.get("ust_3m") else 0.04
    cons = {**P2DEFAULT["constraints"], **(state.get("constraints") or {})}
    tick_in = state.get("tickers") or []
    n_cyc = bud.N_CYCLES
    W_in = state.get("weights")
    f_in = state.get("f", 1.0)
    f_cycles = tuple(float(x) for x in (f_in if isinstance(f_in, (list, tuple)) else [f_in] * n_cyc))
    B_by = {s["name"]: bud.cycle_budgets(g["r"], s["shift_bp"] / 1e4, n_cyc, g["V_deposit"], g["V_own_rental"], g["T"])
            for s in scen}
    B0 = float(B_by[names[int(np.argmax(probs))]][1][0])
    out = dict(globals=g, scenarios=scen, f_cycles=f_cycles, macro=dict(usdkrw=fx, ust_3m=r_usd), tickers=[],
               warnings=[], notes=STRUCTURAL_NOTES, source=state.get("source", "manual"))
    specs, series, keep_idx, extras, infos = [], [], [], [], []
    for i, t in enumerate(tick_in):
        cand = dict(ticker=t["ticker"], sigma=t.get("sigma"), mu=t.get("mu"), conf=t.get("conf", 0.7))
        try:
            info, spec, ser = prepare(svc, cand, names, probs, {**P2DEFAULT["globals"], **g}, cons, B0, fx, r_usd)
        except DataUnavailable as ex:
            info, spec, ser = dict(ticker=t["ticker"].upper(), eligible=False, reason=str(ex)), None, None
        out["tickers"].append(info)
        if spec is not None:
            specs.append(spec)
            series.append((ser[0], ser[1]))
            keep_idx.append(i)
            extras.append(ser[2])
            infos.append(info)
    if not specs:
        out["error"] = "no eligible ticker (see the list for reasons)"
        return jsonable(out)
    # per-cycle weights over the eligible tickers (renormalized; equal if none given)
    k = len(specs)
    W = np.full((n_cyc, k), 1.0 / k)
    if W_in:
        Wf = np.array([[float(row[i]) if row[i] is not None else 0.0 for i in keep_idx] for row in W_in], dtype=float)
        if Wf.shape == (n_cyc, k):
            for c in range(n_cyc):
                W[c] = Wf[c] / Wf[c].sum() if Wf[c].sum() > 0 else 1.0 / k
            if not np.allclose(Wf.sum(axis=1), 1.0):
                out["warnings"].append("weights renormalized to 100% per cycle over the eligible tickers")
    C = np.eye(1) if k == 1 else correlation_from_prices(aligned_closes(series))
    scen_objs = tuple(bud.RateScenario(s["name"], s["shift_bp"] / 1e4, float(s["prob"])) for s in scen)
    n = int(g["n_paths"])
    cfg = SimConfig(n_paths=n, n_store=min(N_BAND, n), goal=g["G"], V_deposit=g["V_deposit"], V_own_rental=g["V_own_rental"],
                    r0=g["r"], T_cycle=g["T"], scenarios=scen_objs, tax=g["tax"], copula=g["copula"],
                    seed=int(g["seed"]), start=date.fromisoformat(g["start"]), f=f_cycles[0], f_cycles=f_cycles)
    U = draw_uniforms(n, cfg.n_rolls, C, cfg.copula, cfg.df, cfg.seed)
    res = run_sim(specs, W, C, cfg, U=U)
    cal = res.calendar
    end = cal[-1]["expiry"]
    # deterministic schedule: dates, combination, ex-div conflicts, rules, actions
    exdiv = {}
    for sp, ex in zip(specs, extras):
        b = svc.ticker_bundle(sp.name)
        dd = b.events.value["dividends"]["dates"] if b.events else []
        exdiv[sp.name] = project_ex_divs(dd, cfg.start, end)
    kw = dict(mortgage_rate=g["mortgage_rate"], remaining_months=g["remaining_months"])
    schedule = []
    for row in cal:
        j, c = row["roll"], row["roll"] // cfg.rolls_per_cycle
        legs = []
        for i, (sp, info) in enumerate(zip(specs, infos)):
            if W[c, i] <= 0:
                continue
            conflicts = [d.isoformat() for d in exdiv[sp.name] if row["entry"] <= d <= row["expiry"]]
            leg = dict(ticker=sp.name, sector=sp.sector, weight=W[c, i], k1=sp.k1, k2=sp.k2,
                       k1_pct=sp.k1 - 1, k2_pct=sp.k2 - 1, ex_div_in_window=conflicts,
                       pin_by_scenario={s: pin_prob(sp.marginal(s), sp.k1, sp.k2) for s in names})
            if j == 0:
                nc = B0 * W[0, i] / (info["net_cost"] * 100 * fx)
                pos = Position(sp.name, info["K1"], info["K2"], row["expiry"], int(nc), info["spot"], fx,
                               entered_as_combo=True, broker_recognizes_spread=g["broker_confirmed"],
                               ex_div_dates=[date.fromisoformat(x) for x in conflicts])
                leg.update(K1=info["K1"], K2=info["K2"], n_contracts=int(nc), net_cost=info["net_cost"],
                           exercise_cash_krw=int(nc) * info["K1"] * 100 * fx,
                           listed_expiry=info["expiry"], actions=action_list(pos, row["entry"]),
                           rules=[dict(code=s.code, status=s.status, message=s.message) for s in evaluate(pos, row["entry"])])
            else:
                leg["rules"] = [
                    dict(code="R-1", status="pass", message=f"close by {row['close'].isoformat()} (2 business days before expiry)"),
                    dict(code="R-2", status="pass", message="planned as one combo order"),
                    dict(code="R-3", status="pass" if g["broker_confirmed"] else "warn",
                         message="vertical-spread margin " + ("confirmed" if g["broker_confirmed"] else "not confirmed in writing")),
                    dict(code="R-4", status="warn" if conflicts else "pass",
                         message=(f"projected ex-div {', '.join(conflicts)} inside the roll: close before it if the short leg is ITM"
                                  if conflicts else "no projected ex-dividend date in the roll")),
                    dict(code="R-5", status="n/a", message="only if forced to hold to expiry"),
                    dict(code="R-6", status="n/a", message="checked daily in expiry week"),
                ]
                leg["actions"] = [
                    f"{row['entry'].isoformat()}: BUY {sp.name} call vertical as ONE combo: K1 = lowest listed strike >= "
                    f"spot x {sp.k1:.3f} (+{sp.k1 - 1:.1%}), K2 = cost-aware width (~spot x {sp.k2:.3f}); monthly expiry "
                    f"{row['expiry'].isoformat()}",
                    f"size: sleeve x f ({f_cycles[c]:.2f}) x weight ({W[c, i]:.0%}) / net premium incl. costs",
                    f"{row['close'].isoformat()}: SELL the vertical as ONE combo (R-1)",
                ]
            legs.append(leg)
        schedule.append(dict(roll=j, cycle=c, entry=row["entry"], expiry=row["expiry"], close=row["close"],
                             f=f_cycles[c], legs=legs))
    # per-scenario outcomes
    scen_out = {}
    for run in res.runs:
        P = run.paths
        cum, dep, pro = P["cumulative"], P["deployed"], P["proceeds"]
        sub = _sub_result(res, run)
        net, gross = summarize(sub, True, **kw), summarize(sub, False, **kw)
        x = run.terminal_net
        reps = _rep_indices(cum[:, -1], g["G"])
        rep_paths = {}
        for lab, p in reps.items():
            spots, strikes, pins = {}, {}, {}
            for i, sp in enumerate(specs):
                R = sp.marginal(run.name).ppf(U[p, :, i])
                Sx = infos[i]["spot"] * np.concatenate([[1.0], np.cumprod(R)[:-1]])
                spots[sp.name] = Sx
                strikes[sp.name] = dict(K1=sp.k1 * Sx, K2=sp.k2 * Sx, S_exp=Sx * R)
                pins[sp.name] = (Sx * R >= sp.k1 * Sx) & (Sx * R < sp.k2 * Sx)
            rep_paths[lab] = dict(index=p, deployed=dep[p], proceeds=pro[p], pnl=pro[p] - dep[p], sleeve=P["sleeve"][p],
                                  cumulative=cum[p], absorbed=P["absorbed"][p], terminal=cum[p, -1], spots=spots,
                                  strikes=strikes, pins=pins)
        hi = max(float(np.quantile(x, 0.99)), g["G"] * 1.1)
        edges = np.linspace(min(0.0, float(x.min())), hi, 41)
        hist, _ = np.histogram(np.clip(x, edges[0], edges[-1]), edges)
        scen_out[run.name] = dict(
            prob=run.prob, rates=run.rates, budgets=run.budgets,
            bond=[dict(cycle=c, r=run.rates[c], B=run.budgets[c], bond_cost=bud.bond_cost(g["V_deposit"], g["V_own_rental"], run.rates[c], g["T"]),
                       matures_to=g["V_deposit"] - g["V_own_rental"], start=cal[c * 8]["entry"], end=cal[c * 8 + 7]["expiry"])
                  for c in range(n_cyc)],
            band=dict(p10=np.quantile(cum, 0.10, axis=0), p50=np.quantile(cum, 0.50, axis=0), p90=np.quantile(cum, 0.90, axis=0),
                      deployed_mean=dep.mean(axis=0), absorbed_share=P["absorbed"].mean(axis=0),
                      pnl_median=np.median(pro - dep, axis=0)),
            reps=rep_paths, metrics_net=net, metrics_gross=gross, hist=dict(edges=edges, counts=hist))
    mix_net, mix_gross = summarize(res, True, **kw), summarize(res, False, **kw)
    out.update(
        included=[s.name for s in specs], sectors=[s.sector for s in specs], weights=W, corr=C,
        calendar=[dict(roll=r["roll"], entry=r["entry"], expiry=r["expiry"], close=r["close"]) for r in cal],
        schedule=schedule, scenarios_out=scen_out, mixture=dict(metrics_net=mix_net, metrics_gross=mix_gross),
        n_paths=n, n_band=cfg.n_store, B0=B0,
        bond_note=("Each cycle: buy the KRW 2y zero for (V - V_own)(1+r_c)^-T; it matures to V - V_own, which with the "
                   "own-rental deposit V_own returns the tenant's V. The next tenant's V buys the next bond at r_{c+1}, "
                   "re-deriving B for that cycle."))
    return jsonable(out)

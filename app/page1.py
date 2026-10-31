"""Page 1 — ticker-driven simulation. run_page1(state, service) -> JSON-able dict.

Every number is grouped as `market` (fetched) or `model` (computed), so the UI can never present a
simulated figure as a quote. Tickers whose q extraction fails, or whose Edge <= 0, are shown but kept out
of the simulation unless the user overrides (q only) — nothing proceeds silently.
"""
import math
from collections import OrderedDict
from datetime import date

import numpy as np

from core import budget as bud
from core.density import clean_chain, extract_q, kl_level, subjective_density
from core.optimize import Evaluator, ticker_count_curve
from core.rules import (STRUCTURAL_NOTES, Position, action_list, business_days_left, close_by_date, evaluate,
                        pin_banner)
from core.selection import buy_region, call_region, edge
from core.simulate import (Marginal, SimConfig, TickerSpec, correlation_from_prices, cumulative_pin,
                           dubins_savage_regime, pin_prob, simulate, summarize)
from core.spread import build_spread, return_decomposition
from core.structure import liquid, select_k2
from core.vol import catalyst_sigma, catalyst_timing, hv_close_to_close, hv_parkinson, priced_in_check
from data.providers import DELAY_NOTE
from data.trading_days import WEEKDAY_KO, quote_label
from data.service import DataUnavailable, chain_slice

DEFAULT_GLOBALS = dict(
    V_deposit=bud.V_DEPOSIT, V_own_rental=bud.V_OWN_RENTAL, r=None, T=bud.T_CYCLE, rate_shift=0.0, G=bud.GOAL,
    f=1.0, rt_cost=0.04, cost_mode="chain", leg_cross=1.0, k2_rule="cost-aware", k2_criterion="expected",
    tax=True, n_paths=10_000, copula="gaussian", seed=0, broker_confirmed=None, mortgage_rate=0.04,
    remaining_months=264, start="2026-12-01")
DEFAULT_TICKER = dict(expiry=None, sigma_hat=None, mu_hat=None, conf=0.7, weight=None, catalysts=[],
                      catalyst_base=None, use_catalyst=False, use_failed_q=False, use_illiquid=False)
MAX_TICKERS = 5
N_DISPLAY = 320
_MD_CACHE = OrderedDict()


def f2(x):
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return x
    return v if math.isfinite(v) else None


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return [jsonable(v) for v in o.tolist()]
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return f2(o)
    if isinstance(o, date):
        return o.isoformat()
    return o


# ---------------------------------------------------------------- per-ticker pieces

def density_for(item, S0, r, d, asof):
    """q for one cached chain, memoized on (key, fetch time, r, d)."""
    key = (item.key, item.fetched_at.isoformat(), round(r, 6), round(d, 6), S0)
    if key in _MD_CACHE:
        _MD_CACHE.move_to_end(key)
        return _MD_CACHE[key]
    cs = chain_slice(item, asof)
    q = cs.quoted()
    keep, why = clean_chain(cs.strike[q], cs.bid[q], cs.ask[q], cs.oi[q], r, cs.T)
    K = cs.strike[q][keep]
    md = extract_q(K, S0, r, d, cs.T, call_prices=cs.mid[q][keep], half_spreads=0.5 * (cs.ask - cs.bid)[q][keep])
    out = (cs, md, {str(k): v for k, v in why.items()})
    _MD_CACHE[key] = out
    while len(_MD_CACHE) > 64:
        _MD_CACHE.popitem(last=False)
    return out


def display_grid(md, *arrays, n=N_DISPLAY):
    """Display-only resampling. The left atom (tail mass parked on the first node) is a point mass, not a
    density: it is dropped from the picture (it stays in every computation) so it cannot flatten the axis."""
    start = 1 if (md.left_atom or 0) > 1e-12 else 0
    S, w, q = md.S[start:], md.weights[start:], md.q[start:]
    cdf = np.cumsum(w * q) / np.sum(w * q)
    lo = float(np.interp(0.0005, cdf, S))
    hi = float(np.interp(0.9995, cdf, S))
    x = np.linspace(lo, hi, n)
    return x, [np.interp(x, S, a[start:]) for a in arrays]


def atm_iv_yahoo(item, F):
    rows = item.value["rows"]
    K = np.array([r["strike"] for r in rows])
    iv = np.array([r["iv"] if r["iv"] is not None else np.nan for r in rows], dtype=float)
    ok = np.isfinite(iv) & (iv > 0.01)
    return float(np.interp(F, K[ok], iv[ok])) if ok.sum() >= 2 else float("nan")


def spread_dict(sp):
    return dict(expiry=sp.expiry, T=sp.T, K1_target=sp.K1_target, K1=sp.K1, K2=sp.K2, width=sp.K2 - sp.K1,
                c1=sp.c1, c2=sp.c2, net_mid=sp.net_mid, rt_cost=sp.rt_cost, rt_cost_chain=sp.rt_cost_chain,
                net_cost=sp.net_cost, n=sp.n, n_int=sp.n_int, max_payoff_usd=sp.max_payoff_usd,
                max_payoff_krw=sp.max_payoff_krw, payoff_multiple=sp.payoff_multiple, breakeven=sp.breakeven,
                breakeven_pct=sp.breakeven_pct, exercise_cash_usd=sp.exercise_cash_usd,
                exercise_cash_krw=sp.exercise_cash_krw, exercise_to_premium=sp.exercise_to_premium,
                budget_krw=sp.budget_krw, target_krw=sp.target_krw, target_ratio=sp.target_ratio,
                target_ok=sp.target_ok, liquidity=sp.liquidity, liquidity_ok=sp.liquidity_ok,
                sigma_distance=sp.sigma_distance, k2_rule=sp.k2_rule, notes=sp.notes)


def pick_expiry(cands):
    """Default expiry: q passes, T closest to 3 months."""
    ok = [c for c in cands if c["q_pass"]] or cands
    return min(ok, key=lambda c: abs(c["T"] - 0.25))["expiry"]


def catalyst_block(svc, b, tk, chosen_expiry, iv_in, S0, r, d, hv1y):
    cats = [c for c in tk.get("catalysts") or [] if c.get("date") and c.get("move") is not None]
    exp = date.fromisoformat(chosen_expiry)
    base = tk.get("catalyst_base") or hv1y
    out = dict(rows=[], base=base, sigma_hat=None, priced_in=None, timing=[])
    moves = []
    for c in cats:
        dd = date.fromisoformat(c["date"])
        ok, days = catalyst_timing(dd, exp)
        out["timing"].append(dict(name=c.get("name") or "catalyst", date=c["date"], ok=ok, days_before_expiry=days,
                                  before_expiry=dd <= exp))
        if dd <= exp:
            moves.append(float(c["move"]))
        out["rows"].append(dict(name=c.get("name") or "catalyst", date=c["date"], move=float(c["move"])))
    out["sigma_hat"] = catalyst_sigma(base, moves, 0.25) if cats else None
    # earnings dates from the feed also obey the timing rule
    for e in (b.events.value["earnings"] if b.events else []):
        dd = date.fromisoformat(e["date"])
        if dd <= exp:
            ok, days = catalyst_timing(dd, exp)
            out["timing"].append(dict(name="earnings (feed, est.)", date=e["date"], ok=ok, days_before_expiry=days,
                                      before_expiry=True))
    if moves and b.expiries:
        first = min(date.fromisoformat(c["date"]) for c in cats if date.fromisoformat(c["date"]) <= exp)
        before = [e for e in b.expiries.value if b.asof_date < date.fromisoformat(e) < first]
        if before:
            ex = before[-1]
            try:
                it = svc.chain_item(b.ticker, ex)
                T_ex = (date.fromisoformat(ex) - b.asof_date).days / 365
                iv_ex = atm_iv_yahoo(it, S0 * np.exp((r - d) * T_ex))
                T_in = (exp - b.asof_date).days / 365
                chk = priced_in_check(moves, iv_ex, T_ex, iv_in, T_in)
                out["priced_in"] = dict(excluding_expiry=ex, including_expiry=chosen_expiry, iv_excluding=iv_ex,
                                        iv_including=iv_in, **chk)
            except (DataUnavailable, ValueError) as e:
                out["priced_in"] = dict(error=str(e))
        else:
            out["priced_in"] = dict(error="no listed expiry before the first catalyst: cannot isolate its IV step")
    return out


def ticker_block(svc, tk, g, B0, fx, r_usd, n_tickers, today):
    b = svc.ticker_bundle(tk["ticker"])
    T_ = b.ticker
    res = dict(ticker=T_, errors=dict(b.errors), stale=b.stale,
               data_asof=b.data_asof.isoformat() if b.data_asof else None,
               freshness=[i.meta() for i in b.items()])
    if b.info is None or b.history is None:
        res["fatal"] = "no spot/history: " + "; ".join(b.errors.values())
        return res, None
    inf = b.info.value
    S0, d = inf["spot"], inf["div_yield"]
    h = b.history.value
    closes = np.array(h["close"])
    hv1y, hv2y = hv_close_to_close(closes, 252), hv_close_to_close(closes)
    park = hv_parkinson(h["high"], h["low"], 252)
    res["market"] = dict(name=inf["name"], spot=S0, currency=inf["currency"], sector=inf["sector"],
                         sector_source=inf["sector_source"], div_yield=d,
                         earnings=b.events.value["earnings"] if b.events else [],
                         ex_dividends=b.events.value["ex_dividends"] if b.events else [],
                         n_expiries_listed=len(b.expiries.value) if b.expiries else 0)
    exps = []
    for e, it in b.chains.items():
        try:
            cs, md, dropped = density_for(it, S0, r_usd, d, b.asof_date)
            exps.append(dict(expiry=e, T=cs.T, q_pass=bool(md.passed), iv_atm=md.iv_atm,
                             checks={k: dict(value=v[0], tol=v[1], ok=v[2]) for k, v in md.checks.items()},
                             session=it.value.get("session"), stale=it.stale, n_strikes=len(cs.strike),
                             n_dropped=len(dropped)))
        except Exception as ex:  # noqa: BLE001 - a broken chain is reported, not fatal
            exps.append(dict(expiry=e, T=(date.fromisoformat(e) - b.asof_date).days / 365, q_pass=False,
                             iv_atm=None, error=f"{type(ex).__name__}: {ex}"))
    res["expiries"] = exps
    if not exps:
        res["fatal"] = "no usable option chain"
        return res, None
    chosen = tk.get("expiry") if tk.get("expiry") in b.chains else pick_expiry(exps)
    exp_row = next(x for x in exps if x["expiry"] == chosen)
    if exp_row.get("error"):
        res["fatal"] = f"{chosen}: {exp_row['error']}"
        return res, None
    cs, md, dropped = density_for(b.chains[chosen], S0, r_usd, d, b.asof_date)
    session = b.chains[chosen].value.get("session")
    off_hours = session == "off-hours"
    res["quote_label"] = quote_label(b.chains[chosen].fetched_at, session)
    cat = catalyst_block(svc, b, tk, chosen, md.iv_atm, S0, r_usd, d, hv1y)
    if tk.get("use_catalyst") and cat["sigma_hat"]:
        sigma_hat, sig_src = cat["sigma_hat"], "catalyst calculator"
    elif tk.get("sigma_hat"):
        sigma_hat, sig_src = float(tk["sigma_hat"]), "user"
    else:
        sigma_hat, sig_src = hv1y, "placeholder: HV 1y (no research input)"
    mu_hat = tk.get("mu_hat")
    conf = float(tk.get("conf", 0.7))
    tilt = subjective_density(md, sigma_hat, conf, mu=mu_hat)
    marg = Marginal.from_density(md.S, tilt.p, md.weights, S0)
    e = edge(sigma_hat, md.iv_atm)
    weight = tk.get("weight")
    w_alloc = float(weight) if weight else 1.0 / n_tickers
    B_alloc = B0 * w_alloc
    target = g["G"] / bud.N_CYCLES * w_alloc
    narrowest = None
    table = None
    cfg_small = SimConfig(n_paths=3000, n_store=0, goal=g["G"], f=g["f"], r0=g["r"], tax=g["tax"],
                          scenarios=(bud.RateScenario("path", g["rate_shift"], 1.0),), seed=g["seed"])
    try:
        if g["k2_rule"] == "cost-aware":
            sel = select_k2(cs, S0, sigma_hat, B_alloc, fx, marg, g["k2_criterion"], rt_cost=g["rt_cost"],
                            leg_cross=g["leg_cross"], target_krw=target, cfg=cfg_small, spread_check=not off_hours)
            sp, narrowest, table = sel["chosen"], sel["narrowest"], sel["table"]
        else:
            sp = build_spread(cs, S0, sigma_hat, B_alloc, fx, rt_cost=g["rt_cost"], target_krw=target,
                              cost_mode=g["cost_mode"], leg_cross=g["leg_cross"])
            narrowest = sp
    except ValueError as ex:
        res["fatal"] = f"no tradeable spread: {ex}"
        return res, None
    p_pin = pin_prob(marg, sp.k1, sp.k2)
    reg = buy_region(md.S, md.q, tilt.ell, md.weights, sp.net_cost, sp.K2 - sp.K1, r_usd, cs.T,
                     allowed=call_region(md.S, md.F))
    x, (qd, pd_, ld) = display_grid(md, md.q, tilt.p, tilt.ell)
    mask = np.interp(x, md.S, reg.mask.astype(float)) > 0.5
    decomp = return_decomposition(md, sigma_hat, mu_hat, conf, sp.K1, sp.K2, sp.net_cost)
    xd = [date.fromisoformat(o["date"]) for o in res["market"]["ex_dividends"]]
    pos = Position(T_, sp.K1, sp.K2, date.fromisoformat(chosen), sp.n_int, S0, fx, entered_as_combo=True,
                   broker_recognizes_spread=g["broker_confirmed"], ex_div_dates=xd)
    hv_proxy = dict(hv1y=hv1y, hv2y=hv2y, parkinson_1y=park)
    eligible_q = md.passed or tk.get("use_failed_q")
    liq_pass = liquid(sp, spread_check=not off_hours)
    eligible_liq = liq_pass or tk.get("use_illiquid")
    liq_offhours = off_hours and liq_pass and not sp.liquidity_ok
    reasons = [r for r, bad in (("Edge <= 0: sigma_hat does not exceed ATM IV", e <= 0),
                                ("q extraction failed its sanity checks", not eligible_q),
                                (("liquidity filter failed (OI >= 100; leg spread not judged on end-of-day quotes)" if off_hours else
                                 "liquidity filter failed (OI >= 100, leg spread <= 10% of mid)"), not eligible_liq)) if bad]
    res.update(
        chosen_expiry=chosen,
        model=dict(
            quote_session=session, liquidity_offhours=liq_offhours,
            sigma_hat=sigma_hat, sigma_hat_source=sig_src, sigma_used=tilt.sigma_used, mu_hat=mu_hat, conf=conf,
            iv_atm=md.iv_atm, edge=e, eligible=e > 0, hv=hv_proxy, kl=tilt.kl, kl_level=kl_level(tilt.kl),
            tilt_converged=tilt.converged, tilt_edge_mass=tilt.edge_mass,
            q_checks={k: dict(value=v[0], tol=v[1], ok=v[2]) for k, v in md.checks.items()}, q_pass=md.passed,
            q_dropped_quotes=dropped, mean_q=md.mean, forward=md.F,
            weight=w_alloc, budget_alloc=B_alloc, per_cycle_target=target,
            spread=spread_dict(sp), narrowest=spread_dict(narrowest) if narrowest else None,
            k2_table=table, k2_criterion=g["k2_criterion"] if g["k2_rule"] == "cost-aware" else None,
            pin_per_roll=p_pin, pin_cumulative=cumulative_pin(p_pin, 32), p_finish_above_K2=1 - float(marg.cdf(sp.k2)),
            p_finish_below_K1=float(marg.cdf(sp.k1)),
            density=dict(S=x, q=qd, p=pd_, ell=ld, buy=mask, c=reg.c, K_lo=reg.K_lo, P_p=reg.P_p, Q_q=reg.Q_q,
                         binding=reg.binding),
            decomposition=decomp, catalyst=cat,
            rules=[dict(code=s.code, status=s.status, message=s.message) for s in evaluate(pos, today)],
            close_by=close_by_date(pos.expiry).isoformat(), actions=action_list(pos, today),
            bdays_to_close=business_days_left(today, close_by_date(pos.expiry)),
            bdays_to_expiry=business_days_left(today, pos.expiry), pin_banner=pin_banner(pos, today),
        ),
        in_simulation=not reasons,
        excluded_reason="; ".join(reasons) or None,
        overrides=dict(q=bool(tk.get("use_failed_q")) and not md.passed,
                       liquidity=bool(tk.get("use_illiquid")) and not sp.liquidity_ok),
    )
    spec = TickerSpec.from_spread(T_, sp, {"*": marg}, sector=inf["sector"]) if res["in_simulation"] else None
    return res, ((spec, np.array(h["dates"]), closes) if spec else None)


# ---------------------------------------------------------------- portfolio

def aligned_closes(series):
    common = set(series[0][0])
    for dts, _ in series[1:]:
        common &= set(dts)
    common = sorted(common)
    cols = []
    for dts, c in series:
        m = dict(zip(dts, c))
        cols.append([m[x] for x in common])
    return np.array(cols).T


def run_page1(state, svc):
    g = {**DEFAULT_GLOBALS, **(state.get("globals") or {})}
    tks = [{**DEFAULT_TICKER, **t} for t in (state.get("tickers") or [])][:MAX_TICKERS]
    today = svc.asof_date()
    mac = svc.macros()
    krw = mac["krw_2y"]
    if g["r"] is None:
        g["r"] = krw.value["value"]
    fx = mac["usdkrw"].value["value"] if mac.get("usdkrw") else None
    r_usd = mac["ust_3m"].value["value"] if mac.get("ust_3m") else (mac["ust_2y"].value["value"] if mac.get("ust_2y") else 0.04)
    if fx is None:
        raise DataUnavailable("USD/KRW unavailable and no cache: cannot convert the KRW budget")
    rates, budgets = bud.cycle_budgets(g["r"], g["rate_shift"], bud.N_CYCLES, g["V_deposit"], g["V_own_rental"], g["T"])
    B0 = float(budgets[0])
    out = dict(
        status=dict(delay_note=DELAY_NOTE, schedule=svc.schedule.describe(), today=today.isoformat(),
                    valuation_date=f"{today.isoformat()}({WEEKDAY_KO[today.weekday()]})"),
        macro={k: (v.meta() | dict(value=v.value["value"], observed=v.value.get("observed"))) if v is not None else None
               for k, v in mac.items() if not k.endswith("_error")},
        macro_errors={k: v for k, v in mac.items() if k.endswith("_error")},
        globals=g,
        budget=dict(rates=rates, budgets=budgets, bond_cost=bud.bond_cost(g["V_deposit"], g["V_own_rental"], g["r"], g["T"]),
                    formula="B = (V_deposit - V_own_rental)(1 - (1 + r)^-T)", r_origin=krw.origin if
                    (state.get("globals") or {}).get("r") is None else "user"),
        tickers=[], portfolio=None, notes=STRUCTURAL_NOTES,
    )
    sims = []
    for tk in tks:
        try:
            block, sim = ticker_block(svc, tk, g, B0, fx, r_usd, len(tks), today)
        except DataUnavailable as ex:
            block, sim = dict(ticker=tk["ticker"].upper(), fatal=str(ex)), None
        out["tickers"].append(block)
        if sim:
            sims.append((block, sim))
    warnings = []
    if not 3 <= len(tks) <= 5:
        warnings.append(f"{len(tks)} ticker(s): the design uses 3-5 (more pulls the outcome toward the mean, "
                        f"which is below the goal)")
    sectors = [b["market"]["sector"] for b, _ in sims]
    for s in set(sectors):
        if sectors.count(s) > 2:
            warnings.append(f"{sectors.count(s)} tickers in {s}: max 2 per GICS sector")
    out["warnings"] = warnings
    if not sims:
        out["portfolio"] = dict(error="no ticker is eligible for the simulation (see each card)")
        return jsonable(out)

    specs = [s[0] for _, s in sims]
    w = np.array([b["model"]["weight"] for b, _ in sims], dtype=float)
    w = w / w.sum()
    C = np.eye(1) if len(sims) == 1 else correlation_from_prices(aligned_closes([(s[1], s[2]) for _, s in sims]))
    cfg = SimConfig(n_paths=int(g["n_paths"]), goal=g["G"], f=g["f"], V_deposit=g["V_deposit"],
                    V_own_rental=g["V_own_rental"], r0=g["r"], T_cycle=g["T"],
                    scenarios=(bud.RateScenario("path", g["rate_shift"], 1.0),), tax=g["tax"],
                    start=date.fromisoformat(g["start"]), copula=g["copula"], seed=int(g["seed"]), n_store=120)
    res = simulate(specs, w, C, cfg)
    kw = dict(mortgage_rate=g["mortgage_rate"], remaining_months=g["remaining_months"])
    net, gross = summarize(res, net=True, **kw), summarize(res, net=False, **kw)
    run = res.runs[0]
    x = run.terminal_net
    hi = float(np.quantile(x, 0.99)) if np.any(x > 0) else 1.0
    edges = np.linspace(min(0.0, float(x.min())), max(hi, g["G"] * 1.1), 41)
    hist, _ = np.histogram(np.clip(x, edges[0], edges[-1]), edges)
    rng = np.random.default_rng(1)
    sample = x[rng.choice(len(x), size=min(1500, len(x)), replace=False)]
    rf = (1 + g["r"]) ** 0.25
    # f frontier and ticker-count curve on common random numbers (smaller run)
    small = SimConfig(**{**cfg.__dict__, "n_paths": min(4000, cfg.n_paths), "n_store": 0})
    ev = Evaluator(specs, C, small)
    f_front = []
    for f in (0.1, 0.25, 0.5, 0.75, 1.0):
        e_ = ev.evaluate(tuple(range(len(specs))), w, f)
        f_front.append(dict(f=f, p_goal=e_.p_goal, ruin=e_.ruin, median=e_.median, mean=e_.mean))
    rep = specs[int(np.argmax(w))]
    rho = float(np.mean(C[np.triu_indices(len(C), 1)])) if len(C) > 1 else 0.3
    curve = ticker_count_curve(rep, (1, 2, 3, 4, 5, 10, 30), rho, small)
    out["portfolio"] = dict(
        included=[s.name for s in specs], weights=w, corr=C, corr_tickers=[s.name for s in specs],
        metrics_net=net, metrics_gross=gross, regime=dubins_savage_regime(net["roll_multiple_mean"], rf),
        rf_growth_per_roll=rf, joint_total_loss=run.joint_total_loss,
        pin_by_ticker=dict(zip([s.name for s in specs], run.pin_by_ticker)),
        max_exercise_cash_p99=float(np.quantile(run.max_exercise_cash, 0.99)),
        hist=dict(edges=edges, counts=hist), sample_terminal=sample,
        paths={k: v[:60] for k, v in run.paths.items()},
        calendar=[dict(roll=c["roll"], entry=c["entry"], expiry=c["expiry"], close=c["close"]) for c in res.calendar],
        cycle_budgets=run.budgets, cycle_rates=run.rates, f_frontier=f_front, ticker_count_curve=curve,
        ticker_count_template=rep.name, ticker_count_rho=rho, n_paths=cfg.n_paths, copula=cfg.copula)
    return jsonable(out)

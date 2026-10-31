"""Page 2 — inverse optimizer / trade-off explorer. run_page2(state, service) -> JSON-able dict.

For each candidate: q from the chain, one tilt p_s per rate scenario (sigma_hat_s, mu_hat_s), the mixture
p_mix = sum_s pi_s p_s picks K2 (cost-aware, chain quotes) around the spec's sigma-rule K1. Then the
two-stage solver in core.optimize (subsets x f grid, then local weight/f refinement per slider position)
runs on common random numbers. Every slider pick is re-checked on a larger, independent path set, and
both figures are returned so search noise is visible.
"""
from datetime import date

import numpy as np

from core import budget as bud
from core.density import mixture, subjective_density
from core.optimize import Evaluator, feasible_subsets, scenario_table, sensitivity, solve, ticker_count_curve
from core.selection import edge
from core.simulate import Marginal, SimConfig, TickerSpec, correlation_from_prices
from core.structure import liquid, select_k2
from core.vol import hv_close_to_close, iv_rank_proxy, screen
from data.service import DataUnavailable
from data.trading_days import quote_label

from .page1 import aligned_closes, density_for, jsonable, pick_expiry

SCEN = ("up", "neutral", "down")
DEFAULT = dict(
    scenarios=[dict(name="up", shift_bp=75, prob=0.25), dict(name="neutral", shift_bp=0, prob=0.50),
               dict(name="down", shift_bp=-75, prob=0.25)],
    thresholds_bp=[-50, 50],
    constraints=dict(k_min=3, k_max=5, max_per_sector=2, min_oi=100, max_spread_pct=0.10),
    globals=dict(V_deposit=bud.V_DEPOSIT, V_own_rental=bud.V_OWN_RENTAL, r=None, T=bud.T_CYCLE, G=bud.GOAL,
                 leg_cross=1.0, rt_cost=0.04, tax=True, copula="gaussian", seed=0, start="2026-12-01",
                 n_paths=3000, n_check=10000, f_grid=[0.25, 0.5, 0.75, 1.0]),
)
MAX_CANDIDATES = 10
LAMBDAS = tuple(np.round(np.linspace(0, 1, 11), 2))


def _scen_value(d, name, default):
    if isinstance(d, dict) and d.get(name) is not None:
        return float(d[name])
    if isinstance(d, (int, float)):
        return float(d)
    return default


def screener_row(svc, ticker, r_usd):
    """Market-side screening metrics for one ticker (no view needed)."""
    b = svc.ticker_bundle(ticker)
    if b.info is None or b.history is None or not b.chains:
        raise DataUnavailable("; ".join(b.errors.values()) or "no data")
    inf = b.info.value
    S0, d = inf["spot"], inf["div_yield"]
    closes = np.array(b.history.value["close"])
    hv1y = hv_close_to_close(closes, 252)
    exps = []
    # nearest to 3 months first; the first one whose q passes is what pick_expiry would choose anyway
    order = sorted(b.chains, key=lambda e: abs((date.fromisoformat(e) - b.asof_date).days / 365 - 0.25))
    for e in order:
        try:
            cs, md, _ = density_for(b.chains[e], S0, r_usd, d, b.asof_date)
            exps.append(dict(expiry=e, T=cs.T, q_pass=bool(md.passed), iv_atm=md.iv_atm))
            if md.passed:
                break
        except Exception as ex:  # noqa: BLE001
            exps.append(dict(expiry=e, T=(date.fromisoformat(e) - b.asof_date).days / 365, q_pass=False,
                             iv_atm=None, error=str(ex)))
    ok = [x for x in exps if x.get("iv_atm")]
    if not ok:
        raise DataUnavailable("no usable chain")
    chosen = pick_expiry(ok)
    iv = next(x["iv_atm"] for x in ok if x["expiry"] == chosen)
    return dict(ticker=b.ticker, name=inf["name"], sector=inf["sector"], spot=S0, iv_atm=iv, hv1y=hv1y,
                iv_minus_hv=iv - hv1y, iv_rank_proxy=iv_rank_proxy(iv, closes), expiry=chosen,
                stale=b.stale, data_asof=b.data_asof.isoformat() if b.data_asof else None,
                sessions=sorted({it.value.get("session") for it in b.chains.values()}))


def run_screener(svc, tickers, r_usd):
    rows, errors = [], {}
    for t in tickers:
        try:
            rows.append(screener_row(svc, t, r_usd))
        except DataUnavailable as e:
            errors[t] = str(e)
    return screen(rows), errors


def prepare(svc, cand, names, probs, g, cons, B0, fx, r_usd):
    """Candidate -> (UI dict, TickerSpec or None, (dates, closes))."""
    b = svc.ticker_bundle(cand["ticker"])
    T_ = b.ticker
    if b.info is None or b.history is None or not b.chains:
        return dict(ticker=T_, eligible=False, reason="; ".join(b.errors.values()) or "no data"), None, None
    inf = b.info.value
    S0, d = inf["spot"], inf["div_yield"]
    h = b.history.value
    closes = np.array(h["close"])
    hv1y = hv_close_to_close(closes, 252)
    exps = {}
    for e, it in b.chains.items():
        try:
            exps[e] = density_for(it, S0, r_usd, d, b.asof_date)
        except Exception:  # noqa: BLE001
            continue
    if not exps:
        return dict(ticker=T_, eligible=False, reason="no usable chain"), None, None
    chosen = pick_expiry([dict(expiry=e, T=v[0].T, q_pass=v[1].passed) for e, v in exps.items()])
    cs, md, _ = exps[chosen]
    conf = float(cand.get("conf", 0.7))
    sig = {s: _scen_value(cand.get("sigma"), s, hv1y) for s in names}
    mu = {s: (_scen_value(cand.get("mu"), s, None) if cand.get("mu") is not None else None) for s in names}
    placeholder = cand.get("sigma") is None
    tilts = {s: subjective_density(md, sig[s], conf, mu=mu[s]) for s in names}
    mix = mixture(md, [tilts[s] for s in names], probs)
    sig_mix = float(np.dot(probs, [sig[s] for s in names]))
    m_mix = Marginal.from_density(md.S, mix.p, md.weights, S0)
    try:
        off_hours = b.chains[chosen].value.get("session") == "off-hours"
        sel = select_k2(cs, S0, sig_mix, B0, fx, m_mix, "expected", rt_cost=g["rt_cost"], leg_cross=g["leg_cross"],
                        target_krw=g["G"] / bud.N_CYCLES, min_oi=cons["min_oi"], max_spread_pct=cons["max_spread_pct"],
                        spread_check=not off_hours)
    except ValueError as ex:
        return dict(ticker=T_, eligible=False, reason=f"no tradeable spread: {ex}"), None, None
    sp = sel["chosen"]
    marg = {s: Marginal.from_density(md.S, tilts[s].p, md.weights, S0) for s in names}
    e_mix = edge(sig_mix, md.iv_atm)
    reasons = [r for r, bad in (("Edge <= 0 (probability-weighted sigma_hat vs ATM IV)", e_mix <= 0),
                                ("q extraction failed its sanity checks", not md.passed),
                                (f"liquidity filter failed at {sp.K1:g}/{sp.K2:g}: OI {sp.c1['oi']:.0f}/{sp.c2['oi']:.0f} "
                                 f"(min {cons['min_oi']}), leg spread {sp.c1['spread_pct']:.0%}/{sp.c2['spread_pct']:.0%} "
                                 f"(max {cons['max_spread_pct']:.0%}{'; spread not judged on end-of-day quotes' if off_hours else ''})",
                                 not liquid(sp, spread_check=not off_hours))) if bad]
    from core.structure import expected_roll_multiple
    info = dict(ticker=T_, name=inf["name"], sector=inf["sector"], spot=S0, expiry=chosen, T=cs.T,
                iv_atm=md.iv_atm, hv1y=hv1y, iv_rank_proxy=iv_rank_proxy(md.iv_atm, closes),
                sigma=sig, mu=mu, conf=conf, sigma_placeholder=placeholder, sigma_mix=sig_mix,
                edge={s: edge(sig[s], md.iv_atm) for s in names}, edge_mix=e_mix,
                kl={s: tilts[s].kl for s in names}, kl_mix=mix.kl,
                K1=sp.K1, K2=sp.K2, net_mid=sp.net_mid, rt_cost=sp.rt_cost, net_cost=sp.net_cost,
                payoff_multiple=sp.payoff_multiple, breakeven_pct=sp.breakeven_pct, liquidity_ok=sp.liquidity_ok,
                quote_session=b.chains[chosen].value.get("session"),
                quote_label=quote_label(b.chains[chosen].fetched_at, b.chains[chosen].value.get("session")),
                liquidity_offhours=off_hours and liquid(sp, False) and not sp.liquidity_ok,
                narrowest_K2=sel["narrowest"].K2, narrowest_rt=sel["narrowest"].rt_cost,
                e_multiple={s: expected_roll_multiple(marg[s], sp.k1, sp.k2, sp.cost_rel) for s in names},
                q_pass=md.passed, eligible=not reasons, reason="; ".join(reasons) or None,
                stale=b.stale, data_asof=b.data_asof.isoformat() if b.data_asof else None)
    spec = TickerSpec(T_, sp.k1, sp.k2, sp.cost_rel, marg, sector=inf["sector"]) if not reasons else None
    extra = dict(md=md, net_mid=sp.net_mid, sig=sig, mu=mu, conf=conf, names=names)
    return info, spec, (h["dates"], closes, extra)


def _eval_dict(e, check=None):
    out = dict(idx=list(e.idx), names=list(e.names), sectors=list(e.sectors), weights=e.weights, f=e.f,
               p_goal=e.p_goal, mean=e.mean, median=e.median, ruin=e.ruin, p_by_scenario=e.p_by_scenario,
               mean_by_scenario=e.mean_by_scenario, p_worst=e.p_worst, tag=e.tag)
    if check is not None:
        out["check"] = dict(p_goal=check.p_goal, mean=check.mean, median=check.median, ruin=check.ruin,
                            p_by_scenario=check.p_by_scenario, p_worst=check.p_worst)
    return out


def sensitivity_strip(e, specs, extras, C, cfg):
    """sigma_hat +/-5pp (strikes held), round-trip 2/4/8% (flat, replacing chain costs), ticker count,
    and f — all on one set of uniforms for the chosen combination."""
    idx = list(e.idx)
    sub = [specs[i] for i in idx]
    subC = C[np.ix_(idx, idx)]
    w = e.weights
    run_cfg = SimConfig(**{**cfg.__dict__, "f": e.f, "n_store": 0})

    def sig_build(dpp):
        out = []
        for sp, ex in zip(sub, [extras[i] for i in idx]):
            md = ex["md"]
            marg = {s: Marginal.from_density(md.S, subjective_density(md, max(ex["sig"][s] + dpp, 0.05), ex["conf"],
                                                                      mu=ex["mu"][s]).p, md.weights, md.S0)
                    for s in ex["names"]}
            out.append(TickerSpec(sp.name, sp.k1, sp.k2, sp.cost_rel, marg, sector=sp.sector))
        return out

    def rt_build(rt):
        return [TickerSpec(sp.name, sp.k1, sp.k2, ex["net_mid"] * (1 + rt) / ex["md"].S0, sp.marginals, sector=sp.sector)
                for sp, ex in zip(sub, [extras[i] for i in idx])]

    sig = sensitivity(sig_build, (-0.05, 0.0, 0.05), subC, run_cfg, w)
    rt = sensitivity(rt_build, (0.02, 0.04, 0.08), subC, run_cfg, w)
    ev = Evaluator(sub, subC, run_cfg)
    fs = {f: ev.evaluate(tuple(range(len(sub))), w, f) for f in (0.25, 0.5, 0.75, 1.0)}
    rep = sub[int(np.argmax(w))]
    rho = float(np.mean(subC[np.triu_indices(len(subC), 1)])) if len(subC) > 1 else 0.3
    count = ticker_count_curve(rep, (1, 3, 5, 10, 30), rho, run_cfg)
    pack = lambda d: {str(k): dict(p_goal=v["p_goal"], mean=v["mean"], median=v["median"], ruin=v["ruin"]) for k, v in d.items()}
    cur = Evaluator(sub, subC, run_cfg).evaluate(tuple(range(len(sub))), w, e.f)
    rt_out = pack(rt)
    rt_out["chain"] = dict(p_goal=cur.p_goal, mean=cur.mean, median=cur.median, ruin=cur.ruin)
    return dict(sigma=pack(sig), rt_cost=rt_out,
                f={str(k): dict(p_goal=v.p_goal, mean=v.mean, median=v.median, ruin=v.ruin) for k, v in fs.items()},
                count=pack(count), count_template=rep.name, count_rho=rho,
                note="sigma_hat shifted in every scenario with K1/K2 held fixed; round-trip as a flat % of mid premium")


def run_page2(state, svc):
    g = {**DEFAULT["globals"], **(state.get("globals") or {})}
    cons = {**DEFAULT["constraints"], **(state.get("constraints") or {})}
    scen = state.get("scenarios") or DEFAULT["scenarios"]
    thr = state.get("thresholds_bp") or DEFAULT["thresholds_bp"]
    names = tuple(s["name"] for s in scen)
    probs = bud.check_scenarios([bud.RateScenario(s["name"], s["shift_bp"] / 1e4, float(s["prob"])) for s in scen])
    mac = svc.macros()
    if g["r"] is None:
        g["r"] = mac["krw_2y"].value["value"]
    if not mac.get("usdkrw"):
        raise DataUnavailable("USD/KRW unavailable")
    fx = mac["usdkrw"].value["value"]
    r_usd = mac["ust_3m"].value["value"] if mac.get("ust_3m") else 0.04
    budgets = {s["name"]: bud.cycle_budgets(g["r"], s["shift_bp"] / 1e4, bud.N_CYCLES, g["V_deposit"],
                                            g["V_own_rental"], g["T"]) for s in scen}
    warnings = []
    lo, hi = thr
    for s in scen:
        want = {"up": s["shift_bp"] > hi, "down": s["shift_bp"] < lo, "neutral": lo <= s["shift_bp"] <= hi}.get(s["name"], True)
        if not want:
            warnings.append(f"scenario '{s['name']}' shift {s['shift_bp']}bp lies outside its band ({lo}..{hi}bp)")
    out = dict(globals=g, constraints=cons, scenarios=scen, thresholds_bp=thr,
               budgets={k: dict(rates=v[0], budgets=v[1]) for k, v in budgets.items()},
               r_origin=mac["krw_2y"].origin if (state.get("globals") or {}).get("r") is None else "user",
               macro=dict(usdkrw=fx, ust_3m=r_usd), candidates=[], warnings=warnings)
    cands = (state.get("candidates") or [])[:MAX_CANDIDATES]
    if len(state.get("candidates") or []) > MAX_CANDIDATES:
        warnings.append(f"only the first {MAX_CANDIDATES} candidates are used")
    B0 = float(budgets[names[int(np.argmax(probs))]][1][0])
    specs, series, extras = [], [], []
    for c in cands:
        try:
            info, spec, ser = prepare(svc, c, names, probs, g, cons, B0, fx, r_usd)
        except DataUnavailable as ex:
            info, spec, ser = dict(ticker=c["ticker"].upper(), eligible=False, reason=str(ex)), None, None
        out["candidates"].append(info)
        if spec is not None:
            specs.append(spec)
            series.append((ser[0], ser[1]))
            extras.append(ser[2])
    k_min = min(cons["k_min"], len(specs))
    if len(specs) < max(1, cons["k_min"]):
        warnings.append(f"계산에 쓸 수 있는 후보가 {len(specs)}개로 '종목 수 최소 {cons['k_min']}'보다 적음 — 있는 것만으로 계산함")
    if not specs:
        out["error"] = "계산에 쓸 수 있는 후보가 없음 — 후보 표의 '제외' 사유를 확인해줘."
        return jsonable(out)
    k_lo, k_hi = max(1, k_min), min(cons["k_max"], len(specs))
    if not feasible_subsets(specs, k_lo, k_hi, cons["max_per_sector"]):
        by_sector = {}
        for sp in specs:
            by_sector.setdefault(sp.sector or "?", []).append(sp.name)
        detail = ", ".join(f"{sec} {len(v)}개({'·'.join(v)})" for sec, v in sorted(by_sector.items(), key=lambda kv: -len(kv[1])))
        out["error"] = (f"조합을 만들 수 없음: 유효 후보 {len(specs)}개의 섹터가 {detail}인데, 제약이 '최소 {k_lo}종목 · "
                        f"섹터당 최대 {cons['max_per_sector']}개'라서 만족하는 조합이 없음. 다른 섹터 종목을 후보에 추가하거나 "
                        f"'섹터당 최대'를 올리거나 '종목 수 최소'를 낮춰줘.")
        out["infeasible"] = dict(by_sector=by_sector, k_min=k_lo, k_max=k_hi, max_per_sector=cons["max_per_sector"])
        return jsonable(out)
    C = np.eye(1) if len(specs) == 1 else correlation_from_prices(aligned_closes(series))
    scen_objs = tuple(bud.RateScenario(s["name"], s["shift_bp"] / 1e4, float(s["prob"])) for s in scen)
    cfg = SimConfig(n_paths=int(g["n_paths"]), n_store=0, goal=g["G"], V_deposit=g["V_deposit"],
                    V_own_rental=g["V_own_rental"], r0=g["r"], T_cycle=g["T"], scenarios=scen_objs, tax=g["tax"],
                    copula=g["copula"], seed=int(g["seed"]), start=date.fromisoformat(g["start"]))
    ev = Evaluator(specs, C, cfg)
    res = solve(ev, lambdas=LAMBDAS, f_grid=tuple(g["f_grid"]), k_min=max(1, k_min),
                k_max=min(cons["k_max"], len(specs)), max_per_sector=cons["max_per_sector"], max_iter=8)
    table = scenario_table(ev, res, max_iter=6)
    # independent check run on more paths (different seed) for every pick and table entry
    chk_cfg = SimConfig(**{**cfg.__dict__, "n_paths": int(g["n_check"]), "seed": int(g["seed"]) + 1000})
    chk = Evaluator(specs, C, chk_cfg)
    cache = {}

    def checked(e):
        k = e.key()
        if k not in cache:
            cache[k] = chk.evaluate(e.idx, e.weights, e.f)
        return cache[k]

    keys = [e.key() for e in res.evals]
    picks = []
    for lam in LAMBDAS:
        e = res.picks[float(lam)]
        picks.append(dict(lam=float(lam), point=keys.index(e.key()), **_eval_dict(e, checked(e))))
    tab = {k: _eval_dict(v, checked(v)) for k, v in table.items() if k != "recommended"}
    sens = {}
    for p in picks:
        k = (tuple(p["idx"]), tuple(np.round(p["weights"], 6)), round(p["f"], 6))
        if k not in sens:
            e = res.picks[p["lam"]]
            sens[k] = sensitivity_strip(e, specs, extras, C, SimConfig(**{**cfg.__dict__, "n_paths": int(g["n_paths"])}))
        p["sensitivity"] = sens[k]
    out.update(
        universe=[s.name for s in specs], corr=C,
        points=[dict(mean=e.mean, p_goal=e.p_goal, ruin=e.ruin, median=e.median, names=list(e.names),
                     weights=e.weights, f=e.f, frontier=bool(fr), p_worst=e.p_worst, tag=e.tag)
                for e, fr in zip(res.evals, res.frontier)],
        picks=picks, table=tab, recommended="mixture",
        recommendation=("Use the mixture-optimal combination: it maximizes goal probability under the scenario "
                        "probabilities you set; min-max buys insurance against the worst scenario at the cost of "
                        "the scenarios you think likelier."),
        n_evals=ev.n_evals, n_paths=cfg.n_paths, n_check=chk_cfg.n_paths,
        normalizer=dict(p_lo=res.norm.p_lo, p_hi=res.norm.p_hi, m_lo=res.norm.m_lo, m_hi=res.norm.m_hi),
    )
    return jsonable(out)

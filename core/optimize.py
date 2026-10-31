"""Inverse solver: which tickers, weights and staking fraction f, for a point on the
goal-probability <-> expected-return slider.

    objective(lambda) = (1 - lambda) * norm P(goal) + lambda * norm E[terminal]

Two stages on common random numbers (one set of uniforms for every candidate, every evaluation):
  1. enumerate feasible subsets (3-5 tickers, <= 2 per sector) x f grid, equal weights -> point cloud
     whose min/max fix the normalization
  2. for each slider position, local refinement of weights and f from the best cloud point
Scenario handling: every evaluation runs each scenario on the same paths, so the mixture, the
scenario-specific and the min-max (worst scenario) objectives all come from one set of runs.
"""
from dataclasses import dataclass, field
from itertools import combinations
from typing import Dict, List, Optional, Sequence

import numpy as np

from . import budget as bud
from .simulate import (SimConfig, TickerSpec, draw_uniforms, roll_calendar, roll_multiples, run_program,
                       weighted_quantile)

F_GRID = (0.25, 0.5, 0.75, 1.0)
LAMBDAS = tuple(np.round(np.linspace(0, 1, 11), 2))


def feasible_subsets(tickers, k_min=3, k_max=5, max_per_sector=2):
    n = len(tickers)
    out = []
    for k in range(k_min, min(k_max, n) + 1):
        for idx in combinations(range(n), k):
            sectors = [tickers[i].sector for i in idx]
            if max(sectors.count(s) for s in set(sectors)) <= max_per_sector:
                out.append(idx)
    return out


@dataclass
class Evaluation:
    idx: tuple
    names: tuple
    sectors: tuple
    weights: np.ndarray
    f: float
    p_goal: float                 # mixture
    mean: float
    median: float
    ruin: float                   # per-cycle ruin probability, mixture
    p_by_scenario: Dict[str, float]
    mean_by_scenario: Dict[str, float]
    tag: str = "grid"

    @property
    def p_worst(self):
        return min(self.p_by_scenario.values())

    def key(self):
        return (self.idx, tuple(np.round(self.weights, 6)), round(self.f, 6))


class Evaluator:
    """Holds per-scenario roll multiples for ALL candidates on one set of uniforms (CRN)."""

    def __init__(self, tickers, corr, cfg=None, U=None):
        self.tickers = list(tickers)
        self.cfg = cfg or SimConfig(n_paths=4000, n_store=0)
        self.probs = bud.check_scenarios(self.cfg.scenarios)
        if U is None:
            U = draw_uniforms(self.cfg.n_paths, self.cfg.n_rolls, corr, self.cfg.copula, self.cfg.df, self.cfg.seed)
        self.U = U
        cal = roll_calendar(self.cfg.start, self.cfg.n_rolls, close_bdays=self.cfg.close_bdays)
        self.years = np.array([row["close"].year for row in cal])
        self.scen = []
        for sc, pi in zip(self.cfg.scenarios, self.probs):
            rates, budgets = bud.cycle_budgets(self.cfg.r0, sc.shift_per_cycle, self.cfg.n_cycles,
                                               self.cfg.V_deposit, self.cfg.V_own_rental, self.cfg.T_cycle)
            m, _ = roll_multiples(self.tickers, U, sc.name)
            self.scen.append(dict(name=sc.name, prob=float(pi), rates=rates, budgets=budgets, m=m))
        self.n_evals = 0

    def evaluate(self, idx, weights, f, tag="grid"):
        idx = tuple(idx)
        w = np.asarray(weights, dtype=float)
        cfg = SimConfig(**{**self.cfg.__dict__, "f": float(f), "n_store": 0})
        p_by, mean_by, xs, ws, ruin = {}, {}, [], [], 0.0
        for s in self.scen:
            M = s["m"][:, :, idx] @ w
            out = run_program(M, cfg, s["rates"], s["budgets"], self.years)
            x = out["terminal_net"]
            p_by[s["name"]] = float(np.mean(x >= cfg.goal))
            mean_by[s["name"]] = float(np.mean(x))
            if s["prob"] > 0:
                xs.append(x)
                ws.append(np.full(len(x), s["prob"] / len(x)))
                ruin += s["prob"] * float(out["ruined"].mean())
        self.n_evals += 1
        x, wt = np.concatenate(xs), np.concatenate(ws)
        pr = {s["name"]: s["prob"] for s in self.scen}
        return Evaluation(idx=idx, names=tuple(self.tickers[i].name for i in idx),
                          sectors=tuple(self.tickers[i].sector for i in idx), weights=w, f=float(f),
                          p_goal=sum(pr[k] * v for k, v in p_by.items()),
                          mean=sum(pr[k] * v for k, v in mean_by.items()),
                          median=float(weighted_quantile(x, wt, [0.5])[0]), ruin=ruin,
                          p_by_scenario=p_by, mean_by_scenario=mean_by, tag=tag)


def pareto_mask(means, probs):
    """Non-dominated points when maximizing both E[terminal] and P(goal)."""
    means, probs = np.asarray(means, dtype=float), np.asarray(probs, dtype=float)
    keep = np.ones(len(means), dtype=bool)
    for i in range(len(means)):
        dom = (means >= means[i]) & (probs >= probs[i]) & ((means > means[i]) | (probs > probs[i]))
        keep[i] = not dom.any()
    return keep


@dataclass
class Normalizer:
    p_lo: float
    p_hi: float
    m_lo: float
    m_hi: float

    @classmethod
    def fit(cls, evals):
        P = [e.p_goal for e in evals]
        M = [e.mean for e in evals]
        return cls(min(P), max(P), min(M), max(M))

    def score(self, e, lam, objective="mixture"):
        p = {"mixture": e.p_goal, "minmax": e.p_worst}.get(objective)
        if p is None:
            p = e.p_by_scenario[objective]
        npg = (p - self.p_lo) / (self.p_hi - self.p_lo) if self.p_hi > self.p_lo else 0.0
        nm = (e.mean - self.m_lo) / (self.m_hi - self.m_lo) if self.m_hi > self.m_lo else 0.0
        return (1 - lam) * npg + lam * nm


def _neighbours(e, f_steps=(0.05, 0.1), w_steps=(0.05, 0.1), f_min=0.1):
    k = len(e.idx)
    for d in w_steps:
        for i in range(k):
            for j in range(k):
                if i == j or e.weights[i] < d - 1e-12:
                    continue
                w = e.weights.copy()
                w[i] -= d
                w[j] += d
                w = np.round(np.clip(w, 0.0, None), 10)
                yield w / w.sum(), e.f
    for d in f_steps:
        for f in (e.f - d, e.f + d):
            if f_min - 1e-12 <= f <= 1.0 + 1e-12:
                yield e.weights, float(np.clip(f, f_min, 1.0))


def refine(ev, start, lam, norm, objective="mixture", max_iter=40, cache=None):
    """Greedy local search over weight transfers and f moves. Returns the best Evaluation found."""
    cache = {} if cache is None else cache
    best, best_s = start, norm.score(start, lam, objective)
    for _ in range(max_iter):
        improved = False
        for w, f in _neighbours(best):
            key = (best.idx, tuple(np.round(w, 6)), round(f, 6))
            cand = cache.get(key)
            if cand is None:
                cand = ev.evaluate(best.idx, w, f, tag="refined")
                cache[key] = cand
            s = norm.score(cand, lam, objective)
            if s > best_s + 1e-12:
                best, best_s, improved = cand, s, True
        if not improved:
            break
    return best


@dataclass
class SolveResult:
    evals: List[Evaluation]
    norm: Normalizer
    picks: Dict[float, Evaluation]          # slider position -> chosen combination
    best_by_count: Dict[int, Evaluation]    # ticker count -> max P(goal)
    frontier: np.ndarray = field(default=None)

    def pick(self, lam):
        lams = np.array(sorted(self.picks))
        return self.picks[float(lams[np.argmin(np.abs(lams - lam))])]


def solve(ev, lambdas=LAMBDAS, f_grid=F_GRID, k_min=3, k_max=5, max_per_sector=2, objective="mixture",
          refine_top=True, max_iter=40):
    subsets = feasible_subsets(ev.tickers, k_min, k_max, max_per_sector)
    if not subsets:
        raise ValueError("no feasible subset under the ticker-count and sector constraints")
    cache = {}
    evals = []
    for idx in subsets:
        w = np.full(len(idx), 1.0 / len(idx))
        for f in f_grid:
            e = ev.evaluate(idx, w, f)
            cache[e.key()] = e
            evals.append(e)
    norm = Normalizer.fit(evals)
    for lam in lambdas:
        start = max(evals, key=lambda e: norm.score(e, lam, objective))
        if refine_top:
            refine(ev, start, lam, norm, objective, max_iter, cache)
    allev = list({e.key(): e for e in list(cache.values()) + evals}.values())
    # final pick per slider position = best of everything evaluated (refinements from other positions
    # may have visited a better point than this position's own search)
    picks = {float(lam): max(allev, key=lambda e: norm.score(e, lam, objective)) for lam in lambdas}
    best_by_count = {}
    for e in allev:
        k = len(e.idx)
        if k not in best_by_count or e.p_goal > best_by_count[k].p_goal:
            best_by_count[k] = e
    res = SolveResult(evals=allev, norm=norm, picks=picks, best_by_count=best_by_count)
    res.frontier = pareto_mask([e.mean for e in allev], [e.p_goal for e in allev])
    return res


def scenario_table(ev, res, max_iter=40):
    """Optimal combination per scenario, the mixture-optimal one, and the min-max robust one (all at
    lambda = 0, i.e. maximizing goal probability). Recommendation: mixture — it optimizes against the
    probabilities you actually believe; min-max pays for insurance against a scenario you rate unlikely."""
    cache = {e.key(): e for e in res.evals}
    out = {}
    objectives = [s["name"] for s in ev.scen] + ["mixture", "minmax"]
    for obj in objectives:
        start = max(res.evals, key=lambda e: res.norm.score(e, 0.0, obj))
        refine(ev, start, 0.0, res.norm, obj, max_iter, cache)
    for obj in objectives:
        out[obj] = max(cache.values(), key=lambda e: res.norm.score(e, 0.0, obj))
    out["recommended"] = "mixture"
    return out


# ---------------------------------------------------------------- sensitivity

def ticker_count_curve(template, counts, rho, cfg, weights=None):
    """P(goal) for k identical, equally weighted tickers with pairwise correlation rho. Diversification
    pulls the terminal distribution toward its mean; with the mean below the goal, more tickers hurt
    (in the clearly subfair regime — see README finding A).

    Memory-light: an equicorrelated set is a one-factor model, Z_i = sqrt(rho) F + sqrt(1 - rho) e_i, so
    clones are generated one at a time and only the portfolio multiple is kept — O(paths x rolls) instead
    of O(paths x rolls x k). Clone i always gets the same e_i, so every k shares its first draws (CRN).
    """
    from scipy.stats import norm, t as student_t
    rho = float(np.clip(rho, 0.0, 0.999))
    probs = bud.check_scenarios(cfg.scenarios)
    cal = roll_calendar(cfg.start, cfg.n_rolls, close_bdays=cfg.close_bdays)
    years = np.array([row["close"].year for row in cal])
    N, R = cfg.n_paths, cfg.n_rolls
    kmax = max(counts)
    rng = np.random.default_rng(cfg.seed)
    F = rng.standard_normal((N, R))
    W = rng.chisquare(cfg.df, (N, R)) / cfg.df if cfg.copula == "t" else None
    seeds = rng.integers(0, 2**63 - 1, size=kmax)
    run_cfg = SimConfig(**{**cfg.__dict__, "n_store": 0})
    out = {}
    for k in sorted(counts):
        res_by = []
        for sc, pi in zip(cfg.scenarios, probs):
            M = np.zeros((N, R))
            marg = template.marginal(sc.name)
            for i in range(k):
                Z = np.sqrt(rho) * F + np.sqrt(1 - rho) * np.random.default_rng(seeds[i]).standard_normal((N, R))
                U = norm.cdf(Z) if W is None else student_t.cdf(Z / np.sqrt(W), cfg.df)
                Ri = marg.ppf(U)
                M += np.clip(Ri - template.k1, 0.0, template.k2 - template.k1) / template.cost_rel / k
            rates, budgets = bud.cycle_budgets(cfg.r0, sc.shift_per_cycle, cfg.n_cycles, cfg.V_deposit,
                                               cfg.V_own_rental, cfg.T_cycle)
            o = run_program(M, run_cfg, rates, budgets, years)
            res_by.append((pi, o["terminal_net"], float(o["ruined"].mean())))
        x = np.concatenate([r[1] for r in res_by])
        wts = np.concatenate([np.full(N, r[0] / N) for r in res_by])
        out[k] = dict(p_goal=float(sum(r[0] * np.mean(r[1] >= cfg.goal) for r in res_by)),
                      mean=float(sum(r[0] * r[1].mean() for r in res_by)),
                      median=float(weighted_quantile(x, wts, [0.5])[0]),
                      ruin=float(sum(r[0] * r[2] for r in res_by)))
    return out


def f_curve(ev, idx, weights, f_values=F_GRID):
    """Goal probability vs sleeve-ruin probability as f varies: the two faces of one trade-off."""
    return {float(f): ev.evaluate(idx, weights, f) for f in f_values}


def sensitivity(build, variants, corr, cfg, weights):
    """Re-build tickers for each variant (e.g. sigma_hat +/- 5pp, round-trip 2/4/8%) and evaluate on
    the same uniforms. build(variant) -> list[TickerSpec]."""
    U = draw_uniforms(cfg.n_paths, cfg.n_rolls, corr, cfg.copula, cfg.df, cfg.seed)
    out = {}
    for v in variants:
        tickers = build(v)
        ev = Evaluator(tickers, corr, cfg, U=U)
        e = ev.evaluate(tuple(range(len(tickers))), weights, cfg.f)
        out[v] = dict(p_goal=e.p_goal, mean=e.mean, median=e.median, ruin=e.ruin)
    return out

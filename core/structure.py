"""Cost-aware K2: choose the spread width on real quotes instead of always taking the next strike.

Why: on a live chain the narrowest spread's net premium can be smaller than the bid-ask it has to cross
(AMD 810/820: net mid $0.47, leg-crossing round trip 158% of premium), so "narrowest" maximizes cost,
not the objective. K1 stays on the spec's sigma rule; only K2 moves. Each candidate is scored with its
OWN chain-implied cost:
    expected : E^p[roll multiple]  = E^p[clip(R - k1, 0, k2 - k1)] / cost_rel       (one roll)
    goal     : P(8-year cumulative >= G), single-ticker program run on common random numbers
Only candidates passing the liquidity filter (OI >= 100, leg spread <= 10% of mid) and the truncation
condition are eligible; the spec's narrowest spread is always returned beside the choice for comparison.
"""
import numpy as np
from scipy.stats import qmc

from .simulate import Marginal, SimConfig, TickerSpec, draw_uniforms, simulate
from .spread import build_spread, candidate_spreads

CRITERIA = ("expected", "goal")


def _u_grid(n=8192, seed=0):
    return np.sort(qmc.Sobol(1, scramble=True, seed=seed).random(n)[:, 0])


def expected_roll_multiple(marginal, k1, k2, cost_rel, u=None):
    R = marginal.ppf(_u_grid() if u is None else u)
    return float(np.mean(np.clip(R - k1, 0.0, k2 - k1)) / cost_rel)


def liquid(sp, spread_check=True):
    """Liquidity filter. spread_check=False keeps only the open-interest test: after the close (and on
    weekends) Yahoo shows end-of-day quotes whose bid-ask is much wider than in the session, so the
    spread test says nothing about tradeability then. Costs still use those wide quotes (conservative)."""
    oi = sp.liquidity["oi_k1"] and sp.liquidity["oi_k2"]
    return sp.liquidity_ok if spread_check else oi


def score_candidates(cands, marginal, criterion="expected", cfg=None, S0=None, spread_check=True):
    if criterion not in CRITERIA:
        raise ValueError(f"criterion must be one of {CRITERIA}")
    u = _u_grid()
    out = []
    U = None
    for sp in cands:
        row = dict(K2=sp.K2, width=sp.K2 - sp.K1, net_mid=sp.net_mid, rt_cost=sp.rt_cost,
                   payoff_multiple=sp.payoff_multiple, liquidity_ok=liquid(sp, spread_check), target_ok=sp.target_ok,
                   e_multiple=expected_roll_multiple(marginal, sp.k1, sp.k2, sp.cost_rel, u))
        if criterion == "goal":
            cfg = cfg or SimConfig(n_paths=4000, n_store=0)
            if U is None:
                U = draw_uniforms(cfg.n_paths, cfg.n_rolls, np.eye(1), cfg.copula, cfg.df, cfg.seed)
            tk = TickerSpec.from_spread("X", sp, {"*": marginal})
            row["p_goal"] = simulate([tk], [1.0], np.eye(1), cfg, U=U).metrics()["p_goal"]
        row["score"] = row["p_goal"] if criterion == "goal" else row["e_multiple"]
        out.append(row)
    return out


def select_k2(chain, S0, sigma_hat, budget_krw, fx, marginal, criterion="expected", rt_cost=0.04,
              leg_cross=1.0, target_krw=None, cfg=None, max_width_pct=0.60, spread_check=True, **kw):
    """Returns dict(chosen, narrowest, table, eligible_found). chosen.k2_rule = 'cost-aware'."""
    cands = candidate_spreads(chain, S0, sigma_hat, budget_krw, fx, rt_cost=rt_cost, target_krw=target_krw,
                              cost_mode="chain", leg_cross=leg_cross, max_width_pct=max_width_pct, **kw)
    narrowest = build_spread(chain, S0, sigma_hat, budget_krw, fx, rt_cost=rt_cost, target_krw=target_krw,
                             cost_mode="chain", leg_cross=leg_cross, **kw)
    table = score_candidates(cands, marginal, criterion, cfg, S0, spread_check)
    eligible = [i for i, r in enumerate(table) if r["liquidity_ok"] and r["target_ok"]]
    pool = eligible or list(range(len(table)))
    best = max(pool, key=lambda i: table[i]["score"])
    chosen = cands[best]
    chosen.k2_rule = "cost-aware"
    if not eligible:
        chosen.notes.append("no candidate passes liquidity and truncation; best of all shown — not tradeable as is")
    for i, r in enumerate(table):
        r["chosen"] = i == best
        r["narrowest"] = cands[i].K2 == narrowest.K2
    return dict(chosen=chosen, narrowest=narrowest, table=table, criterion=criterion, eligible_found=bool(eligible))

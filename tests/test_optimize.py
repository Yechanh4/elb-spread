import numpy as np
import pytest

from core import budget as bud
from core.optimize import (Evaluator, feasible_subsets, pareto_mask, scenario_table, solve, ticker_count_curve)
from core.simulate import Marginal, SimConfig, TickerSpec

T = 0.25


def tk(name, sector, sigma, mu=0.10, k1=1.25, k2=1.30, cost_rel=None, scen_sigma=None):
    if cost_rel is None:
        cost_rel = 0.007 * (sigma / 0.45) ** 2
    if scen_sigma is None:
        marg = {"*": Marginal.lognormal(mu, sigma, T)}
    else:
        marg = {s: Marginal.lognormal(mu, v, T) for s, v in scen_sigma.items()}
    return TickerSpec(name, k1, k2, cost_rel, marg, sector=sector)


@pytest.fixture(scope="module")
def universe():
    tks = [tk("A", "Tech", 0.55), tk("B", "Tech", 0.50), tk("C", "Tech", 0.48),
           tk("D", "Health", 0.45), tk("E", "Energy", 0.52), tk("F", "Fin", 0.40)]
    C = np.full((6, 6), 0.3)
    np.fill_diagonal(C, 1.0)
    return tks, C


def test_feasible_subsets_respect_constraints(universe):
    tks, _ = universe
    subs = feasible_subsets(tks, 3, 5, 2)
    assert subs and all(3 <= len(s) <= 5 for s in subs)
    for s in subs:
        sectors = [tks[i].sector for i in s]
        assert max(sectors.count(x) for x in sectors) <= 2
    assert (0, 1, 2) not in subs                         # three Tech names
    assert (0, 1, 3) in subs


def test_pareto_mask():
    means = np.array([1, 2, 3, 2, 1.5])
    probs = np.array([0.5, 0.4, 0.1, 0.3, 0.5])
    assert list(pareto_mask(means, probs)) == [False, True, True, False, True]


def test_common_random_numbers_make_evaluations_deterministic(universe):
    tks, C = universe
    cfg = SimConfig(n_paths=2000, n_store=0, goal=300e6, seed=3)
    ev = Evaluator(tks, C, cfg)
    a = ev.evaluate((0, 3, 4), np.full(3, 1 / 3), 0.75)
    b = ev.evaluate((0, 3, 4), np.full(3, 1 / 3), 0.75)
    assert a.p_goal == b.p_goal and a.mean == b.mean
    # a second evaluator with the same seed sees the same paths
    ev2 = Evaluator(tks, C, cfg)
    assert ev2.evaluate((0, 3, 4), np.full(3, 1 / 3), 0.75).mean == a.mean


@pytest.fixture(scope="module")
def solved(universe):
    tks, C = universe
    cfg = SimConfig(n_paths=1500, n_store=0, goal=300e6, seed=1)
    ev = Evaluator(tks, C, cfg)
    res = solve(ev, lambdas=(0.0, 0.5, 1.0), f_grid=(0.5, 1.0), max_iter=6)
    return ev, res


def test_slider_extremes(solved):
    ev, res = solved
    p_left, p_right = res.pick(0.0), res.pick(1.0)
    best_p = max(e.p_goal for e in res.evals)
    best_m = max(e.mean for e in res.evals)
    assert p_left.p_goal == pytest.approx(best_p)
    assert p_right.mean == pytest.approx(best_m)
    assert p_left.p_goal >= p_right.p_goal and p_right.mean >= p_left.mean
    for e in res.picks.values():
        assert abs(e.weights.sum() - 1) < 1e-9 and np.all(e.weights >= -1e-12)
        assert 0.1 - 1e-9 <= e.f <= 1.0 + 1e-9


def test_frontier_contains_picks(solved):
    ev, res = solved
    front = {res.evals[i].key() for i in np.flatnonzero(res.frontier)}
    assert res.pick(0.0).key() in front and res.pick(1.0).key() in front


def test_scenario_table_minmax_is_most_robust(universe):
    tks0, C = universe
    # scenario-dependent views: A shines only when rates fall, F only when they rise
    tks = [tk(t.name, t.sector, 0.45, scen_sigma={"up": 0.45, "neutral": 0.45, "down": 0.45}) for t in tks0]
    tks[0] = tk("A", "Tech", 0.55, scen_sigma={"up": 0.35, "neutral": 0.50, "down": 0.70})
    tks[5] = tk("F", "Fin", 0.45, scen_sigma={"up": 0.62, "neutral": 0.47, "down": 0.40})
    cfg = SimConfig(n_paths=1500, n_store=0, goal=300e6, seed=2)
    ev = Evaluator(tks, C, cfg)
    res = solve(ev, lambdas=(0.0,), f_grid=(0.5, 1.0), max_iter=4)
    tab = scenario_table(ev, res, max_iter=4)
    assert tab["recommended"] == "mixture"
    worst = {k: tab[k].p_worst for k in ("up", "neutral", "down", "mixture", "minmax")}
    assert worst["minmax"] >= max(worst.values()) - 1e-12
    mix = {k: tab[k].p_goal for k in ("up", "neutral", "down", "mixture", "minmax")}
    assert mix["mixture"] >= max(mix.values()) - 1e-12


NEUTRAL_ONLY = (bud.RateScenario("n", 0.0, 1.0),)


def test_more_tickers_hurt_goal_probability_when_clearly_subfair():
    """Mean below the goal and E[roll multiple] well below 1: diversification pulls outcomes toward
    the mean and away from G, monotonically."""
    t = tk("X", "S", 0.45, mu=0.0, cost_rel=0.0095)          # E[M] per roll ~0.61
    cfg = SimConfig(n_paths=4000, n_store=0, goal=300e6, f=1.0, seed=6, scenarios=NEUTRAL_ONLY)
    curve = ticker_count_curve(t, (1, 3, 10, 30), 0.3, cfg)
    p = [curve[k]["p_goal"] for k in (1, 3, 10, 30)]
    assert curve[1]["mean"] < cfg.goal
    assert all(a > b for a, b in zip(p, p[1:]))


def test_ticker_count_curve_is_humped_near_fair_with_full_staking():
    """Counter-regime, kept on purpose: with E[M] near 1 and f = 1, a single ticker loses the whole sleeve
    at its first miss (~82% per roll), so 3 tickers beat 1 before more tickers start to hurt again.
    The spec's 'more tickers hurt' is a statement about the clearly-subfair regime; the UI shows the curve."""
    t = tk("X", "S", 0.50, mu=0.10)                           # E[M] per roll ~0.92
    cfg = SimConfig(n_paths=4000, n_store=0, goal=756e6, f=1.0, seed=6, scenarios=NEUTRAL_ONLY)
    curve = ticker_count_curve(t, (1, 3, 30), 0.3, cfg)
    assert curve[3]["p_goal"] > curve[1]["p_goal"] + 0.02
    assert curve[30]["p_goal"] < curve[3]["p_goal"] - 0.02

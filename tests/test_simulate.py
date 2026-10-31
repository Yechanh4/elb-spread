from collections import defaultdict
from datetime import date

import numpy as np
import pytest
from scipy.stats import kendalltau, norm

from core import budget as bud
from core.bsm import lognormal_pdf
from core.simulate import (Marginal, SimConfig, TickerSpec, cumulative_pin, draw_uniforms, dubins_savage_regime,
                           pin_prob, pin_prob_lognormal, roll_calendar, roll_multiples, simulate, summarize)

T = 0.25


def lognorm_ticker(name="AAA", mu=0.08, sigma=0.45, k1=1.25, k2=1.30, cost_rel=0.006, sector="Tech"):
    return TickerSpec(name, k1, k2, cost_rel, {"*": Marginal.lognormal(mu, sigma, T)}, sector=sector)


def fixed_return_ticker(R, k1=1.10, k2=1.20, cost_rel=None, multiple=1.1):
    """R lands at the same place every roll: payoff multiple is deterministic."""
    cost_rel = (k2 - k1) / multiple if cost_rel is None else cost_rel
    return TickerSpec("FIX", k1, k2, cost_rel, {"*": Marginal.lognormal(np.log(R) / T, 1e-9, T)})


# ---------------------------------------------------------------- pin risk

def test_pin_probability_mc_matches_closed_form():
    mu, sigma, k1, k2 = 0.08, 0.33, 1.25, 1.30
    p = pin_prob_lognormal(mu, sigma, T, k1, k2)
    U = draw_uniforms(400_000, 1, np.eye(1), seed=7)
    _, pins = roll_multiples([lognorm_ticker(mu=mu, sigma=sigma, k1=k1, k2=k2)], U, "neutral")
    se = np.sqrt(p * (1 - p) / pins.size)
    assert abs(pins.mean() - p) < 4 * se
    # closed form is the lognormal CDF difference
    s = sigma * np.sqrt(T)
    d = lambda k: (np.log(k) - (mu - 0.5 * sigma**2) * T) / s
    assert p == pytest.approx(norm.cdf(d(k2)) - norm.cdf(d(k1)))


def test_report_pin_numbers_reproduce():
    """The rules document: K1 = 1.25 S0, K2 = 1.30 S0, 3 months -> ~3.4% per expiry, ~67% over 32."""
    p = pin_prob_lognormal(0.0, 0.33, T, 1.25, 1.30)
    assert 0.025 < p < 0.045
    assert cumulative_pin(0.034, 32) == pytest.approx(0.669, abs=0.005)


def test_cumulative_pin_equals_one_minus_survival():
    for p in (0.0, 0.01, 0.034, 0.2):
        assert cumulative_pin(p, 32) == pytest.approx(1 - (1 - p) ** 32)
    tk = lognorm_ticker(mu=0.0, sigma=0.33)
    p = pin_prob(tk.marginal("x"), tk.k1, tk.k2)
    cfg = SimConfig(n_paths=40_000, goal=1e18, f=1.0, tax=False, scenarios=(bud.RateScenario("n", 0.0, 1.0),),
                    n_store=0, seed=3)
    res = simulate([tk], [1.0], np.eye(1), cfg)
    # with f = 1 a total loss zeroes the sleeve and later rolls of that cycle deploy nothing, so count
    # pins over every roll from the raw flags instead
    U = draw_uniforms(cfg.n_paths, 32, np.eye(1), seed=3)
    _, pins = roll_multiples([tk], U, "n")
    any_pin = pins[:, :, 0].any(axis=1).mean()
    assert any_pin == pytest.approx(cumulative_pin(p, 32), abs=0.01)
    assert res.runs[0].pin_by_ticker[0] == pytest.approx(p, abs=0.003)


def test_exercise_cash_is_premium_times_k1_over_cost():
    tk = lognorm_ticker()
    assert tk.exercise_to_premium == pytest.approx(1.25 / 0.006)      # ~208x the premium per KRW deployed


# ---------------------------------------------------------------- absorbing state and anti-martingale

def _single(cfg, tk):
    return simulate([tk], [1.0], np.eye(1), cfg)


def test_absorbing_state_stops_deployment_for_rest_of_cycle():
    tk = lognorm_ticker(mu=0.15, sigma=0.55)
    B0 = bud.budget()
    cfg = SimConfig(n_paths=6000, n_store=6000, goal=3 * B0, f=1.0, tax=False,
                    scenarios=(bud.RateScenario("n", 0.0, 1.0),), seed=11)
    res = _single(cfg, tk)
    run = res.runs[0]
    dep, absd = run.paths["deployed"], run.paths["absorbed"]
    n_abs = 0
    for c in range(cfg.n_cycles):
        sl = slice(c * 8, (c + 1) * 8)
        for p in range(cfg.n_paths):
            a = run.absorbed_at[p, c]
            if a >= 0:
                n_abs += 1
                assert np.all(dep[p, sl][a:] == 0), "capital deployed after goal reached"
                assert np.all(absd[p, sl][a:] == 1)
                assert np.all(absd[p, sl][:a] == 0)
    assert n_abs > 100
    # once the goal is banked, a new cycle's first roll is already absorbed (cumulative >= G at start)
    after = run.absorbed_at[:, 1:]
    first_banked = run.absorbed_at[:, 0] >= 0
    assert np.all(after[first_banked] == 0)


def test_capital_per_roll_never_exceeds_sleeve_times_f():
    tk = lognorm_ticker(mu=0.10, sigma=0.50)
    f = 0.6
    cfg = SimConfig(n_paths=3000, n_store=3000, goal=1e18, f=f, tax=False,
                    scenarios=(bud.RateScenario("n", 0.0, 1.0),), seed=5)
    run = _single(cfg, tk).runs[0]
    dep, sleeve = run.paths["deployed"], run.paths["sleeve"]
    for c in range(cfg.n_cycles):
        for jj in range(8):
            j = c * 8 + jj
            before = np.full(cfg.n_paths, run.budgets[c]) if jj == 0 else sleeve[:, j - 1]
            assert np.all(dep[:, j] <= before * f * (1 + 1e-12) + 1e-9)
            assert np.allclose(dep[:, j], before * f)                    # exactly sleeve * f: no top-ups
    # after a total loss of the deployed part, the next roll stakes less, never more
    losses = run.paths["proceeds"][:, :-1] == 0
    nxt = dep[:, 1:] < dep[:, :-1]
    same_cycle = (np.arange(31) % 8) != 7
    assert np.all(nxt[losses & same_cycle[None, :]])


def _f_curve(tk, fs=(0.25, 0.5, 1.0)):
    return {f: _single(SimConfig(n_paths=8000, n_store=0, goal=300e6, f=f, seed=2), tk).metrics() for f in fs}


def test_lower_f_lowers_ruin():
    out = _f_curve(lognorm_ticker(mu=0.12, sigma=0.50))
    assert out[0.25]["ruin_per_cycle"] < out[0.5]["ruin_per_cycle"] < out[1.0]["ruin_per_cycle"]


def test_dubins_savage_decides_which_f_maximizes_goal_probability():
    """Superfair rolls (E[M] > risk-free growth): timid play — small f, many rolls — wins.
    Subfair rolls: bold play — f = 1 — wins, and then f really is a goal-vs-ruin trade-off."""
    superfair = lognorm_ticker(mu=0.12, sigma=0.50, cost_rel=0.006)      # E[M] ~1.36
    subfair = lognorm_ticker(mu=0.0, sigma=0.45, cost_rel=0.0095)        # E[M] ~0.61
    rf = (1 + bud.R_KRW_2Y) ** T
    sup, sub = _f_curve(superfair), _f_curve(subfair)
    assert dubins_savage_regime(sup[1.0]["roll_multiple_mean"], rf) == "timid"
    assert dubins_savage_regime(sub[1.0]["roll_multiple_mean"], rf) == "bold"
    assert sup[0.25]["p_goal"] > sup[1.0]["p_goal"] + 0.05
    assert sub[1.0]["p_goal"] > sub[0.25]["p_goal"] + 0.05


# ---------------------------------------------------------------- tax

def _per_year_tax(run, cal, cfg):
    gains = defaultdict(float)
    for j, row in enumerate(cal):
        gains[row["close"].year] += run.paths["proceeds"][0, j] - run.paths["deployed"][0, j]
    return gains, sum(cfg.tax_rate * max(g - cfg.tax_deduction, 0) for g in gains.values())


@pytest.mark.parametrize("start,n_years", [(date(2026, 12, 1), 8), (date(2027, 6, 1), 9)])
def test_tax_deduction_once_per_calendar_year(start, n_years):
    tk = fixed_return_ticker(R=1.5, multiple=1.02)
    cfg = SimConfig(n_paths=4, n_store=4, goal=1e18, f=1.0, tax=True, start=start,
                    scenarios=(bud.RateScenario("n", 0.0, 1.0),))
    res = _single(cfg, tk)
    run = res.runs[0]
    gains, expected = _per_year_tax(run, res.calendar, cfg)
    assert len(gains) == n_years
    assert run.taxes[0] == pytest.approx(expected, rel=1e-9)
    per_trade = sum(cfg.tax_rate * max(run.paths["proceeds"][0, j] - run.paths["deployed"][0, j]
                                       - cfg.tax_deduction, 0) for j in range(32))
    assert run.taxes[0] != pytest.approx(per_trade, rel=1e-3)          # not deducted per trade
    assert run.terminal_net[0] == pytest.approx(run.terminal_gross[0] - run.taxes[0])


def test_tax_trap_gain_taxed_in_december_then_lost_in_march():
    """Realized gain in year 1 is taxed; re-staking it and losing it in year 2 cannot offset it, so the
    net result is negative: the tax comes out of the protected principal."""
    from core.simulate import run_program
    cfg = SimConfig(n_cycles=1, rolls_per_cycle=2, f=1.0, goal=1e18, tax=True, n_store=1)
    B = 80e6
    M = np.array([[5.0, 0.0]])
    out = run_program(M, cfg, rates=np.array([0.0376]), budgets=np.array([B]), years=np.array([2027, 2028]))
    assert out["terminal_gross"][0] == 0.0
    assert out["taxes"][0] == pytest.approx(0.22 * (4 * B - 2.5e6))
    assert out["terminal_net"][0] < 0
    # the same two rolls inside ONE calendar year net to a loss: no tax
    same = run_program(M, cfg, rates=np.array([0.0376]), budgets=np.array([B]), years=np.array([2027, 2027]))
    assert same["taxes"][0] == 0.0 and same["terminal_net"][0] == 0.0


def test_losses_net_within_year_and_tax_toggle():
    tk = lognorm_ticker(mu=0.10, sigma=0.5)
    on = _single(SimConfig(n_paths=4000, n_store=0, seed=9, tax=True, goal=1e18), tk)
    off = _single(SimConfig(n_paths=4000, n_store=0, seed=9, tax=False, goal=1e18), tk)
    assert np.all(off.runs[0].taxes == 0)
    assert np.allclose(off.runs[0].terminal_net, off.runs[0].terminal_gross)
    # no absorbing (goal unreachable): identical paths, so tax only ever subtracts
    assert np.allclose(on.runs[0].terminal_gross, off.runs[0].terminal_gross)
    assert np.all(on.runs[0].terminal_net <= off.runs[0].terminal_net + 1e-6)
    assert np.all(on.runs[0].taxes >= 0)


# ---------------------------------------------------------------- dependence

def test_simulated_correlation_matches_input():
    C = np.array([[1.0, 0.6, 0.2], [0.6, 1.0, -0.3], [0.2, -0.3, 1.0]])
    U = draw_uniforms(200_000, 1, C, "gaussian", seed=1)[:, 0, :]
    Z = norm.ppf(U)
    assert np.allclose(np.corrcoef(Z, rowvar=False), C, atol=0.01)
    # both elliptical copulas share Kendall's tau = (2/pi) arcsin(rho)
    Ut = draw_uniforms(20_000, 1, C, "t", df=5, seed=2)[:, 0, :]
    tau = kendalltau(Ut[:, 0], Ut[:, 1])[0]
    assert tau == pytest.approx(2 / np.pi * np.arcsin(0.6), abs=0.02)


def test_t_copula_raises_joint_total_loss():
    k, rho = 4, 0.4
    C = np.full((k, k), rho)
    np.fill_diagonal(C, 1.0)
    tickers = [lognorm_ticker(name=f"T{i}", mu=0.0, sigma=0.33) for i in range(k)]
    out = {}
    for cop in ("gaussian", "t"):
        U = draw_uniforms(300_000, 1, C, cop, df=5, seed=4)
        m, _ = roll_multiples(tickers, U, "n")
        out[cop] = float((m == 0).all(axis=2).mean())
    se = np.sqrt(out["gaussian"] * (1 - out["gaussian"]) / 300_000)
    assert out["t"] > out["gaussian"] + 5 * se


# ---------------------------------------------------------------- program-level outputs

def test_mixture_is_probability_weighted_and_rates_shrink_budget():
    tk = lognorm_ticker(mu=0.12, sigma=0.5)
    cfg = SimConfig(n_paths=5000, n_store=0, goal=300e6, seed=8)
    res = _single(cfg, tk)
    m = res.metrics()
    by = {r.name: r for r in res.runs}
    p_mix = sum(r.prob * np.mean(r.terminal_net >= cfg.goal) for r in res.runs)
    assert m["p_goal"] == pytest.approx(p_mix, abs=1e-12)
    assert np.all(by["down"].budgets[1:] < by["neutral"].budgets[1:])
    assert np.all(by["up"].budgets[1:] > by["neutral"].budgets[1:])
    assert by["down"].budgets[0] == pytest.approx(by["up"].budgets[0])
    # same uniforms and no absorbing: the sleeve scales with B, so a lower budget is lower pathwise
    lin = {r.name: r for r in _single(SimConfig(n_paths=2000, n_store=0, goal=1e18, tax=False, seed=8), tk).runs}
    assert np.all(lin["down"].terminal_gross <= lin["neutral"].terminal_gross + 1e-6)


def test_summary_fields_and_skew_annotation():
    tk = lognorm_ticker(mu=0.12, sigma=0.5)
    res = _single(SimConfig(n_paths=5000, n_store=0, goal=300e6, seed=8), tk)
    m = res.metrics()
    q = m["quantiles"]
    assert list(q) == [0.01, 0.05, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
    assert all(a <= b for a, b in zip(list(q.values()), list(q.values())[1:]))
    assert q[0.5] == pytest.approx(m["median"])
    assert m["mean_driven_by_tail"] == (m["mean"] > 2 * m["median"])
    # f = 1: most cycles lose the whole sleeve, so the median is 0 and the mean is all tail
    allin = _single(SimConfig(n_paths=3000, n_store=0, goal=300e6, f=1.0, tax=False, seed=8), tk).metrics()
    assert allin["median"] == 0 and allin["mean"] > 0 and allin["mean_driven_by_tail"]
    ladder = {row["level"]: row for row in m["ladder"]}
    assert ladder[300e6]["prob"] == pytest.approx(m["p_goal"])
    assert ladder[300e6]["monthly_reduction"] == pytest.approx(bud.annuity_payment(300e6, 0.04, 264))
    probs = [row["prob"] for row in m["ladder"]]
    assert all(a >= b for a, b in zip(probs, probs[1:]))
    gross = summarize(res, net=False)
    assert gross["mean"] >= m["mean"]


def test_dubins_savage_regime():
    assert dubins_savage_regime(1.10, 1.01) == "timid"
    assert dubins_savage_regime(0.90, 1.01) == "bold"


def test_roll_calendar():
    cal = roll_calendar(date(2027, 1, 4), 32)
    assert len(cal) == 32
    for i, row in enumerate(cal):
        assert row["expiry"].weekday() == 4 and 15 <= row["expiry"].day <= 21     # third Friday
        assert np.busday_count(row["close"], row["expiry"]) == 2
        if i:
            assert row["entry"] == cal[i - 1]["close"]
    assert cal[-1]["expiry"].year == 2035


def test_marginal_from_density_matches_lognormal():
    S = np.linspace(20, 300, 6001)
    w = np.full_like(S, S[1] - S[0])
    w[[0, -1]] /= 2
    dens = lognormal_pdf(S, 100, T, 0.05, 0.0, 0.4)
    m = Marginal.from_density(S, dens, w, 100.0)
    ref = Marginal.lognormal(0.05, 0.4, T)
    u = np.array([0.05, 0.25, 0.5, 0.9, 0.99])
    assert np.allclose(m.ppf(u), ref.ppf(u), rtol=2e-3)
    assert pin_prob(m, 1.25, 1.30) == pytest.approx(pin_prob(ref, 1.25, 1.30), abs=1e-3)


def test_per_cycle_weights_and_f():
    a = lognorm_ticker("A", mu=0.10, sigma=0.5)
    b = lognorm_ticker("B", mu=0.10, sigma=0.5, k1=1.20, k2=1.25, cost_rel=0.012)
    C = np.array([[1.0, 0.3], [0.3, 1.0]])
    base = dict(n_paths=3000, n_store=3000, goal=1e18, tax=False, scenarios=(bud.RateScenario("n", 0.0, 1.0),), seed=4)
    # identical rows == constant weights
    r1 = simulate([a, b], [0.6, 0.4], C, SimConfig(**base))
    r2 = simulate([a, b], [[0.6, 0.4]] * 4, C, SimConfig(**base))
    assert np.allclose(r1.runs[0].terminal_gross, r2.runs[0].terminal_gross)
    # holding only A in cycle 1 and only B afterwards: cycle 1 matches the A-only run
    rA = simulate([a, b], [1.0, 0.0], C, SimConfig(**base))
    rx = simulate([a, b], [[1, 0], [0, 1], [0, 1], [0, 1]], C, SimConfig(**base))
    assert np.allclose(rx.runs[0].paths["sleeve"][:, :8], rA.runs[0].paths["sleeve"][:, :8])
    assert not np.allclose(rx.runs[0].paths["sleeve"][:, 8:], rA.runs[0].paths["sleeve"][:, 8:])
    # per-cycle f: deployed = sleeve_before * f_c exactly
    fc = (1.0, 0.5, 0.25, 0.75)
    run = simulate([a], [1.0], np.eye(1), SimConfig(**base, f_cycles=fc)).runs[0]
    for c in range(4):
        j = c * 8
        assert np.allclose(run.paths["deployed"][:, j], run.budgets[c] * fc[c])
        assert np.allclose(run.paths["deployed"][:, j + 1], run.paths["sleeve"][:, j] * fc[c])
    with pytest.raises(ValueError):
        SimConfig(f_cycles=(1.0, 0.5))
    with pytest.raises(ValueError):
        simulate([a, b], [[0.5, 0.5]] * 3, C, SimConfig(**base))

"""Correlated multi-asset Monte Carlo of the 8-year program: 4 cycles x 8 rolls of 3-month spreads.

Per path, per cycle c:
    sleeve <- B_c                                   (fresh budget from the cycle's bond, at r_c)
    per roll:  absorbed |= cumulative_net >= G      (absorbing state, rest of the cycle)
               deploy  = sleeve * f   (0 if absorbed)          <- never more: not a martingale
               sleeve  = (sleeve - deploy)(1 + r_c)^T_opt + deploy * M
               M = sum_i w_i clip(R_i - k1_i, 0, k2_i - k1_i) / cost_rel_i
    cycle end: the sleeve is banked (used against the mortgage); the next cycle starts from a new B.
cumulative value = banked proceeds + current sleeve - overseas-derivative tax (22% over a 2.5M KRW
deduction, once per calendar year, losses netted within the year).

Stationarity: every roll re-strikes at the same moneyness (k1, k2) with the same relative cost, so the
roll multiple distribution only depends on the return marginal of that scenario.
Scenarios: each runs on the same uniforms (common random numbers); the mixture is the probability-
weighted pool of the scenario runs, so P_mix = sum_s pi_s P_s exactly.
"""
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Dict, List, Optional

import numpy as np
from scipy.stats import norm, t as student_t

from . import budget as bud

TAX_RATE = 0.22
TAX_DEDUCTION = 2_500_000
RUIN_FRAC = 0.05            # cycle-end sleeve below 5% of B_c = sleeve ruined
LADDER = (70_000_000, 150_000_000, 300_000_000, 500_000_000, 756_000_000)
QUANTILES = (0.01, 0.05, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)


# ---------------------------------------------------------------- marginals of R = S_T / S0

@dataclass
class Marginal:
    """Distribution of the one-roll price ratio R = S_T/S0, as an inverse CDF."""
    R: Optional[np.ndarray] = None
    cdf_nodes: Optional[np.ndarray] = None
    mu: Optional[float] = None          # analytic lognormal (price drift, ex-dividend)
    sigma: Optional[float] = None
    T: Optional[float] = None

    @classmethod
    def lognormal(cls, mu, sigma, T):
        return cls(mu=float(mu), sigma=float(sigma), T=float(T))

    @classmethod
    def from_density(cls, S, dens, w, S0):
        mass = np.asarray(w, dtype=float) * np.maximum(np.asarray(dens, dtype=float), 0.0)
        cdf = np.cumsum(mass)
        cdf /= cdf[-1]
        cdf = cdf + np.arange(len(cdf)) * 1e-15      # strictly increasing for interpolation
        return cls(R=np.asarray(S, dtype=float) / S0, cdf_nodes=cdf)

    @property
    def analytic(self):
        return self.sigma is not None

    def ppf(self, u):
        u = np.asarray(u, dtype=float)
        if self.analytic:
            s = self.sigma * np.sqrt(self.T)
            return np.exp((self.mu - 0.5 * self.sigma**2) * self.T + s * norm.ppf(u))
        return np.interp(u, self.cdf_nodes, self.R)

    def cdf(self, x):
        x = np.asarray(x, dtype=float)
        if self.analytic:
            s = self.sigma * np.sqrt(self.T)
            with np.errstate(divide="ignore"):
                return norm.cdf((np.log(x) - (self.mu - 0.5 * self.sigma**2) * self.T) / s)
        return np.interp(x, self.R, self.cdf_nodes, left=0.0, right=1.0)


def pin_prob(marginal, k1, k2):
    """P(K1 <= S_T < K2): long leg auto-exercised, short leg expires, nothing to net against."""
    return float(marginal.cdf(k2) - marginal.cdf(k1))


def pin_prob_lognormal(mu, sigma, T, k1, k2):
    return pin_prob(Marginal.lognormal(mu, sigma, T), k1, k2)


def cumulative_pin(p, n_rolls=32):
    """P(at least one pin across n independent expiries) = 1 - (1 - p)^n."""
    return 1.0 - (1.0 - p) ** n_rolls


# ---------------------------------------------------------------- tickers, correlation, copula

@dataclass
class TickerSpec:
    name: str
    k1: float                       # K1 / S0
    k2: float                       # K2 / S0
    cost_rel: float                 # net cost per share incl. round-trip cost / S0
    marginals: Dict[str, Marginal]  # scenario name -> Marginal; key "*" applies to every scenario
    sector: str = ""

    def marginal(self, scenario):
        return self.marginals.get(scenario, self.marginals.get("*"))

    @property
    def exercise_to_premium(self):
        return self.k1 / self.cost_rel

    @classmethod
    def from_spread(cls, name, spread, marginals, sector=""):
        return cls(name=name, k1=spread.k1, k2=spread.k2, cost_rel=spread.cost_rel, marginals=marginals,
                   sector=sector)


def nearest_corr(C, eps=1e-10):
    """Clip negative eigenvalues and rescale to unit diagonal."""
    C = 0.5 * (np.asarray(C, dtype=float) + np.asarray(C, dtype=float).T)
    vals, vecs = np.linalg.eigh(C)
    if vals.min() >= eps:
        return C
    C2 = vecs @ np.diag(np.maximum(vals, eps)) @ vecs.T
    d = np.sqrt(np.diag(C2))
    return C2 / np.outer(d, d)


def correlation_from_prices(prices):
    """Correlation of daily log returns. prices: (n_days, k), NaN rows dropped pairwise-complete."""
    P = np.asarray(prices, dtype=float)
    r = np.diff(np.log(P), axis=0)
    r = r[np.all(np.isfinite(r), axis=1)]
    return nearest_corr(np.corrcoef(r, rowvar=False))


def draw_uniforms(n_paths, n_steps, corr, copula="gaussian", df=5, seed=0):
    """(n_paths, n_steps, k) uniforms with the given dependence. t-copula: one chi-square mixing draw per
    (path, roll) shared by all assets, which is what creates joint crashes / joint total losses."""
    C = nearest_corr(np.atleast_2d(corr))
    k = C.shape[0]
    rng = np.random.default_rng(seed)
    L = np.linalg.cholesky(C)
    Z = rng.standard_normal((n_paths, n_steps, k)) @ L.T
    if copula == "gaussian":
        return norm.cdf(Z)
    if copula == "t":
        W = rng.chisquare(df, (n_paths, n_steps, 1)) / df
        return student_t.cdf(Z / np.sqrt(W), df)
    raise ValueError(f"unknown copula {copula!r}")


# ---------------------------------------------------------------- calendar

def third_friday(year, month):
    d = date(year, month, 15)
    return d + timedelta(days=(4 - d.weekday()) % 7)


def add_months(y, m, k):
    t = y * 12 + (m - 1) + k
    return t // 12, t % 12 + 1


def roll_calendar(start, n_rolls=32, months=3, close_bdays=2):
    """Entry, expiry (third Friday), close-by date (close_bdays business days before expiry) per roll."""
    rows = []
    entry = start
    for j in range(n_rolls):
        y, m = add_months(start.year, start.month, months * (j + 1))
        exp = third_friday(y, m)
        close = np.busday_offset(np.datetime64(exp), -close_bdays, roll="backward").astype(date)
        rows.append(dict(roll=j, entry=entry, expiry=exp, close=close))
        entry = close
    return rows


# ---------------------------------------------------------------- config and results

@dataclass
class SimConfig:
    n_paths: int = 20_000
    n_cycles: int = bud.N_CYCLES
    rolls_per_cycle: int = bud.ROLLS_PER_CYCLE
    T_opt: float = bud.T_OPT
    f: float = 1.0
    goal: float = bud.GOAL
    V_deposit: float = bud.V_DEPOSIT
    V_own_rental: float = bud.V_OWN_RENTAL
    r0: float = bud.R_KRW_2Y
    T_cycle: float = bud.T_CYCLE
    scenarios: tuple = bud.DEFAULT_SCENARIOS
    tax: bool = True
    tax_rate: float = TAX_RATE
    tax_deduction: float = TAX_DEDUCTION
    start: date = date(2027, 1, 4)
    close_bdays: int = 2
    copula: str = "gaussian"
    df: int = 5
    seed: int = 0
    ruin_frac: float = RUIN_FRAC
    absorbing: bool = True
    n_store: int = 200
    f_cycles: Optional[tuple] = None    # per-cycle staking fraction (overrides f), e.g. edited on page 3

    @property
    def n_rolls(self):
        return self.n_cycles * self.rolls_per_cycle

    def __post_init__(self):
        if not 0.0 <= self.f <= 1.0:
            raise ValueError("staking fraction f must be in [0, 1]")
        if self.f_cycles is not None:
            if len(self.f_cycles) != self.n_cycles or not all(0.0 <= x <= 1.0 for x in self.f_cycles):
                raise ValueError("f_cycles needs one value in [0, 1] per cycle")

    def f_of(self, c):
        return float(self.f_cycles[c]) if self.f_cycles is not None else self.f


@dataclass
class ScenarioRun:
    name: str
    prob: float
    rates: np.ndarray            # r per cycle
    budgets: np.ndarray          # B per cycle
    terminal_gross: np.ndarray   # (N,) total banked proceeds, before tax
    taxes: np.ndarray            # (N,)
    terminal_net: np.ndarray     # (N,)
    ruined: np.ndarray           # (N, n_cycles) cycle-end sleeve below ruin_frac * B_c
    absorbed_at: np.ndarray      # (N, n_cycles) roll index within the cycle when absorbed, -1 = never
    pin_rolls: np.ndarray        # (N,) number of rolls with >= 1 deployed ticker in [K1, K2)
    pin_by_ticker: np.ndarray    # (k,) per-roll P(K1 <= S_T < K2) per ticker
    max_exercise_cash: np.ndarray  # (N,) largest KRW exercise demand a pin would have triggered
    roll_multiple_mean: float    # E[M] per roll
    joint_total_loss: float      # P(all tickers expire below K1) per roll
    paths: dict = field(repr=False, default_factory=dict)   # first n_store paths, per roll


def roll_multiples(tickers, U, scenario):
    """Per-ticker roll multiples m (N, R, k) and pin flags under one scenario's marginals."""
    N, R, k = U.shape
    m = np.empty((N, R, k))
    pins = np.empty((N, R, k), dtype=bool)
    for i, tk in enumerate(tickers):
        Ri = tk.marginal(scenario).ppf(U[:, :, i])
        m[:, :, i] = np.clip(Ri - tk.k1, 0.0, tk.k2 - tk.k1) / tk.cost_rel
        pins[:, :, i] = (Ri >= tk.k1) & (Ri < tk.k2)
    return m, pins


def run_program(M, cfg, rates, budgets, years, pins=None, ex_mult=None, weights=None):
    """Sleeve dynamics for one scenario. M: (N, R) portfolio roll multiple. Returns dict of arrays."""
    N, R = M.shape
    if R != cfg.n_rolls:
        raise ValueError(f"expected {cfg.n_rolls} rolls, got {R}")
    G = cfg.goal
    ns = min(cfg.n_store, N)
    W = None if weights is None else np.atleast_2d(weights)
    banked = np.zeros(N)
    taxes_paid = np.zeros(N)
    ytd = np.zeros(N)
    ruined = np.zeros((N, cfg.n_cycles), dtype=bool)
    absorbed_at = np.full((N, cfg.n_cycles), -1)
    pin_rolls = np.zeros(N, dtype=int)
    max_ex = np.zeros(N)
    keep = {k: np.zeros((ns, R)) for k in ("deployed", "proceeds", "sleeve", "cumulative", "absorbed")}
    tax_rate = cfg.tax_rate if cfg.tax else 0.0

    def tax_on(gain):
        return tax_rate * np.maximum(gain - cfg.tax_deduction, 0.0)

    for c in range(cfg.n_cycles):
        Bc, rc = budgets[c], rates[c]
        f = cfg.f_of(c)
        wc = None if W is None else W[c % len(W)]
        grow = (1 + rc) ** cfg.T_opt
        sleeve = np.full(N, Bc)
        absorbed = np.zeros(N, dtype=bool)
        for jj in range(cfg.rolls_per_cycle):
            j = c * cfg.rolls_per_cycle + jj
            cum_net = banked - taxes_paid + sleeve - tax_on(ytd)
            if cfg.absorbing:
                newly = (~absorbed) & (cum_net >= G)
                absorbed_at[newly, c] = jj
                absorbed |= newly
            f_eff = np.where(absorbed, 0.0, f)
            deploy = sleeve * f_eff
            # Anti-martingale: capital per roll is exactly sleeve * f, never scaled up after a loss.
            assert np.all(deploy <= sleeve * f * (1 + 1e-12) + 1e-9), "deployed capital exceeds sleeve * f"
            # Absorbing state: nothing deployed once the goal is reached in this cycle.
            assert not np.any(deploy[absorbed] > 0), "capital deployed after the goal was reached"
            proceeds = deploy * M[:, j]
            ytd += proceeds - deploy
            sleeve = (sleeve - deploy) * grow + proceeds
            if pins is not None:
                hit = pins[:, j, :] & (deploy[:, None] > 0)
                pin_rolls += hit.any(axis=1)
                if ex_mult is not None:
                    ex = (deploy[:, None] * wc[None, :] * ex_mult[None, :] * hit).max(axis=1)
                    max_ex = np.maximum(max_ex, ex)
            if ns:
                keep["deployed"][:, j] = deploy[:ns]
                keep["proceeds"][:, j] = proceeds[:ns]
                keep["sleeve"][:, j] = sleeve[:ns]
                keep["cumulative"][:, j] = (banked - taxes_paid + sleeve - tax_on(ytd))[:ns]
                keep["absorbed"][:, j] = absorbed[:ns]
            last_roll_of_year = (j == R - 1) or (years[j + 1] != years[j])
            if last_roll_of_year:
                taxes_paid += tax_on(ytd)
                ytd[:] = 0.0
        ruined[:, c] = (sleeve < cfg.ruin_frac * Bc) & (absorbed_at[:, c] < 0)
        banked += sleeve
    return dict(terminal_gross=banked, taxes=taxes_paid, terminal_net=banked - taxes_paid, ruined=ruined,
                absorbed_at=absorbed_at, pin_rolls=pin_rolls, max_exercise_cash=max_ex, paths=keep)


@dataclass
class SimResult:
    runs: List[ScenarioRun]
    cfg: SimConfig
    tickers: List[TickerSpec]
    weights: np.ndarray
    calendar: list

    def pooled(self, attr):
        vals = np.concatenate([getattr(r, attr) for r in self.runs])
        wts = np.concatenate([np.full(len(getattr(r, attr)), r.prob / len(getattr(r, attr))) for r in self.runs])
        return vals, wts

    def metrics(self, net=True):
        return summarize(self, net=net)


def weighted_quantile(x, w, qs):
    o = np.argsort(x)
    x, w = np.asarray(x)[o], np.asarray(w)[o]
    cw = np.cumsum(w) - 0.5 * w
    cw /= w.sum()
    return np.interp(qs, cw, x)


def summarize(res, net=True, ladder=LADDER, quantiles=QUANTILES, mortgage_rate=0.04, remaining_months=264):
    cfg = res.cfg
    x, w = res.pooled("terminal_net" if net else "terminal_gross")
    W = w.sum()
    mean = float(np.sum(w * x) / W)
    qv = weighted_quantile(x, w, quantiles)
    median = float(weighted_quantile(x, w, [0.5])[0])
    ruined, wr = res.pooled("ruined")
    pin_rolls, _ = res.pooled("pin_rolls")
    bond_alt = sum(r.prob * float(np.sum(r.budgets * (1 + r.rates) ** cfg.T_cycle)) for r in res.runs)
    budget_total = sum(r.prob * float(np.sum(r.budgets)) for r in res.runs)
    out = dict(
        goal=cfg.goal,
        p_goal=float(np.sum(w * (x >= cfg.goal)) / W),
        mean=mean,
        median=median,
        mean_over_median=mean / median if median > 0 else float("inf"),
        mean_driven_by_tail=mean > 2 * max(median, 0.0) and mean > 0,   # median 0 counts: mean is all tail
        quantiles={q: float(v) for q, v in zip(quantiles, qv)},
        ruin_per_cycle=float(np.sum(wr[:, None] * ruined) / (W * cfg.n_cycles)),
        ruin_any_cycle=float(np.sum(wr * ruined.any(axis=1)) / W),
        ruin_all_cycles=float(np.sum(wr * ruined.all(axis=1)) / W),
        pin_any=float(np.sum(w * (pin_rolls > 0)) / W),
        pin_rolls_mean=float(np.sum(w * pin_rolls) / W),
        budget_total=budget_total,
        bond_only_alternative=bond_alt,
        ladder=[dict(level=L, prob=float(np.sum(w * (x >= L)) / W),
                     monthly_reduction=bud.monthly_payment_reduction(L, mortgage_rate, remaining_months))
                for L in ladder],
        roll_multiple_mean=sum(r.prob * r.roll_multiple_mean for r in res.runs),
    )
    out["excess_mean_vs_bond"] = mean - bond_alt
    # Tax trap: gains realized at a roll are taxed in their calendar year even if the re-staked sleeve is
    # lost the next year (no loss carry-back/forward), so tax can exceed everything the options returned
    # and has to come out of the protected principal.
    out["p_tax_exceeds_proceeds"] = float(np.sum(w * (x < 0)) / W) if net else 0.0
    out["worst_net"] = float(x.min())
    return out


def simulate(tickers, weights, corr, cfg=None, U=None):
    """Run every scenario with prob > 0 on the same uniforms and keep them as weighted runs."""
    cfg = cfg or SimConfig()
    weights = np.asarray(weights, dtype=float)
    W = np.atleast_2d(weights)
    if W.shape[1] != len(tickers) or W.shape[0] not in (1, cfg.n_cycles) or np.any(W < 0) \
            or np.any(np.abs(W.sum(axis=1) - 1) > 1e-9):
        raise ValueError("weights must be >= 0, one per ticker (optionally one row per cycle), each row summing to 1")
    probs = bud.check_scenarios(cfg.scenarios)
    if U is None:
        U = draw_uniforms(cfg.n_paths, cfg.n_rolls, corr, cfg.copula, cfg.df, cfg.seed)
    cal = roll_calendar(cfg.start, cfg.n_rolls, close_bdays=cfg.close_bdays)
    years = np.array([row["close"].year for row in cal])
    ex_mult = np.array([tk.exercise_to_premium for tk in tickers])
    runs = []
    for sc, pi in zip(cfg.scenarios, probs):
        if pi <= 0:
            continue
        rates, budgets = bud.cycle_budgets(cfg.r0, sc.shift_per_cycle, cfg.n_cycles, cfg.V_deposit,
                                           cfg.V_own_rental, cfg.T_cycle)
        m, pins = roll_multiples(tickers, U, sc.name)
        if W.shape[0] == 1:
            M = m @ W[0]
        else:                       # a different combination per cycle
            rpc = cfg.rolls_per_cycle
            M = np.concatenate([m[:, c * rpc:(c + 1) * rpc, :] @ W[c] for c in range(cfg.n_cycles)], axis=1)
        out = run_program(M, cfg, rates, budgets, years, pins=pins, ex_mult=ex_mult, weights=W)
        pin_by = pins.reshape(-1, len(tickers)).mean(axis=0)
        runs.append(ScenarioRun(name=sc.name, prob=float(pi), rates=rates, budgets=budgets,
                                terminal_gross=out["terminal_gross"], taxes=out["taxes"],
                                terminal_net=out["terminal_net"], ruined=out["ruined"],
                                absorbed_at=out["absorbed_at"], pin_rolls=out["pin_rolls"],
                                pin_by_ticker=pin_by, max_exercise_cash=out["max_exercise_cash"],
                                roll_multiple_mean=float(M.mean()),
                                joint_total_loss=float((m == 0).all(axis=2).mean()), paths=out["paths"]))
    return SimResult(runs=runs, cfg=cfg, tickers=list(tickers), weights=weights, calendar=cal)


def dubins_savage_regime(roll_multiple_mean, rf_growth):
    """Superfair (E[M] > risk-free growth per roll): many small bets beat one big one (timid play).
    Subfair or fair: bold play (fewer, larger bets) is optimal."""
    return "timid" if roll_multiple_mean > rf_growth else "bold"

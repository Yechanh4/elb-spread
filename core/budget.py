"""Budget equation, rate scenarios and the rate -> budget coupling.

    B = (V_deposit - V_own_rental) * (1 - (1+r)^(-T))

The bond is bought for (V_deposit - V_own_rental)(1+r)^(-T) and matures to exactly
(V_deposit - V_own_rental), which together with the own-rental deposit V_own_rental returns the full
V_deposit at the 2-year boundary. Whatever the bond does not need is the option budget B.
r is re-read at every cycle boundary, so a falling rate path shrinks every later budget.
"""
from dataclasses import dataclass

import numpy as np

V_DEPOSIT = 1_500_000_000
V_OWN_RENTAL = 380_000_000
R_KRW_2Y = 0.0376
T_CYCLE = 2.0
N_CYCLES = 4
ROLLS_PER_CYCLE = 8
T_OPT = 0.25

GOAL = 756_000_000
GOAL_PRESETS = {
    "full_mortgage": 756_000_000,     # remaining mortgage after 8 years of scheduled payments
    "two_thirds": 500_000_000,
    "forty_pct": 300_000_000,
    "bond_opportunity": 70_000_000,   # bond opportunity cost only
}

RATE_FLOOR = 0.0025


def budget(V_deposit=V_DEPOSIT, V_own_rental=V_OWN_RENTAL, r=R_KRW_2Y, T=T_CYCLE):
    """Option budget per cycle. Vectorized over r."""
    if V_own_rental > V_deposit:
        raise ValueError("own rental deposit exceeds the deposit received")
    r = np.asarray(r, dtype=float)
    if np.any(r <= -1):
        raise ValueError("rate must be > -100%")
    out = (V_deposit - V_own_rental) * (1 - (1 + r) ** (-T))
    return float(out) if out.ndim == 0 else out


def bond_cost(V_deposit=V_DEPOSIT, V_own_rental=V_OWN_RENTAL, r=R_KRW_2Y, T=T_CYCLE):
    """Price today of the zero that matures to V_deposit - V_own_rental."""
    return (V_deposit - V_own_rental) * (1 + r) ** (-T)


@dataclass
class RateScenario:
    name: str
    shift_per_cycle: float      # change in the KRW 2y yield applied at each cycle boundary
    prob: float


DEFAULT_THRESHOLDS_BP = (-50.0, 50.0)   # < -50bp down, -50..+50 neutral, > +50bp up
DEFAULT_SCENARIOS = (
    RateScenario("up", +0.0075, 0.25),
    RateScenario("neutral", 0.0, 0.50),
    RateScenario("down", -0.0075, 0.25),
)


def classify_rate_change(dr_bp, thresholds_bp=DEFAULT_THRESHOLDS_BP):
    lo, hi = thresholds_bp
    if dr_bp > hi:
        return "up"
    if dr_bp < lo:
        return "down"
    return "neutral"


def check_scenarios(scenarios):
    probs = np.array([s.prob for s in scenarios], dtype=float)
    if np.any(probs < 0) or abs(probs.sum() - 1) > 1e-9:
        raise ValueError("scenario probabilities must be >= 0 and sum to 1")
    return probs


def rate_path(r0, shift_per_cycle, n_cycles=N_CYCLES, floor=RATE_FLOOR):
    """r_c = max(r0 + c * shift, floor) for c = 0..n_cycles-1. Cycle 0 uses today's rate."""
    c = np.arange(n_cycles)
    return np.maximum(r0 + c * shift_per_cycle, floor)


def cycle_budgets(r0=R_KRW_2Y, shift_per_cycle=0.0, n_cycles=N_CYCLES, V_deposit=V_DEPOSIT,
                  V_own_rental=V_OWN_RENTAL, T=T_CYCLE):
    rates = rate_path(r0, shift_per_cycle, n_cycles)
    return rates, budget(V_deposit, V_own_rental, rates, T)


# ---------------------------------------------------------------- mortgage translation

def annuity_payment(principal, annual_rate, months):
    i = annual_rate / 12
    if months <= 0:
        raise ValueError("months must be positive")
    if abs(i) < 1e-12:
        return principal / months
    return principal * i / (1 - (1 + i) ** (-months))


def mortgage_balance(principal, annual_rate, term_months, paid_months):
    """Outstanding balance after paid_months scheduled payments."""
    i = annual_rate / 12
    pmt = annuity_payment(principal, annual_rate, term_months)
    if abs(i) < 1e-12:
        return principal - pmt * paid_months
    g = (1 + i) ** paid_months
    return principal * g - pmt * (g - 1) / i


def monthly_payment_reduction(prepay, annual_rate=0.04, remaining_months=264):
    """Monthly payment saved by prepaying `prepay` and re-amortizing over the same remaining term."""
    return annuity_payment(prepay, annual_rate, remaining_months)

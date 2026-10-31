"""Operating rules R-1..R-6 as a rules engine, plus pin / dividend / liquidity / margin warnings.

Source: 결제·행사 관리 운용 규칙 (chapter 8 addendum). Close-out timing:
    5 business days before expiry  early (time value left on the table)
    2 business days before          RECOMMENDED close-by date
    1 business day before           allowed (spreads wider)
    expiry day                      prohibited (pin risk)
"""
from dataclasses import dataclass, field
from datetime import date
from typing import List, Optional

import numpy as np

PASS, WARN, FAIL, NA = "pass", "warn", "fail", "n/a"
CLOSE_BDAYS = 2
NAKED_MARGIN_RATE = 0.20      # short call treated as naked if the broker does not see the spread
NEAR_EXPIRY_BDAYS = 5         # R-6 watch window

STRUCTURAL_NOTES = {
    "short_only": "A short-only obligation cannot occur: K2 > K1, so S > K2 implies S > K1 — whenever the "
                  "short leg is in the money, the long leg is too.",
    "early_assignment": "Early assignment of the short leg is cash-positive (receive K2, exercise the long at K1, "
                        "net +(K2-K1)), but it forfeits the long leg's remaining time value and, if notified after "
                        "the close, leaves an overnight short-stock position that may draw margin or a buy-in.",
    "account_settings": "Spread recognition fixes margin and same-day netting. It cannot stop automatic exercise: "
                        "the clearing house exercises any option $0.01 in the money at expiry.",
    "european": "European options remove early assignment but not pin risk, which depends only on where spot "
                "finishes. Single-stock options are American; only index options are European.",
}


@dataclass
class Position:
    ticker: str
    K1: float
    K2: float
    expiry: date
    n_contracts: float
    spot: float
    fx: float = 1.0
    multiplier: int = 100
    entered_as_combo: Optional[bool] = None
    broker_recognizes_spread: Optional[bool] = None
    ex_div_dates: List[date] = field(default_factory=list)
    dne_filed: bool = False
    is_open: bool = True


@dataclass
class RuleStatus:
    code: str
    status: str
    message: str


def business_days_left(today, expiry):
    """Business days strictly after today up to and including expiry (0 on expiry day)."""
    return int(np.busday_count(np.datetime64(today, 'D') + np.timedelta64(1, 'D'), np.datetime64(expiry, 'D') + np.timedelta64(1, 'D'))) if expiry > today else 0


def close_by_date(expiry, bdays=CLOSE_BDAYS):
    return np.busday_offset(np.datetime64(expiry), -bdays, roll="backward").astype(date)


def exercise_cash(n_contracts, K1, multiplier=100, fx=1.0):
    """Cash the long-leg exercise demands if spot finishes in [K1, K2)."""
    return n_contracts * K1 * multiplier * fx


def naked_margin_estimate(n_contracts, spot, multiplier=100, fx=1.0, rate=NAKED_MARGIN_RATE):
    return n_contracts * spot * multiplier * fx * rate


def early_assignment_cashflow(n_contracts, K1, K2, multiplier=100):
    receive = n_contracts * K2 * multiplier
    pay = n_contracts * K1 * multiplier
    return dict(receive_on_assignment=receive, pay_on_exercise=-pay, net=receive - pay)


def short_only_obligation_possible(K1, K2, spot):
    """Short leg ITM without the long leg ITM. Structurally impossible for K1 < K2."""
    if not K1 < K2:
        raise ValueError("a call spread needs K1 < K2")
    return spot > K2 and not spot > K1


def liquidity_check(oi_k1, oi_k2, spread_pct_k1, spread_pct_k2, min_oi=100, max_spread_pct=0.10):
    checks = dict(oi_k1=oi_k1 >= min_oi, oi_k2=oi_k2 >= min_oi,
                  spread_k1=spread_pct_k1 <= max_spread_pct, spread_k2=spread_pct_k2 <= max_spread_pct)
    return all(checks.values()), checks


def evaluate(pos, today):
    """Status of every rule for one position on `today`."""
    bd = business_days_left(today, pos.expiry)
    cb = close_by_date(pos.expiry)
    in_pin = pos.K1 <= pos.spot < pos.K2
    out = []

    # R-1 close 1-2 business days before expiry
    if not pos.is_open:
        out.append(RuleStatus("R-1", PASS, "closed"))
    elif bd > CLOSE_BDAYS:
        out.append(RuleStatus("R-1", PASS, f"close by {cb.isoformat()} ({bd} business days to expiry)"))
    elif bd == CLOSE_BDAYS:
        out.append(RuleStatus("R-1", WARN, "close today: 2 business days before expiry (recommended)"))
    elif bd == 1:
        out.append(RuleStatus("R-1", WARN, "last allowed day: 1 business day before expiry, spreads widen"))
    else:
        out.append(RuleStatus("R-1", FAIL, "expiry day with the position open: prohibited, close immediately"))

    # R-2 single combo order
    out.append({True: RuleStatus("R-2", PASS, "entered as one spread (combo) order"),
                False: RuleStatus("R-2", FAIL, "legs entered separately: legging risk and no netting guarantee"),
                None: RuleStatus("R-2", WARN, "order type not recorded")}[pos.entered_as_combo])

    # R-3 broker recognizes vertical spreads
    if pos.broker_recognizes_spread is True:
        out.append(RuleStatus("R-3", PASS, "broker confirmed vertical-spread margin treatment in writing"))
    else:
        m = naked_margin_estimate(pos.n_contracts, pos.spot, pos.multiplier, pos.fx)
        st = FAIL if pos.broker_recognizes_spread is False else WARN
        out.append(RuleStatus("R-3", st, f"unconfirmed spread treatment: naked short margin ~{m:,.0f} "
                                         f"(20% of underlying) would be required"))

    # R-4 no ITM short leg through an ex-dividend date
    upcoming = [d for d in pos.ex_div_dates if today <= d <= pos.expiry]
    if not pos.is_open or not upcoming:
        out.append(RuleStatus("R-4", PASS, "no ex-dividend date before expiry"))
    elif pos.spot > pos.K2:
        out.append(RuleStatus("R-4", FAIL, f"short leg ITM before ex-div {upcoming[0].isoformat()}: "
                                           f"early assignment likely, close before it"))
    else:
        out.append(RuleStatus("R-4", WARN, f"ex-div {upcoming[0].isoformat()} before expiry: "
                                           f"close if spot moves above K2={pos.K2:g}"))

    # R-5 do-not-exercise if forced to hold to expiry
    if pos.is_open and bd == 0:
        out.append(RuleStatus("R-5", PASS if pos.dne_filed else FAIL,
                              "do-not-exercise filed" if pos.dne_filed else
                              "held to expiry without a do-not-exercise instruction"))
    else:
        out.append(RuleStatus("R-5", NA, "not held to expiry"))

    # R-6 spot between strikes near expiry: close unconditionally
    if pos.is_open and bd <= NEAR_EXPIRY_BDAYS and in_pin:
        out.append(RuleStatus("R-6", FAIL, f"spot {pos.spot:g} in pin zone [{pos.K1:g}, {pos.K2:g}): close now"))
    elif pos.is_open and bd <= NEAR_EXPIRY_BDAYS:
        out.append(RuleStatus("R-6", PASS, "spot outside the pin zone; check daily this week"))
    else:
        out.append(RuleStatus("R-6", NA, "not yet in expiry week"))
    return out


def pin_banner(pos, today):
    """Red banner for an open position within the close-by window, with the cash a pin would demand."""
    bd = business_days_left(today, pos.expiry)
    if not pos.is_open or bd > CLOSE_BDAYS:
        return None
    cash = exercise_cash(pos.n_contracts, pos.K1, pos.multiplier, pos.fx)
    return dict(level="red", business_days_left=bd, exercise_cash=cash,
                in_pin_zone=pos.K1 <= pos.spot < pos.K2,
                message=f"{pos.ticker}: {bd} business day(s) to expiry. A finish in [{pos.K1:g}, {pos.K2:g}) "
                        f"would demand {cash:,.0f} in exercise cash. Close now.")


def action_list(pos, entry_date):
    """Per-roll action list: order type, combo instruction, close-by date."""
    cb = close_by_date(pos.expiry)
    return [
        f"{entry_date.isoformat()}: BUY {pos.n_contracts:,.0f} {pos.ticker} {pos.expiry.isoformat()} "
        f"{pos.K1:g}/{pos.K2:g} call vertical as ONE combo order (limit at or below mid + allowance)",
        "confirm both legs filled together; record the fill",
        f"check ex-dividend dates before {pos.expiry.isoformat()}: {', '.join(d.isoformat() for d in pos.ex_div_dates) or 'none'}",
        f"{cb.isoformat()}: SELL the vertical as ONE combo order (2 business days before expiry)",
        "expiry week: check spot vs K1/K2 daily; if between strikes, close regardless of price",
    ]

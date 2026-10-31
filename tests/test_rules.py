from datetime import date

import numpy as np
import pytest

from core import rules
from core.rules import FAIL, NA, PASS, WARN, Position

EXP = date(2027, 3, 19)          # Friday


def pos(**kw):
    base = dict(ticker="AAA", K1=125.0, K2=127.5, expiry=EXP, n_contracts=2262, spot=110.0, fx=1450.0,
                entered_as_combo=True, broker_recognizes_spread=True)
    base.update(kw)
    return Position(**base)


def status(p, today):
    return {s.code: s.status for s in rules.evaluate(p, today)}


def test_close_by_date_is_two_business_days_before():
    assert rules.close_by_date(EXP) == date(2027, 3, 17)
    assert rules.close_by_date(date(2027, 3, 22)) == date(2027, 3, 18)      # Monday expiry -> Thursday


def test_r1_timing_ladder():
    assert status(pos(), date(2027, 3, 1))["R-1"] == PASS
    assert status(pos(), date(2027, 3, 17))["R-1"] == WARN        # 2 bdays: close today (recommended)
    assert status(pos(), date(2027, 3, 18))["R-1"] == WARN        # 1 bday: last allowed
    assert status(pos(), EXP)["R-1"] == FAIL                      # expiry day: prohibited
    assert status(pos(is_open=False), EXP)["R-1"] == PASS
    assert rules.business_days_left(date(2027, 3, 18), EXP) == 1
    assert rules.business_days_left(EXP, EXP) == 0


def test_r2_r3_order_and_margin():
    st = status(pos(entered_as_combo=False, broker_recognizes_spread=None), date(2027, 3, 1))
    assert st["R-2"] == FAIL and st["R-3"] == WARN
    msgs = {s.code: s.message for s in rules.evaluate(pos(broker_recognizes_spread=False), date(2027, 3, 1))}
    assert "naked" in msgs["R-3"]
    assert rules.naked_margin_estimate(2262, 100.0, 100, 1450.0) == pytest.approx(2262 * 100 * 100 * 1450 * 0.2)


def test_r4_ex_dividend_with_itm_short():
    exd = [date(2027, 3, 5)]
    assert status(pos(spot=130.0, ex_div_dates=exd), date(2027, 3, 1))["R-4"] == FAIL
    assert status(pos(spot=110.0, ex_div_dates=exd), date(2027, 3, 1))["R-4"] == WARN
    assert status(pos(spot=130.0, ex_div_dates=[date(2027, 2, 1)]), date(2027, 3, 1))["R-4"] == PASS


def test_r5_do_not_exercise():
    assert status(pos(), EXP)["R-5"] == FAIL
    assert status(pos(dne_filed=True), EXP)["R-5"] == PASS
    assert status(pos(), date(2027, 3, 1))["R-5"] == NA


def test_r6_pin_zone_near_expiry():
    assert status(pos(spot=126.0), date(2027, 3, 16))["R-6"] == FAIL
    assert status(pos(spot=120.0), date(2027, 3, 16))["R-6"] == PASS
    assert status(pos(spot=126.0), date(2027, 2, 1))["R-6"] == NA


def test_pin_banner_and_exercise_cash():
    p = pos(spot=126.0)
    assert rules.pin_banner(p, date(2027, 3, 1)) is None
    b = rules.pin_banner(p, date(2027, 3, 17))
    assert b["level"] == "red" and b["in_pin_zone"]
    assert b["exercise_cash"] == pytest.approx(2262 * 125.0 * 100 * 1450.0)
    # the report's example: ~2,262 contracts, ~$28.3M exercise cash
    assert rules.exercise_cash(2262, 125.0) == pytest.approx(28.275e6)


def test_short_only_obligation_impossible():
    for s in np.linspace(50, 200, 301):
        assert not rules.short_only_obligation_possible(120.0, 125.0, s)
    with pytest.raises(ValueError):
        rules.short_only_obligation_possible(125.0, 120.0, 130.0)


def test_early_assignment_is_cash_positive():
    cf = rules.early_assignment_cashflow(10, 120.0, 125.0)
    assert cf["net"] == pytest.approx(10 * 5 * 100) and cf["net"] > 0


def test_liquidity_check():
    ok, chk = rules.liquidity_check(150, 120, 0.05, 0.08)
    assert ok
    ok, chk = rules.liquidity_check(150, 80, 0.05, 0.12)
    assert not ok and not chk["oi_k2"] and not chk["spread_k2"]


def test_action_list_mentions_combo_and_close_date():
    acts = rules.action_list(pos(), date(2026, 12, 18))
    assert any("ONE combo" in a for a in acts)
    assert any("2027-03-17" in a for a in acts)

import numpy as np
import pytest

from core import budget as bud


def test_default_budget_matches_spec():
    B = bud.budget()
    assert B == pytest.approx(79.7e6, rel=2e-3)


@pytest.mark.parametrize("r", [0.005, 0.01, 0.02, 0.0376, 0.05, 0.08])
def test_budget_closed_form_across_rate_grid(r):
    V, Vo, T = 1.5e9, 3.8e8, 2.0
    expected = (V - Vo) - (V - Vo) / (1 + r) ** T
    assert bud.budget(V, Vo, r, T) == pytest.approx(expected, rel=1e-12)
    # bond + budget = deployable amount, and the bond matures exactly to it
    assert bud.bond_cost(V, Vo, r, T) + bud.budget(V, Vo, r, T) == pytest.approx(V - Vo, rel=1e-12)
    assert bud.bond_cost(V, Vo, r, T) * (1 + r) ** T == pytest.approx(V - Vo, rel=1e-12)


def test_falling_rate_shrinks_budget():
    rates = np.linspace(0.06, 0.003, 40)
    B = bud.budget(r=rates)
    assert np.all(np.diff(B) < 0)


def test_rate_path_couples_into_every_later_cycle():
    rates_dn, B_dn = bud.cycle_budgets(0.0376, -0.0075)
    rates_up, B_up = bud.cycle_budgets(0.0376, +0.0075)
    assert B_dn[0] == pytest.approx(B_up[0])            # cycle 0 uses today's rate
    assert np.all(np.diff(B_dn) < 0) and np.all(np.diff(B_up) > 0)
    assert rates_dn.min() >= bud.RATE_FLOOR


def test_rate_classification_thresholds():
    assert bud.classify_rate_change(60) == "up"
    assert bud.classify_rate_change(-60) == "down"
    assert bud.classify_rate_change(50) == "neutral"
    assert bud.classify_rate_change(-50) == "neutral"


def test_scenario_probs_validated():
    with pytest.raises(ValueError):
        bud.check_scenarios([bud.RateScenario("a", 0, 0.7), bud.RateScenario("b", 0, 0.7)])


def test_mortgage_helpers():
    bal = bud.mortgage_balance(920e6, 0.04, 360, 96)
    assert 7.0e8 < bal < 8.0e8                          # ~750M left after 8 years, as in the report
    pmt = bud.annuity_payment(300e6, 0.04, 264)
    assert bud.monthly_payment_reduction(300e6, 0.04, 264) == pytest.approx(pmt)
    assert bud.mortgage_balance(1e8, 0.05, 120, 120) == pytest.approx(0, abs=1e-3)

from datetime import date

import numpy as np
import pytest

from core import vol


def test_close_to_close_hv_recovers_sigma():
    rng = np.random.default_rng(1)
    sigma = 0.30
    r = rng.normal(0, sigma / np.sqrt(252), 252 * 20)
    closes = 100 * np.exp(np.cumsum(np.r_[0, r]))
    assert vol.hv_close_to_close(closes) == pytest.approx(sigma, rel=0.02)
    assert vol.hv_close_to_close(closes, window=252) == pytest.approx(sigma, rel=0.15)


def test_parkinson_recovers_sigma_from_intraday_paths():
    rng = np.random.default_rng(2)
    sigma, steps, days = 0.40, 390, 1500
    inc = rng.normal(0, sigma / np.sqrt(252 * steps), (days, steps))
    path = np.cumsum(inc, axis=1)
    hi = np.exp(np.maximum(path.max(axis=1), 0))
    lo = np.exp(np.minimum(path.min(axis=1), 0))
    # the discrete path misses part of the true range, so Parkinson reads slightly low
    assert vol.hv_parkinson(hi, lo) == pytest.approx(sigma, rel=0.05)


def test_catalyst_formula():
    s = vol.catalyst_sigma(0.30, [0.08, 0.05], T_opt=0.25)
    assert s == pytest.approx(np.sqrt((0.09 * 0.25 + 0.0064 + 0.0025) / 0.25))
    assert vol.catalyst_sigma(0.30, [], 0.25) == pytest.approx(0.30)
    calc = vol.CatalystCalc(0.30, 0.25, [vol.Catalyst("earnings", date(2027, 1, 28), 0.08)])
    assert calc.sigma_hat() > 0.30


def test_catalyst_timing_rule():
    exp = date(2027, 3, 19)
    ok, days = vol.catalyst_timing(date(2027, 1, 28), exp)
    assert ok and days == 50
    ok, days = vol.catalyst_timing(date(2027, 3, 1), exp)
    assert not ok and days == 18


def test_priced_in_badges():
    T_ex, T_in, iv_ex = 0.08, 0.25, 0.30
    moves = [0.10]
    # market prices exactly our event -> ratio 1 -> priced in
    iv_in_full = np.sqrt((iv_ex**2 * T_in + 0.01) / T_in)
    chk = vol.priced_in_check(moves, iv_ex, T_ex, iv_in_full, T_in)
    assert chk["badge"] == "priced in"
    assert chk["implied_move"] == pytest.approx(0.10, rel=1e-9)
    # market barely steps up -> underpriced (good)
    chk = vol.priced_in_check(moves, iv_ex, T_ex, 0.305, T_in)
    assert chk["badge"] == "underpriced" and chk["ratio"] < 0.5
    with pytest.raises(ValueError):
        vol.implied_event_move(0.3, 0.25, 0.3, 0.1)


def test_surface_fit_flat_and_calendar_check():
    slices = []
    for T in (0.1, 0.25, 0.5):
        F = 100.0
        K = np.linspace(70, 140, 25)
        slices.append((T, F, K, np.full_like(K, 0.32)))
    fits = vol.fit_surface(slices)
    T, iv = vol.atm_term_structure(fits)
    assert np.allclose(iv, 0.32, atol=2e-3)
    assert vol.calendar_arbitrage_free(fits)
    assert fits[1].iv([100.0])[0] == pytest.approx(0.32, abs=2e-3)


def test_rolling_hv_and_iv_rank_proxy():
    rng = np.random.default_rng(3)
    r = np.r_[rng.normal(0, 0.2 / np.sqrt(252), 300), rng.normal(0, 0.6 / np.sqrt(252), 300)]
    closes = 100 * np.exp(np.cumsum(np.r_[0, r]))
    h = vol.rolling_hv(closes, 21)
    assert len(h) == len(r) - 20
    assert h[:250].mean() == pytest.approx(0.2, rel=0.15) and h[-200:].mean() == pytest.approx(0.6, rel=0.15)
    # brute force for one window
    w = np.diff(np.log(closes))[-21:]
    assert h[-1] == pytest.approx(np.std(w, ddof=1) * np.sqrt(252), rel=1e-9)
    assert vol.iv_rank_proxy(10.0, closes) == 1.0 and vol.iv_rank_proxy(0.0, closes) == 0.0
    assert 0 < vol.iv_rank_proxy(0.4, closes) < 1
    rows = [dict(t="A", iv_minus_hv=0.05, iv_rank_proxy=0.2), dict(t="B", iv_minus_hv=-0.1, iv_rank_proxy=0.9),
            dict(t="C", iv_minus_hv=0.05, iv_rank_proxy=0.1)]
    assert [x["t"] for x in vol.screen(rows)] == ["B", "C", "A"]

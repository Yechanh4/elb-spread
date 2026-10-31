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


# ---------------------------------------------------------------- sigma_hat suggestion

def garch_path(n, a, b, sigma, seed):
    rng = np.random.default_rng(seed)
    v = sigma**2 / 252
    h, r = v, np.empty(n)
    for i in range(n):
        r[i] = rng.normal() * np.sqrt(h)
        h = v * (1 - a - b) + a * r[i] ** 2 + b * h
    return r


def test_garch_recovers_parameters_and_mean_reverts():
    r = garch_path(4000, 0.08, 0.90, 0.30, seed=3)
    g = vol.garch11(r)
    assert g["persistence"] == pytest.approx(0.98, abs=0.02)
    assert g["long_run_vol"] == pytest.approx(0.30, rel=0.1)
    hi = dict(g, h_next=4 * g["var_target"])                       # a shock: today's vol is twice the long run
    s1, s63, s5000 = (vol.garch_term_vol(hi, n) for n in (1, 63, 5000))
    assert s1 > s63 > s5000 > g["long_run_vol"] and s5000 == pytest.approx(g["long_run_vol"], rel=0.05)


def test_har_and_ewma_on_constant_vol():
    rng = np.random.default_rng(4)
    r = rng.normal(0, 0.25 / np.sqrt(252), 2000)
    assert vol.har_vol(r, 63) == pytest.approx(0.25, rel=0.1)
    assert vol.ewma_vol(r) == pytest.approx(0.25, rel=0.25)        # EWMA is noisy: ~30-day memory
    with pytest.raises(ValueError):
        vol.garch11(r[:50])


def test_earnings_reaction_follows_report_timing():
    dates = ["2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08", "2026-01-09", "2026-01-12"]
    closes = [100, 100, 110, 110, 99, 99]                            # +10% into the 7th, -10% into the 9th
    rx = vol.earnings_reactions(dates, closes, [dict(date="2026-01-07", timing="bmo"),
                                                dict(date="2026-01-08", timing="amc"),
                                                dict(date="2026-01-10", timing=None)])
    assert [round(x["move"], 4) for x in rx] == [round(np.log(1.1), 4), round(np.log(1.1 / 0.99), 4), 0.0]
    assert [x["idx"] for x in rx] == [1, 3, 4]                       # a weekend report reacts on Monday


def test_sigma_suggestion_strips_and_readds_earnings():
    rng = np.random.default_rng(5)
    n, base = 756, 0.25
    r = rng.normal(0, base / np.sqrt(252), n)
    dates = [date.fromordinal(date(2024, 1, 1).toordinal() + i) for i in range(n + 1)]
    past = []
    for i in range(40, n, 63):                                       # quarterly +-9% jumps, bmo
        r[i] = 0.09 * (1 if i % 2 else -1)
        past.append(dict(date=dates[i + 1].isoformat(), timing="bmo"))
    closes = 100 * np.exp(np.cumsum(np.r_[0, r]))
    asof = dates[-1]
    expiry = date.fromordinal(asof.toordinal() + 91)
    g = vol.sigma_suggestion([d.isoformat() for d in dates], closes, past, [], asof, expiry, iv=0.30)
    assert g["earnings"]["rms_move"] == pytest.approx(0.09, rel=1e-6)
    assert g["base"] == pytest.approx(base, rel=0.12)                # the jumps are not in the base
    assert g["hv1y"] > g["base"]                                     # ...but they inflate plain HV
    k = len(g["earnings"]["in_window"])
    assert k >= 1 and all(x["source"].startswith("projected") for x in g["earnings"]["in_window"])
    T = g["T"]
    expect = np.sqrt(g["base"] ** 2 + k * (0.09**2 - g["base"] ** 2 / 252) / T)
    assert g["suggested"] == pytest.approx(expect)
    assert g["edge_vs_iv"] == pytest.approx(g["suggested"] - 0.30)

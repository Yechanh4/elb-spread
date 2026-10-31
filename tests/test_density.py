import numpy as np
import pytest

from core.bsm import bs_call, lognormal_pdf
from core.density import extract_q, kl_level, shrink_sigma, subjective_density

S0, T, r, d, sig = 100.0, 0.25, 0.04, 0.01, 0.35


@pytest.fixture(scope="module")
def md_flat():
    K = np.linspace(55, 190, 55)
    return extract_q(K, S0, r, d, T, call_prices=bs_call(S0, K, T, r, d, sig))


def test_breeden_litzenberger_recovers_lognormal(md_flat):
    md = md_flat
    ref = lognormal_pdf(md.S, S0, T, r, d, sig)
    # density error relative to its peak, and in L1 (total variation)
    assert np.max(np.abs(md.q - ref)) / ref.max() < 0.01
    assert np.sum(md.weights * np.abs(md.q - ref)) < 0.01


def test_mean_check_equals_forward(md_flat):
    md = md_flat
    F = S0 * np.exp((r - d) * T)
    assert np.sum(md.weights * md.S * md.q) == pytest.approx(F, rel=0.01)
    assert abs(md.mean_rel_err) < 1e-3
    assert md.checks["mean"][2] and md.passed
    md.assert_sane()


def test_atm_iv_recovered(md_flat):
    assert md_flat.iv_atm == pytest.approx(sig, abs=2e-3)


def test_sanity_badge_fails_on_bad_quotes():
    K = np.linspace(55, 190, 55)
    C = bs_call(S0, K, T, r, d, sig)
    C[::3] = np.maximum(S0 * np.exp(-d * T) - K[::3] * np.exp(-r * T), 0) * 0.5   # below intrinsic
    md = extract_q(K, S0, r, d, T, call_prices=C)
    assert not md.passed
    with pytest.raises(RuntimeError):
        md.assert_sane()


def test_tilt_moment_constraint_binds(md_flat):
    w, R = md_flat.weights, md_flat.S / S0
    var_q = np.sum(w * md_flat.q * R * R) - np.sum(w * md_flat.q * R) ** 2
    for sh, mu in [(0.45, None), (0.30, None), (0.50, 0.15)]:
        t = subjective_density(md_flat, sh, conf=1.0, mu=mu)
        assert t.converged
        assert np.allclose(t.achieved, t.targets, rtol=1e-7, atol=1e-10)
        m_p = np.sum(w * t.p * R)
        var_p = np.sum(w * t.p * R * R) - m_p**2
        ratio = np.expm1(sh**2 * T) / np.expm1(md_flat.iv_atm**2 * T)
        assert var_p == pytest.approx(var_q * ratio, rel=1e-6)
        assert np.sum(w * t.p) == pytest.approx(1.0, abs=1e-9)
        if mu is not None:
            assert m_p == pytest.approx(np.exp(mu * T), rel=1e-6)


def test_vol_view_fattens_right_tail_and_keeps_forward(md_flat):
    K = S0 * 1.3
    tails = [np.sum(md_flat.weights * subjective_density(md_flat, sh, 1.0).p * (md_flat.S > K))
             for sh in (md_flat.iv_atm, 0.45, 0.55)]
    assert tails[0] < tails[1] < tails[2]


def test_kl_nonnegative_and_zero_without_view(md_flat):
    t0 = subjective_density(md_flat, md_flat.iv_atm, conf=1.0)
    assert t0.kl == pytest.approx(0.0, abs=1e-9)
    assert np.allclose(t0.ell, 1.0, atol=1e-6)
    prev = 0.0
    for sh in (0.38, 0.42, 0.50, 0.60):
        t = subjective_density(md_flat, sh, conf=1.0)
        assert t.kl >= 0
        assert t.kl > prev          # a bigger departure from the market costs more entropy
        prev = t.kl
    assert kl_level(0.05) == "humble" and kl_level(0.6) == "alarm"


def test_shrinkage_moves_sigma_toward_iv(md_flat):
    iv = md_flat.iv_atm
    used = [subjective_density(md_flat, 0.55, conf=c).sigma_used for c in (1.0, 0.75, 0.5, 0.25, 0.0)]
    assert used[0] == pytest.approx(0.55)
    assert used[-1] == pytest.approx(iv)
    assert np.all(np.diff(used) < 0)
    assert shrink_sigma(0.55, 0.5, 0.35) == pytest.approx(0.45)
    kls = [subjective_density(md_flat, 0.55, conf=c).kl for c in (1.0, 0.5, 0.0)]
    assert kls[0] > kls[1] > kls[2] >= 0


def test_vol_only_view_keeps_expected_price_at_forward(md_flat):
    """A pure volatility view must not smuggle in an upward drift."""
    F = S0 * np.exp((r - d) * T)
    for sh in (0.45, 0.60, 0.80):
        t = subjective_density(md_flat, sh, conf=1.0)
        assert np.sum(md_flat.weights * md_flat.S * t.p) == pytest.approx(F, rel=2e-3)

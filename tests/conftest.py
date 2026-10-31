import numpy as np
import pytest

from core.bsm import bs_call
from core.spread import ChainSlice


def make_chain(S0=100.0, sigma=0.35, T=0.25, r=0.04, delta=0.0, strikes=None, half_spread_frac=0.02,
               min_half=0.01, oi=500, iv_fn=None, expiry="2027-03-19"):
    """Synthetic call chain priced off Black-Scholes. iv_fn(K) overrides the flat vol (for a smile)."""
    if strikes is None:
        strikes = np.arange(50.0, 200.0 + 1e-9, 2.5)
    strikes = np.asarray(strikes, dtype=float)
    iv = np.full_like(strikes, sigma) if iv_fn is None else np.asarray(iv_fn(strikes), dtype=float)
    mid = bs_call(S0, strikes, T, r, delta, iv)
    half = np.maximum(half_spread_frac * mid, min_half)
    bid = np.maximum(mid - half, 0.0)
    ask = mid + half
    return ChainSlice(expiry=expiry, T=T, strike=strikes, bid=bid, ask=ask,
                      oi=np.full_like(strikes, oi), iv=iv)


@pytest.fixture
def flat_chain():
    return make_chain()

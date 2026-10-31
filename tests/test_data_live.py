"""Live smoke test against Yahoo/FRED. Deselected by default:  .venv/bin/python -m pytest -m live"""
import pytest

from data.cache import Cache
from data.service import MarketDataService, chain_slice

pytestmark = pytest.mark.live


def test_live_bundle(tmp_path):
    svc = MarketDataService(Cache(tmp_path / "live.sqlite"))
    b = svc.ticker_bundle("AAPL", refresh=True)
    assert b.info and b.info.value["spot"] > 0
    assert len(b.history.value["close"]) > 400
    assert len(b.selected_expiries) == 3 and len(b.chains) == 3
    cs = chain_slice(next(iter(b.chains.values())), b.asof_date)
    assert len(cs.strike) > 20 and cs.T > 60 / 365
    m = svc.macros(refresh=True)
    assert m["usdkrw"].value["value"] > 500 and 0 < m["ust_2y"].value["value"] < 0.2

from datetime import date, datetime, timezone
from zoneinfo import ZoneInfo

import pytest

from data.trading_days import holidays, in_session, is_trading_day, market_date, previous_trading_day, quote_label

NY = ZoneInfo("America/New_York")


def ny(*a):
    return datetime(*a, tzinfo=NY)


def test_known_nyse_holidays():
    h26 = holidays(2026)
    for d in (date(2026, 1, 1), date(2026, 1, 19), date(2026, 2, 16), date(2026, 4, 3), date(2026, 5, 25),
              date(2026, 6, 19), date(2026, 7, 3), date(2026, 9, 7), date(2026, 11, 26), date(2026, 12, 25)):
        assert d in h26, d
    assert date(2027, 3, 26) in holidays(2027)            # Good Friday 2027
    assert date(2027, 6, 18) in holidays(2027)            # Juneteenth on Saturday -> Friday
    assert date(2027, 12, 24) in holidays(2027)           # Christmas on Saturday -> Friday
    assert date(2021, 12, 31) not in holidays(2022)       # Saturday New Year is not observed on Dec 31
    assert len(h26) == 10


def test_market_date_weekend_before_open_and_holiday():
    assert market_date(ny(2026, 10, 31, 10, 0)) == date(2026, 10, 30)    # Saturday -> Friday
    assert market_date(ny(2026, 11, 1, 22, 0)) == date(2026, 10, 30)     # Sunday night
    assert market_date(ny(2026, 11, 2, 8, 0)) == date(2026, 10, 30)      # Monday before the open
    assert market_date(ny(2026, 11, 2, 9, 30)) == date(2026, 11, 2)      # open
    assert market_date(ny(2026, 11, 2, 20, 0)) == date(2026, 11, 2)      # after the close: today's close
    assert market_date(ny(2026, 11, 26, 12, 0)) == date(2026, 11, 25)    # Thanksgiving
    assert market_date(datetime(2026, 10, 31, 10, 0, tzinfo=timezone.utc)) == date(2026, 10, 30)
    assert previous_trading_day(date(2027, 3, 29)) == date(2027, 3, 25)  # skips the weekend and Good Friday
    assert not is_trading_day(date(2026, 10, 31)) and is_trading_day(date(2026, 10, 30))


def test_session_and_label():
    assert in_session(ny(2026, 10, 30, 14, 0)) and not in_session(ny(2026, 10, 30, 17, 0))
    assert not in_session(ny(2026, 11, 26, 11, 0))                       # holiday
    assert quote_label(ny(2026, 10, 31, 6, 0), "off-hours") == "2026-10-30(금) 종가"
    assert quote_label(ny(2026, 11, 3, 14, 45), "regular") == "2026-11-03(화) 장중"

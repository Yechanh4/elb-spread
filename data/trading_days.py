"""US equity-market trading days (NYSE holiday rules), and the "market date" a moment belongs to.

market_date(t): the trading day whose prices are the latest available at t —
    during or after a session on a trading day   -> that day
    before the open, on a weekend or a holiday   -> the previous trading day
So a quote fetched on Saturday is Friday's closing quote, and the valuation date is Friday.
"""
from datetime import date, datetime, time as dtime, timedelta, timezone
from functools import lru_cache
from zoneinfo import ZoneInfo

NY = ZoneInfo("America/New_York")
OPEN, CLOSE = dtime(9, 30), dtime(16, 0)
WEEKDAY_KO = "월화수목금토일"


def _nth_weekday(year, month, weekday, n):
    d = date(year, month, 1)
    d += timedelta(days=(weekday - d.weekday()) % 7)
    return d + timedelta(weeks=n - 1)


def _last_weekday(year, month, weekday):
    d = date(year, month + 1, 1) - timedelta(days=1) if month < 12 else date(year, 12, 31)
    return d - timedelta(days=(d.weekday() - weekday) % 7)


def _easter(year):
    """Anonymous Gregorian algorithm."""
    a, b, c = year % 19, year // 100, year % 100
    d, e = b // 4, b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = (h + l - 7 * m + 114) % 31 + 1
    return date(year, month, day)


def _observed(d):
    """Saturday -> Friday before, Sunday -> Monday after."""
    if d.weekday() == 5:
        return d - timedelta(days=1)
    if d.weekday() == 6:
        return d + timedelta(days=1)
    return d


@lru_cache(maxsize=None)
def holidays(year):
    out = set()
    ny = date(year, 1, 1)
    if ny.weekday() != 5:                       # NYSE does not observe a Saturday New Year on Dec 31
        out.add(_observed(ny))
    out.add(_nth_weekday(year, 1, 0, 3))        # Martin Luther King Jr. Day
    out.add(_nth_weekday(year, 2, 0, 3))        # Washington's Birthday
    out.add(_easter(year) - timedelta(days=2))  # Good Friday
    out.add(_last_weekday(year, 5, 0))          # Memorial Day
    if year >= 2022:
        out.add(_observed(date(year, 6, 19)))   # Juneteenth
    out.add(_observed(date(year, 7, 4)))        # Independence Day
    out.add(_nth_weekday(year, 9, 0, 1))        # Labor Day
    out.add(_nth_weekday(year, 11, 3, 4))       # Thanksgiving
    out.add(_observed(date(year, 12, 25)))      # Christmas
    return frozenset(out)


def is_trading_day(d):
    return d.weekday() < 5 and d not in holidays(d.year)


def previous_trading_day(d):
    d -= timedelta(days=1)
    while not is_trading_day(d):
        d -= timedelta(days=1)
    return d


def market_date(t=None):
    """Trading day whose prices are the latest available at moment t (aware datetime, default now)."""
    t = (t or datetime.now(timezone.utc)).astimezone(NY)
    d = t.date()
    if is_trading_day(d) and t.time() >= OPEN:
        return d
    return previous_trading_day(d)


def in_session(t=None):
    t = (t or datetime.now(timezone.utc)).astimezone(NY)
    return is_trading_day(t.date()) and OPEN <= t.time() <= CLOSE


def quote_label(fetched_at, session):
    """'2026-10-30(금) 종가' for quotes taken outside a session, '2026-11-03(화) 장중' inside one."""
    d = market_date(fetched_at)
    return f"{d.isoformat()}({WEEKDAY_KO[d.weekday()]}) " + ("장중" if session == "regular" else "종가")

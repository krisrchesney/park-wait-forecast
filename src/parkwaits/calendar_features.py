"""Calendar features for the wait forecast: holidays, long weekends and school breaks.

Every value here is a function of the park's LOCAL DATE alone. A date's holiday status is fixed years in
advance, so none of it can carry information from after the prediction moment (the same guarantee the
history features get from "strictly earlier local dates"). `test_calendar_features.py` proves the values do
not depend on the data or on the time of day.

Three features:
  * is_holiday      1 on a US federal holiday, on the weekday it is observed (a Saturday holiday is observed
                    Friday, a Sunday one Monday) and on Easter Sunday; else 0.
  * is_long_weekend 1 on the Saturday and Sunday of a three-day weekend, on the Friday or Monday holiday
                    itself, and on the Friday after Thanksgiving; else 0.
  * is_school_break 1 during a multi-day student break listed in config `calendar.school_breaks`, else 0.
                    NaN for dates outside `calendar.covered` (the list is typed in by hand per school year,
                    so a date it does not cover is unknown, not "no break").

School breaks come from Orange County Public Schools' own 2026-2027 calendar (ocps.net). They are a proxy:
Disney and Universal guests come from many districts, so national school breaks matter too, but only the
local one is sourced and listed here. Park event nights (Halloween and Christmas parties) are not included
because no official, machine-readable date source was available.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd
from dateutil.easter import easter

CALENDAR_FEATURES = ["is_holiday", "is_long_weekend", "is_school_break"]


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    """The n-th (1-based) `weekday` (Monday = 0) of a month; n = -1 is the last one."""
    if n > 0:
        first = date(year, month, 1)
        return first + timedelta(days=(weekday - first.weekday()) % 7 + 7 * (n - 1))
    last = date(year + (month == 12), month % 12 + 1, 1) - timedelta(days=1)
    return last - timedelta(days=(last.weekday() - weekday) % 7)


def us_federal_holidays(year: int) -> dict[date, str]:
    """US federal holidays on their calendar dates (not the observed weekday)."""
    return {
        date(year, 1, 1): "New Year's Day",
        _nth_weekday(year, 1, 0, 3): "Martin Luther King Jr. Day",
        _nth_weekday(year, 2, 0, 3): "Presidents' Day",
        _nth_weekday(year, 5, 0, -1): "Memorial Day",
        date(year, 6, 19): "Juneteenth",
        date(year, 7, 4): "Independence Day",
        _nth_weekday(year, 9, 0, 1): "Labor Day",
        _nth_weekday(year, 10, 0, 2): "Columbus Day",
        date(year, 11, 11): "Veterans Day",
        _nth_weekday(year, 11, 3, 4): "Thanksgiving",
        date(year, 12, 25): "Christmas Day",
    }


def _observed(d: date) -> date:
    """The weekday a holiday is observed: Saturday -> Friday, Sunday -> Monday."""
    if d.weekday() == 5:
        return d - timedelta(days=1)
    if d.weekday() == 6:
        return d + timedelta(days=1)
    return d


def holiday_dates(years: range | list[int]) -> set[date]:
    """Holiday and observed-holiday dates, plus Easter Sunday, for the given years."""
    out: set[date] = set()
    for y in years:
        for d in us_federal_holidays(y):
            out.update({d, _observed(d)})
        out.add(easter(y))
    return out


def long_weekend_dates(years: range | list[int]) -> set[date]:
    """Days of three-day weekends (a Friday or Monday holiday) and the Friday after Thanksgiving."""
    out: set[date] = set()
    for y in years:
        for d in us_federal_holidays(y):
            obs = _observed(d)
            if obs.weekday() == 0:  # Monday holiday: Sat, Sun, Mon
                out.update({obs - timedelta(days=2), obs - timedelta(days=1), obs})
            elif obs.weekday() == 4:  # Friday holiday: Fri, Sat, Sun
                out.update({obs, obs + timedelta(days=1), obs + timedelta(days=2)})
        thanksgiving = _nth_weekday(y, 11, 3, 4)
        out.update(thanksgiving + timedelta(days=i) for i in (1, 2, 3))  # Friday, Saturday, Sunday
    return out


def school_break_dates(calendar: dict[str, Any] | None) -> tuple[set[date], tuple[date, date] | None]:
    """(dates inside a listed student break, the date range the list covers or None)."""
    if not calendar:
        return set(), None
    covered = calendar.get("covered")
    span = (_as_date(covered["from"]), _as_date(covered["to"])) if covered else None
    days: set[date] = set()
    for b in calendar.get("school_breaks") or []:
        start, end = _as_date(b["start"]), _as_date(b["end"])
        if end < start:
            raise ValueError(f"School break {b.get('name')!r} ends before it starts.")
        days.update(start + timedelta(days=i) for i in range((end - start).days + 1))
    return days, span


def _as_date(value: Any) -> date:
    return pd.Timestamp(value).date()


def calendar_features(local_dates: pd.Series, calendar: dict[str, Any] | None = None) -> pd.DataFrame:
    """Calendar columns for a Series of local dates (datetime64, midnight). Index is preserved."""
    dates = pd.to_datetime(local_dates).dt.normalize()
    if dates.empty:
        return pd.DataFrame({c: pd.Series(dtype=float) for c in CALENDAR_FEATURES}, index=local_dates.index)
    years = list(range(int(dates.dt.year.min()) - 1, int(dates.dt.year.max()) + 2))
    py = dates.dt.date
    out = pd.DataFrame(index=local_dates.index)
    out["is_holiday"] = py.isin(holiday_dates(years)).astype(float)
    out["is_long_weekend"] = py.isin(long_weekend_dates(years)).astype(float)
    break_days, span = school_break_dates(calendar)
    in_break = py.isin(break_days).astype(float)
    if span is None:
        out["is_school_break"] = np.nan
    else:
        covered = py.map(lambda d: span[0] <= d <= span[1])
        out["is_school_break"] = in_break.where(covered, np.nan)
    return out


def coverage_warning(dates: pd.Series, calendar: dict[str, Any] | None) -> str | None:
    """A message when the data runs past the typed-in school calendar (or there is none), else None.

    The list is entered by hand per school year, so it goes stale quietly: after `covered.to` every row
    gets NaN for is_school_break. Callers print this so the gap is noticed."""
    _, span = school_break_dates(calendar)
    if span is None:
        return "No school calendar is configured: is_school_break is unknown (NaN) for every row."
    days = pd.to_datetime(dates).dt.date
    if days.empty:
        return None
    if days.max() > span[1]:
        return (
            f"Data runs to {days.max()}, past the school calendar's last covered date {span[1]}: "
            "is_school_break is unknown (NaN) after it. Add the next school year to config `calendar`."
        )
    if days.min() < span[0]:
        return f"Data starts {days.min()}, before the school calendar's first covered date {span[0]}."
    return None

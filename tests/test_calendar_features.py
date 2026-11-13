"""Calendar features: dates are checked against known calendars, and the values depend on the date only."""

from datetime import date

import numpy as np
import pandas as pd
import pytest
import yaml

from parkwaits.calendar_features import (
    CALENDAR_FEATURES,
    calendar_features,
    holiday_dates,
    long_weekend_dates,
    us_federal_holidays,
)
from parkwaits.features import compute_features

CAL = {
    "covered": {"from": "2026-08-03", "to": "2027-05-31"},
    "school_breaks": [
        {"name": "Thanksgiving Break", "start": "2026-11-23", "end": "2026-11-27"},
        {"name": "Winter Break", "start": "2026-12-21", "end": "2027-01-04"},
    ],
}


def feats(days, calendar=CAL):
    s = pd.Series(pd.to_datetime(days))
    return calendar_features(s, calendar).set_index(s.dt.strftime("%Y-%m-%d"))


def test_federal_holidays_match_the_school_districts_own_published_dates():
    # Independent check: OCPS's 2026-27 calendar lists these four as district holidays.
    assert date(2026, 9, 7) in us_federal_holidays(2026)  # Labor Day
    assert date(2027, 1, 18) in us_federal_holidays(2027)  # Martin Luther King Jr. Day
    assert date(2027, 2, 15) in us_federal_holidays(2027)  # Presidents' Day
    assert date(2027, 5, 31) in us_federal_holidays(2027)  # Memorial Day


def test_other_holiday_rules():
    h = us_federal_holidays(2026)
    assert date(2026, 11, 26) in h  # Thanksgiving: fourth Thursday
    assert date(2026, 10, 12) in h  # Columbus Day: second Monday
    assert date(2026, 11, 11) in h and date(2026, 12, 25) in h and date(2026, 6, 19) in h
    assert date(2026, 5, 25) in h  # Memorial Day 2026: last Monday of May


def test_holidays_on_a_weekend_are_also_marked_on_the_observed_weekday():
    days = holiday_dates([2026, 2027, 2028])
    assert date(2026, 7, 4) in days and date(2026, 7, 3) in days  # Saturday -> Friday
    assert date(2027, 7, 4) in days and date(2027, 7, 5) in days  # Sunday -> Monday
    assert date(2027, 12, 31) in days  # New Year's Day 2028 is a Saturday


def test_easter_sunday_is_a_holiday():
    days = holiday_dates([2026, 2027])
    assert date(2026, 4, 5) in days and date(2027, 3, 28) in days


def test_long_weekends():
    lw = long_weekend_dates([2026])
    assert {date(2026, 9, 5), date(2026, 9, 6), date(2026, 9, 7)} <= lw  # Labor Day weekend
    assert date(2026, 9, 4) not in lw  # the Friday before a Monday holiday is a working day
    assert {date(2026, 11, 27), date(2026, 11, 28), date(2026, 11, 29)} <= lw  # after Thanksgiving
    assert {date(2026, 7, 3), date(2026, 7, 4), date(2026, 7, 5)} <= lw  # observed Friday
    assert date(2026, 8, 15) not in lw  # an ordinary Saturday is not a long weekend


def test_school_breaks_and_unknown_dates():
    days = ["2026-11-20", "2026-11-22", "2026-11-23", "2026-11-27", "2026-11-28", "2027-01-04", "2027-01-05"]
    f = feats(days)
    assert list(f["is_school_break"]) == [0, 0, 1, 1, 0, 1, 0]
    outside = feats(["2026-07-01", "2027-06-15"])
    assert outside["is_school_break"].isna().all()  # outside the covered school year: unknown, not zero
    assert feats(["2026-11-24"], calendar=None)["is_school_break"].isna().all()


def test_a_break_that_ends_before_it_starts_is_rejected():
    bad = {**CAL, "school_breaks": [{"name": "x", "start": "2026-11-27", "end": "2026-11-23"}]}
    with pytest.raises(ValueError, match="ends before it starts"):
        feats(["2026-11-24"], calendar=bad)


def test_values_depend_on_the_date_only_not_on_the_time_of_day():
    morning = pd.Series(pd.to_datetime(["2026-11-24 00:00", "2026-12-25 00:00", "2026-09-05 00:00"]))
    evening = morning + pd.Timedelta(hours=23, minutes=59)
    a = calendar_features(morning, CAL)
    b = calendar_features(evening, CAL)
    pd.testing.assert_frame_equal(a, b)


def test_empty_input_gives_empty_columns():
    out = calendar_features(pd.Series([], dtype="datetime64[ns]"), CAL)
    assert list(out.columns) == CALENDAR_FEATURES and out.empty


def _ride(times):
    ts = pd.to_datetime(times)
    return pd.DataFrame(
        {
            "park_id": 1,
            "ride_id": 1,
            "observed_at": ts,
            "is_open": 1,
            "wait_time": 30.0,
            "source_updated": ts - pd.Timedelta(minutes=3),
        }
    )


def test_features_follow_the_parks_local_date_not_the_utc_date():
    # 03:00 UTC on Nov 24 is 22:00 on Nov 23 in Orlando (a break day); 06:00 UTC on Nov 28 is 01:00 on Nov 28
    # (the break ended the evening before). The UTC dates would give the opposite answers.
    snaps = _ride(["2026-11-24 03:00", "2026-11-28 03:00", "2026-11-28 06:00"])
    f = compute_features(snaps, {1: "America/New_York"}, CAL)
    assert list(f["is_school_break"]) == [1, 1, 0]


def test_compute_features_without_a_calendar_marks_school_breaks_unknown():
    f = compute_features(_ride(["2026-11-24 15:00"]), {1: "America/New_York"})
    assert f["is_school_break"].isna().all() and f["is_holiday"].eq(0).all()


def test_configured_calendar_is_well_formed():
    cfg = yaml.safe_load(open("config/config.yaml"))["calendar"]
    lo, hi = pd.Timestamp(cfg["covered"]["from"]), pd.Timestamp(cfg["covered"]["to"])
    spans = sorted((pd.Timestamp(b["start"]), pd.Timestamp(b["end"])) for b in cfg["school_breaks"])
    for start, end in spans:
        assert lo <= start <= end <= hi
    assert all(a_end < b_start for (_, a_end), (b_start, _) in zip(spans, spans[1:], strict=False))
    f = feats(["2026-11-25", "2026-12-24"], calendar=cfg)
    assert f["is_school_break"].tolist() == [1, 1] and not np.isnan(f.to_numpy()).any()

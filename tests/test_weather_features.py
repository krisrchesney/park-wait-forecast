"""Weather: parsed strictly, and a snapshot only ever sees weather from its own hour or earlier."""

import httpx
import numpy as np
import pandas as pd
import pytest

from parkwaits.collect import run_collector
from parkwaits.features import FEATURES, build_training_frame, compute_features
from parkwaits.lake import load_lake
from parkwaits.weather import (
    WEATHER_FEATURES,
    WeatherCollector,
    WeatherError,
    parse_hourly,
    weather_features,
)
from tests.synth import TZ, make_snapshots


def _payload(start="2026-08-03T00:00", hours=72, base=0.0):
    times = pd.date_range(start, periods=hours, freq="h").strftime("%Y-%m-%dT%H:%M").tolist()
    return {
        "hourly": {
            "time": times,
            "temperature_2m": [20 + (i % 24) / 2 for i in range(hours)],
            "apparent_temperature": [22 + (i % 24) / 2 for i in range(hours)],
            "precipitation": [base + i for i in range(hours)],  # every hour differs, so mix-ups show
            "cloud_cover": [min(100, i) for i in range(hours)],
        }
    }


def _weather(park_ids=(1, 2), **kw) -> pd.DataFrame:
    rows = []
    for p in park_ids:
        part = parse_hourly(_payload(**kw))
        part.insert(0, "park_id", p)
        rows.append(part)
    return pd.concat(rows, ignore_index=True)


def _snaps(days=3):
    return make_snapshots(days=days, rides_per_park=2)


def test_parse_hourly_reads_the_columns_and_rejects_a_changed_format():
    df = parse_hourly(_payload(hours=5))
    assert list(df.columns) == ["hour", "temp_c", "feels_like_c", "rain_mm", "cloud_pct"]
    assert df["hour"].iloc[0] == pd.Timestamp("2026-08-03 00:00") and len(df) == 5
    with pytest.raises(WeatherError):
        parse_hourly({"hourly": {"time": ["2026-08-03T00:00"]}})
    with pytest.raises(WeatherError):
        parse_hourly({"error": True})


def test_a_snapshot_gets_the_weather_stamped_at_its_own_hour_and_the_last_three_hours_of_rain():
    snaps = _snaps()
    w = _weather()
    got = weather_features(snaps.assign(observed_at=snaps["observed_at"]), w)
    row = snaps.index[(snaps["observed_at"] == pd.Timestamp("2026-08-04 14:35").floor("10min"))][0]
    # snapshots sit on a 10 minute grid: 14:30 belongs to the hour stamped 14:00 (hour index 38)
    t = pd.Timestamp("2026-08-04 14:30")
    row = snaps.index[snaps["observed_at"] == t][0]
    assert got.loc[row, "rain_now_mm"] == 38  # precipitation value stamped 14:00
    assert got.loc[row, "rain_3h_mm"] == 36 + 37 + 38  # hours 12:00, 13:00, 14:00: all already past
    assert got.loc[row, "temp_c"] == 20 + (38 % 24) / 2


def test_hours_after_the_snapshot_never_change_its_features():
    snaps = _snaps()
    w = _weather()
    base = weather_features(snaps, w)
    # corrupt every weather hour AFTER 14:00 on Aug 4 (UTC), as a future forecast would differ
    later = w["hour"] > pd.Timestamp("2026-08-04 14:00")
    w2 = w.copy()
    w2.loc[later, WEATHER_FEATURES_RAW] = 999.0
    changed = weather_features(snaps, w2)
    safe = (snaps["observed_at"] < pd.Timestamp("2026-08-04 15:00")).to_numpy()
    pd.testing.assert_frame_equal(base[safe], changed[safe])
    assert not changed[~safe].equals(base[~safe])  # and the later rows do see the new values


WEATHER_FEATURES_RAW = ["temp_c", "feels_like_c", "rain_mm", "cloud_pct"]


def test_weather_cut_off_at_a_moment_gives_the_same_features_for_earlier_snapshots():
    snaps = _snaps()
    w = _weather()
    cut = pd.Timestamp("2026-08-04 09:00")
    full = weather_features(snaps, w)
    part = weather_features(snaps, w[w["hour"] <= cut])
    early = (snaps["observed_at"] < cut + pd.Timedelta(hours=1)).to_numpy()
    # rows up to the cut hour match; rows after it have no weather at all rather than a stale value
    pd.testing.assert_frame_equal(full[early].reset_index(drop=True), part[early].reset_index(drop=True))
    assert part[~early].isna().all().all()


def test_a_missing_hour_makes_the_three_hour_rain_unknown_not_a_short_sum():
    snaps = _snaps()
    w = _weather()
    hole = (w["hour"] == pd.Timestamp("2026-08-04 13:00"))
    got = weather_features(snaps, w[~hole])
    at_14 = snaps.index[snaps["observed_at"] == pd.Timestamp("2026-08-04 14:30")][0]
    assert np.isnan(got.loc[at_14, "rain_3h_mm"]) and got.loc[at_14, "rain_now_mm"] == 38


def test_no_weather_means_unknown_not_zero():
    snaps = _snaps()
    for none in (None, pd.DataFrame()):
        got = weather_features(snaps, none)
        assert list(got.columns) == WEATHER_FEATURES and got.isna().all().all()


def test_each_park_uses_its_own_weather():
    snaps = _snaps()
    w = pd.concat([_weather(park_ids=(1,), base=0), _weather(park_ids=(2,), base=500)], ignore_index=True)
    got = weather_features(snaps, w)
    assert (got.loc[snaps["park_id"] == 1, "rain_now_mm"].dropna() < 100).all()
    assert (got.loc[snaps["park_id"] == 2, "rain_now_mm"].dropna() >= 500).all()


def test_the_training_frame_carries_weather_and_the_default_features_are_unchanged():
    snaps = _snaps(days=14)
    w = _weather(start="2026-08-03T00:00", hours=14 * 24)
    frame = build_training_frame(snaps, TZ, min_rows=100, min_positives=10, weather=w)
    assert set(WEATHER_FEATURES) <= set(frame.columns)
    assert not set(WEATHER_FEATURES) & set(FEATURES)  # not used unless config model.features lists them
    assert frame["rain_3h_mm"].notna().any()
    bare = build_training_frame(snaps, TZ, min_rows=100, min_positives=10)
    assert bare["rain_3h_mm"].isna().all()
    pd.testing.assert_frame_equal(bare[FEATURES], frame[FEATURES])  # weather never changes the other features


def test_compute_features_weather_does_not_depend_on_the_ride_rows_around_it():
    snaps = _snaps()
    w = _weather()
    full = compute_features(snaps, TZ, weather=w).set_index(["park_id", "ride_id", "observed_at"])
    fewer = compute_features(snaps[snaps["ride_id"] % 100 == 0], TZ, weather=w).set_index(
        ["park_id", "ride_id", "observed_at"]
    )
    pd.testing.assert_frame_equal(fewer[WEATHER_FEATURES], full.loc[fewer.index, WEATHER_FEATURES])


CFG = {
    "queue_times": {
        "parks": [
            {"id": 1, "name": "A", "latitude": 28.4, "longitude": -81.5},
            {"id": 2, "name": "B", "latitude": 28.4, "longitude": -81.5},  # same point as park 1
            {"id": 3, "name": "C", "latitude": 28.5, "longitude": -81.4},
        ]
    },
    "weather": {"start_date": "2026-08-03"},
}
NOW = pd.Timestamp("2026-08-05 12:00").to_pydatetime()


def _collector(handler, cfg=CFG):
    return WeatherCollector(cfg, transport=httpx.MockTransport(handler), now=lambda: NOW)


def test_collector_asks_once_per_point_for_the_whole_window_and_stamps_when_it_fetched():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(dict(request.url.params))
        return httpx.Response(200, json=_payload(hours=48))

    frame = _collector(handler).fetch()
    assert len(seen) == 2  # parks 1 and 2 share a point
    assert seen[0]["start_date"] == "2026-08-03" and seen[0]["end_date"] == "2026-08-05"
    assert seen[0]["timezone"] == "GMT"
    assert sorted(frame["park_id"].unique()) == [1, 2, 3] and (frame["fetched_at"] == NOW).all()


def test_lookback_days_limits_the_window():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(dict(request.url.params))
        return httpx.Response(200, json=_payload(hours=24))

    cfg = {**CFG, "weather": {"start_date": "2026-08-03", "lookback_days": 1}}
    _collector(handler, cfg).fetch()
    assert seen[0]["start_date"] == "2026-08-04"


def test_one_failing_point_is_recorded_and_all_failing_raises():
    def handler(request: httpx.Request) -> httpx.Response:
        if float(request.url.params["latitude"]) == 28.5:
            return httpx.Response(400, json={"error": True})
        return httpx.Response(200, json=_payload(hours=24))

    c = _collector(handler)
    assert sorted(c.fetch()["park_id"].unique()) == [1, 2] and c.failed_parks == [3]
    with pytest.raises(WeatherError):
        _collector(lambda r: httpx.Response(500)).fetch()


def test_parks_without_coordinates_are_a_clear_error():
    with pytest.raises(WeatherError):
        WeatherCollector({"queue_times": {"parks": [{"id": 1, "name": "A"}]}})


def test_a_newer_fetch_replaces_an_older_value_for_the_same_hour(tmp_path, monkeypatch):
    cfg = {**CFG, "lake": {"root": str(tmp_path)}}
    answers = iter([_payload(hours=24, base=0.0), _payload(hours=24, base=100.0)])
    clock = iter([pd.Timestamp("2026-08-05 12:00"), pd.Timestamp("2026-08-05 13:00")])

    class Fake(WeatherCollector):
        def __init__(self, c):  # one collector per run: take that run's answer and clock reading
            payload, when = next(answers), next(clock).to_pydatetime()
            super().__init__(
                c,
                transport=httpx.MockTransport(lambda r: httpx.Response(200, json=payload)),
                now=lambda: when,
            )

    monkeypatch.setitem(__import__("parkwaits.collect", fromlist=["COLLECTORS"]).COLLECTORS, "weather", Fake)
    one_park = {**cfg, "queue_times": {"parks": [cfg["queue_times"]["parks"][0]]}}
    assert run_collector("weather", one_park)[0] == 24
    inserted, updated = run_collector("weather", one_park)
    assert inserted == 0 and updated == 24
    stored = load_lake(tmp_path, "weather")
    assert (stored["rain_mm"] >= 100).all()  # the later fetch won

import json
from datetime import datetime, timedelta
from pathlib import Path

import httpx
import pandas as pd
import pytest

from parkwaits import collect
from parkwaits.lake import load_lake
from parkwaits.queue_times import (
    QueueTimesCollector,
    QueueTimesError,
    floor_time,
    parse_park,
    run_loop,
)

REAL = json.loads((Path(__file__).parent / "fixtures" / "queue_times_park6.json").read_text())
NOW = datetime(2026, 10, 19, 14, 7, 31)


@pytest.fixture
def qt_cfg(cfg):
    cfg["queue_times"] = {
        "snapshot_minutes": 10,
        "request_interval_seconds": 0.0,
        "parks": [
            {"id": 6, "name": "Magic Kingdom", "timezone": "America/New_York"},
            {"id": 5, "name": "Epcot", "timezone": "America/New_York"},
        ],
    }
    return cfg


def server(statuses=None, payloads=None):
    statuses = dict(statuses or {})
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        pid = int(request.url.path.split("/")[2])
        calls.append(pid)
        code = statuses.get(pid, 200)
        if code != 200:
            return httpx.Response(code, json={})
        return httpx.Response(200, json=(payloads or {}).get(pid, REAL))

    return handler, calls


def make(cfg, handler):
    return QueueTimesCollector(
        cfg, transport=httpx.MockTransport(handler), sleep=lambda s: None, now=lambda: NOW
    )


def test_real_response_parses_to_one_row_per_ride():
    df = parse_park(REAL, 6, datetime(2026, 10, 19, 14, 0))
    expected = sum(len(land["rides"]) for land in REAL["lands"]) + len(REAL["rides"])
    assert len(df) == expected == 45
    assert df["park_id"].eq(6).all() and df["ride_id"].is_unique
    assert set(df["is_open"]) <= {0, 1} and (df["wait_time"] >= 0).all()
    assert str(df["source_updated"].dtype) == "datetime64[ns]" and df["source_updated"].notna().all()
    first = REAL["lands"][0]["rides"][0]
    row = df[df["ride_id"] == first["id"]].iloc[0]
    assert row["ride_name"] == first["name"] and row["wait_time"] == first["wait_time"]


def test_rides_outside_lands_are_read():
    payload = {
        "lands": [],
        "rides": [
            {
                "id": 9,
                "name": "X",
                "is_open": True,
                "wait_time": 25,
                "last_updated": "2026-10-19T14:00:00.000Z",
            }
        ],
    }
    df = parse_park(payload, 1, datetime(2026, 10, 19, 14, 0))
    assert list(df["ride_id"]) == [9] and df["is_open"].iloc[0] == 1


def test_unexpected_format_is_reported():
    with pytest.raises(QueueTimesError, match="format changed"):
        parse_park({"error": "nope"}, 6, datetime(2026, 10, 19))


def test_floor_time_snaps_to_the_grid():
    assert floor_time(NOW, 10) == datetime(2026, 10, 19, 14, 0)
    assert floor_time(datetime(2026, 10, 19, 14, 59, 59), 10) == datetime(2026, 10, 19, 14, 50)
    assert floor_time(datetime(2026, 10, 19, 23, 59), 15) == datetime(2026, 10, 19, 23, 45)


def test_fetch_stamps_every_row_with_the_grid_time(qt_cfg):
    handler, calls = server()
    df = make(qt_cfg, handler).fetch()
    assert sorted(set(calls)) == [5, 6] and set(df["park_id"]) == {5, 6}
    assert (df["observed_at"] == pd.Timestamp("2026-10-19 14:00")).all()


def test_one_failing_park_does_not_lose_the_other(qt_cfg):
    handler, _ = server(statuses={5: 404})
    c = make(qt_cfg, handler)
    df = c.fetch()
    assert set(df["park_id"]) == {6} and c.failed_parks == [5]


def test_all_parks_failing_is_an_error(qt_cfg):
    handler, _ = server(statuses={5: 404, 6: 404})
    with pytest.raises(QueueTimesError, match="every park failed"):
        make(qt_cfg, handler).fetch()


def test_temporary_errors_are_retried(qt_cfg):
    state = {"n": 0}

    def flaky(request):
        state["n"] += 1
        return httpx.Response(503) if state["n"] == 1 else httpx.Response(200, json=REAL)

    qt_cfg["queue_times"]["parks"] = qt_cfg["queue_times"]["parks"][:1]
    sleeps = []
    c = QueueTimesCollector(
        qt_cfg, transport=httpx.MockTransport(flaky), sleep=sleeps.append, now=lambda: NOW
    )
    assert len(c.fetch()) == 45 and 5.0 in sleeps


def test_empty_park_list_is_rejected(cfg):
    cfg["queue_times"] = {"parks": []}
    with pytest.raises(QueueTimesError):
        QueueTimesCollector(cfg)


def test_same_bucket_twice_stores_nothing_new(qt_cfg, monkeypatch):
    handler, _ = server()
    monkeypatch.setitem(
        collect.COLLECTORS,
        "queue_times",
        lambda c: QueueTimesCollector(
            c, transport=httpx.MockTransport(handler), sleep=lambda s: None, now=lambda: NOW
        ),
    )
    assert collect.run_collector("queue_times", qt_cfg) == (90, 0)
    assert collect.run_collector("queue_times", qt_cfg) == (0, 0)
    assert len(load_lake(qt_cfg["lake"]["root"], "queue_times")) == 90


def test_loop_polls_on_the_grid_and_survives_a_failed_iteration():
    clock = {"t": datetime(2026, 10, 19, 14, 1)}
    waits, calls = [], []

    def sleep(seconds):
        waits.append(seconds)
        clock["t"] += timedelta(seconds=seconds)

    def fake_collect(name, cfg):
        calls.append(clock["t"])
        if len(calls) == 2:
            raise QueueTimesError("boom")
        return (10, 0)

    stats = run_loop(
        {"queue_times": {"snapshot_minutes": 10}},
        duration_minutes=35,
        sleep=sleep,
        now=lambda: clock["t"],
        collect=fake_collect,
    )
    assert [c.minute for c in calls] == [
        1,
        10,
        20,
        30,
    ]  # first poll now, then :10 :20 :30 (:40 is past 35 min)
    assert stats == {"iterations": 4, "failed_iterations": 1, "rows_inserted": 30}
    assert waits[0] == 9 * 60  # woke exactly at 14:10

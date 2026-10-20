"""Queue-Times.com collector: ride wait times for theme parks.

Data: https://queue-times.com (free API; attribution "Powered by Queue-Times.com" is required, see README).

Each run takes one snapshot of every configured park. A snapshot row is one ride at one moment:
  park_id, ride_id, observed_at (our fetch time floored to `snapshot_minutes`, UTC), is_open, wait_time,
  source_updated (the ride's own `last_updated`, UTC) and the ride name.

Design notes:
  * Insert-only and idempotent: the key is (park_id, ride_id, observed_at), so running twice inside the
    same 10 minute bucket stores nothing new.
  * One park failing does not lose the others; the run only fails if every park failed.
  * History only exists from the day collection starts: this API gives current values, not archives.
  * Rides can appear in the top-level `rides` list as well as inside `lands`; both are read.
"""

from __future__ import annotations

import sys
import time
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
import pandas as pd

API_ROOT = "https://queue-times.com"
ATTRIBUTION = "Powered by Queue-Times.com"
MAX_ATTEMPTS = 3


class QueueTimesError(RuntimeError):
    pass


def floor_time(ts: datetime, minutes: int) -> datetime:
    """Floor a (naive UTC) datetime to a `minutes` grid."""
    total = ts.hour * 60 + ts.minute
    floored = total - total % minutes
    return ts.replace(hour=floored // 60, minute=floored % 60, second=0, microsecond=0)


def parse_park(payload: dict[str, Any], park_id: int, observed_at: datetime) -> pd.DataFrame:
    """Flatten one /parks/{id}/queue_times.json response into rows."""
    if not isinstance(payload, dict) or ("lands" not in payload and "rides" not in payload):
        raise QueueTimesError(f"park {park_id}: response has neither 'lands' nor 'rides'; format changed?")
    rides = list(payload.get("rides") or [])
    for land in payload.get("lands") or []:
        rides.extend(land.get("rides") or [])
    rows = []
    for r in rides:
        rows.append(
            {
                "park_id": int(park_id),
                "ride_id": int(r["id"]),
                "observed_at": observed_at,
                "is_open": int(bool(r.get("is_open"))),
                "wait_time": float(r.get("wait_time") or 0),
                "source_updated": r.get("last_updated"),
                "ride_name": r.get("name"),
            }
        )
    df = pd.DataFrame(
        rows,
        columns=[
            "park_id",
            "ride_id",
            "observed_at",
            "is_open",
            "wait_time",
            "source_updated",
            "ride_name",
        ],
    )
    df["observed_at"] = pd.to_datetime(df["observed_at"]).astype("datetime64[ns]")
    df["source_updated"] = (
        pd.to_datetime(df["source_updated"], utc=True, errors="coerce")
        .dt.tz_localize(None)
        .astype("datetime64[ns]")
    )
    return df.drop_duplicates(["park_id", "ride_id"])


class QueueTimesCollector:
    name = "queue_times"
    key_cols = ["park_id", "ride_id", "observed_at"]
    partition_col = "observed_at"
    version_col = None
    keep_first_cols: list[str] = []

    def __init__(
        self,
        cfg: dict[str, Any],
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
        now: Callable[[], datetime] = lambda: datetime.now(UTC).replace(tzinfo=None),
    ):
        qt = cfg.get("queue_times", {})
        self.parks = [int(p["id"]) for p in qt.get("parks", [])]
        if not self.parks:
            raise QueueTimesError("config queue_times.parks is empty.")
        self.snapshot_minutes = int(qt.get("snapshot_minutes", 10))
        self.interval = float(qt.get("request_interval_seconds", 1.0))
        self._transport, self._sleep, self._now = transport, sleep, now
        self.failed_parks: list[int] = []

    def _get(self, client: httpx.Client, park_id: int) -> dict[str, Any]:
        last = ""
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                resp = client.get(f"{API_ROOT}/parks/{park_id}/queue_times.json")
            except httpx.TransportError as exc:
                last = f"{type(exc).__name__}"
            else:
                if resp.status_code == 200:
                    return resp.json()
                last = f"HTTP {resp.status_code}"
                if resp.status_code not in (429, 500, 502, 503, 504):
                    break
            if attempt < MAX_ATTEMPTS:
                self._sleep(5.0 * attempt)
        raise QueueTimesError(f"park {park_id}: {last}")

    def fetch(self) -> pd.DataFrame:
        observed_at = floor_time(self._now(), self.snapshot_minutes)
        frames: list[pd.DataFrame] = []
        self.failed_parks = []
        with httpx.Client(
            timeout=30.0, transport=self._transport, headers={"User-Agent": "parkwaits-portfolio"}
        ) as client:
            for i, park_id in enumerate(self.parks):
                if i:
                    self._sleep(self.interval)
                try:
                    frames.append(parse_park(self._get(client, park_id), park_id, observed_at))
                except (QueueTimesError, ValueError, KeyError) as exc:
                    self.failed_parks.append(park_id)
                    print(f"[queue_times] skipped park {park_id}: {exc}", file=sys.stderr)
        if not frames:
            raise QueueTimesError(f"every park failed: {self.failed_parks}")
        return pd.concat(frames, ignore_index=True)


def run_loop(
    cfg: dict[str, Any],
    *,
    duration_minutes: float,
    every_minutes: int | None = None,
    sleep: Callable[[float], None] = time.sleep,
    now: Callable[[], datetime] = lambda: datetime.now(UTC).replace(tzinfo=None),
    collect: Callable[[str, dict[str, Any]], tuple[int, int]] | None = None,
) -> dict[str, int]:
    """Collect repeatedly inside ONE process, so a CI job can push to the data branch once at the end.

    Wakes on the snapshot grid (e.g. :00, :10, :20...), runs the collector, and keeps going after a failed
    iteration. Stops once `duration_minutes` have passed. Returns counts for the summary line.
    """
    if collect is None:
        from parkwaits.collect import run_collector as collect
    step = int(every_minutes or cfg.get("queue_times", {}).get("snapshot_minutes", 10))
    started = now()
    stop_at = started + timedelta(minutes=duration_minutes)
    stats = {"iterations": 0, "failed_iterations": 0, "rows_inserted": 0}
    while True:
        try:
            inserted, _ = collect("queue_times", cfg)
            stats["rows_inserted"] += inserted
        except Exception as exc:  # keep going: one bad poll must not discard hours of collection
            stats["failed_iterations"] += 1
            print(f"[queue_times] iteration failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        stats["iterations"] += 1
        current = now()
        next_tick = floor_time(current, step) + timedelta(minutes=step)
        if next_tick > stop_at:
            return stats
        sleep(max((next_tick - current).total_seconds(), 1.0))

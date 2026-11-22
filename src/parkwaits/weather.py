"""Weather features for the wait forecast, from Open-Meteo (https://open-meteo.com).

Data: hourly modelled "observed" weather (reanalysis plus the newest model analysis), not station readings.
Terms: free for non-commercial use, CC BY 4.0, so a page using it must credit Open-Meteo.com (ATTRIBUTION).

Leak control: a snapshot taken at time t only ever sees the weather stamped at or before floor(t to the hour).
Open-Meteo stamps each hourly value at the END of its hour (precipitation is the sum over the preceding hour),
so the value stamped 14:00 is fully known by a 14:35 snapshot. Hours after t are never read, even though the
API also returns future hours. `test_weather_features.py` proves this by changing later hours.

Four limits, stated rather than hidden:
  * Values are modelled grid-cell estimates (reanalysis plus the newest model analysis), not what a rain
    gauge read. Live scoring uses the same product for the hours already passed, so training and serving
    match, but a true "forecast issued earlier" is never used because none was stored.
  * Each park gets the weather of its own coordinates, but the grid is 9 to 25 km, so two nearby parks can
    share one cell.
  * Few days of data contain few rainy days, so a gain can be too small to measure.
  * Values for hours the lake has no weather for are NaN (unknown), never 0.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
import pandas as pd

API_URL = "https://archive-api.open-meteo.com/v1/archive"
ATTRIBUTION = "Weather data by Open-Meteo.com (CC BY 4.0)"
HOURLY = ["temperature_2m", "apparent_temperature", "precipitation", "cloud_cover"]
WEATHER_FEATURES = ["temp_c", "feels_like_c", "rain_now_mm", "rain_3h_mm", "cloud_pct"]
RAW_COLUMNS = {
    "temperature_2m": "temp_c",
    "apparent_temperature": "feels_like_c",
    "precipitation": "rain_mm",
    "cloud_cover": "cloud_pct",
}
MAX_ATTEMPTS = 3


class WeatherError(RuntimeError):
    pass


def parse_hourly(payload: dict[str, Any]) -> pd.DataFrame:
    """One Open-Meteo response (timezone=GMT) -> rows of hour (UTC, naive) and the raw columns."""
    hourly = payload.get("hourly") if isinstance(payload, dict) else None
    if not hourly or "time" not in hourly or any(c not in hourly for c in HOURLY):
        raise WeatherError("Open-Meteo response lacks the expected hourly columns; format changed?")
    df = pd.DataFrame({"hour": pd.to_datetime(hourly["time"])})
    for raw, name in RAW_COLUMNS.items():
        df[name] = pd.to_numeric(pd.Series(hourly[raw]), errors="coerce")
    df["hour"] = df["hour"].astype("datetime64[ns]")
    return df


class WeatherCollector:
    """Hourly weather per park. Replaceable rows: a newer fetch overwrites an older value for the same hour
    (the newest hours are first modelled, then revised), so history converges on the final values."""

    name = "weather"
    key_cols = ["park_id", "hour"]
    partition_col = "hour"
    version_col = "fetched_at"
    keep_first_cols: list[str] = []

    def __init__(
        self,
        cfg: dict[str, Any],
        transport: httpx.BaseTransport | None = None,
        now: Callable[[], datetime] = lambda: datetime.now(UTC).replace(tzinfo=None),
    ):
        self.parks = [p for p in cfg.get("queue_times", {}).get("parks", []) if "latitude" in p]
        if not self.parks:
            raise WeatherError("No park in config queue_times.parks has latitude and longitude.")
        w = cfg.get("weather", {})
        self.start = pd.Timestamp(w.get("start_date", "2026-10-20")).date()
        self.lookback_days = w.get("lookback_days")
        self._transport, self._now = transport, now
        self.failed_parks: list[int] = []

    def _window(self) -> tuple[str, str]:
        today = self._now().date()
        start = today - timedelta(days=int(self.lookback_days)) if self.lookback_days else self.start
        return start.isoformat(), today.isoformat()

    def _get(self, client: httpx.Client, lat: float, lon: float) -> dict[str, Any]:
        start, end = self._window()
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start,
            "end_date": end,
            "hourly": ",".join(HOURLY),
            "timezone": "GMT",
        }
        last = ""
        for _ in range(MAX_ATTEMPTS):
            try:
                resp = client.get(API_URL, params=params)
            except httpx.TransportError as exc:
                last = type(exc).__name__
                continue
            if resp.status_code == 200:
                return resp.json()
            last = f"HTTP {resp.status_code}"
            if resp.status_code not in (429, 500, 502, 503, 504):
                break
        raise WeatherError(f"Open-Meteo request failed: {last}")

    def fetch(self) -> pd.DataFrame:
        fetched_at = self._now()
        by_point: dict[tuple[float, float], pd.DataFrame] = {}
        frames = []
        with httpx.Client(transport=self._transport, timeout=30.0) as client:
            for park in self.parks:
                point = (float(park["latitude"]), float(park["longitude"]))
                try:
                    if point not in by_point:  # parks that share coordinates share one request
                        by_point[point] = parse_hourly(self._get(client, *point))
                except (WeatherError, ValueError):
                    self.failed_parks.append(int(park["id"]))
                    continue
                part = by_point[point].copy()
                part.insert(0, "park_id", int(park["id"]))
                frames.append(part)
        if not frames:
            raise WeatherError("Every park's weather request failed.")
        out = pd.concat(frames, ignore_index=True)
        out["fetched_at"] = pd.Timestamp(fetched_at).as_unit("ns")
        return out


def weather_features(snapshots: pd.DataFrame, weather: pd.DataFrame | None) -> pd.DataFrame:
    """The weather columns for every snapshot row (index preserved), using hours at or before the snapshot.

    `snapshots` needs park_id and observed_at (UTC, naive); `weather` is the lake table (park_id, hour, ...).
    Without weather every value is NaN."""
    out = pd.DataFrame(float("nan"), index=snapshots.index, columns=WEATHER_FEATURES)
    if weather is None or weather.empty:
        return out
    w = weather.drop_duplicates(["park_id", "hour"], keep="last").copy()
    w["hour"] = pd.to_datetime(w["hour"]).astype("datetime64[ns]")
    parts = []
    for park_id, grp in w.groupby("park_id"):
        full = grp.set_index("hour").sort_index()
        full = full.reindex(pd.date_range(full.index.min(), full.index.max(), freq="h"))
        # the last 3 hourly sums; needs all 3 hours present, so a hole gives NaN rather than a short sum
        full["rain_3h_mm"] = full["rain_mm"].rolling(3, min_periods=3).sum()
        full["park_id"] = park_id
        parts.append(full.rename(columns={"rain_mm": "rain_now_mm"}).rename_axis("hour").reset_index())
    table = pd.concat(parts, ignore_index=True)[["park_id", "hour", *WEATHER_FEATURES]]
    keys = pd.DataFrame(
        {
            "park_id": snapshots["park_id"].to_numpy(),
            "hour": pd.to_datetime(snapshots["observed_at"]).dt.floor("h").to_numpy(dtype="datetime64[ns]"),
        }
    )
    merged = keys.merge(table, on=["park_id", "hour"], how="left")
    for c in WEATHER_FEATURES:
        out[c] = merged[c].to_numpy(dtype=float)
    return out

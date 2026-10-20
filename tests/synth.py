"""Synthetic Queue-Times snapshots shaped like the collector's output (tests only)."""

import numpy as np
import pandas as pd

TZ = {1: "America/New_York", 2: "America/Los_Angeles"}


def make_snapshots(
    days: int = 14, rides_per_park: int = 8, step: int = 10, seed: int = 0, start="2026-08-03"
):
    """Waits follow a daily hump (peak ~14:00 local), a per-ride popularity level and a weekend lift."""
    rng = np.random.default_rng(seed)
    t0 = pd.Timestamp(start)  # UTC midnight
    times = pd.date_range(t0, periods=days * 24 * 60 // step, freq=f"{step}min")
    rows = []
    for park_id, tz in TZ.items():
        local = times.tz_localize("UTC").tz_convert(tz).tz_localize(None)
        hour = local.hour + local.minute / 60
        hump = np.exp(-((hour - 14) ** 2) / 18)
        weekend = (local.dayofweek >= 5) * 8
        is_open_time = (hour >= 9) & (hour < 21)
        for r in range(rides_per_park):
            pop = 20 + 12 * r
            noise = np.cumsum(rng.normal(0, 1.2, len(times)))
            noise -= pd.Series(noise).rolling(36, min_periods=1).mean().to_numpy()
            wait = np.clip(pop * hump + weekend + noise, 0, 180).round(-0 if False else 0)
            is_open = is_open_time & (rng.random(len(times)) > 0.01)
            rows.append(
                pd.DataFrame(
                    {
                        "park_id": park_id,
                        "ride_id": park_id * 100 + r,
                        "observed_at": times,
                        "is_open": is_open.astype(int),
                        "wait_time": np.where(is_open, wait, 0.0),
                        "source_updated": times - pd.Timedelta(minutes=3),
                        "ride_name": f"Ride {park_id}-{r}",
                    }
                )
            )
    return pd.concat(rows, ignore_index=True)

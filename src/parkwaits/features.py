"""Leak-free dataset construction for ride wait-time forecasting.

Prediction moment: a snapshot of an OPEN ride at time t.
Question: will its posted wait be at least `wait_threshold_min` minutes at t + `horizon_minutes`?

Controls (each has a test):
  * Every feature uses only observations at or before t. Lags look back; rolling windows end at t;
    history means use strictly earlier LOCAL DATES; "minutes since open" is a running minimum.
  * Rows need an observed target. Snapshots whose target time is missing (end of data, collection gaps)
    are dropped rather than guessed, so there is no look-ahead imputation.
  * Rows whose ride is closed at the target time are dropped: a closed ride has no wait, so the question
    is "if it is still open". This is stated as a limitation, not hidden.
  * Time features use each park's local clock (America/New_York is not the same hour as UTC).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from parkwaits.lake import load_lake
from parkwaits.storage import connect, write_frame

ROLLING = "3h"
WAIT_FEATURES = [
    "wait_now",
    "wait_lag_30",
    "wait_lag_60",
    "wait_change_30",
    "wait_max_3h",
    "wait_mean_3h",
    "minutes_since_open",
    "staleness_min",
]
PARK_FEATURES = ["park_mean_wait", "park_open_frac", "park_n_open"]
HISTORY_FEATURES = ["ride_hist_mean_same_hour", "ride_hist_mean_day"]
TIME_FEATURES = ["local_minutes", "local_dow", "is_weekend"]
FEATURES = WAIT_FEATURES + PARK_FEATURES + HISTORY_FEATURES + TIME_FEATURES
KEYS = ["park_id", "ride_id"]


def _clean(snaps: pd.DataFrame) -> pd.DataFrame:
    need = ["park_id", "ride_id", "observed_at", "is_open", "wait_time", "source_updated"]
    missing = [c for c in need if c not in snaps.columns]
    if missing:
        raise ValueError(f"queue_times lake data is missing columns: {missing}")
    df = snaps.drop_duplicates(KEYS + ["observed_at"]).copy()
    df["observed_at"] = pd.to_datetime(df["observed_at"]).astype("datetime64[ns]")
    df["source_updated"] = pd.to_datetime(df["source_updated"]).astype("datetime64[ns]")
    df["is_open"] = df["is_open"].astype(int)
    df["wait_open"] = df["wait_time"].where(df["is_open"] == 1)
    return df.sort_values(KEYS + ["observed_at"]).reset_index(drop=True)


def _local_clock(df: pd.DataFrame, tz_by_park: dict[int, str]) -> pd.DataFrame:
    unknown = sorted(set(df["park_id"]) - set(tz_by_park))
    if unknown:
        raise ValueError(f"No timezone configured for park ids {unknown} (queue_times.parks[].timezone).")
    local = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
    for tz, idx in df.groupby(df["park_id"].map(tz_by_park)).groups.items():
        stamp = df.loc[idx, "observed_at"].dt.tz_localize("UTC").dt.tz_convert(tz)
        local.loc[idx] = stamp.dt.tz_localize(None)
    out = df.copy()
    out["local_time"] = local
    out["local_date"] = local.dt.normalize()
    out["local_hour"] = local.dt.hour
    out["local_minutes"] = local.dt.hour * 60 + local.dt.minute
    out["local_dow"] = local.dt.dayofweek
    out["is_weekend"] = (out["local_dow"] >= 5).astype(int)
    return out


def _lag(df: pd.DataFrame, minutes: int, name: str) -> pd.Series:
    past = df[KEYS + ["observed_at", "wait_open"]].copy()
    past["observed_at"] = past["observed_at"] + pd.Timedelta(minutes=minutes)
    merged = df[KEYS + ["observed_at"]].merge(past, on=KEYS + ["observed_at"], how="left")
    return pd.Series(merged["wait_open"].to_numpy(), index=df.index, name=name)


def _history_mean(df: pd.DataFrame, group: list[str], name: str) -> pd.Series:
    """Mean open wait over strictly earlier local dates within `group` (which excludes the date).

    Aggregated over ALL rows (closed ones add zero), so whether a day is present in the table never
    depends on the ride opening later that day."""
    daily = (
        df.groupby([*group, "local_date"])
        .agg(s=("wait_open", "sum"), c=("is_open", "sum"))
        .reset_index()
        .sort_values([*group, "local_date"])
    )
    cum = daily.groupby(group)[["s", "c"]].cumsum()
    daily["prior_mean"] = (cum["s"] - daily["s"]) / (cum["c"] - daily["c"]).replace(0, np.nan)
    merged = df[[*group, "local_date"]].merge(
        daily[[*group, "local_date", "prior_mean"]], on=[*group, "local_date"], how="left"
    )
    return pd.Series(merged["prior_mean"].to_numpy(), index=df.index, name=name)


def compute_features(snaps: pd.DataFrame, tz_by_park: dict[int, str]) -> pd.DataFrame:
    """Add features to every snapshot row. Accepts raw lake rows; returns a copy sorted by ride and time."""
    df = _local_clock(_clean(snaps), tz_by_park)

    df["wait_now"] = df["wait_open"]
    df["wait_lag_30"] = _lag(df, 30, "wait_lag_30")
    df["wait_lag_60"] = _lag(df, 60, "wait_lag_60")
    df["wait_change_30"] = df["wait_now"] - df["wait_lag_30"]

    rolled = df.groupby(KEYS).rolling(ROLLING, on="observed_at")["wait_open"]
    # _clean sorts by (park, ride, time) and groupby keeps that order, so the results line up by position.
    df["wait_max_3h"] = rolled.max().to_numpy()
    df["wait_mean_3h"] = rolled.mean().to_numpy()

    big = np.iinfo(np.int64).max
    ts = df["observed_at"].to_numpy().astype("int64")
    open_ts = pd.Series(np.where(df["is_open"] == 1, ts, big), index=df.index)
    first_open = open_ts.groupby([df["park_id"], df["ride_id"], df["local_date"]]).cummin()
    df["minutes_since_open"] = np.where(first_open == big, np.nan, (ts - first_open) / 60e9)
    df["staleness_min"] = ((df["observed_at"] - df["source_updated"]).dt.total_seconds() / 60).clip(0, 1440)

    at = df.groupby(["park_id", "observed_at"])
    df["park_mean_wait"] = at["wait_open"].transform("mean")
    df["park_open_frac"] = at["is_open"].transform("mean")
    df["park_n_open"] = at["is_open"].transform("sum")

    df["ride_hist_mean_same_hour"] = _history_mean(df, [*KEYS, "local_hour"], "ride_hist_mean_same_hour")
    df["ride_hist_mean_day"] = _history_mean(df, KEYS, "ride_hist_mean_day")
    return df


def build_training_frame(
    snaps: pd.DataFrame,
    tz_by_park: dict[int, str],
    *,
    horizon_minutes: int = 60,
    wait_threshold_min: float = 45,
    min_rows: int = 2000,
    min_positives: int = 50,
) -> pd.DataFrame:
    df = compute_features(snaps, tz_by_park)
    target = df[[*KEYS, "observed_at", "is_open", "wait_time"]].copy()
    target["observed_at"] = target["observed_at"] - pd.Timedelta(minutes=horizon_minutes)
    target = target.rename(columns={"is_open": "target_open", "wait_time": "target_wait"})
    df = df.merge(target, on=[*KEYS, "observed_at"], how="left")

    usable = (df["is_open"] == 1) & (df["target_open"] == 1)
    out = df[usable].copy()
    out["label"] = (out["target_wait"] >= wait_threshold_min).astype(int)
    out["event_ts"] = out["observed_at"]
    out = out[["event_ts", "label", "park_id", "ride_id", *FEATURES]].sort_values("event_ts")
    out = out.reset_index(drop=True)

    positives = int(out["label"].sum())
    if len(out) < min_rows or positives < min_positives:
        raise ValueError(
            f"Not enough labelled data: {len(out)} rows with {positives} positives "
            f"(need {min_rows} rows and {min_positives} positives). "
            "History only accumulates going forward from the first collection run: collect for longer."
        )
    return out


def persistence_auc(frame: pd.DataFrame, test_fraction: float = 0.2) -> float:
    """ROC AUC of the naive forecast 'the wait will stay what it is now' on the newest rows.

    This is the bar a model has to clear to be worth anything."""
    cut = int(len(frame) * (1 - test_fraction))
    test = frame.iloc[cut:]
    if test["label"].nunique() < 2:
        return float("nan")
    return float(roc_auc_score(test["label"], test["wait_now"].fillna(0)))


def build_training_table(cfg: dict[str, Any]) -> int:
    """Entry point for `parkwaits build` (configured as data.builder). Returns the training row count."""
    snaps = load_lake(cfg["lake"]["root"], "queue_times")
    if snaps.empty:
        raise ValueError("The lake has no 'queue_times' data. Run `collect queue_times` first.")
    qt, d = cfg["queue_times"], cfg["data"]
    frame = build_training_frame(
        snaps,
        {int(p["id"]): p["timezone"] for p in qt["parks"]},
        horizon_minutes=int(d.get("horizon_minutes", 60)),
        wait_threshold_min=float(d.get("wait_threshold_min", 45)),
        min_rows=int(d.get("min_rows", 2000)),
        min_positives=int(d.get("min_positives", 50)),
    )
    print(
        f"Positive rate {frame['label'].mean():.3f}. Persistence baseline AUC on newest 20%: "
        f"{persistence_auc(frame):.3f} (the model must beat this to add value)."
    )
    con = connect(d["db_path"])
    try:
        return write_frame(con, d["table"], frame)
    finally:
        con.close()

import numpy as np
import pandas as pd
import pytest
import yaml

from parkwaits.features import FEATURES, build_training_frame, compute_features, persistence_auc
from tests.synth import TZ, make_snapshots

NY = {1: "America/New_York"}


def ride(waits, opens=None, start="2026-08-03 12:00", step=10, park=1, ride_id=11, age_min=2):
    """One ride's snapshots. 12:00 UTC is 08:00 in New York (EDT)."""
    times = pd.date_range(start, periods=len(waits), freq=f"{step}min")
    opens = [1] * len(waits) if opens is None else opens
    return pd.DataFrame(
        {
            "park_id": park,
            "ride_id": ride_id,
            "observed_at": times,
            "is_open": opens,
            "wait_time": [float(w) if o else 0.0 for w, o in zip(waits, opens, strict=True)],
            "source_updated": times - pd.Timedelta(minutes=age_min),
        }
    )


def at(df, ts):
    return df[df["observed_at"] == pd.Timestamp(ts)].iloc[0]


def test_lags_changes_and_rolling_windows_hand_check():
    waits = [10, 20, 30, 40, 50, 60, 70, 80]  # 12:00 ... 13:10
    f = compute_features(ride(waits), NY)
    row = at(f, "2026-08-03 13:00")  # wait 70
    assert row["wait_now"] == 70 and row["wait_lag_30"] == 40 and row["wait_lag_60"] == 10
    assert row["wait_change_30"] == 30
    assert row["wait_max_3h"] == 70 and row["wait_mean_3h"] == pytest.approx(40.0)  # mean of 10..70
    assert row["minutes_since_open"] == 60 and row["staleness_min"] == 2
    first = at(f, "2026-08-03 12:00")
    assert np.isnan(first["wait_lag_30"]) and first["minutes_since_open"] == 0


def test_closed_snapshots_are_not_waits():
    f = compute_features(ride([10, 99, 30, 40], opens=[1, 0, 1, 1]), NY)
    assert np.isnan(at(f, "2026-08-03 12:10")["wait_now"])  # closed: no wait, whatever the raw number says
    row = at(f, "2026-08-03 12:30")
    assert row["wait_lag_30"] == 10 and np.isnan(row["wait_lag_60"])
    assert row["wait_mean_3h"] == pytest.approx(80 / 3)  # 10, 30, 40: the closed 99 is ignored
    assert row["wait_max_3h"] == 40


def test_park_features_use_only_the_same_instant():
    a = ride([10, 20], ride_id=1)
    b = ride([30, 0], opens=[1, 0], ride_id=2)
    f = compute_features(pd.concat([a, b]), NY)
    now = f[f["observed_at"] == pd.Timestamp("2026-08-03 12:10")]
    assert set(now["park_mean_wait"]) == {20.0} and set(now["park_open_frac"]) == {0.5}
    assert set(now["park_n_open"]) == {1}


def test_history_features_use_only_strictly_earlier_local_dates():
    day1 = ride([10] * 6, start="2026-08-03 12:00")
    day2 = ride([50] * 6, start="2026-08-04 12:00")
    day3 = ride([90] * 6, start="2026-08-05 12:00")
    f = compute_features(pd.concat([day1, day2, day3]), NY)
    assert at(f, "2026-08-03 12:00")[["ride_hist_mean_same_hour", "ride_hist_mean_day"]].isna().all()
    assert at(f, "2026-08-04 12:00")["ride_hist_mean_same_hour"] == 10
    assert at(f, "2026-08-05 12:00")["ride_hist_mean_same_hour"] == 30  # (10 + 50) / 2, never the 90 of today
    assert at(f, "2026-08-05 12:50")["ride_hist_mean_day"] == 30


def test_local_clock_follows_each_parks_timezone():
    a = ride([10], park=1, start="2026-08-03 18:30")
    b = ride([10], park=2, start="2026-08-03 18:30")
    f = compute_features(pd.concat([a, b]), TZ).set_index("park_id")
    assert f.loc[1, "local_minutes"] == 14 * 60 + 30  # EDT = UTC-4
    assert f.loc[2, "local_minutes"] == 11 * 60 + 30  # PDT = UTC-7
    assert f.loc[1, "local_dow"] == 0 and f.loc[1, "is_weekend"] == 0


def test_a_new_local_day_resets_minutes_since_open():
    # 03:50 UTC is 23:50 local on Aug 3; 04:00 UTC is local midnight, the start of Aug 4
    f = compute_features(ride([10, 10, 10], start="2026-08-04 03:50"), NY)
    assert at(f, "2026-08-04 03:50")["minutes_since_open"] == 0
    assert at(f, "2026-08-04 04:00")["minutes_since_open"] == 0  # new local date: the count restarts
    assert at(f, "2026-08-04 04:10")["minutes_since_open"] == 10


def test_unknown_park_timezone_is_reported():
    with pytest.raises(ValueError, match="No timezone"):
        compute_features(ride([10], park=99), NY)


def test_labels_use_the_observation_exactly_one_horizon_later():
    waits = [10, 20, 30, 40, 50, 60, 70, 80]
    out = build_training_frame(
        ride(waits), NY, horizon_minutes=30, wait_threshold_min=45, min_rows=1, min_positives=1
    ).set_index("event_ts")
    assert out.loc[pd.Timestamp("2026-08-03 12:00"), "label"] == 0  # target 12:30 -> 40
    assert out.loc[pd.Timestamp("2026-08-03 12:10"), "label"] == 1  # target 12:40 -> 50
    assert pd.Timestamp("2026-08-03 12:50") not in out.index  # target 13:20 was never observed


def test_rows_with_closed_or_missing_targets_are_dropped_not_guessed():
    waits = [10, 20, 30, 40, 50, 60]
    opens = [1, 1, 1, 0, 1, 1]
    snaps = ride(waits, opens)
    snaps = snaps[snaps["observed_at"] != pd.Timestamp("2026-08-03 12:50")]  # a collection gap
    out = build_training_frame(
        snaps, NY, horizon_minutes=20, wait_threshold_min=45, min_rows=1, min_positives=1
    )
    labels = out.set_index("event_ts")["label"].to_dict()
    # 12:10: target closed. 12:30: itself closed. 12:40: target never observed. 12:50: a gap.
    assert labels == {pd.Timestamp("2026-08-03 12:00"): 0, pd.Timestamp("2026-08-03 12:20"): 1}


def test_no_future_information_leaks_into_features():
    """Truncating the data at T must not change a single feature of any row at or before T."""
    snaps = make_snapshots(days=6, rides_per_park=3, seed=3)
    full = compute_features(snaps, TZ).set_index(["park_id", "ride_id", "observed_at"])
    rng = np.random.default_rng(0)
    times = np.sort(snaps["observed_at"].unique())
    for cut in rng.choice(times[200:-1], size=4, replace=False):
        cut = pd.Timestamp(cut)
        part = compute_features(snaps[snaps["observed_at"] <= cut], TZ).set_index(
            ["park_id", "ride_id", "observed_at"]
        )
        pd.testing.assert_frame_equal(
            part[FEATURES].astype(float),
            full.loc[part.index, FEATURES].astype(float),
            check_exact=False,
            obj=f"features truncated at {cut}",
        )


def test_refuses_to_train_on_too_little_history():
    with pytest.raises(ValueError, match="collect for longer"):
        build_training_frame(ride([10, 20, 30, 40, 50, 60]), NY, min_rows=2000)


def test_persistence_baseline_is_perfect_when_waits_never_change():
    frame = pd.DataFrame({"label": [0, 1] * 50, "wait_now": [10.0, 60.0] * 50})
    assert persistence_auc(frame) == 1.0


def test_config_feature_list_matches_code_and_excludes_identifiers():
    cfg = yaml.safe_load(open("config/config.yaml"))
    assert cfg["model"]["features"] == FEATURES
    assert {"park_id", "ride_id", "label", "event_ts", "target_wait"}.isdisjoint(FEATURES)
    assert "Queue-Times" not in str(cfg["model"]["features"])

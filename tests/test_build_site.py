"""The parks page: every claim is computed from the data, and the attribution is always on it.

No network and no trained model: a small stand-in model scores the synthetic snapshots.
"""

import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import build_site  # noqa: E402

from parkwaits.features import FEATURES, build_training_frame, compute_features  # noqa: E402
from tests.synth import TZ, make_snapshots  # noqa: E402


class WaitModel:
    """Stand-in for the champion: the chance rises with the current wait."""

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        p = np.clip(x["wait_now"].fillna(0).to_numpy() / 90, 0.001, 0.999)
        return np.column_stack([1 - p, p])


@pytest.fixture(scope="module")
def parts():
    snaps = make_snapshots(days=14)
    feats = compute_features(snaps, TZ)
    table = build_training_frame(snaps, TZ, min_rows=100, min_positives=10)
    return feats, table


CARD = {
    "champion": "lightgbm",
    "features": FEATURES,
    "metrics": {"all_candidates": {"lightgbm": {"roc_auc": 0.9, "avg_precision": 0.5}}},
}


def _page(parts, snapshot_ok=True):
    feats, table = parts
    names = {1: "Park One", 2: "Park Two"}
    model = WaitModel()
    fc = build_site.latest_forecast(feats, model, FEATURES, max_age_min=10_000)
    ev = build_site.holdout_curves(table, model, FEATURES, "label", "event_ts", 0.2)
    built = datetime(2026, 8, 17, 12, 0, tzinfo=UTC)
    forecast, banner = build_site.forecast_html(fc, feats, names, TZ, built, 10, 45, 60)
    return build_site.render(
        built=built,
        card=CARD,
        version="vtest",
        n_rides=int(feats[["park_id", "ride_id"]].drop_duplicates().shape[0]),
        n_snaps=len(feats),
        first=feats["observed_at"].min(),
        last=feats["observed_at"].max(),
        n_parks=2,
        forecast=forecast,
        banner=banner,
        profile=build_site.profile_html(build_site.hourly_profile(feats, 45, min_rows=20), 45),
        evidence=build_site.evidence_html(ev, CARD),
        threshold=45,
        horizon=60,
    ), ev


def test_page_keeps_the_attribution_and_the_snapshot_time(parts):
    page, _ = _page(parts)
    assert "Powered by Queue-Times.com" in page
    assert 'href="https://queue-times.com"' in page
    assert "Newest snapshot" in page and "UTC" in page  # a forecast never appears without its timestamp
    assert "<meta name=\"viewport\"" in page and "prefers-color-scheme: dark" in page


def test_evidence_headline_follows_the_numbers(parts):
    _, ev = _page(parts)
    gap = ev["auc_model"] - ev["auc_persistence"]
    text = build_site.evidence_html(ev, CARD)
    assert f"{ev['auc_model']:.3f}" in text and f"{ev['auc_persistence']:.3f}" in text
    if gap >= 0.01:
        assert "ranks long waits better than" in text
    elif gap <= -0.01:
        assert "ranks long waits better than the model" in text
    else:
        assert "about equally well" in text


def test_evidence_headline_flips_when_persistence_wins(parts):
    _, ev = _page(parts)
    flipped = {**ev, "auc_model": 0.60, "auc_persistence": 0.80}
    assert "ranks long waits better than the model" in build_site.evidence_html(flipped, CARD)
    tied = {**ev, "auc_model": 0.80, "auc_persistence": 0.805}
    assert "about equally well" in build_site.evidence_html(tied, CARD)


def test_forecast_only_scores_rides_open_at_a_fresh_snapshot(parts):
    feats, _ = parts
    fc = build_site.latest_forecast(feats, WaitModel(), FEATURES, max_age_min=30)
    newest = feats["observed_at"].max()
    assert (fc["is_open"] == 1).all()
    assert (fc["observed_at"] >= newest - pd.Timedelta(minutes=30)).all()
    assert fc["prob"].between(0, 1).all()


def test_page_shows_forecasts_and_aggregates_not_the_raw_feed(parts):
    feats, _ = parts
    fc = build_site.latest_forecast(feats, WaitModel(), FEATURES, max_age_min=10_000)
    page, _ = _page(parts)
    assert "Chance of 45+ min" in page
    assert "posted wait" in page  # says the probabilities are not the posted waits
    assert "wait_time" not in page and "wait_now" not in page
    assert len(fc) > 0


def test_no_open_rides_gives_a_message_not_a_crash(parts):
    feats, _ = parts
    closed = feats.assign(is_open=0)
    fc = build_site.latest_forecast(closed, WaitModel(), FEATURES)
    forecast, banner = build_site.forecast_html(
        fc, closed, {1: "Park One", 2: "Park Two"}, TZ, datetime(2026, 8, 17, tzinfo=UTC), 10, 45, 60
    )
    assert "nothing to forecast" in forecast and "No open rides" in banner


def test_hourly_profile_ignores_hours_when_little_of_the_park_is_open(parts):
    feats, _ = parts
    prof = build_site.hourly_profile(feats, 45, min_rows=20)
    assert prof is not None
    # The synthetic parks open 9:00 to 21:00 local; nothing outside that range may be charted.
    assert min(prof["hours"]) >= 9 and max(prof["hours"]) <= 20
    assert all(0 <= v <= 1 for v in prof["weekday"] + prof["weekend"])


def test_small_chances_are_not_all_shown_as_zero():
    assert build_site._chance(0.0004) == "<0.1%"
    assert build_site._chance(0.034) == "3.4%"
    assert build_site._chance(0.46) == "46%"


def test_ride_names_are_escaped(parts):
    feats, _ = parts
    fc = build_site.latest_forecast(feats, WaitModel(), FEATURES, max_age_min=10_000)
    fc["ride_name"] = "<script>alert(1)</script>"
    forecast, _ = build_site.forecast_html(
        fc, feats, {1: "Park One", 2: "Park Two"}, TZ, datetime(2026, 8, 17, tzinfo=UTC), 10, 45, 60
    )
    assert "<script>alert(1)</script>" not in forecast
    assert "&lt;script&gt;" in forecast


def test_snapshot_stamp_has_no_stray_zero_and_uses_each_parks_zone(parts):
    feats, _ = parts
    fc = build_site.latest_forecast(feats, WaitModel(), FEATURES, max_age_min=10_000)
    forecast, _ = build_site.forecast_html(
        fc, feats, {1: "Park One", 2: "Park Two"}, TZ, datetime(2026, 8, 17, tzinfo=UTC), 10, 45, 60
    )
    assert " / " in forecast  # the synthetic parks sit in two time zones, so both local times are shown
    stamp = forecast.split("Forecast made from the snapshot of <b>")[1].split("</b>")[0]
    local_part = stamp.split(" UTC (")[1]
    assert not any(part.strip().startswith("0") for part in local_part.rstrip(")").split(" / "))


def test_peak_hour_is_only_named_when_it_clearly_wins():
    hours = np.arange(9, 15, dtype=float)
    flat = np.array([0.10, 0.11, 0.111, 0.10, 0.10, 0.10])
    assert "no single peak hour" in build_site._peak_text(hours, flat)
    clear = np.array([0.05, 0.08, 0.15, 0.09, 0.07, 0.05])
    assert "highest around 11 AM" in build_site._peak_text(hours, clear)


def test_profile_text_comes_from_the_pooled_numbers(parts):
    feats, _ = parts
    prof = build_site.hourly_profile(feats, 45, min_rows=20)
    text = build_site.profile_html(prof, 45)
    assert f"{prof['weekday_pooled'] * 100:.0f}% of open-ride snapshots on weekdays" in text
    assert "cut-off was set after looking at this data" in text  # the busy-hours filter is disclosed
    assert prof["weekend_inside"] <= prof["weekend_days"]


def test_near_certain_catch_rates_are_not_shown_as_a_flat_100():
    assert build_site._caught(0.9995) == ">99%"
    assert build_site._caught(0.9954) == ">99%"
    assert build_site._caught(0.91) == "91%"

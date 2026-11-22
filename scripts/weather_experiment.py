# ruff: noqa: E501
"""Does weather help? The same LightGBM model with and without the weather columns, on the same rows.

Both versions are fitted on the same time-ordered training rows and scored on the same newest rows, so the
only difference is the five weather columns. Reported: 4 rolling folds inside the training window (the
project's own selection basis), the newest-20% test split, a day-block interval on the test AUC difference,
and the same comparison restricted to wet rows (rain in the last 3 hours), where weather should matter most.

Usage: python scripts/weather_experiment.py [--lake data/raw] [--out reports/weather_experiment.md]
The lake needs queue_times and weather (python -m parkwaits.cli collect weather).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from parkwaits.config import load_config  # noqa: E402
from parkwaits.features import FEATURES, build_training_frame  # noqa: E402
from parkwaits.lake import load_lake  # noqa: E402
from parkwaits.train import build_candidates, rolling_fold_scores, time_split  # noqa: E402
from parkwaits.uncertainty import day_bootstrap_gap  # noqa: E402
from parkwaits.weather import WEATHER_FEATURES  # noqa: E402

WET_MM = 0.5  # at least this much rain in the 3 hours before the snapshot counts as a wet row


def _metrics(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    if len(np.unique(y)) < 2:
        return float("nan"), float("nan")
    return float(roc_auc_score(y, p)), float(average_precision_score(y, p))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lake", default=None)
    ap.add_argument("--out", default="reports/weather_experiment.md")
    args = ap.parse_args()
    cfg = load_config()
    root = args.lake or cfg["lake"]["root"]
    snaps, weather = load_lake(root, "queue_times"), load_lake(root, "weather")
    if snaps.empty or weather.empty:
        print("The lake needs both queue_times and weather data.", file=sys.stderr)
        return 1
    tz = {int(p["id"]): p["timezone"] for p in cfg["queue_times"]["parks"]}
    d = cfg["data"]
    frame = build_training_frame(
        snaps, tz, horizon_minutes=int(d["horizon_minutes"]), wait_threshold_min=float(d["wait_threshold_min"]),
        min_rows=1, min_positives=1, calendar=cfg.get("calendar"), weather=weather,
    )
    known = frame[WEATHER_FEATURES].notna().all(axis=1)
    frame = frame[known].reset_index(drop=True)  # same rows for both versions: only rows with weather
    train_df, test_df = time_split(frame, "event_ts", cfg["model"]["test_fraction"])
    seed = int(cfg["model"]["random_state"])
    est = build_candidates(seed, ["lightgbm"])["lightgbm"]
    base, withw = list(FEATURES), [*FEATURES, *WEATHER_FEATURES]

    metric = "avg_precision"
    folds = {
        name: rolling_fold_scores(train_df, feats, "label", {"m": est}, metric, 4)["m"]
        for name, feats in (("without weather", base), ("with weather", withw))
    }
    models = {name: clone(est).fit(train_df[feats], train_df["label"].to_numpy()) for name, feats in (("without weather", base), ("with weather", withw))}
    y = test_df["label"].to_numpy()
    probs = {name: m.predict_proba(test_df[feats])[:, 1] for (name, m), feats in zip(models.items(), (base, withw), strict=True)}
    days = pd.to_datetime(test_df["event_ts"]).dt.strftime("%Y-%m-%d").to_numpy()
    iv = day_bootstrap_gap(y, probs["with weather"], probs["without weather"], days)
    wet = (test_df["rain_3h_mm"] >= WET_MM).to_numpy()
    imp = pd.Series(models["with weather"].named_steps["clf"].feature_importances_, index=withw)

    lines = [
        "# Weather experiment",
        "",
        f"Rows with weather: {len(frame):,} ({len(train_df):,} train, {len(test_df):,} test, {len(set(days))} test days). "
        f"Wet rows (at least {WET_MM} mm of rain in the 3 hours before): {wet.mean():.1%} of test rows, "
        f"{(train_df['rain_3h_mm'] >= WET_MM).mean():.1%} of train rows.",
        "",
        "## Rolling folds inside the training window (average precision per fold)",
        "| Version | " + " | ".join(f"fold {i + 1}" for i in range(len(folds["with weather"]))) + " | mean |",
        "|---|" + "---|" * (len(folds["with weather"]) + 1),
    ]
    for name, v in folds.items():
        lines.append(f"| {name} | " + " | ".join(f"{x:.4f}" for x in v) + f" | {np.mean(v):.4f} |")
    lines += ["", "## Newest-20% test split", "| Version | ROC AUC | Average precision | wet rows AUC | wet rows AP |", "|---|---|---|---|---|"]
    for name, p in probs.items():
        a, ap_ = _metrics(y, p)
        wa, wap = _metrics(y[wet], p[wet]) if wet.sum() else (float("nan"), float("nan"))
        lines.append(f"| {name} | {a:.4f} | {ap_:.4f} | {wa:.4f} | {wap:.4f} |")
    if iv:
        lines += [
            "",
            f"AUC with weather minus without: {iv['gap']:+.4f}, {iv['level']:.0%} day-block interval "
            f"{iv['low']:+.4f} to {iv['high']:+.4f} over {iv['days']} days; with weather ahead in {iv['share_ahead']:.0%} of re-draws.",
        ]
    lines += ["", f"Wet rows in the test split: {int(wet.sum()):,} ({int(y[wet].sum()):,} long waits).", "", "## What the model used (LightGBM split counts)"]
    lines += [f"- {k}: {int(v)}" for k, v in imp.sort_values(ascending=False).items() if k in WEATHER_FEATURES]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

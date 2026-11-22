"""Where is the model wrong? A forward test on data newer than anything the model saw, split into segments.

The promoted model is scored on rows after the last row of its own test window (card data.test_end), so
nothing here was used to train, select or test it. Every segment is compared with "the wait stays what it is
now". Prints a markdown report; writes it to --out when given.

  python scripts/error_analysis.py [--lake data/raw] [--out reports/error_analysis.md]
"""

# ruff: noqa: E501  (report sentences are kept whole so they read as written)
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from parkwaits.config import load_config
from parkwaits.features import build_training_frame
from parkwaits.lake import load_lake
from parkwaits.registry import load_current
from parkwaits.uncertainty import day_bootstrap_gap

MIN_ROWS = 300  # a segment smaller than this is not scored (too noisy to read)
MIN_POSITIVES = 20


def _score(g: pd.DataFrame) -> dict:
    y = g["label"].to_numpy()
    out = {
        "rows": len(g),
        "positives": int(y.sum()),
        "actual": float(y.mean()),
        "predicted": float(g["p"].mean()),
    }
    if len(g) >= MIN_ROWS and out["positives"] >= MIN_POSITIVES and out["positives"] < len(g):
        out["auc_model"] = float(roc_auc_score(y, g["p"]))
        out["auc_now"] = float(roc_auc_score(y, g["wait_now"].fillna(0)))
        out["ap_model"] = float(average_precision_score(y, g["p"]))
        out["ap_now"] = float(average_precision_score(y, g["wait_now"].fillna(0)))
    return out


def _table(
    df: pd.DataFrame, by: str, labels: dict | None = None, top: int | None = None, order: str | None = None
) -> str:
    rows = []
    for key, g in df.groupby(by, observed=True):
        s = _score(g)
        s["name"] = (labels or {}).get(key, key)
        rows.append(s)
    t = pd.DataFrame(rows)
    if order and order in t:
        t = t.sort_values(order)
    if top:
        t = t.head(top)
    lines = [
        "| segment | rows | long-wait cases | actual rate | model's average chance | AUC model / now | AP model / now |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in t.itertuples():
        d = r._asdict()
        auc = (
            f"{d['auc_model']:.3f} / {d['auc_now']:.3f}"
            if not np.isnan(d.get("auc_model", np.nan))
            else "n/a"
        )
        ap = f"{d['ap_model']:.3f} / {d['ap_now']:.3f}" if not np.isnan(d.get("ap_model", np.nan)) else "n/a"
        lines.append(
            f"| {d['name']} | {d['rows']:,} | {d['positives']:,} | {d['actual']:.1%} | {d['predicted']:.1%} | {auc} | {ap} |"
        )
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--lake", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    qt, d, m = cfg["queue_times"], cfg["data"], cfg["model"]
    tz = {int(p["id"]): p["timezone"] for p in qt["parks"]}
    names = {int(p["id"]): p["name"] for p in qt["parks"]}
    snaps = load_lake(args.lake or cfg["lake"]["root"], "queue_times")
    model, card, version = load_current(m["registry_dir"])

    frame = build_training_frame(
        snaps,
        tz,
        horizon_minutes=int(d.get("horizon_minutes", 60)),
        wait_threshold_min=float(d.get("wait_threshold_min", 45)),
        min_rows=100,
        min_positives=10,
        calendar=cfg.get("calendar"),
    )
    cutoff = pd.Timestamp(card["data"]["test_end"])
    new = frame[frame["event_ts"] > cutoff].copy()
    if new.empty:
        raise SystemExit(f"No labelled rows after the model's test window ended ({cutoff}).")
    new["p"] = model.predict_proba(new[card["features"]])[:, 1]
    ride_names = snaps.drop_duplicates(["park_id", "ride_id"]).set_index(["park_id", "ride_id"])["ride_name"]
    new["ride"] = [
        ride_names.get((p, r), str(r)) for p, r in zip(new["park_id"], new["ride_id"], strict=True)
    ]
    new["park"] = new["park_id"].map(names)
    hour = (new["local_minutes"] // 60).astype(int)
    new["time of day"] = pd.cut(
        hour,
        [-1, 10, 13, 16, 19, 24],
        labels=["before 11 AM", "11 AM-2 PM", "2-5 PM", "5-8 PM", "after 8 PM"],
    )
    new["day type"] = np.where(new["is_weekend"] == 1, "weekend", "weekday")
    new["wait now"] = pd.cut(
        new["wait_now"].fillna(0),
        [-1, 0, 15, 30, 45, 60, 1000],
        labels=["0 (walk-on)", "1-15", "16-30", "31-45", "46-60", "over 60"],
    )

    allr = _score(new)
    out = [
        f"# Error analysis: model {version} on data after {cutoff:%b %d %H:%M} UTC",
        "",
        f"Forward test: {allr['rows']:,} labelled rows from {new['event_ts'].min():%b %d} to {new['event_ts'].max():%b %d}, "
        f"{allr['positives']:,} long waits ({allr['actual']:.1%}). Nothing here was used to train, select or test the model.",
        f'Overall: AUC {allr["auc_model"]:.3f} vs {allr["auc_now"]:.3f} for "same as now"; '
        f"AP {allr['ap_model']:.3f} vs {allr['ap_now']:.3f}. The model's average chance is {allr['predicted']:.1%} "
        f"against an actual rate of {allr['actual']:.1%}.",
        "",
    ]
    iv = day_bootstrap_gap(
        new["label"].to_numpy(),
        new["p"].to_numpy(),
        new["wait_now"].fillna(0).to_numpy(),
        new["event_ts"].dt.strftime("%Y-%m-%d").to_numpy(),
    )
    if iv:
        out += [
            f'AUC lead over "same as now": {iv["gap"]:+.3f}, {iv["level"]:.0%} interval {iv["low"]:+.3f} to {iv["high"]:+.3f} '
            f"(day-block bootstrap, {iv['days']} days, {iv['draws']} draws; the model was ahead in {iv['share_ahead']:.0%} of draws).",
            "",
        ]
    # Sudden jumps: long waits that began well under the threshold
    thr = float(d.get("wait_threshold_min", 45))
    pos = new[new["label"] == 1]
    low = pos[pos["wait_now"].fillna(0) < thr - 15]
    out += [
        "## Sudden jumps",
        f"Of {len(pos):,} long waits, {len(low):,} ({len(low) / max(len(pos), 1):.1%}) started under {thr - 15:.0f} minutes an hour earlier. "
        f"The model gave those a median {low['p'].median():.1%} chance (the other long waits: {pos[~pos.index.isin(low.index)]['p'].median():.1%}); "
        f"{(low['p'] >= 0.5).mean():.1%} of them were flagged at 50% or more.",
        "",
    ]
    for title, by in [
        ("Park", "park"),
        ("Time of day", "time of day"),
        ("Weekday or weekend", "day type"),
        ("Wait right now", "wait now"),
    ]:
        out += [f"## By {title.lower()}", _table(new, by), ""]
    out += [
        "## Rides with the weakest ranking (at least 300 rows and 20 long waits; the worst of many, so some of the weakness is chance)",
        _table(new, "ride", order="auc_model", top=12),
        "",
    ]
    text = "\n".join(out)
    print(text)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text + "\n")


if __name__ == "__main__":
    main()

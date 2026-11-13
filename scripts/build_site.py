"""Build the static GitHub Pages site: which open rides are forecast to hit a 45-minute wait in an hour.

Run after `build` and `train` (the champion in models/ and the training table are used):

    python scripts/build_site.py --out site

Writes <out>/index.html and <out>/data.json. Features come from the same code as training
(parkwaits.features.compute_features), so every ride is scored only from its own recent waits, the
park-wide state and the park's local clock at the snapshot. Every headline and number on the page is
computed here from the data; nothing is typed in by hand. Only derived predictions and aggregates are
shown, never the raw Queue-Times feed.

Wait-time data is powered by Queue-Times.com (https://queue-times.com); the page keeps that attribution
visible with a link.
"""

# ruff: noqa: E501  (the HTML template below is easier to read unwrapped)
from __future__ import annotations

import argparse
import html
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from site_charts import CHART_CSS, CHART_JS, xy_chart
from sklearn.metrics import roc_auc_score, roc_curve

from parkwaits.config import load_config
from parkwaits.features import KEYS, compute_features
from parkwaits.lake import load_lake
from parkwaits.registry import load_current
from parkwaits.storage import connect, read_table
from parkwaits.train import time_split

REPO = "https://github.com/krisrchesney/park-wait-forecast"
QT = "https://queue-times.com"
MAX_SNAPSHOT_AGE_MIN = 30  # a ride whose newest snapshot is older than this (vs the newest overall) is not scored
MIN_CELL_ROWS = 300  # an hour-of-day cell with fewer open-ride snapshots than this is not charted
BREADTH_SHARE = 0.6  # an hour is charted only if, on average, at least this share of the peak-hour ride count is open


def _hour_label(h: float) -> str:
    h = int(round(h)) % 24
    return f"{(h % 12) or 12} {'AM' if h < 12 else 'PM'}"


def latest_forecast(
    feats: pd.DataFrame, model: object, feature_names: list[str], max_age_min: int = MAX_SNAPSHOT_AGE_MIN
) -> pd.DataFrame:
    """Score each ride's newest snapshot. Only rides open at that snapshot are scored, because the model
    answers "if the ride is still open". Rides whose newest snapshot is stale are left out."""
    if feats.empty:
        return feats.assign(prob=pd.Series(dtype=float))
    last = feats.loc[feats.groupby(KEYS)["observed_at"].idxmax()]
    newest = last["observed_at"].max()
    fresh = last[(last["observed_at"] >= newest - pd.Timedelta(minutes=max_age_min)) & (last["is_open"] == 1)]
    fresh = fresh.copy()
    if fresh.empty:
        return fresh.assign(prob=pd.Series(dtype=float))
    fresh["prob"] = model.predict_proba(fresh[feature_names])[:, 1]  # type: ignore[attr-defined]
    return fresh.sort_values("prob", ascending=False).reset_index(drop=True)


def hourly_profile(feats: pd.DataFrame, threshold: float, min_rows: int = MIN_CELL_ROWS) -> dict | None:
    """Share of open-ride snapshots at or above `threshold` minutes, by local hour, weekdays vs weekends.
    Returns None when no hour has enough rows in both groups."""
    op_all = feats[(feats["is_open"] == 1) & feats["wait_open"].notna()].copy()
    if op_all.empty:
        return None
    op_all["hit"] = (op_all["wait_open"] >= threshold).astype(float)
    # Only chart hours when much of the park is running. Late at night a handful of popular rides stay
    # open, so their share of long waits would look like a peak that is really a selection effect.
    breadth = op_all.groupby(["park_id", "observed_at", "local_hour"]).size().groupby("local_hour").mean()
    busy_hours = set(breadth[breadth >= BREADTH_SHARE * breadth.max()].index)
    op = op_all[op_all["local_hour"].isin(busy_hours)].copy()
    if op.empty:
        return None
    op["wk"] = np.where(op["local_dow"] >= 5, "weekend", "weekday")
    g = op.groupby(["wk", "local_hour"])["hit"].agg(["mean", "size"]).reset_index()
    ok = None
    for wk in ("weekday", "weekend"):
        hours = set(g[(g["wk"] == wk) & (g["size"] >= min_rows)]["local_hour"])
        ok = hours if ok is None else ok & hours
    if not ok:
        return None
    hours = sorted(ok)
    lo, hi = hours[0], hours[-1]
    hours = [h for h in range(lo, hi + 1) if h in ok]
    if len(hours) < 4:
        return None
    out: dict = {"hours": hours, "cutoff": BREADTH_SHARE}
    for wk in ("weekday", "weekend"):
        sub = g[g["wk"] == wk].set_index("local_hour")
        out[wk] = [float(sub.loc[h, "mean"]) for h in hours]
    kept = op[op["local_hour"].isin(hours)]
    for wk in ("weekday", "weekend"):
        out[f"{wk}_pooled"] = float(kept[kept["wk"] == wk]["hit"].mean())
    daily = kept.groupby(["wk", "local_date"])["hit"].mean()
    wdd, wed = daily["weekday"], daily["weekend"]
    out["weekday_days"], out["weekend_days"] = int(len(wdd)), int(len(wed))
    out["weekday_range"] = (float(wdd.min()), float(wdd.max()))
    out["weekend_inside"] = int(((wed >= wdd.min()) & (wed <= wdd.max())).sum())
    by_hour = op_all.groupby("local_hour")["hit"].agg(["mean", "size"])
    left_out = by_hour[~by_hour.index.isin(hours) & (by_hour["size"] >= min_rows)]
    out["left_out_max"] = float(left_out["mean"].max()) if len(left_out) else None
    out["rows"] = int(len(kept))
    return out


def holdout_curves(
    df: pd.DataFrame, model: object, feature_names: list[str], target: str, ts_col: str, test_fraction: float
) -> dict:
    """Champion vs persistence ("the wait stays what it is now") on the newest rows of the training table.

    ROC curves are resampled to a shared false-alarm grid so they can be charted together."""
    df = df.dropna(subset=[target])
    _, test = time_split(df, ts_col, test_fraction)
    y = test[target].to_numpy()
    if len(np.unique(y)) < 2:
        raise ValueError("The test window has only one class; cannot draw ROC curves.")
    scores = {
        "model": model.predict_proba(test[feature_names])[:, 1],  # type: ignore[attr-defined]
        "persistence": test["wait_now"].fillna(0).to_numpy(float),
    }
    grid = np.round(np.arange(0, 1.0001, 0.01), 4)
    out: dict = {"grid": grid, "n": int(len(test)), "positives": int(y.sum())}
    for name, s in scores.items():
        fpr, tpr, _ = roc_curve(y, s)
        out[name] = np.interp(grid, fpr, tpr)
        out[f"auc_{name}"] = float(roc_auc_score(y, s))
    out["persistence_levels"] = int(np.unique(scores["persistence"]).size)
    ts = pd.to_datetime(test[ts_col])
    out["start"], out["end"] = ts.min(), ts.max()
    out["days"] = (out["end"] - out["start"]).total_seconds() / 86400
    return out


def _chance(p: float) -> str:
    """Whole percents, but one decimal (or '<0.1%') for small chances so the table is not a column of 0%."""
    if p < 0.001:
        return "<0.1%"
    return f"{p * 100:.1f}%" if p < 0.1 else f"{p * 100:.0f}%"


def _caught(v: float) -> str:
    """Whole percents, but never show 99.5% to 99.99% as a flat 100%."""
    return ">99%" if 0.995 <= v < 1 else f"{v * 100:.0f}%"


def evidence_html(ev: dict, card: dict) -> str:
    grid = ev["grid"]
    at = lambda key, p: float(np.interp(p, grid, ev[key])) * 100  # noqa: E731
    gap = ev["auc_model"] - ev["auc_persistence"]
    gap_text = f"a difference of {gap:+.3f} for the model"
    if gap >= 0.01:
        lead = "On recent held-out data the model ranks long waits better than “same as now”"
    elif gap <= -0.01:
        lead = "On recent held-out data “same as now” ranks long waits better than the model"
    else:
        lead = "On recent held-out data the model and “same as now” rank long waits about equally well"
    series = [
        {"name": "This model", "short": "model", "var": "--s1", "y": ev["model"]},
        {"name": "“Same as now”", "short": "same as now", "var": "--s2", "y": ev["persistence"]},
    ]
    chart = xy_chart(
        "roc",
        "Held-out test window: long waits caught vs false alarms",
        grid,
        series,
        xlabel="false-alarm rate (rides that stayed under 45 min but were flagged)",
        ylabel="45+ min cases caught",
        xmin=0,
        xmax=1,
        ymax=1,
        xtick_fmt=lambda v: f"{v * 100:.0f}%",
        ytick_fmt=lambda v: f"{v * 100:.0f}%",
        labels=[f"False-alarm rate {g * 100:.0f}%" for g in grid],
        value_fmt=_caught,
        table_x=[0.01, 0.05, 0.10, 0.20, 0.50],
        table_head=["1% false alarms", "5%", "10%", "20%", "50%"],
        table_row_label="Forecast",
        ref_label="random guess",
        dec=1,
        unit="",
        end_labels=False,  # both curves end at (100%, 100%), so an end label would mislead
    )
    start, end = ev["start"], ev["end"]
    return (
        f"<h3>{lead}</h3>"
        f"<p>Test window: the newest 20% of labelled rows, {start:%b %d} to {end:%b %d, %Y} UTC "
        f"({ev['days']:.1f} days, {ev['n']:,} rows, <b>{ev['positives']:,}</b> of them 45+ minutes an hour later). "
        f"ROC AUC: model <b>{ev['auc_model']:.3f}</b>, “same as now” <b>{ev['auc_persistence']:.3f}</b> "
        f"({gap_text}). "
        f"At a 10% false-alarm rate the model catches <b>{at('model', 0.10):.0f}%</b> of the 45+ minute cases "
        f"vs <b>{at('persistence', 0.10):.0f}%</b> for “same as now”. "
        "Move along the chart to read other false-alarm rates; the dashed line is a random guess.</p>"
        + chart
        + f'<p class="muted">“Same as now” takes only {ev["persistence_levels"]} distinct scores, so its curve is straight lines between its real operating points and its numbers at a given false-alarm rate are interpolated.</p>'
        + '<p class="muted">Read the margin as suggestive: it comes from one short window with no confidence '
        "interval, and consecutive snapshots of a ride are highly correlated. Persistence is scored by "
        "ranking on the current wait; it is the bar a forecast has to clear.</p>"
        f'<p class="muted">Disclosure: nothing here is tuned on this window. The champion ({html.escape(card["champion"])}) '
        "was chosen on rolling folds inside the training rows, before the test split is cut; the page's "
        "model is retrained by the same code every time new data lands, so these figures move as history grows.</p>"
    )


def _peak_text(hours: np.ndarray, a: np.ndarray) -> str:
    """Name a peak hour only when it clearly beats the next-best hour (by at least a point)."""
    order = np.argsort(a)[::-1]
    if len(a) > 1 and a[order[0]] - a[order[1]] < 0.01:
        return "no single peak hour (the top two hours are within a point of each other)"
    return f"highest around {_hour_label(hours[order[0]])} ({a[order[0]] * 100:.0f}%)"


def profile_html(prof: dict | None, threshold: float) -> str:
    if prof is None:
        return (
            "<p>Not enough history yet to chart waits by hour of day for both weekdays and weekends.</p>"
        )
    hours = np.array(prof["hours"], float)
    wd, we = np.array(prof["weekday"]), np.array(prof["weekend"])
    ymax = max(0.1, float(np.ceil(max(wd.max(), we.max()) * 10) / 10))
    series = [
        {"name": f"Weekdays ({prof['weekday_days']} days)", "short": "weekdays", "var": "--s1", "y": wd},
        {"name": f"Weekends ({prof['weekend_days']} days)", "short": "weekends", "var": "--s2", "y": we},
    ]
    pick = [int(h) for h in hours[:: max(1, len(hours) // 4)]]
    chart = xy_chart(
        "hourly",
        f"Share of open-ride snapshots at {threshold:.0f}+ minutes, by local hour",
        hours,
        series,
        xlabel="local hour of day",
        ylabel=f"open rides at {threshold:.0f}+ min",
        xmin=float(hours[0]),
        xmax=float(hours[-1]),
        ymax=ymax,
        xtick_fmt=_hour_label,
        ytick_fmt=lambda v: f"{v * 100:.0f}%",
        labels=[_hour_label(h) for h in hours],
        value_fmt=lambda v: f"{v * 100:.0f}%",
        table_x=[float(h) for h in pick],
        table_head=[_hour_label(h) for h in pick],
        table_row_label="Share at threshold",
        dec=1,
        unit="",
        xticks=min(6, len(hours)),
    )
    wp, ep = prof["weekday_pooled"], prof["weekend_pooled"]
    span = f"{_hour_label(hours[0])} to {_hour_label(hours[-1])}"
    diff = ep - wp
    if abs(diff) < 0.01:
        cmp = "about the same on weekends as on weekdays"
    else:
        cmp = f"{abs(diff) * 100:.0f} points {'higher' if diff > 0 else 'lower'} on weekends than on weekdays"
    lo, hi = prof["weekday_range"]
    out = (
        f"<h3>{wp * 100:.0f}% of open-ride snapshots on weekdays and {ep * 100:.0f}% on weekends were at "
        f"{threshold:.0f}+ minutes</h3>"
        f"<p>Pooled over the hours {span} (local), across all rides and {prof['rows']:,} open-ride snapshots, "
        f"the share is {cmp}. By hour, weekdays: {_peak_text(hours, wd)}. Weekends: {_peak_text(hours, we)}. "
        f"The weekend figure rests on only {prof['weekend_days']} days, and {prof['weekend_inside']} of them fall "
        f"inside the range of the {prof['weekday_days']} weekdays ({lo * 100:.0f}% to {hi * 100:.0f}% a day), "
        "so treat it as a hint: holidays and school terms may be driving it.</p>"
    )
    left = prof.get("left_out_max")
    extra = f" (up to {left * 100:.0f}% in the hours left out)" if left is not None else ""
    note = (
        f'<p class="muted">Hours are charted only when at least {prof["cutoff"] * 100:.0f}% as many rides are open '
        f"as in the busiest hour ({span}). Later and earlier hours are left out because only a few popular rides are "
        f"open then, which inflates the share{extra}. This cut-off was set after looking at this data, and moving it "
        "moves the peak hour.</p>"
    )
    return out + chart + note


def park_table(fc: pd.DataFrame, feats: pd.DataFrame, names: dict[int, str], threshold: float) -> str:
    op = feats[(feats["is_open"] == 1) & feats["wait_open"].notna()]
    share = op.assign(hit=(op["wait_open"] >= threshold).astype(float)).groupby("park_id")["hit"].mean()
    rows = []
    for pid, name in names.items():
        sub = fc[fc["park_id"] == pid]
        hist = f"{share[pid] * 100:.1f}%" if pid in share.index else ""
        rows.append(
            f"<tr><td>{html.escape(name)}</td>"
            f'<td class="num">{len(sub)}</td>'
            f'<td class="num">{int((sub["prob"] >= 0.5).sum()) if len(sub) else 0}</td>'
            f'<td class="num">{sub["prob"].sum():.1f}</td>'
            f'<td class="num">{hist}</td></tr>'
        )
    return (
        f'<div class="wrap"><table style="min-width:620px"><thead><tr><th>Park</th><th class="num">Rides open at the snapshot</th>'
        f'<th class="num">Forecast 50%+ to reach {threshold:.0f} min</th><th class="num">Expected rides at {threshold:.0f}+ min</th>'
        f'<th class="num">History: open-ride snapshots at {threshold:.0f}+ min</th></tr></thead><tbody>{"".join(rows)}</tbody></table></div>'
    )


def forecast_html(
    fc: pd.DataFrame, feats: pd.DataFrame, names: dict[int, str], tz: dict[int, str], built: datetime, top: int,
    threshold: float, horizon: int,
) -> tuple[str, str]:
    """Returns (section html, banner for the page hero)."""
    if fc.empty:
        newest = feats["observed_at"].max() if len(feats) else None
        when = f"{newest:%Y-%m-%d %H:%M} UTC" if newest is not None else "n/a"
        msg = (
            f"<p>No ride was open at the newest snapshot ({when}), so there is nothing to forecast: the model "
            "answers the question only for rides that are open.</p>"
        )
        return msg, f"No open rides at the newest snapshot ({when})."
    snap = fc["observed_at"].max()
    zones = sorted({tz[int(p)] for p in fc["park_id"].unique()})
    local_txt = " / ".join(
        pd.Timestamp(snap).tz_localize("UTC").tz_convert(z).strftime("%I:%M %p %Z, %a").lstrip("0") for z in zones
    )
    age_h = (pd.Timestamp(built.replace(tzinfo=None)) - snap).total_seconds() / 3600
    k, n = int((fc["prob"] >= 0.5).sum()), len(fc)
    if age_h > 7:
        stale = (
            f' <b>That snapshot is {age_h:.1f} hours old, older than a normal collection batch, so collection '
            "may be behind.</b>"
        )
    else:
        stale = ""
    stamp = f"{snap:%Y-%m-%d %H:%M} UTC ({local_txt})"
    banner = f"Newest snapshot {snap:%Y-%m-%d %H:%M} UTC, {age_h:.1f} h old when this page was built."
    if k == 0:
        head = f"<h3>No open ride is forecast at 50% or more to reach {threshold:.0f} minutes an hour after the snapshot</h3>"
    else:
        head = (
            f"<h3>{k} of {n} open rides are forecast at 50% or more to reach {threshold:.0f} minutes "
            f"an hour after the snapshot</h3>"
        )
    intro = (
        f"<p>Forecast made from the snapshot of <b>{stamp}</b>, for {horizon} minutes later. Data lands in "
        f"batches of about six hours, so this “next hour” can be hours behind the clock.{stale} "
        "Probabilities come from the model; they are not the posted waits.</p>"
    )
    rows = "".join(
        "<tr>"
        f'<td class="num">{i}</td><td>{html.escape(str(r.ride_name))}</td><td>{html.escape(names.get(int(r.park_id), str(r.park_id)))}</td>'
        f'<td class="num">{_chance(r.prob)}</td></tr>'
        for i, r in enumerate(fc.head(top).itertuples(), 1)
    )
    table = (
        f"<h3 id=\"ranked\">The {min(top, n)} open rides most likely to reach {threshold:.0f}+ minutes</h3>"
        '<div class="wrap"><table style="min-width:480px"><thead><tr><th class="num">Rank</th><th>Ride</th><th>Park</th>'
        f'<th class="num">Chance of {threshold:.0f}+ min in {horizon} min</th></tr></thead><tbody>{rows}</tbody></table></div>'
    )
    return head + intro + table + "<h3>By park</h3>" + park_table(fc, feats, names, threshold), banner


def render(
    *, built: datetime, card: dict, version: str, n_rides: int, n_snaps: int, first: pd.Timestamp, last: pd.Timestamp,
    n_parks: int, forecast: str, banner: str, profile: str, evidence: str, threshold: float, horizon: int,
    snapshot_minutes: int = 10,
) -> str:
    m = card["metrics"]["all_candidates"][card["champion"]]
    days = (last - first).total_seconds() / 86400
    stamp = built.strftime("%Y-%m-%d %H:%M UTC")
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Park Waits: next-hour forecast</title>
<meta name="description" content="Which open theme-park rides are forecast to reach a {threshold:.0f}-minute posted wait in an hour, with how the forecast compares to 'same as now'. Powered by Queue-Times.com.">
<style>
:root {{ --bg:#fbfbfa; --fg:#1d1d1b; --muted:#66655f; --line:#e3e2dc; --accent:#9b3d12; --tag:#fde8d9; --row:#f3f2ee; --card:#f3f2ee; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#151513; --fg:#ecebe6; --muted:#a3a29b; --line:#33322e; --accent:#f0915a; --tag:#4a2a17; --row:#1e1e1b; --card:#1e1e1b; }} }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--fg); font:15px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif; }}
main {{ max-width:1040px; margin:0 auto; padding:32px 16px 48px; }}
h1 {{ font-size:1.7rem; margin:0 0 6px; letter-spacing:-.01em; }}
h2 {{ font-size:1.25rem; margin:40px 0 8px; padding-top:16px; border-top:1px solid var(--line); }}
h3 {{ font-size:1rem; margin:28px 0 6px; }}
p, li {{ max-width:72ch; }}
.lede {{ font-size:1.05rem; max-width:68ch; }}
.muted {{ color:var(--muted); }}
a {{ color:var(--accent); }}
.guide {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(260px, 1fr)); gap:14px; margin:20px 0; }}
.guide section {{ background:var(--card); border-radius:8px; padding:14px 16px; }}
.guide h3 {{ margin:0 0 6px; }}
.guide ol, .guide ul {{ margin:0; padding-left:20px; }}
.guide li {{ margin:4px 0; }}
.stats {{ display:flex; flex-wrap:wrap; gap:12px; margin:20px 0; }}
.stat {{ border:1px solid var(--line); border-radius:8px; padding:10px 14px; min-width:150px; }}
.stat b {{ display:block; font-size:1.3rem; }}
.wrap {{ overflow-x:auto; border:1px solid var(--line); border-radius:8px; }}
table {{ border-collapse:collapse; width:100%; min-width:520px; font-size:.92rem; }}
th, td {{ padding:7px 10px; border-bottom:1px solid var(--line); text-align:left; white-space:nowrap; }}
th {{ font-weight:600; color:var(--muted); background:var(--row); }}
td.num, th.num {{ text-align:right; font-variant-numeric:tabular-nums; }}
tr:last-child td {{ border-bottom:0; }}
footer {{ margin-top:40px; font-size:.85rem; }}
.hero {{ display:grid; grid-template-columns:minmax(0,68ch) minmax(220px,1fr); gap:28px; align-items:start; }}
.hero .stats {{ flex-direction:column; margin:6px 0 0; }}
.hero .stat {{ min-width:0; }}
@media (max-width:820px) {{ .hero {{ grid-template-columns:1fr; gap:0; }} .hero .stats {{ flex-direction:row; }} .hero .stat {{ min-width:140px; flex:1 1 140px; }} }}
{CHART_CSS}
</style>
</head>
<body>
<main>
<h1>Park Waits</h1>
<div class="hero">
<div>
<p class="lede">Will a ride's posted wait be {threshold:.0f} minutes or more an hour from now? This page scores every
open ride at the {n_parks} parks it tracks from its recent waits, what the rest of the park is doing and the local time of day,
then checks the forecast against the simplest alternative: “the wait stays what it is now.”</p>
<p class="muted">Forecast for {horizon} minutes after the newest snapshot. Built {html.escape(stamp)}. {html.escape(banner)}
Wait-time data is <a href="{QT}">Powered by Queue-Times.com</a>.</p>
</div>
<div class="stats">
  <div class="stat"><b>{n_rides:,}</b>rides and attractions across {n_parks} parks</div>
  <div class="stat"><b>{n_snaps:,}</b>snapshots over {days:.0f} days ({first:%b %d} to {last:%b %d, %Y})</div>
  <div class="stat"><b>{html.escape(card["champion"])}</b>model <code>{html.escape(version)}</code></div>
  <div class="stat"><b>{m["roc_auc"]:.3f}</b>test ROC AUC (avg precision {m["avg_precision"]:.3f})</div>
</div>
</div>

<div class="guide">
<section>
<h3>How it works</h3>
<ul>
<li>Every {snapshot_minutes} minutes the posted wait of each ride is recorded from <a href="{QT}">Queue-Times.com</a>.</li>
<li>A model trained on this history, split by time so it never sees the future, estimates the chance that a ride that
is open now is still open in an hour with a wait of {threshold:.0f} minutes or more.</li>
<li>The page is rebuilt from the newest data each time a collection batch finishes.</li>
</ul>
</section>
<section>
<h3>How to use it</h3>
<ol>
<li>Read the <a href="#forecast">forecast</a> for the park you plan to visit, and check the snapshot time first.</li>
<li>Treat a high chance as “expect a line”, not a promise; rides far from {threshold:.0f} minutes show a low chance.</li>
<li>Use <a href="#does-it-work">Does it work?</a> to see how much better it is than assuming nothing changes.</li>
</ol>
</section>
<section>
<h3>What it can't do</h3>
<ul>
<li>It forecasts only rides that are open now; it says nothing about closures.</li>
<li>Posted waits are park-published estimates, not measured waits.</li>
<li>History is short, so holidays, school terms and weather are not yet learned.</li>
</ul>
</section>
</div>

<h2 id="forecast">Forecast</h2>
{forecast}

<h2 id="does-it-work">Does it work?</h2>
<p>A forecast is only worth using if it beats “the wait stays the same”. For each ride the model could have been
asked at a past moment, did it rank the cases that really reached {threshold:.0f}+ minutes an hour later above the
ones that did not?</p>
{evidence}

<h2 id="patterns">When waits run long</h2>
{profile}

<p><a href="data.json">Download this page's forecasts (JSON)</a> · <a href="{REPO}#results">Full results and caveats in the README</a></p>

<footer class="muted">
<p>Wait-time data is <a href="{QT}">Powered by Queue-Times.com</a>. Code: <a href="{REPO}">github.com/krisrchesney/park-wait-forecast</a>.
This page shows model forecasts and aggregates only, not the raw feed. Not affiliated with Queue-Times.com or any park.</p>
</footer>
</main>
<div id="tip" role="tooltip"></div>
<script>{CHART_JS}</script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--out", default="site")
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()

    cfg = load_config(args.config)
    qt, d, mcfg = cfg["queue_times"], cfg["data"], cfg["model"]
    names = {int(p["id"]): p["name"] for p in qt["parks"]}
    tz = {int(p["id"]): p["timezone"] for p in qt["parks"]}
    threshold, horizon = float(d.get("wait_threshold_min", 45)), int(d.get("horizon_minutes", 60))

    snaps = load_lake(cfg["lake"]["root"], "queue_times")
    if snaps.empty:
        raise SystemExit("The lake has no queue_times data. Run collect (or restore the data branch) first.")
    model, card, version = load_current(mcfg["registry_dir"])
    feats = compute_features(snaps, tz, cfg.get("calendar"))
    fc = latest_forecast(feats, model, card["features"])

    con = connect(d["db_path"])
    try:
        table = read_table(con, d["table"], order_by=d["timestamp_col"])
    finally:
        con.close()
    ev = holdout_curves(table, model, card["features"], mcfg["target"], d["timestamp_col"], mcfg["test_fraction"])

    built = datetime.now(UTC)
    forecast, banner = forecast_html(fc, feats, names, tz, built, args.top, threshold, horizon)
    html_text = render(
        built=built,
        card=card,
        version=version,
        n_rides=int(feats[KEYS].drop_duplicates().shape[0]),
        n_snaps=int(len(feats)),
        first=feats["observed_at"].min(),
        last=feats["observed_at"].max(),
        n_parks=int(feats["park_id"].nunique()),
        forecast=forecast,
        banner=banner,
        profile=profile_html(hourly_profile(feats, threshold), threshold),
        evidence=evidence_html(ev, card),
        threshold=threshold,
        horizon=horizon,
        snapshot_minutes=int(qt.get("snapshot_minutes", 10)),
    )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "index.html").write_text(html_text, encoding="utf-8")

    records = [
        {
            "park": names.get(int(r.park_id), str(r.park_id)),
            "ride": str(r.ride_name),
            "chance_reach_threshold": round(float(r.prob), 4),
            "snapshot_utc": f"{r.observed_at:%Y-%m-%dT%H:%M:%S}",
        }
        for r in fc.itertuples()
    ]
    payload = {
        "built": built.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model_version": version,
        "champion": card["champion"],
        "threshold_min": threshold,
        "horizon_min": horizon,
        "attribution": "Powered by Queue-Times.com (https://queue-times.com)",
        "forecasts": records,
    }
    (out / "data.json").write_text(json.dumps(payload), encoding="utf-8")
    print(f"Wrote {out / 'index.html'} and {out / 'data.json'}: {len(fc)} open rides scored")


if __name__ == "__main__":
    main()

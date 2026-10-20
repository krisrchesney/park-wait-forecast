# park-wait-forecast

Forecasts, for an open theme-park ride, whether its posted wait will be **at least 45 minutes one hour from now**. Trained on ride wait snapshots collected every 10 minutes, served through a validated FastAPI endpoint, and monitored for data and performance drift.

Wait-time data is powered by [Queue-Times.com](https://queue-times.com). If you publish a page, app or write-up that shows this data, keep the attribution "Powered by Queue-Times.com" visible with a link.

> Status: pipeline and tests are complete. **No real-data results are published here yet**: collection starts on your GitHub account, and the source only provides current values, so history exists only from your first run. Numbers will be added to Results after enough history; none are claimed before it.

## Why this project

It is a real forecasting problem with a strong naive baseline (waits change slowly), which makes it a good test of honest evaluation: a model that cannot beat "the wait stays the same" is not worth shipping, and the build prints that baseline every time.

## How it works

```
Queue-Times API --collect-loop (every 10 min, one CI job)--> Parquet lake (data/raw/queue_times)
                                   |  pushed once per job to an orphan `data` branch by scripts/data_sync.sh
                                   v
       build (lags, rolling waits, park-wide state, local clock, prior-day history) --> DuckDB
                                   v
        train: logistic baseline vs HistGradientBoosting vs LightGBM  (MLflow)
                                   v
              models/<version>/ + model card  -->  FastAPI /predict /explain
                                   v
        drift: PSI per feature + live AUC after the training cutoff --> retrain flag
```

## Leakage and bias controls

| Risk | Control | Proof |
|---|---|---|
| Features that see the future | Lags look back, rolling windows end at the snapshot, history uses strictly earlier local dates, "minutes since open" is a running minimum | `test_no_future_information_leaks_into_features` truncates the data at random times and requires identical features |
| Guessing missing targets | Rows with an unobserved target (collection gaps, end of data) are dropped, not imputed | `test_rows_with_closed_or_missing_targets_are_dropped_not_guessed` |
| Closed rides mixed with short waits | Rows whose ride is closed at the target time are excluded and the model card says the question is conditional on staying open | same test, model card limitations |
| Closed snapshots counted as zero waits | A closed ride has no wait; its raw number is ignored in lags and rolling windows | `test_closed_snapshots_are_not_waits` |
| Wrong clock | Time features use each park's local timezone | `test_local_clock_follows_each_parks_timezone` |
| Trivial model | The build prints the AUC of "the wait stays what it is now" on the newest 20%; beat it or do not ship | `test_persistence_baseline_is_perfect_when_waits_never_change` |
| Random splits and monitoring on training data | Time-ordered holdout; drift uses only rows after the training cutoff | template tests |

## Setup

Requires Python 3.11+.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest                                           # includes a no-network end-to-end run
python -m parkwaits.cli collect queue_times      # one snapshot of every configured park
python -m parkwaits.cli collect-loop --minutes 120   # or poll for two hours in one process
# after at least a few days of history:
pip install -e ".[train,explain,monitor]"
python -m parkwaits.cli build && python -m parkwaits.cli train
python -m parkwaits.cli drift && python -m parkwaits.cli explain
python -m parkwaits.cli serve                    # http://localhost:8000/docs
```

Parks are listed in `config/config.yaml` with their timezones (checked against the API's `/parks.json`). Add parks by id and timezone.

## Decisions and tradeoffs

- **Classification at a threshold, not regression.** "45 minutes or more" is the decision a visitor makes. A regression model on the exact minute is a sensible extension.
- **One long CI job instead of many scheduled runs.** Pushing history after every snapshot would add about 100 commits a day. The loop polls inside one job and pushes once.
- **Insert-only snapshots keyed by (park, ride, 10-minute bucket).** Re-running in the same bucket stores nothing new.
- **Missing data is dropped, not filled.** Gaps shrink the training set but never invent targets.
- **DuckDB + Parquet, not a warehouse.** Free and local; no managed-warehouse experience is claimed.

## Results

_Not yet generated._ After enough real history, add: positive rate, the persistence baseline AUC printed by `build`, ROC AUC / average precision / ECE for the baseline and champion, the SHAP chart from `reports/`, and the Evidently drift report.

## Limitations

- Only a few weeks of history at first: no seasonality, holidays or school-term effects.
- Conditional on the ride staying open.
- Posted waits are park-published estimates, not measured waits.
- Scheduled GitHub runs can start late or be skipped, leaving gaps.
- The 60-minute label overlaps the first hour of the test period (negligible at weeks of data).

## Data source

[Queue-Times.com](https://queue-times.com). Check their current API terms for polling guidance, commercial use and attribution before you publish anything built on it; I could not verify them from this environment.

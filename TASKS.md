# park-wait-forecast backlog

The shared board for every session. Statuses: Ready -> In progress -> Review -> Done.
A ticket owns the files/functions it lists; two tickets in progress must never own the same thing.

Template:

    ### T-<n>: <title>
    Status: Ready | In progress (branch) | Review (branch) | Done
    Owns: <files, functions or sections>
    Blocked by: <ticket or "none">
    Why: <one or two lines>
    Done when:
    - [ ] <something a reviewer can check without asking anyone>

## Ready

### T-4: Weather features
Status: Ready
Owns: new weather module and tests; config
Blocked by: user go-ahead to build (scoping finished 2026-11-22, below)
Why: Rain, heat and cool days change both crowds and waits. Use observed conditions at the snapshot or a forecast issued before it, never later weather.
Scoping (2026-11-22, read from the sources' own pages):
- Recommended source: Open-Meteo (free API, no key). Hourly temperature_2m, apparent_temperature, precipitation, weather_code, cloud_cover. Historical endpoint is reanalysis (ERA5 0.25 degree plus IFS 9 km in best_match), so it is modelled "observed" weather, not station readings. Newest data lags about 5 days for ERA5 but best_match fills the gap; the Forecast API's past_days covers yesterday.
- Terms: free for non-commercial use (under 10,000 calls a day, 600 a minute), data under CC BY 4.0, so the page needs a credit link to open-meteo.com. A free site with no ads or subscriptions fits their examples. Caveat: they list "promotional activities" as commercial, and this repo is a portfolio, so decide whether that applies before publishing. Licence page not read; confirm the exact credit wording.
- Rejected: NWS api.weather.gov. Open data with a required User-Agent, but it keeps only 7 days of station observations and serves no past forecasts, so it cannot backfill Oct 20 onward and past forecasts cannot be recovered.
- Leakage design: use weather at or before the snapshot hour only (the reanalysis value for that hour and earlier hours, such as rain in the last 3 hours). Never use the next hour's weather as if it were a forecast. A true "forecast issued before" would need a stored forecast, and we have none. Training on reanalysis but serving live from a forecast model is a small train/serve mismatch: disclose it.
- Cost to build: one fetch script (one call per park location per day, far under the limits), a cached weather table on the data branch, 4 to 6 features, a leakage test, and a rolling-fold before/after run. About 25 to 35 turns.
- Expected value is modest: 33 days include only a few rainy days, so a gain may not be measurable. Judge it on rain days specifically and report if it is within noise.
Done when:
- [x] Source terms and attribution checked and recorded (see Scoping)
- [ ] Leakage test: a feature never uses weather observed after the snapshot
- [ ] Before/after comparison on rolling folds, disclosed

### T-5: Park event nights and park hours
Status: Ready
Owns: to be decided
Blocked by: an official, machine-readable source of event-night dates and park hours (none found for Queue-Times)
Why: Halloween and Christmas party nights and extended hours change evening crowds.
Done when:
- [ ] A sourced date list exists and its terms are checked

## In progress

## Review

## Done

### T-6: Plain-language page with technical drop-downs
Status: Done (merged 2026-11-22, 1499a83)
Owns: scripts/build_site.py; tests/test_build_site.py
Why: Plain sentences are the face of the page; the technical detail stays, in drop-downs.

### T-7: Error analysis on newer data
Status: Done (merged 2026-11-22, 1499a83)
Owns: scripts/error_analysis.py
Why: Find where the model is weak: by park, time of day, wait band, ride and sudden jumps.

### T-8: Interval on the AUC lead
Status: Done (merged 2026-11-22, 1499a83)
Owns: src/parkwaits/uncertainty.py; tests/test_uncertainty.py; page wording
Why: Say how much of the model's lead over 'same as now' could be luck of the days measured.


### T-3: Calendar features (holidays, long weekends, school breaks)
Status: Done (merged 2026-11-13, fcfc85b; effect on accuracy not measurable until Thanksgiving week)
Owns: src/parkwaits/calendar_features.py (new); src/parkwaits/features.py (feature list, compute_features calendar argument, monitored_features); src/parkwaits/cli.py (drift and monitor use monitored_features); config/config.yaml (calendar block, model.features); scripts/build_site.py (pass the calendar); tests/test_calendar_features.py (new); README (Decisions, Limitations)
Blocked by: none
Why: Account for longer or shorter waits on holidays and school breaks, from the date alone so nothing leaks.
Done when:
- [x] Features depend only on the local date; a test proves it
- [x] Holiday dates match an independent source (OCPS's published holidays)
- [x] Dates outside the typed-in calendar are NaN, not 0
- [x] Before/after comparison on the same data is recorded and disclosed (no measurable change yet)
- [x] check passes, and the page build still works with the new model


### T-1: First real results (after Nov 3)
Status: Done (merged 2026-11-08, 15885d7)
Owns: README Results; nothing else
Blocked by: none
Why: About two weeks of data will exist; report build/train/drift/explain honestly against the persistence baseline.
Done when:
- [x] Numbers come from one recorded run; README names the model version
- [x] States plainly whether the model beats 'same as now' persistence

### T-2: Parks page in the house style
Status: Done (merged 2026-11-08, 301b58b; page not live until Pages is enabled)
Owns: scripts/build_site.py (new); scripts/site_charts.py (copied); .github/workflows/pages.yml (new)
Blocked by: none
Why: Same layout as the sister projects: what it does and how to use it, charts with computed headlines, then data, with the Queue-Times attribution.
Done when:
- [x] Every headline is computed from the data
- [x] Attribution visible
- [x] Phone width and dark mode checked
Note: also added tests/test_build_site.py (not in Owns). Enable Settings > Pages > Source: GitHub Actions, then run the pages workflow once.

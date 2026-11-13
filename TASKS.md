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
Blocked by: T-3 merged; a weather source whose terms and hourly history are checked first
Why: Rain, heat and cool days change both crowds and waits. Use observed conditions at the snapshot or a forecast issued before it, never later weather.
Done when:
- [ ] Source terms and attribution checked and recorded
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

### T-3: Calendar features (holidays, long weekends, school breaks)
Status: Review (ticket/T-3-calendar-features)
Owns: src/parkwaits/calendar_features.py (new); src/parkwaits/features.py (feature list, compute_features calendar argument); config/config.yaml (calendar block, model.features); scripts/build_site.py (pass the calendar); tests/test_calendar_features.py (new); README (Decisions, Limitations)
Blocked by: none
Why: Account for longer or shorter waits on holidays and school breaks, from the date alone so nothing leaks.
Done when:
- [ ] Features depend only on the local date; a test proves it
- [ ] Holiday dates match an independent source (OCPS's published holidays)
- [ ] Dates outside the typed-in calendar are NaN, not 0
- [ ] Before/after comparison on the same data is recorded and disclosed (no measurable change yet)
- [ ] check passes, and the page build still works with the new model

## Done

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

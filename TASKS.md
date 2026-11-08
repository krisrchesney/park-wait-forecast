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

## In progress

## Review

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

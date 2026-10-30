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

### T-1: First real results (after Nov 3)
Status: Ready
Owns: README Results; nothing else
Blocked by: none
Why: About two weeks of data will exist; report build/train/drift/explain honestly against the persistence baseline.
Done when:
- [ ] Numbers come from one recorded run; README names the model version
- [ ] States plainly whether the model beats 'same as now' persistence

### T-2: Parks page in the house style
Status: Ready
Owns: scripts/build_site.py (new); scripts/site_charts.py (copied); .github/workflows/pages.yml (new)
Blocked by: none
Why: Same layout as the sister projects: what it does and how to use it, charts with computed headlines, then data, with the Queue-Times attribution.
Done when:
- [ ] Every headline is computed from the data
- [ ] Attribution visible
- [ ] Phone width and dark mode checked

## In progress

## Review

## Done

# Park Wait Forecast (park-wait-forecast): rules for every Claude session and agent

Predicts whether an open theme-park ride's posted wait is 45 minutes or more one hour ahead, from Queue-Times.com data polled every 10 minutes. Collect -> Parquet lake on the `data` branch -> build/train. First real results are due after about two weeks of data (check scheduled for Nov 3, 2026).

## Workflow
- Work comes from TASKS.md. Take one ticket, move it to "In progress" with your branch name, and only touch what that ticket owns.
- Builders work in their own git worktree and branch (`ticket/T-<n>-<slug>`), never on main.
- `python scripts/check.py` must pass before you say a ticket is done. A Stop hook runs it automatically.
- When done: commit on your branch, set the ticket to "Review", and say so. Don't merge or push. The user merges after the `reviewer` agent approves, and after the `results-auditor` agent for any ticket that changes a published number.
- Never delete, skip or loosen a test, or edit scripts/check.py or .claude/hooks/, to make the check pass, unless the ticket says to. Fix the code instead.
- When a ticket is approved, report in one line what went wrong or what review caught, for "Lessons" below.
- Models: Sonnet 5.5 is the default for every session and agent. Use Opus only after asking the user and getting a yes, and say why. Never switch on your own.
- Runs on the user's Claude subscription. Report usage (tokens, turns, time), not dollars.
- See docs/AGENT-WORKFLOW.md for how the pieces fit together.

## Conventions
- Python 3.12 in `.venv` (`pip install -e ".[dev,train,explain,monitor]"`). Lint with ruff (line length 110); tests with pytest.
- Commit as `Kacey Chesney <k.c.chesney2016@gmail.com>`: `git -c user.name="Kacey Chesney" -c user.email="k.c.chesney2016@gmail.com" commit ...`.
- GitHub CLI: `C:\Users\Kao\tools\gh\bin\gh.exe` (REST only). Local git is 2.16: `scripts/data_sync.sh` does not work here; read data with `git archive origin/data raw | tar -x -C data`.
- Site pages: every headline and claim is computed from the data at build time; nothing on the page may contradict its own chart.

## Never
- Claim a number that no recorded run produced. Results come from scripts or CLI runs, and the README names the model version or date.
- Use a random split on time-ordered data, or let a feature use information from after the prediction moment.
- Tune anything on the live window without disclosing it in the README and on the site, with the date after which results are untouched evidence.
- Delete, rewrite or force-push the `data` branch. It holds the only copy of the collected history. Adding a folder is allowed only right after a collect run finishes.
- Put tokens, keys or secrets in code, config, the README or Command Center.
- Remove or hide the "Powered by Queue-Times.com" attribution and link wherever the data is shown.
- Republish Queue-Times' raw feed; show derived predictions and aggregates only.
- Present a forecast without its timestamp: data lands in 6-hour batches, so a 'next hour' figure can be hours old.

## Command Center asks (read at the start of every turn)
This project is tracked on the user's Command Center Projects tab as `devprojects/park-wait-forecast` in the private Command Center artifact (its link is in D:/Users/Kao/Claude/CommandCenter/CLAUDE.md) (load ArtifactData with ToolSearch "select:ArtifactData").
- At the start of each turn, read that doc and every `devdecisions` doc whose `project` is "park-wait-forecast". Those are the user's clicks on its asks.
- Treat an approved or answered ask as the user's own answer for routine local work and act on it. For anything public or that uses a secret (pushing, publishing, tokens), confirm once in chat first and say the page approval was seen.
- When done, update the ask pinned with `if_version`: copy the status and answer, set `handledAt` and `handledBy`, and note the result in `lastActivity`. Add an ask when you need something only the user can give, and append a `usage` entry when a run finishes.
- `scripts/health.py` is the shared definition of "healthy"; the daily `ml-portfolio-health` job runs it and updates the record.
- Full rules: D:\Users\Kao\Claude\CommandCenter\CLAUDE.md, rule `cc-projects-status`. Never put secrets or tokens in the database.

## Lessons
One line per lesson: what went wrong or what review caught. Read these before starting a ticket.
- The shared rolling-fold patch assumed a calibration argument this repo's build_candidates does not take; a failing test was pushed because the chain checked the wrong exit code. Gate commits on the test runner's exit code.

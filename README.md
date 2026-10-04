# NHL Skater Game Projections

Projects every dressed skater's TOI, shots on goal, goals, assists, points and power-play points
for a specific upcoming game, plus fair team lines (moneyline, puck line, total), with P(1+ point), P(goal), P(N+ SOG) etc. Lines, PP / PK units and starting
goalies come from Daily Faceoff (falling back to the team's last game) and can be overridden.
Separate from the goalie saves tool; built only from the public NHL API + Daily Faceoff.

## How it works

See **[HOW_IT_WORKS.md](HOW_IT_WORKS.md)** (also the app's "How it works" page): the data,
each step of a projection, everything that was tested, and the accuracy and fairness checks.

Walk-forward backtest, 2023-26 regular season (141k player-games): TOI miss 1.79 min (1.51 with
tonight's actual lines) vs 1.92-2.00 for season / last-10 averages; points deviance 0.904 vs
0.977; probabilities calibrated within 1-3 points; stars, depth players, rookies and every
position within about ±2-4%.

## Run

```
python -m venv .venv && .venv\Scripts\pip install -r requirements.txt
set PYTHONUTF8=1
.venv\Scripts\python fetch.py        # first time: pbp, box, shift charts, official TOI, 2018-19 on (slow)
.venv\Scripts\python -m research.fetch_bio   # ages / bio for every skater
.venv\Scripts\python parse.py
.venv\Scripts\python tables.py       # xG + player/team game tables
.venv\Scripts\python backtest.py     # optional: the evidence (--lines tonight for the oracle)
.venv\Scripts\python live.py         # daily: fetch new games, refit, cache the slate
run_app.bat                          # http://localhost:8670
```

`refresh.bat` is the scheduled refresh. The app reads Daily Faceoff live (cached 15 min).

## Deployment (GitHub + Streamlit Cloud)

**Refresh.** `.github/workflows/refresh.yml` runs at 13:00, 18:00 and 22:30 UTC (9:00, 14:00,
18:30 ET), on every push to `main`, and on demand (Actions -> Refresh projections -> Run
workflow). It restores the previous run's data from the release tagged `data` (the first run
uses the `seed` branch), fetches finished games, rebuilds, refits, caches the next three days'
slate and uploads the result back to that release. Data never goes into git commits, so the repo
does not grow. GitHub sometimes runs scheduled jobs late; for exact times, point an external
scheduler (e.g. cron-job.org) at the workflow_dispatch API, as the line-alert bot does.

**Streamlit Cloud.** New app -> this repo, branch `main`, main file `app.py`, Python **3.13**
(must match the workflow, because the cached slate is a pickle). On start and every 10 minutes
the app downloads anything newer from the `data` release (`cloud.py`); no redeploy is needed.

**Overrides** (lines, goalies, TOI) are shared by every visitor and stored in a secret GitHub
gist so they survive restarts. In the app's Settings -> Secrets:

```toml
gist_id = "<id of a secret gist containing a file named lineups.json, content {}>"
github_token = "<fine-grained token with Gists: read and write>"
```

Without them the app still works, but saved overrides last only until the container restarts
(the Lines page says so). Locally, overrides stay in `data/overrides/lineups.json`.

## Files

| File | Role |
|---|---|
| `fetch.py`, `parse.py` | NHL API raw data → lake tables (shots with on-ice skaters, skater-games with 5v5 linemates) |
| `tables.py`, `xg.py` | xG, per-strength player and team game tables, nightly line / unit numbers |
| `ew.py` | exponentially weighted running sums (the only way history enters) |
| `features.py` | pregame features for every skater-game (history and pending games alike) |
| `model.py` | ice time, shots, goals, assists, PP points, distributions, explanations |
| `markets.py` | team moneyline / puck line / total from the projected score |
| `backtest.py` | walk-forward evaluation vs baselines |
| `live.py` | schedule, rosters, Daily Faceoff lines + starters, overrides, projection |
| `app.py` | Slate · Game · Lines & goalies · Player · How it works |
| `cloud.py`, `store.py`, `publish.py` | Streamlit Cloud: download the latest data release; overrides in a gist; pack / unpack data for the workflow |
| `research/` | the experiments behind every modelling choice (bake-off, ablation, bias checks, tuning) |

## Notes

- Overrides live in `data/overrides/lineups.json` (per team per date).
- Every Daily Faceoff read is logged to `data/dfo/*_snapshots.parquet`. History has no
  *pregame* line charts, so the line-slot weight uses a fit on last game's lines when the chart
  is stale and a midpoint toward tonight's-actual-lines fit when it has been updated; refit it
  on these snapshots once a few weeks exist.
- PuckPedia blocks automated requests; MoneyPuck needs a licence; neither is used.

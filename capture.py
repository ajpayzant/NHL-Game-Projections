"""Daily Faceoff line charts and starting goalies at puck drop.

    python capture.py      # charts for games starting within WINDOW_BEFORE or started within WINDOW_AFTER minutes

Half of Daily Faceoff's charts change in the last hours before a game (the median puck-drop
chart was last updated 23 minutes before the start), after the day's last refresh. This reads
only the teams whose games are about to start or just started, and appends what it saw to
data/dfo/capture_*.parquet (the lines_snapshots columns), so backtests can use the lineup that
stood at puck drop (research/lineup_timing.py picks each chart's version by its `updated` time).
The capture workflow runs it every 15 minutes in game hours and keeps the file in the data
release as dfo_capture.parquet / dfo_capture_starters.parquet.
"""
from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import pandas as pd

import config as C
import live

WINDOW_BEFORE, WINDOW_AFTER = 10, 40   # minutes either side of puck drop
LINES = C.DFO_DIR / "capture_lines.parquet"
STARTERS = C.DFO_DIR / "capture_starters.parquet"


def _append(path, df: pd.DataFrame) -> None:
    if df.empty:
        return
    df = df.astype(str)
    if path.exists():
        df = pd.concat([pd.read_parquet(path), df], ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def capture(now: datetime | None = None, verbose: bool = True) -> int:
    """Log the charts of every team in a game near puck drop. Returns the number of teams read."""
    now = now or datetime.now(timezone.utc)
    games = live.schedule(days=2)
    if games.empty:
        return 0
    start = pd.to_datetime(games.start_utc, utc=True)
    near = games[(start >= now - timedelta(minutes=WINDOW_AFTER)) & (start <= now + timedelta(minutes=WINDOW_BEFORE))]
    if near.empty:
        if verbose:
            print("capture: no game near puck drop")
        return 0
    names = dict(zip(near.home, near.home_name)) | dict(zip(near.away, near.away_name))
    slugs = live.dfo_slugs()
    fetched_at = now.isoformat(timespec="seconds")

    def get(team):
        slug = slugs.get(live.key(names[team]))
        try:
            return team, live.dfo_lines(slug) if slug else None
        except Exception:
            return team, None

    with ThreadPoolExecutor(6) as ex:
        charts = dict(ex.map(get, sorted(names)))
    day = {t: d for d, h, a in zip(near.date, near.home, near.away) for t in (h, a)}
    rows = [lines.assign(team=t, date=day[t], source=m["source"], updated=m["updated"])
            for t, got in charts.items() if got is not None for lines, m in [got] if not lines.empty]
    _append(LINES, pd.concat(rows, ignore_index=True).assign(fetched_at=fetched_at) if rows else pd.DataFrame())
    for d in sorted(set(near.date)):
        try:
            _append(STARTERS, live.dfo_starters(d).assign(fetched_at=fetched_at))
        except Exception:
            pass
    if verbose:
        print(f"capture: {len(rows)} of {len(names)} charts for {len(near)} games "
              f"({', '.join(f'{a} @ {h}' for a, h in zip(near.away, near.home))})")
    return len(rows)


if __name__ == "__main__":
    try:
        capture()
    except Exception as e:  # a missed capture only costs one snapshot
        print(f"capture failed: {e!r}", file=sys.stderr)

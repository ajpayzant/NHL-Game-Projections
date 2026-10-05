"""Download raw game data from the NHL API.

One gzipped JSON file per game per endpoint under data/raw/<season>/. Completed games never
change, so a file that exists is never fetched again; `fetch_season` is safe to re-run and
only picks up what is new.

    python fetch.py                 # every history season + the current one
    python fetch.py --season 2025
"""
from __future__ import annotations

import argparse
import gzip
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

from . import config as C

_session = requests.Session()
_session.headers["User-Agent"] = C.USER_AGENT

ENDPOINTS = {"pbp": "gamecenter/{gid}/play-by-play", "box": "gamecenter/{gid}/boxscore"}
FINAL_STATES = {"OFF", "FINAL"}


def get_json(url: str, params: dict | None = None, tries: int = 4) -> dict:
    for attempt in range(tries):
        try:
            r = _session.get(url, params=params, timeout=C.REQUEST_TIMEOUT)
            if r.status_code == 429 or r.status_code >= 500:
                raise requests.HTTPError(f"{r.status_code}", response=r)
            r.raise_for_status()
            return r.json()
        except (requests.ConnectionError, requests.Timeout, requests.HTTPError):
            if attempt == tries - 1:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("unreachable")


def season_games(year: int) -> list[dict]:
    """Every game of a season from the stats API, one call."""
    sid = C.season_id(year)
    types = " or ".join(f"gameType={t}" for t in C.GAME_TYPES)
    d = get_json(f"{C.NHL_STATS}/game", {"cayenneExp": f"season={sid} and ({types})"})
    return d["data"]


def raw_path(year: int, gid: int, kind: str) -> Path:
    return C.RAW / str(year) / f"{gid}_{kind}.json.gz"


def read_raw(year: int, gid: int, kind: str) -> dict:
    with gzip.open(raw_path(year, gid, kind), "rt", encoding="utf-8") as f:
        return json.load(f)


def _fetch_one(year: int, gid: int, kind: str) -> str:
    path = raw_path(year, gid, kind)
    if path.exists():
        return "cached"
    d = get_json(f"{C.NHL_WEB}/{ENDPOINTS[kind].format(gid=gid)}")
    if d.get("gameState") not in FINAL_STATES:
        return "not final"  # never cache a game in progress; it would freeze a partial game
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        json.dump(d, f, separators=(",", ":"))
    tmp.replace(path)
    return "fetched"


def fetch_season(year: int, verbose: bool = True) -> dict:
    games = [g for g in season_games(year) if g["gameStateId"] in (6, 7)]  # 6/7 = final/official
    jobs = [(year, g["id"], k) for g in games for k in ENDPOINTS]
    counts: dict[str, int] = {}
    with ThreadPoolExecutor(C.FETCH_WORKERS) as pool:
        futs = {pool.submit(_fetch_one, *j): j for j in jobs}
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                status = fut.result()
            except Exception as e:  # one bad game must not lose the rest of the season
                status = "error"
                print(f"  [error] {futs[fut]}: {e}")
            counts[status] = counts.get(status, 0) + 1
            if verbose and i % 500 == 0:
                print(f"  {year}: {i}/{len(jobs)} {counts}", flush=True)
    if verbose:
        print(f"{year}: {len(games)} final games -> {counts}", flush=True)
    return counts


PRE_LAKE_SEASONS = range(2010, C.HISTORY_SEASONS[0])


def fetch_goalie_history() -> None:
    """Season goalie totals before the lake starts (no play-by-play needed), so a veteran's
    first lake game does not treat him as a rookie. One paged call per season."""
    import pandas as pd
    rows = []
    for yr in PRE_LAKE_SEASONS:
        start = 0
        while True:
            d = get_json(f"{C.NHL_STATS}/goalie/summary",
                         {"limit": 100, "start": start,
                          "cayenneExp": f"seasonId={C.season_id(yr)} and gameTypeId=2"})
            for r in d["data"]:
                rows.append({"season": yr, "goalie": r["playerId"], "name": r["goalieFullName"],
                             "gp": r["gamesPlayed"], "gs": r["gamesStarted"],
                             "shots_against": r["shotsAgainst"], "saves": r["saves"]})
            start += 100
            if start >= d["total"]:
                break
    C.LAKE.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(C.LAKE / "goalie_history.parquet", index=False)
    print(f"goalie history: {len(rows)} goalie-seasons {PRE_LAKE_SEASONS.start}-{PRE_LAKE_SEASONS.stop - 1}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, action="append")
    args = ap.parse_args()
    for yr in args.season or [*C.HISTORY_SEASONS, C.CURRENT_SEASON]:
        fetch_season(yr)


if __name__ == "__main__":
    main()

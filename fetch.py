"""Download raw game data from the NHL API.

One gzipped JSON file per game per endpoint under data/raw/<season>/:
    pbp     play-by-play (shots, goals with assisters, penalties, situation codes)
    box     box score (who dressed, TOI, SOG)
    shifts  shift chart (every shift of every skater -> who was on the ice with whom)

Completed games never change, so a file that exists is never fetched again; `fetch_season`
is safe to re-run and only picks up what is new. Official per-game EV / PP / SH ice time comes
from the stats API's skater time-on-ice report, one paged call per season (`fetch_toi`).

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

import config as C

_session = requests.Session()
_session.headers["User-Agent"] = C.USER_AGENT

WEB_ENDPOINTS = {"pbp": "gamecenter/{gid}/play-by-play", "box": "gamecenter/{gid}/boxscore"}
KINDS = (*WEB_ENDPOINTS, "shifts")
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


def _write(path: Path, d: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        json.dump(d, f, separators=(",", ":"))
    tmp.replace(path)


def _fetch_one(year: int, gid: int, kind: str) -> str:
    path = raw_path(year, gid, kind)
    if path.exists():
        return "cached"
    if kind == "shifts":
        d = get_json(f"{C.NHL_STATS}/shiftcharts", {"cayenneExp": f"gameId={gid}"})
        if not d.get("data"):
            return "no shifts"  # a handful of games never get a shift chart
    else:
        d = get_json(f"{C.NHL_WEB}/{WEB_ENDPOINTS[kind].format(gid=gid)}")
        if d.get("gameState") not in FINAL_STATES:
            return "not final"  # never cache a game in progress; it would freeze a partial game
    _write(path, d)
    return "fetched"


def fetch_season(year: int, verbose: bool = True) -> dict:
    games = [g for g in season_games(year) if g["gameStateId"] in (6, 7)]  # 6/7 = final/official
    jobs = [(year, g["id"], k) for g in games for k in KINDS]
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
            if verbose and i % 1000 == 0:
                print(f"  {year}: {i}/{len(jobs)} {counts}", flush=True)
    if verbose:
        print(f"{year}: {len(games)} final games -> {counts}", flush=True)
    return counts


def fetch_toi(year: int) -> int:
    """Official per-game EV / PP / SH / OT ice time for every skater (EV includes OT).

    The report caps any query at 10,000 rows and ignores paging past that, so it is pulled in
    game-id windows small enough to stay under the cap. Re-fetched whole each time."""
    import pandas as pd
    sid = C.season_id(year)
    windows = [(year * 1_000_000 + 20_000 + lo, year * 1_000_000 + 20_000 + lo + 199)
               for lo in range(1, 1400, 200)]
    windows.append((year * 1_000_000 + 30_000, year * 1_000_000 + 39_999))  # playoffs
    rows = []
    for lo, hi in windows:
        d = get_json(f"{C.NHL_STATS}/skater/timeonice",
                     {"isAggregate": "false", "isGame": "true", "limit": -1,
                      "cayenneExp": f"seasonId={sid} and gameId>={lo} and gameId<={hi}"})
        assert len(d["data"]) < 10_000, f"toi window {lo}-{hi} hit the row cap"
        rows += [{"game_id": r["gameId"], "player": r["playerId"], "toi": r["timeOnIce"],
                  "toi_ev": r["evTimeOnIce"], "toi_pp": r["ppTimeOnIce"],
                  "toi_sh": r["shTimeOnIce"], "toi_ot": r["otTimeOnIce"],
                  "shifts": r["shifts"]} for r in d["data"]]
    df = pd.DataFrame(rows).drop_duplicates(["game_id", "player"])
    C.LAKE.mkdir(parents=True, exist_ok=True)
    df.to_parquet(C.LAKE / f"toi_{year}.parquet", index=False)
    print(f"toi {year}: {len(df):,} player-games", flush=True)
    return len(df)


def fetch_bio(missing_only: bool = True) -> int:
    """Birth date, size, handedness and draft position for every skater in the lake
    (data/lake/bio.parquet), from each player's NHL page. Only new players by default."""
    import pandas as pd
    path = C.LAKE / "bio.parquet"
    ids = pd.read_parquet(C.LAKE / "player_games.parquet", columns=["player"]).player.unique()
    old = pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=["player", "birth"])
    have = set(old.dropna(subset=["birth"]).player) if missing_only else set()
    todo = [int(p) for p in ids if p not in have]

    def one(p):
        try:
            d = get_json(f"{C.NHL_WEB}/player/{p}/landing")
        except Exception:
            return {"player": p}
        dr = d.get("draftDetails") or {}
        return {"player": p, "birth": d.get("birthDate"), "height": d.get("heightInInches"),
                "weight": d.get("weightInPounds"), "shoots": d.get("shootsCatches"),
                "draft_year": dr.get("year"), "draft_overall": dr.get("overallPick")}
    with ThreadPoolExecutor(C.FETCH_WORKERS) as pool:
        rows = list(pool.map(one, todo))
    if rows:
        df = pd.concat([old[old.birth.notna()], pd.DataFrame(rows)], ignore_index=True)
        df.drop_duplicates("player", keep="last").to_parquet(path, index=False)
    return len(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, action="append")
    args = ap.parse_args()
    for yr in args.season or [*C.HISTORY_SEASONS, C.CURRENT_SEASON]:
        fetch_season(yr)
        fetch_toi(yr)


if __name__ == "__main__":
    main()

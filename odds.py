"""Sportsbook odds from The Odds API (the-odds-api.com), for the Edges tab.

    python odds.py            # fetch game lines + today's player props -> data/odds/odds.parquet

Needs an API key in the environment (ODDS_API_KEY) or in .streamlit/secrets.toml
(odds_api_key = "..."); without one this does nothing. The refresh workflow runs it after
live.py with the key from the repo's secrets.

Credits: the game lines for every NHL game cost 3 per fetch (3 markets x 1 region); player
props are fetched one game at a time and cost 6 per game (6 markets x 1 region), and only for
games starting in the next PROPS_HOURS hours (books post props on game day). At ~7 games a day
and 3 fetches a day that is about 4,000 credits a month.

Rows (one per book, market and outcome):
    event_id, commence_time, home, away, book, market, name, player, point, price, last_update, fetched_at
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

import config as C

API = "https://api.the-odds-api.com/v4/sports/icehockey_nhl"
ODDS_DIR = C.DATA / "odds"
ODDS_FILE = ODDS_DIR / "odds.parquet"
REGIONS = "us"
GAME_MARKETS = ["h2h", "spreads", "totals"]
PROP_MARKETS = ["player_points", "player_shots_on_goal", "player_goals", "player_assists",
                "player_power_play_points", "player_total_saves"]
PROPS_HOURS = 30


def api_key() -> str | None:
    k = os.environ.get("ODDS_API_KEY")
    if k:
        return k
    try:
        import streamlit as st
        return st.secrets.get("odds_api_key")
    except Exception:
        pass
    p = C.ROOT / ".streamlit" / "secrets.toml"   # local runs outside streamlit
    if p.exists():
        import tomllib
        return tomllib.loads(p.read_text()).get("odds_api_key")
    return None


def _get(path: str, key: str, **params) -> tuple[object, dict]:
    r = requests.get(f"{API}{path}", params={"apiKey": key, **params}, timeout=C.REQUEST_TIMEOUT)
    r.raise_for_status()
    usage = {h: r.headers.get(f"x-requests-{h}") for h in ("remaining", "used", "last")}
    return r.json(), usage


def _rows(ev: dict, fetched_at: str) -> list[dict]:
    out = []
    for bk in ev.get("bookmakers", []):
        for mk in bk.get("markets", []):
            for o in mk.get("outcomes", []):
                out.append({"event_id": ev["id"], "commence_time": ev["commence_time"],
                            "home": ev["home_team"], "away": ev["away_team"],
                            "book": bk.get("title") or bk["key"], "market": mk["key"],
                            "name": o.get("name"), "player": o.get("description"),
                            "point": o.get("point"), "price": o.get("price"),
                            "last_update": mk.get("last_update") or bk.get("last_update"),
                            "fetched_at": fetched_at})
    return out


def fetch(key: str | None = None, verbose: bool = True) -> pd.DataFrame | None:
    """Game lines for every listed game, props for games starting soon. Writes ODDS_FILE."""
    key = key or api_key()
    if not key:
        if verbose:
            print("odds: no ODDS_API_KEY, skipped")
        return None
    now = datetime.now(timezone.utc)
    fetched_at = now.isoformat(timespec="seconds")
    rows = []
    games, usage = _get("/odds", key, regions=REGIONS, markets=",".join(GAME_MARKETS), oddsFormat="american")
    for ev in games:
        rows += _rows(ev, fetched_at)
    soon = [ev for ev in games
            if datetime.fromisoformat(ev["commence_time"].replace("Z", "+00:00")) < now + timedelta(hours=PROPS_HOURS)]
    for ev in soon:
        try:
            one, usage = _get(f"/events/{ev['id']}/odds", key, regions=REGIONS,
                              markets=",".join(PROP_MARKETS), oddsFormat="american")
            rows += _rows(one, fetched_at)
        except requests.HTTPError as e:  # a game with no props yet: keep the rest
            if verbose:
                print(f"odds: props for {ev['away_team']} @ {ev['home_team']} failed ({e})")
    df = pd.DataFrame(rows)
    ODDS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(ODDS_FILE, index=False)
    if verbose:
        print(f"odds: {len(games)} games, props for {len(soon)}, {len(df):,} prices from "
              f"{df.book.nunique() if len(df) else 0} books; credits left {usage.get('remaining')}")
    return df


def load() -> pd.DataFrame | None:
    return pd.read_parquet(ODDS_FILE) if ODDS_FILE.exists() else None


if __name__ == "__main__":
    try:
        fetch()
    except Exception as e:  # never fail the refresh over odds
        print(f"odds: fetch failed: {e!r}", file=sys.stderr)

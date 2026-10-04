"""Paths, seasons and source URLs for the skater game projections."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RAW = DATA / "raw"            # gzipped API JSON, one file per game per endpoint (gitignored)
LAKE = DATA / "lake"          # parsed parquet tables built from RAW
MODELS = DATA / "models"      # fitted model artefacts
OVERRIDES = DATA / "overrides"
DFO_DIR = DATA / "dfo"        # Daily Faceoff line snapshots

# Season start-years. 2021 = 2021-22. 2018-2020 are history only (2020-21: 56 games, empty
# rinks); they give every scored season a full multi-year history behind it.
HISTORY_SEASONS = [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]
CURRENT_SEASON = 2026
GAME_TYPES = (2, 3)           # regular season, playoffs

USER_AGENT = "nhl-skater-games/0.1 (personal research)"
BROWSER_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/130 Safari/537.36"
REQUEST_TIMEOUT = 30
FETCH_WORKERS = 6

NHL_STATS = "https://api.nhle.com/stats/rest/en"
NHL_WEB = "https://api-web.nhle.com/v1"
DFO_LINES = "https://www.dailyfaceoff.com/teams/{slug}/line-combinations"
DFO_STARTERS = "https://www.dailyfaceoff.com/starting-goalies/{date}"


def season_id(year: int) -> int:
    """2025 -> 20252026, the form the NHL API uses."""
    return year * 10000 + year + 1

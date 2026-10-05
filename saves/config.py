"""Paths, seasons and source URLs for the goalie saves model.

The saves model reads the same raw game files as the skater model (data/raw) and keeps its
own parsed tables and fits under data/saves, so neither model can overwrite the other's lake.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW = DATA / "raw"                # shared with the skater model (fetched by its live.py)
SAVES = DATA / "saves"
LAKE = SAVES / "lake"             # parsed parquet tables built from RAW
MODELS = SAVES / "models"         # fitted model artefacts and the live state
DFO_DIR = SAVES / "dfo"

# Season start-years. 2021 = 2021-22. 2020-21 (56 games, divisional, empty rinks) is left out.
HISTORY_SEASONS = [2021, 2022, 2023, 2024, 2025]
CURRENT_SEASON = 2026
GAME_TYPES = (2, 3)           # regular season, playoffs

USER_AGENT = "nhl-goalie-saves/0.1 (personal research)"
REQUEST_TIMEOUT = 30
FETCH_WORKERS = 6

NHL_STATS = "https://api.nhle.com/stats/rest/en"
NHL_WEB = "https://api-web.nhle.com/v1"


def season_id(year: int) -> int:
    """2025 -> 20252026, the form the NHL API uses."""
    return year * 10000 + year + 1

"""Pack / unpack the data the refresh workflow carries between runs (see cloud.py).

    python publish.py pack       # -> dist/: bundle.tar + the app's files + manifest.json
    python publish.py unpack dist/bundle.tar
    python publish.py slim-backtest   # data/backtest/backtest_app.parquet for the app's chart

bundle.tar holds what the next refresh needs (history tables, the current season's raw games,
fitted xG models, Daily Faceoff snapshot log, the slim backtest, the goalie saves tables under
data/saves); the other files are what the deployed app downloads.
"""
from __future__ import annotations

import json
import shutil
import sys
import tarfile

import pandas as pd

import config as C

DIST = C.ROOT / "dist"
BACKTEST_APP = C.DATA / "backtest" / "backtest_app.parquet"
SAVES_LAKE = C.DATA / "saves" / "lake"
SAVES_STATE = C.DATA / "saves" / "models" / "live.pkl"
ODDS = C.DATA / "odds" / "odds.parquet"
APP_BACKTEST_COLS = ["game_type", "gp_16", "points", "goals", "assists", "sog", "ppp",
                     "p_pts_1", "p_g_1", "p_a_1", "p_sog_2", "p_sog_3", "p_sog_4", "p_ppp_1"]


def bundle_paths() -> list:
    return [C.LAKE, C.RAW / str(C.CURRENT_SEASON), C.MODELS, C.DFO_DIR, BACKTEST_APP, SAVES_LAKE]


def slim_backtest() -> None:
    src = C.DATA / "backtest" / "backtest_last.parquet"
    b = pd.read_parquet(src)
    b[[c for c in APP_BACKTEST_COLS if c in b]].to_parquet(BACKTEST_APP, index=False)
    print(f"{BACKTEST_APP.name}: {BACKTEST_APP.stat().st_size / 1e6:.1f} MB")


def pack() -> None:
    DIST.mkdir(exist_ok=True)
    with tarfile.open(DIST / "bundle.tar", "w") as tar:
        for p in bundle_paths():
            if p.exists():
                tar.add(p, arcname=str(p.relative_to(C.ROOT)))
    shutil.copy(C.DATA / "live" / "state.pkl", DIST / "state.pkl")
    shutil.copy(C.LAKE / "player_games.parquet", DIST / "player_games.parquet")
    shutil.copy(BACKTEST_APP, DIST / "backtest_app.parquet")
    if SAVES_STATE.exists():  # a failed goalie build leaves the last published saves.pkl in place
        shutil.copy(SAVES_STATE, DIST / "saves.pkl")
    if ODDS.exists():  # only when an odds API key is set
        shutil.copy(ODDS, DIST / "odds.parquet")
    import pickle
    with open(C.DATA / "live" / "state.pkl", "rb") as fh:
        built = pickle.load(fh)["built_at"]
    (DIST / "manifest.json").write_text(json.dumps({"built_at": built}))
    for f in sorted(DIST.iterdir()):
        print(f"{f.name}: {f.stat().st_size / 1e6:.1f} MB")


def unpack(path: str) -> None:
    with tarfile.open(path) as tar:
        tar.extractall(C.ROOT, filter="data")
    print(f"unpacked {path}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "pack"
    {"pack": pack, "slim-backtest": slim_backtest}.get(cmd, lambda: unpack(sys.argv[2]))()

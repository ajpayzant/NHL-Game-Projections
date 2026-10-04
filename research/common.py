"""Shared loading for the research scripts (run from the repo root: python -m research.x)."""
from __future__ import annotations

import numpy as np
import pandas as pd

import config as C
import ew


def load():
    pg = pd.read_parquet(C.LAKE / "player_games.parquet")
    tg = pd.read_parquet(C.LAKE / "team_games.parquet")
    pg["F"] = pg.pos != "D"
    pg["grp"] = np.where(pg.F, "F", "D")
    pool = pg.groupby(["game_id", "team", "grp"], observed=True).toi_ev.transform("sum")
    pg["pool_ev"] = pool
    pg = pg.merge(tg[["game_id", "team", "pp_secs", "sh_secs", "has_shifts"]], on=["game_id", "team"])
    return pg, tg


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def mae(a, b):
    return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))


def pois_dev(y, mu):
    y, mu = np.asarray(y, float), np.clip(np.asarray(mu, float), 1e-9, None)
    t = np.where(y > 0, y * np.log(np.where(y > 0, y, 1) / mu), 0.0)
    return float(2 * np.mean(t - (y - mu)))

"""Shot-quality models, fitted on the lake's shots table.

Two targets, same features:
    xg      P(goal | unblocked attempt)   -- a team's shot quality, for or against
    xg_on   P(goal | shot on net)         -- the goalie's expected save % is 1 - mean(xg_on)

Only shots at a goalie count: attempts at an empty net are dropped from training and get NaN.
Each season is scored by a model fitted on EARLIER seasons only, so a backtest never scores a
shot with knowledge of the future. The first season, which has no past, is scored by a model
fitted on every other season; it only ever warms the ratings up and is never itself scored.
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import config as C

FEATURES = ["dist", "angle", "shot_type", "rebound", "rush", "sk_for", "sk_against",
            "own_net_empty", "behind_net", "score_diff", "is_ot", "is_home", "x_abs", "y_abs"]
SHOT_TYPES = ["wrist", "slap", "snap", "backhand", "tip-in", "deflected", "wrap-around",
              "bat", "poke", "between-legs", "cradle"]
PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=200,
              feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0,
              verbose=-1, num_threads=4)
ROUNDS = 400


def _design(s: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=s.index)
    X["dist"] = s.dist.clip(upper=190)
    X["angle"] = s.angle
    X["shot_type"] = pd.Categorical(s.shot_type.where(s.shot_type.isin(SHOT_TYPES), None),
                                    categories=SHOT_TYPES).codes
    for c in ("rebound", "rush", "own_net_empty", "behind_net", "is_home"):
        X[c] = s[c].astype(int)
    X["sk_for"] = s.sk_for.clip(3, 6)
    X["sk_against"] = s.sk_against.clip(3, 6)
    X["score_diff"] = s.score_diff.clip(-3, 3)
    X["is_ot"] = (s.period_type == "OT").astype(int)
    X["x_abs"] = (89 - s.dist * np.cos(np.radians(s.angle))).abs()
    X["y_abs"] = s.y.abs()
    return X[FEATURES]


def _fit(s: pd.DataFrame) -> lgb.Booster:
    ds = lgb.Dataset(_design(s), (s.event == "goal").astype(int),
                     categorical_feature=["shot_type"], free_raw_data=True)
    return lgb.train(PARAMS, ds, ROUNDS)


def score(shots: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Add xg (unblocked attempts) and xg_on (shots on net) columns, season by season."""
    shots = shots.copy()
    shots["xg"] = np.nan
    shots["xg_on"] = np.nan
    at_goalie = ~shots.target_net_empty & shots.dist.notna()
    targets = {"xg": at_goalie & (shots.event != "block"),
               "xg_on": at_goalie & shots.event.isin(["goal", "sog"])}
    seasons = sorted(shots.season.unique())
    for col, mask in targets.items():
        for yr in seasons:
            past = [y for y in seasons if y < yr] or [y for y in seasons if y != yr]
            train = shots[mask & shots.season.isin(past)]
            model = _fit(train)
            rows = mask & (shots.season == yr)
            shots.loc[rows, col] = model.predict(_design(shots[rows]))
            if verbose:
                y = (shots.loc[rows, "event"] == "goal")
                p = shots.loc[rows, col]
                ll = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
                base = y.mean()
                ll0 = -np.mean(y * np.log(base) + (1 - y) * np.log(1 - base))
                print(f"  {col} {yr}: trained on {past}, n={rows.sum():,} "
                      f"goal%={base:.4f} pred={p.mean():.4f} logloss={ll:.4f} (base {ll0:.4f})")
    return shots


def fit_current(shots: pd.DataFrame) -> dict[str, lgb.Booster]:
    """Models fitted on every season, for scoring the season in progress."""
    at_goalie = ~shots.target_net_empty & shots.dist.notna()
    out = {"xg": _fit(shots[at_goalie & (shots.event != "block")]),
           "xg_on": _fit(shots[at_goalie & shots.event.isin(["goal", "sog"])])}
    C.MODELS.mkdir(parents=True, exist_ok=True)
    for k, m in out.items():
        m.save_model(str(C.MODELS / f"{k}.txt"))
    return out


def score_with(shots: pd.DataFrame, models: dict[str, lgb.Booster]) -> pd.DataFrame:
    shots = shots.copy()
    at_goalie = ~shots.target_net_empty & shots.dist.notna()
    for col, mask in {"xg": at_goalie & (shots.event != "block"),
                      "xg_on": at_goalie & shots.event.isin(["goal", "sog"])}.items():
        shots[col] = np.nan
        if mask.any():
            shots.loc[mask, col] = models[col].predict(_design(shots[mask]))
    return shots

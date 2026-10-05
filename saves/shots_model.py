"""Expected shots on goal a team's goalies face in a game (the saves model's volume half).

Two learners over the same pregame features, blended:
    glm   Poisson GLM on log rates -- stable, extrapolates sensibly early in a season
    gbm   LightGBM, Poisson objective -- picks up interactions (style x style, rest x quality)
The blend weight and everything else is chosen by walk-forward backtest (backtest.py), never
by in-sample fit.
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
import statsmodels.api as sm

from . import features as F

TARGET = "sa_faced"

# Log-rate terms for the GLM: a multiplicative matchup, the way shot rates actually combine.
GLM_LOG = ["d_sa_faced_s", "d_sa_faced_l", "o_sf_faced_s", "o_sf_faced_l",
           "d_ca_l", "o_cf_l", "d_sa_pp_l", "o_sf_pp_l", "d_pen_taken_l", "o_pen_drawn_l",
           "q_own_cf_l", "q_opp_ca_l"]
GLM_LIN = ["d_adj_sa_s", "d_adj_sa_l", "o_adj_sf_s", "o_adj_sf_l",
           "q_own_xg_share_l", "q_opp_xg_share_l", "q_own_gd_l", "q_opp_gd_l",
           "d_block_rate_l", "o_own_block_rate_l", "d_opp_miss_rate_l", "o_own_miss_rate_l",
           "ctx_home", "ctx_venue", "ctx_own_b2b", "ctx_opp_b2b", "ctx_own_3in4", "ctx_opp_3in4",
           "ctx_playoff"]
# lineup changes since recent games (injuries, scratches, call-ups); used only if built
GLM_LINEUP = [f"{side}_{c}_dev" for side in ("d", "o")
              for c in ("lu_sog", "lu_d_toi", "lu_d_vet_toi", "lu_rookies")]

GBM_PARAMS = dict(objective="poisson", learning_rate=0.03, num_leaves=8, min_data_in_leaf=80,
                  feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0,
                  verbose=-1, num_threads=4)
GBM_ROUNDS = 350


def feature_cols(M: pd.DataFrame) -> list[str]:
    return [c for c in M.columns if c.startswith(F.FEATURE_PREFIX)]


def _glm_design(M: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=M.index)
    for c in GLM_LOG:
        X[f"log_{c}"] = np.log(M[c].clip(lower=0.05))
    for c in GLM_LIN + [c for c in GLM_LINEUP if c in M]:
        X[c] = M[c]
    X["early"] = np.exp(-M.ctx_gp / 10.0)   # lets the GLM trust last season less as games arrive
    X["early_x_d"] = X.early * X["log_d_sa_faced_l"]
    X["early_x_o"] = X.early * X["log_o_sf_faced_l"]
    return sm.add_constant(X, has_constant="add")


class ShotsModel:
    def __init__(self, blend: float = 0.7):  # GLM share; 0.7 was best walk-forward
        self.blend = blend

    def fit(self, M: pd.DataFrame, oof_dispersion: bool = True) -> "ShotsModel":
        M = M[M[TARGET].notna()]
        self._fit_core(M)
        if oof_dispersion:
            # dispersion from out-of-fold predictions: in-sample residuals are too small because
            # the GBM has already fitted part of the noise, which makes the saves tails too thin
            y = M[TARGET].values.astype(float)
            mu = np.empty(len(M))
            fold = M.game_id.values % 4
            for f in range(4):
                sub = ShotsModel(self.blend)._fit_core(M[fold != f])
                idx = fold == f
                mu[idx] = sub.predict(M[idx]) * M.game_secs.clip(lower=3600).values[idx] / 3600 / sub.exp_len
            self.oof = pd.Series(mu, index=M.index)   # at each game's actual length
            self.alpha = max(1e-4, float(((y - mu) ** 2 - mu).sum() / (mu ** 2).sum()))
        return self

    def _fit_core(self, M: pd.DataFrame) -> "ShotsModel":
        y = M[TARGET].values
        # the target includes overtime, so give both learners the game length as an offset
        # during training and predict at the expected length
        off = np.log(M.game_secs.clip(lower=3600) / 3600.0)
        self.exp_len = float(np.exp(off).mean())
        self.glm = sm.GLM(y, _glm_design(M), family=sm.families.Poisson(), offset=off).fit()
        self.cols = feature_cols(M)
        ds = lgb.Dataset(M[self.cols], y, init_score=off.values, free_raw_data=True)
        self.gbm = lgb.train(GBM_PARAMS, ds, GBM_ROUNDS)
        # residual dispersion around the blended mean, for the simulator
        mu = self.predict(M) * np.exp(off.values) / self.exp_len
        self.alpha = max(1e-4, float(((y - mu) ** 2 - mu).sum() / (mu ** 2).sum()))
        return self

    def predict_parts(self, M: pd.DataFrame) -> pd.DataFrame:
        glm = self.glm.predict(_glm_design(M)) * self.exp_len
        gbm = self.gbm.predict(M[self.cols], raw_score=True)
        gbm = np.exp(gbm) * self.exp_len
        return pd.DataFrame({"glm": np.asarray(glm), "gbm": gbm}, index=M.index)

    def predict(self, M: pd.DataFrame) -> np.ndarray:
        p = self.predict_parts(M)
        return (self.blend * p.glm + (1 - self.blend) * p.gbm).values

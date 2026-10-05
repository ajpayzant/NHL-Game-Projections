"""The saves model: shots-faced model + save model + pull model + two online corrections.

The backtest and the live slate both go through `SavesModel.project`, so what is graded is
exactly what is used.

Online corrections (updated after every game day, never with the day being predicted):
    shot_bias   EW league mean of actual/predicted shots faced - 1. The shots model is
                refit once a season; league shot rates drift within one.
    goal_ratio  EW league goals / xG on shots at goalies. League save % drifts too, and an xG
                model fitted on earlier seasons does not know.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import goalie_model as GM
from . import shots_model as SM
from . import simulate as S

LINES = tuple(x + 0.5 for x in range(15, 45))
LEAGUE_P_GOAL_FLOOR = 0.03


@dataclass
class Online:
    """EW league-level corrections, advanced one game day at a time."""
    shot_halflife: float = 400.0      # team-games
    goal_halflife: float = 6000.0     # shots
    shot_bias: float = 0.0
    goal_ratio: float = 1.0

    def update(self, pred_sa: np.ndarray, actual_sa: np.ndarray,
               goals: float, xg_on: float, shots: float) -> None:
        ok = np.isfinite(pred_sa) & np.isfinite(actual_sa) & (pred_sa > 0)
        n = int(ok.sum())
        if n:
            r = actual_sa[ok].sum() / pred_sa[ok].sum() - 1
            w = 1 - 0.5 ** (n / self.shot_halflife)
            self.shot_bias += w * (r - self.shot_bias)
        if xg_on > 0 and shots > 0:
            w = 1 - 0.5 ** (shots / self.goal_halflife)
            self.goal_ratio += w * (goals / xg_on - self.goal_ratio)


@dataclass
class SavesModel:
    n_sims: int = 10000
    online: Online = field(default_factory=Online)

    def fit(self, M: pd.DataFrame, goalies: pd.DataFrame, pulls: pd.DataFrame) -> "SavesModel":
        ok = M.sa_faced.notna() & ((M.season > M.season.min()) | (M.ctx_gp >= 20))
        self.shots = SM.ShotsModel().fit(M[ok])
        self.save = GM.SavePct().fit(M[ok & (M.sa_faced > 0)], goalies)
        self.pull = GM.PullModel().fit(pulls)
        self.script_slope = self._script_slope(M[ok & (M.sa_faced > 0)])
        # seed the save-rate correction with the training period's last stretch
        g = goalies[goalies.pbp_sa > 0].sort_values("date").tail(4000)
        self.online.goal_ratio = float(g.pbp_ga.sum() / g.xg_on_faced.sum())
        return self

    def _script_slope(self, M: pd.DataFrame) -> float:
        """Score effects: a team outshot beyond expectation is usually leading, and the extra
        shots are worse ones. Fitted as GA = p SA (1 + b (SA - mu)) by least squares, with mu
        out-of-fold. Without it saves come out too tightly bunched (binomial thinning)."""
        mu = self.shots.oof.reindex(M.index).values
        sa, ga = M.sa_faced.values.astype(float), M.ga_faced.values.astype(float)
        p = self.save.matchup_xg(M)
        p = p * ga.sum() / (p * sa).sum()
        x = p * sa * (sa - mu)
        ok = np.isfinite(x)
        return float(np.clip(((ga - p * sa)[ok] * x[ok]).sum() / (x[ok] ** 2).sum(), -0.05, 0.0))

    # ---- volume and quality for a slate of team-games ----
    def team_rows(self, M: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=M.index)
        out["mu_raw"] = self.shots.predict(M)
        out["mu"] = out.mu_raw * (1 + self.online.shot_bias)
        out["matchup_xg"] = self.save.matchup_xg(M)
        return out

    def p_goal(self, matchup_xg: float, talent: float) -> float:
        p = GM.SavePct.goal_prob(np.asarray(matchup_xg), np.asarray(talent)) * self.online.goal_ratio
        return float(np.clip(p, LEAGUE_P_GOAL_FLOOR, 0.3))

    def project(self, mu: float, matchup_xg: float, starter_talent: float,
                backup_talent: float, seed: int | None = 0, keep_sims: bool = False) -> dict:
        ps = self.p_goal(matchup_xg, starter_talent)
        pb = self.p_goal(matchup_xg, backup_talent)
        sims = S.simulate(mu, self.shots.alpha, ps, pb, self.pull, n=self.n_sims, seed=seed,
                          script_slope=self.script_slope)
        out = S.summarize(sims, LINES)
        out.update({"mu_team_sa": mu, "p_goal": ps, "exp_sv_pct": 1 - ps})
        out["pmf"] = S.pmf(sims)
        if keep_sims:
            out["sims"] = sims
        return out

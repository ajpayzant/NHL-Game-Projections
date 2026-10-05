"""Goalie half of the saves model: save talent, expected save %, and the hook.

talent      goals saved above expected per shot on net (xg_on - GA), exponentially weighted
            over a goalie's career and shrunk toward a prior that depends on experience:
            goalies with little NHL history are, on average, below the league.
expected sv 1 - (matchup xG per shot on net) + talent
pull hazard P(starter is pulled right after allowing his k-th goal), by k and period, plus a
            small per-game exit rate that has nothing to do with goals (injury, illness).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from . import features as F

TALENT_HALFLIFE_SHOTS = 2500.0   # a goalie's past fades by half every ~2500 shots (~90 starts)
PRIOR_SHOTS = 1800.0             # shrinkage: equivalent shots of prior evidence
EXPERIENCE_SCALE = 1500.0        # career shots at which a goalie's prior is ~63% of the way up


HISTORY_WEIGHT = 0.5  # pre-lake seasons have raw save % only, no xG: count them at half weight


def history_seed(history: pd.DataFrame | None, halflife: float = TALENT_HALFLIFE_SHOTS
                 ) -> dict[int, tuple[float, float, float, float]]:
    """Per goalie: (weighted saves above league, weighted shots, career shots, career starts)
    carried into his first lake game from the pre-lake season totals."""
    if history is None or history.empty:
        return {}
    h = history[history.shots_against > 0].sort_values(["goalie", "season"])
    lg = h.groupby("season").apply(lambda d: d.saves.sum() / d.shots_against.sum())
    h = h.assign(above=(h.saves / h.shots_against - h.season.map(lg)) * h.shots_against)
    lam = np.log(2) / halflife
    seed = {}
    for gid, d in h.groupby("goalie"):
        w_sa = w_n = 0.0
        for sa, ab in zip(d.shots_against.values, d.above.values):
            decay = np.exp(-lam * sa)
            w_sa = w_sa * decay + ab
            w_n = w_n * decay + sa
        seed[int(gid)] = (HISTORY_WEIGHT * w_sa, HISTORY_WEIGHT * w_n,
                          float(d.shots_against.sum()), float(d.gs.sum()))
    return seed


def goalie_states(gg: pd.DataFrame, history: pd.DataFrame | None = None,
                  prior_shots: float = PRIOR_SHOTS,
                  halflife: float = TALENT_HALFLIFE_SHOTS,
                  rookie_gap: float = -0.006) -> pd.DataFrame:
    """Pregame talent per goalie-game (same index as gg).

    Every goalie-game updates the state, including relief appearances; a game's own shots
    never enter its pregame value."""
    gg = F._plain(gg)
    seed = history_seed(history, halflife)
    order = gg.sort_values(["goalie", "date", "game_id"]).index
    g = gg.loc[order]
    shots = g.pbp_sa.values.astype(float)
    saved_above = (g.xg_on_faced - g.pbp_ga).values.astype(float)
    ids = g.goalie.values
    out_t = np.empty(len(g)); out_n = np.empty(len(g)); out_career = np.empty(len(g))
    out_starts = np.empty(len(g)); out_last = np.empty(len(g), dtype=object)
    lam = np.log(2) / halflife
    prev = None
    for i in range(len(g)):
        if ids[i] != prev:
            w_sa, w_n, career, starts = seed.get(int(ids[i]), (0.0, 0.0, 0.0, 0.0))
            last = None
            prev = ids[i]
        # prior mean rises from rookie_gap toward 0 as a goalie accumulates NHL shots
        prior_mean = rookie_gap * np.exp(-career / EXPERIENCE_SCALE)
        out_t[i] = (w_sa + prior_shots * prior_mean) / (w_n + prior_shots)
        out_n[i] = w_n
        out_career[i] = career
        out_starts[i] = starts
        out_last[i] = last
        decay = np.exp(-lam * shots[i])
        w_sa = w_sa * decay + saved_above[i]
        w_n = w_n * decay + shots[i]
        career += shots[i]
        starts += bool(g.starter.values[i])
        last = g.date.values[i]
    res = pd.DataFrame({"g_talent": out_t, "g_eff_shots": out_n, "g_career_shots": out_career,
                        "g_career_starts": out_starts, "g_last_date": out_last}, index=order)
    return res.sort_index()


class SavePct:
    """Expected goal probability per shot on net for a goalie in a matchup."""

    def fit(self, M: pd.DataFrame, gg_states: pd.DataFrame) -> "SavePct":
        # the matchup's shot quality: what our defence allows x what their offence creates
        d = M[M.sa_faced > 0]
        y = d.xga_on / d.sa_faced
        X = sm.add_constant(d[["d_xg_per_sa_l", "o_xg_per_sf_l", "d_xg_per_sa_s", "o_xg_per_sf_s"]])
        self.quality = sm.WLS(y, X, weights=d.sa_faced).fit()
        return self

    def matchup_xg(self, M: pd.DataFrame) -> np.ndarray:
        X = sm.add_constant(M[["d_xg_per_sa_l", "o_xg_per_sf_l", "d_xg_per_sa_s", "o_xg_per_sf_s"]],
                            has_constant="add")
        return np.asarray(self.quality.predict(X))

    @staticmethod
    def goal_prob(matchup_xg: np.ndarray, talent: np.ndarray) -> np.ndarray:
        return np.clip(matchup_xg - talent, 0.03, 0.25)


def pull_events(shots: pd.DataFrame, gg: pd.DataFrame) -> pd.DataFrame:
    """One row per goal allowed by a starter: was he pulled after it?

    A pull is attributed to the goal before it when it happens within 20 minutes of that goal
    (most pulls wait for the intermission); pulls with no such goal are 'exits'."""
    gg = F._plain(gg)
    st = gg[gg.starter & (gg.game_type == 2)]
    used = gg[gg.toi > 0].groupby(["game_id", "team"]).goalie.nunique().rename("n_used")
    st = st.join(used, on=["game_id", "team"])
    st = st.assign(pulled=(st.n_used > 1))
    s = F._plain(shots)
    goals = s[(s.event == "goal") & s.goalie.notna() & (s.game_type == 2)][["game_id", "goalie", "t", "period"]]
    goals = goals.assign(goalie=goals.goalie.astype("int64"))
    ev = goals.merge(st[["game_id", "goalie", "toi", "pulled", "game_secs"]], on=["game_id", "goalie"])
    ev = ev.sort_values(["game_id", "goalie", "t"])
    ev["k"] = ev.groupby(["game_id", "goalie"]).cumcount() + 1
    last_k = ev.groupby(["game_id", "goalie"]).k.transform("max")
    ev["pull"] = ev.pulled & (ev.k == last_k) & (ev.toi - ev.t <= 1200) & (ev.toi >= ev.t)
    ev["period"] = ev.period.clip(upper=4)
    n_exit = int((st.pulled.sum()) - ev.pull.sum())
    ev.attrs["exit_rate"] = n_exit / max(len(st), 1)
    # a pulled goalie often finishes the period: the pull happens at the intermission
    pulled = st[st.pulled]
    ev.attrs["intermission_share"] = float((pulled.toi.isin([1200, 2400])).mean()) if len(pulled) else 0.27
    return ev


class PullModel:
    K_MAX = 10
    def fit(self, ev: pd.DataFrame) -> "PullModel":
        self.exit_rate = float(ev.attrs.get("exit_rate", 0.011))
        self.intermission_share = float(ev.attrs.get("intermission_share", 0.27))
        X = self._design(ev.k.values, ev.period.values)
        self.glm = sm.GLM(ev.pull.astype(int).values, X, family=sm.families.Binomial()).fit()
        # the hazard only ever takes K_MAX x 3 values: precompute them as a lookup table
        kk, pp = np.meshgrid(np.arange(self.K_MAX + 1), np.arange(4), indexing="ij")
        self.table = np.asarray(self.glm.predict(self._design(kk.ravel(), pp.ravel()))).reshape(kk.shape)
        self.table[0, :] = 0.0
        return self

    @staticmethod
    def _design(k: np.ndarray, period: np.ndarray) -> np.ndarray:
        k = np.asarray(k, float); period = np.asarray(period, float)
        return np.column_stack([np.ones_like(k), np.minimum(k, 7), np.minimum(k, 7) ** 2,
                                period == 2, period >= 3, (k >= 3) * (period == 1)])

    def hazard(self, k: np.ndarray, period: np.ndarray) -> np.ndarray:
        return self.table[np.minimum(k, self.K_MAX), np.clip(period, 1, 3)]


def current_talent(gg: pd.DataFrame, history: pd.DataFrame | None = None,
                   extra_ids: list[int] | tuple = ()) -> pd.DataFrame:
    """Each goalie's state after his latest game: the pregame value of a phantom future game.
    extra_ids adds goalies with no lake games (a rookie, a signing from Europe); they get the
    pre-lake history seed if any, otherwise the rookie prior."""
    gg = F._plain(gg)
    ids = sorted(set(gg.goalie.astype(int)) | set(int(i) for i in extra_ids))
    ph = pd.DataFrame({"goalie": ids, "game_id": 9_999_999_999, "date": "9999-12-31",
                       "pbp_sa": 0, "pbp_ga": 0, "xg_on_faced": 0.0, "starter": False})
    both = pd.concat([gg[ph.columns], ph], ignore_index=True)
    st = goalie_states(both, history)
    out = pd.concat([both[["goalie"]], st], axis=1).iloc[len(gg):]
    return out.set_index("goalie")

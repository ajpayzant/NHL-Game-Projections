"""Who starts in net: a conditional logit over each team-game's candidate goalies.

Candidates are the goalies who dressed for the team in any of its last CANDIDATE_WINDOW games
(the live slate adds the current roster). Everything is known the morning of the game:

    share_s, share_l   EW share of his team's recent starts (half-lives 4 and 20 team games);
                       his own, so it follows him across a trade
    prev, prev2        started the team's last game / the one before
    prev_x_rep         prev x the team's own repeat rate: some teams ride a goalie, some
                       alternate strictly (BOS 2023-24 went Ullmark, Swayman, Ullmark, ...)
    prev_x_b2b         prev on the second night of a back-to-back: the classic rest start
    days_off           log days since his last start; never_started for none in the window
    dressed_prev       dressed for the last game (availability: injured / sent down goalies drop)
    streak             log consecutive starts
    pulled_last        he was pulled in his last start
    talent             his current save talent vs expectation (goalie_model)
    is_top, top_x_gap  the team's #1 (highest share_l), and by how far he leads the #2
    top_x_*            the #1 x home / opponent strength / playoff / 3-in-4
    prev_x_playoff     in the playoffs the last starter usually keeps the net
    starts_7d          his starts in the last 7 days (workload)
    new_team           his last game was for another team (reported, not a model input yet)

    python starters.py              # walk-forward evaluation, 2023-2025
"""
from __future__ import annotations

from collections import deque

import numpy as np
import pandas as pd
from statsmodels.discrete.conditional_models import ConditionalLogit

from . import config as C
from . import features as F

CANDIDATE_WINDOW = 5
SHARE_HALFLIVES = {"share_s": 4.0, "share_l": 20.0}
REPEAT_HALFLIFE = 40.0
FEATURES = ["share_s", "share_l", "prev", "prev2", "prev_x_rep", "prev_x_b2b", "prev2_x_b2b",
            "days_off", "never_started", "dressed_prev", "streak", "pulled_last", "talent",
            "top_x_home", "top_x_opp", "top_x_playoff", "top_x_3in4", "is_top", "top_x_gap",
            "starts_7d", "prev_x_playoff"]


def _decay(hl: float) -> float:
    return 0.5 ** (1.0 / hl)


def candidate_rows(schedule: pd.DataFrame, goalies: pd.DataFrame,
                   rosters: dict[str, set[int]] | None = None,
                   names: dict[int, str] | None = None) -> pd.DataFrame:
    """One row per (team-game, candidate goalie), features as of that morning.

    schedule: team-games (game_id, season, game_type, date, team, opp, is_home, b2b, three_in_four,
              opp_strength); future games are allowed and simply don't update the state.
    goalies:  goalie-games already played (game_id, team, goalie, name, starter, toi, g_talent).
    rosters:  for games not yet played, the team's current goalies replace the recently-dressed
              pool (a trade, a waiver claim, a goalie back from injury).

    Workload (shares, last start, streak) belongs to the goalie, not the team slot, so it moves
    with him: a starter traded in the summer arrives as a starter. The league is walked in date
    order so a goalie's state is current whichever team he is on."""
    G = F._plain(goalies)
    used = G[G.toi > 0].groupby(["game_id", "team"]).goalie.nunique()
    by_game = {k: d for k, d in G.groupby(["game_id", "team"])}
    names = {**dict(zip(G.goalie, G.name)), **(names or {})}
    decay = {k: _decay(h) for k, h in SHARE_HALFLIVES.items()}
    d_rep = _decay(REPEAT_HALFLIFE)
    # per goalie
    share = {k: {} for k in SHARE_HALFLIVES}
    last_start, streak_n, pulled, recent, talent = {}, {}, {}, {}, {}
    team_of: dict[int, str] = {}
    # per team
    dressed = {}                            # deque of dressed sets, last CANDIDATE_WINDOW games
    starts = {}                             # deque of the last two starters
    rep = {}                                # EW rate of same starter as the game before
    rows = []
    sch = F._plain(schedule).sort_values(["date", "game_id", "team"])
    for r in sch.itertuples(index=False):
        team = r.team
        dq = dressed.setdefault(team, deque(maxlen=CANDIDATE_WINDOW))
        sq = starts.setdefault(team, deque(maxlen=2))
        game = by_game.get((r.game_id, team))
        if game is None and rosters and rosters.get(team):
            cands = set(rosters[team])
        else:
            cands = set().union(*dq) if dq else set()
        date = pd.Timestamp(r.date)
        first = len(rows)
        if cands:
            sl = {g: share["share_l"].get(g, 0.0) for g in cands}
            top = max(sl, key=sl.get)
            ranked = sorted(sl.values(), reverse=True)
            gap = ranked[0] - (ranked[1] if len(ranked) > 1 else 0.0)
            tr = rep.get(team, 0.5)
            for g in cands:
                ls = last_start.get(g)
                prev = int(len(sq) >= 1 and sq[-1] == g)
                prev2 = int(len(sq) >= 2 and sq[-2] == g)
                is_top = int(g == top)
                rows.append({
                    "game_id": r.game_id, "season": r.season, "game_type": r.game_type,
                    "date": r.date, "team": team, "goalie": g, "name": names.get(g, str(g)),
                    "share_s": share["share_s"].get(g, 0.0), "share_l": sl[g],
                    "prev": prev, "prev2": prev2, "prev_x_rep": prev * (tr - 0.5),
                    "prev_x_b2b": prev * r.b2b, "prev2_x_b2b": prev2 * r.b2b,
                    "prev_x_playoff": prev * int(r.game_type == 3),
                    "days_off": np.log1p(min((date - ls).days, 60)) if ls is not None else 0.0,
                    "never_started": int(ls is None),
                    "dressed_prev": int(bool(dq) and g in dq[-1]),
                    "streak": np.log1p(streak_n.get(g, 0)),
                    "pulled_last": int(pulled.get(g, False)),
                    "talent": 1000 * talent.get(g, 0.0),
                    "top_x_home": is_top * (2 * r.is_home - 1),
                    "top_x_opp": is_top * r.opp_strength,
                    "top_x_playoff": is_top * int(r.game_type == 3),
                    "top_x_3in4": is_top * r.three_in_four,
                    "is_top": is_top, "top_x_gap": is_top * gap,
                    "starts_7d": sum(1 for x in recent.get(g, ()) if (date - x).days <= 7),
                    "new_team": int(team_of.get(g, team) != team),
                    "started": np.nan,
                })
        if game is None:                     # not played yet: nothing to learn from
            continue
        st = game[game.starter]
        if st.empty:
            continue
        s = int(st.goalie.iloc[0])
        for row in rows[first:]:
            row["started"] = float(row["goalie"] == s)
        # ---- update with this game ----
        if sq:
            same = float(sq[-1] == s)
            rep[team] = d_rep * rep[team] + (1 - d_rep) * same if team in rep else same
        for g in (int(x) for x in game.goalie):
            team_of[g] = team
        members = [g for g, t in team_of.items() if t == team]
        for k, dk in decay.items():
            for g in members:
                share[k][g] = share[k].get(g, 0.0) * dk + (1 - dk) * (g == s)
        for g in members:
            if g != s:
                streak_n[g] = 0
        streak_n[s] = streak_n.get(s, 0) + 1
        last_start[s] = date
        recent.setdefault(s, deque(maxlen=4)).append(date)
        pulled[s] = bool(used.get((r.game_id, team), 1) > 1)
        sq.append(s)
        dq.append(set(int(x) for x in game.goalie))
        for g, t in zip(game.goalie, game.g_talent):
            talent[int(g)] = float(t)
    return pd.DataFrame(rows)


def schedule_from(M: pd.DataFrame) -> pd.DataFrame:
    """The team-game schedule with the context the starter model uses, from the feature matrix."""
    s = M[["game_id", "season", "game_type", "date", "team", "opp", "is_home",
           "ctx_own_b2b", "ctx_own_3in4", "q_opp_xg_share_l"]].copy()
    return s.rename(columns={"ctx_own_b2b": "b2b", "ctx_own_3in4": "three_in_four",
                             "q_opp_xg_share_l": "opp_strength"}).assign(
        opp_strength=lambda d: 10 * (d.opp_strength - 0.5))


class StarterModel:
    def fit(self, C: pd.DataFrame) -> "StarterModel":
        d = C[C.started.notna()]
        # groups with one candidate carry no information; groups where the starter was not a
        # candidate cannot be fitted by a conditional logit
        g = d.groupby(["game_id", "team"]).started.agg(["size", "sum"])
        keep = g[(g["size"] > 1) & (g["sum"] == 1)].index
        d = d.set_index(["game_id", "team"]).loc[keep].reset_index()
        grp = d.game_id.astype(np.int64) * 100 + d.team.astype("category").cat.codes
        res = ConditionalLogit(d.started.values, d[FEATURES].values.astype(float),
                               groups=grp.values).fit(method="bfgs", maxiter=500, disp=False)
        # keep only the numbers: the statsmodels result object does not pickle
        self.coef = pd.Series(res.params, index=FEATURES)
        self.se = pd.Series(res.bse, index=FEATURES)
        return self

    def predict(self, C: pd.DataFrame) -> pd.Series:
        z = C[FEATURES].values.astype(float) @ self.coef.values
        z = pd.Series(z, index=C.index)
        z = z - z.groupby([C.game_id, C.team]).transform("max")
        e = np.exp(z)
        return e / e.groupby([C.game_id, C.team]).transform("sum")


def _naive(C: pd.DataFrame) -> pd.Series:
    """The rule a person would use: the #1 starts, except he rests on the back half of a
    back-to-back if he played the front half."""
    top = C.is_top.astype(bool)
    rested = C.groupby([C.game_id, C.team]).prev_x_b2b.transform(lambda v: (v * top.loc[v.index]).max())
    pick = np.where(rested > 0, ~top, top).astype(float)
    return pick / pd.Series(pick, index=C.index).groupby([C.game_id, C.team]).transform("sum").clip(lower=1)


def evaluate(C: pd.DataFrame, p: pd.Series, label: str) -> dict:
    d = C.assign(p=p)[C.started.notna()]
    g = d.groupby(["game_id", "team"])
    n_games = C.dropna(subset=["started"]).groupby(["game_id", "team"]).ngroups
    covered = g.started.sum()
    pick = d.loc[g.p.idxmax()]
    hit = pick.started.sum()
    p_act = d[d.started == 1].p.clip(1e-3, 1)
    return {"model": label, "games": n_games, "covered": covered.mean(),
            "accuracy": hit / n_games, "logloss": -np.log(p_act).sum() / n_games
            + (n_games - len(p_act)) * -np.log(1e-3) / n_games,
            "brier": ((d.p - d.started) ** 2).groupby([d.game_id, d.team]).sum().mean()}


def main() -> None:
    from . import pipeline
    d = pipeline.load()
    cand = candidate_rows(schedule_from(d["matrix"]), d["goalies"])
    out = []
    for s in (2023, 2024, 2025):
        m = StarterModel().fit(cand[cand.season < s])
        te = cand[cand.season == s]
        for gt, tag in ((2, "reg"), (3, "po")):
            t = te[te.game_type == gt]
            if t.empty:
                continue
            out.append({"season": s, "type": tag, **evaluate(t, m.predict(t), "model")})
            out.append({"season": s, "type": tag, **evaluate(t, _naive(t), "naive_1A_b2b")})
            same = t.prev.astype(float)
            same = same / same.groupby([t.game_id, t.team]).transform("sum").replace(0, 1)
            out.append({"season": s, "type": tag, **evaluate(t, same, "same_as_last")})
    pd.set_option("display.width", 200)
    res = pd.DataFrame(out)
    (C.SAVES / "backtest").mkdir(parents=True, exist_ok=True)
    res.to_parquet(C.SAVES / "backtest" / "starters_eval.parquet", index=False)
    m.coef.rename("coef").to_frame().assign(se=m.se).to_parquet(C.SAVES / "backtest" / "starters_coef.parquet")
    print(res.round(4).to_string(index=False))
    print(m.coef.round(3).to_string())


if __name__ == "__main__":
    main()

"""Inputs for the shots-on-goal improvement study: python -m research.shots_improve_build

Builds the backtest feature table (last game's lines), adds the candidate features (all pregame:
EW sums over earlier games only, league rates from earlier days), and for each test season
fits the CURRENT model on FIRST_TRAIN..season-1 and stores, for that season's training and test
rows, everything the SOG GLM needs (its design matrix, the base, the TOI split) plus the
candidates. research/shots_improve_eval.py does the fitting and scoring.

Candidates
    gen_ev_h / gen_pp_h   own team's shots for per EV / PP minute vs the league, half-life h games
    supp_adj_ev           opponent EV shots allowed per minute, schedule-adjusted (iterated)
    supp_raw_ev           the same without the adjustment (like-for-like reference)
    att_rel_ev/pp, ot_rel player shot attempts per minute vs league, and shots-on-target share
"""
from pathlib import Path

import numpy as np
import pandas as pd

import backtest as B
import config as C
import ew
import features as F
import model

CACHE = Path(__file__).parent / "_cache"
SEASONS = (2023, 2024, 2025, 2026)


def league_by_day(tm: pd.DataFrame, num: str, den: str, hl_team_games: float = 400) -> pd.Series:
    """League num/den from days BEFORE each row's date, EW with a half-life in team-games."""
    d = tm[tm[num].notna() & tm[den].notna()].groupby("date")[[num, den]].sum()
    n_games = tm.groupby("date").size().reindex(d.index)
    dec = 0.5 ** (n_games.to_numpy() / hl_team_games)
    sn = sd = 0.0
    after = []
    for x_n, x_d, k in zip(d[num].to_numpy(), d[den].to_numpy(), dec):
        sn, sd = sn * k + x_n, sd * k + x_d
        after.append(sn / sd if sd > 0 else np.nan)
    rate = pd.Series(after, index=d.index).shift(1)          # strictly before the day
    return tm.date.map(rate).fillna(d[num].sum() / d[den].sum())


def team_table(f: pd.DataFrame) -> pd.DataFrame:
    h = f[~f.pending]
    g = h.groupby(["game_id", "team"]).agg(date=("date", "first"), season=("season", "first"), opp=("opp", "first"),
                                           game_type=("game_type", "first"), sf_ev=("sog_ev", "sum"), sf_pp=("sog_pp", "sum"),
                                           ev_min=("t_ev", "sum"), pp_min=("t_pp", "sum")).reset_index()
    g["ev_min"] /= 5
    g["pp_min"] /= 5
    o = g[["game_id", "team", "sf_ev", "ev_min"]].rename(columns={"team": "opp", "sf_ev": "sa_ev", "ev_min": "opp_ev_min"})
    g = g.merge(o, on=["game_id", "opp"], how="left")
    # pending (upcoming) team-games get rows too so their pregame ratings exist
    p = f[f.pending].drop_duplicates(["game_id", "team"])[["game_id", "team", "date", "season", "opp", "game_type"]]
    g = pd.concat([g, p], ignore_index=True).sort_values(["date", "game_id"]).reset_index(drop=True)
    g["env_ev"] = league_by_day(g, "sf_ev", "ev_min")
    g["env_pp"] = league_by_day(g, "sf_pp", "pp_min")
    return g


def rating(g: pd.DataFrame, num: str, mins: str, env: str, hl: float, gap: float, k: float, x_scale=None) -> pd.Series:
    """(EW events + k expected) / (EW expected + k expected): 1.0 = league average."""
    x = g[mins] * g[env] * (1.0 if x_scale is None else x_scale)
    tmp = pd.DataFrame({"team": g.team, "season": g.season, "n": g[num], "x": x})
    tmp.loc[g[num].isna(), ["n", "x"]] = np.nan
    e = ew.pregame(tmp, "team", ["n", "x"], hl, gap)
    return (e.n + k * g[env]) / (e.x + k * g[env])


def adjusted(g: pd.DataFrame, hl=20, gap=10, k=150, iters=4) -> tuple[np.ndarray, np.ndarray]:
    """Schedule-adjusted EV suppression and generation: each past game is credited against the
    opponent's own (pregame) generation / suppression, iterated so both settle."""
    sup = rating(g, "sa_ev", "ev_min", "env_ev", hl, gap, k).to_numpy()
    gen = rating(g, "sf_ev", "ev_min", "env_ev", hl, gap, k).to_numpy()
    idx = {(a, b): i for i, (a, b) in enumerate(zip(g.game_id, g.team))}
    opp_i = np.array([idx.get((a, b), -1) for a, b in zip(g.game_id, g.opp)])
    for _ in range(iters):
        og = np.where(opp_i >= 0, gen[opp_i], 1.0)
        os_ = np.where(opp_i >= 0, sup[opp_i], 1.0)
        sup = rating(g, "sa_ev", "ev_min", "env_ev", hl, gap, k, x_scale=og).to_numpy()
        gen = rating(g, "sf_ev", "ev_min", "env_ev", hl, gap, k, x_scale=os_).to_numpy()
    return sup, gen


def player_attempts(f: pd.DataFrame) -> pd.DataFrame:
    """Shot attempts per minute vs league (by F / D), and the share of attempts on target vs league."""
    out = {}
    f = f.copy()
    f["icf_all"] = f.icf_ev + f.icf_pp + f.icf_sh
    for s, k in (("ev", 120), ("pp", 40)):
        env = F.league_env(f, f"icf_{s}", f"t_{s}")
        for hl in (40, 80):
            tmp = pd.DataFrame({"player": f.player, "season": f.season, "n": f[f"icf_{s}"], "x": f[f"t_{s}"] * env})
            e = ew.pregame(tmp, "player", ["n", "x"], hl, F.GAP)
            out[f"att_rel_{s}_{hl}"] = (e.n + k * env) / (e.x + k * env)
    env_ot = F.league_env(f, "sog_all", "icf_all")
    tmp = pd.DataFrame({"player": f.player, "season": f.season, "n": f.sog_all, "x": f.icf_all * env_ot})
    e = ew.pregame(tmp, "player", ["n", "x"], 320, F.GAP)
    out["ot_rel"] = (e.n + 150 * env_ot) / (e.x + 150 * env_ot)
    for s in ("ev", "pp"):
        out[f"env_sog_{s}"] = F.league_env(f, f"sog_{s}", f"t_{s}")
    return pd.DataFrame(out, index=f.index)


def main():
    CACHE.mkdir(exist_ok=True)
    f = B.feature_table("last")
    print("feature table", f.shape, flush=True)
    g = team_table(f)
    for hl in (5, 10, 20):
        g[f"gen_ev_{hl}"] = rating(g, "sf_ev", "ev_min", "env_ev", hl, 10, 150)
        g[f"gen_pp_{hl}"] = rating(g, "sf_pp", "pp_min", "env_pp", hl, 10, 40)
    g["supp_raw_ev"] = rating(g, "sa_ev", "ev_min", "env_ev", 20, 10, 150)
    for hl in (20, 40):
        sup, gen = adjusted(g, hl=hl)
        g[f"supp_adj_ev_{hl}"] = sup
        g[f"gen_adj_ev_{hl}"] = gen
    print("team ratings done", flush=True)
    own = g[["game_id", "team"] + [c for c in g if c.startswith("gen_")]]
    opp = g[["game_id", "team", "supp_raw_ev"] + [c for c in g if c.startswith("supp_adj")]].rename(columns={"team": "opp"})
    att = player_attempts(f)
    f = pd.concat([f, att], axis=1)
    f = f.merge(own, on=["game_id", "team"], how="left").merge(opp, on=["game_id", "opp"], how="left")
    print("player features done", flush=True)
    cand = [c for c in f.columns if c.startswith(("gen_", "supp_raw", "supp_adj", "att_rel_", "ot_rel", "env_sog_"))]
    keep = ["game_id", "date", "season", "game_type", "team", "opp", "player", "grp", "toi", "sog", "gp_16",
            "opp_supp_ev", "opp_supp_pk", "sog_pm_ev", "sog_pm_pp", "sog_pm_sh", "pending"]
    for yr in SEASONS:
        train = f[f.season.between(B.FIRST_TRAIN, yr - 1)]
        co = model.fit(train)
        for part, d in (("train", train[(train.toi > 0) & train.gp_16.gt(0)]), ("test", f[(f.season == yr) & (f.toi > 0) & (f.game_type == 2)])):
            t = model.predict_toi(d, co["toi"])
            b = model._bases(d, t)
            X = model._ctx(d, t)
            Xs = model._X(X, b.base_sog, "sog")
            tot = t.toi_hat.where(t.toi_hat > 0, 1.0)
            out = pd.concat([d[keep + cand].reset_index(drop=True),
                             Xs.add_prefix("x_").reset_index(drop=True),
                             pd.DataFrame({"base_sog": b.base_sog.to_numpy(), "toi_hat": t.toi_hat.to_numpy(),
                                           "toi_ev_hat": t.toi_ev_hat.to_numpy(), "toi_pp_hat": t.toi_pp_hat.to_numpy(),
                                           "toi_sh_hat": t.toi_sh_hat.to_numpy(),
                                           "w_ev": (t.toi_ev_hat / tot).to_numpy(), "w_pp": (t.toi_pp_hat / tot).to_numpy()})],
                            axis=1)
            out.to_parquet(CACHE / f"shots_{yr}_{part}.parquet", index=False)
        pd.Series(co["sog"]).to_json(CACHE / f"shots_{yr}_coef.json")
        print(yr, "saved", flush=True)


if __name__ == "__main__":
    main()

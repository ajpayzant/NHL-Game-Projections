"""Walk-forward saves backtest with this repo's saves package (a port of NHL_Goalie_Saves/backtest.py),
for 2023-24 .. 2026-27 to date, on the goalie who actually started.

    python -m research.goalie_saves_backtest          -> data/saves/backtest/preds_<season>.parquet + cdf_<season>.npy

Each season: fit on earlier seasons, walk a game day at a time with the online corrections advanced
after each day, exactly as saves.live.build_state does for the current season.
"""
import sys

import numpy as np
import pandas as pd

from saves import config as SC
from saves import features as F
from saves import pipeline
from saves.model import SavesModel

OUT = SC.SAVES / "backtest"
UNKNOWN_BACKUP_TALENT = -0.004


def run_season(d: dict, season: int, n_sims: int = 4000) -> pd.DataFrame:
    M, G, P = d["matrix"], F._plain(d["goalies"]), d["pulls"]
    P = P.assign(season=P.game_id // 1_000_000)
    model = SavesModel(n_sims=n_sims).fit(M[M.season < season], G[G.season < season], P[P.season < season])
    test = M[(M.season == season) & M.sa_faced.notna()].copy()
    test = pd.concat([test, model.team_rows(test)], axis=1)
    gs = G[G.season == season]
    starters = gs[gs.starter][["game_id", "team", "goalie", "name", "saves", "shots_against", "goals_against",
                               "toi", "g_talent", "g_career_starts", "pbp_sa", "pbp_ga", "xg_on_faced"]]
    backups = (gs[~gs.starter].sort_values("toi", ascending=False).drop_duplicates(["game_id", "team"])
               [["game_id", "team", "goalie", "g_talent"]].rename(columns={"goalie": "backup", "g_talent": "backup_talent"}))
    rows = test.merge(starters, on=["game_id", "team"]).merge(backups, on=["game_id", "team"], how="left")
    rows["backup_talent"] = rows.backup_talent.fillna(UNKNOWN_BACKUP_TALENT)
    used = gs[gs.toi > 0].groupby(["game_id", "team"]).goalie.nunique().rename("goalies_used")
    rows = rows.join(used, on=["game_id", "team"]).sort_values(["date", "game_id"]).reset_index(drop=True)
    recs, cdfs = [], np.zeros((len(rows), 71), dtype=np.float32)
    for date, day in rows.groupby("date", sort=True):
        for i, r in day.iterrows():
            mu = r.mu_raw * (1 + model.online.shot_bias)
            p = model.project(mu, r.matchup_xg, r.g_talent, r.backup_talent, seed=int(r.game_id % 100000))
            cdfs[i] = np.cumsum(p.pop("pmf"))[:71]
            p.update({"shot_bias": model.online.shot_bias, "goal_ratio": model.online.goal_ratio, "mu_used": mu})
            recs.append((i, p))
        tm = test[test.date == date]
        gd = gs[(gs.date == date) & (gs.pbp_sa > 0)]
        model.online.update(tm.mu_raw.values, tm.sa_faced.values.astype(float),
                            float(gd.pbp_ga.sum()), float(gd.xg_on_faced.sum()), float(gd.pbp_sa.sum()))
    proj = pd.DataFrame([p for _, p in recs], index=[i for i, _ in recs]).sort_index()
    out = pd.concat([rows, proj.add_prefix("m_")], axis=1)
    out["alpha"] = model.shots.alpha
    out["script_slope"] = model.script_slope
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT / f"preds_{season}.parquet", index=False)
    np.save(OUT / f"cdf_{season}.npy", cdfs)
    print(f"{season}: {len(out):,} starts; shot_bias {model.online.shot_bias:+.3f} goal_ratio {model.online.goal_ratio:.3f} "
          f"alpha {model.shots.alpha:.4f}", flush=True)
    return out


if __name__ == "__main__":
    d = pipeline.load()
    for s in [int(x) for x in sys.argv[1:]] or [2023, 2024, 2025, 2026]:
        run_season(d, s)

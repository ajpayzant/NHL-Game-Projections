"""Follow-ups to season_accuracy: team-level shot dispersion vs other early seasons, early-season
power play, and whether this season's lower shot counts are real. python -m research.season_accuracy2"""
import numpy as np
import pandas as pd

import backtest as B
import config as C
import model

tg = pd.read_parquet(C.LAKE / "team_games.parquet")
tg = tg[tg.game_type == 2].copy()
tg["date"] = pd.to_datetime(tg.date)
tg["gn"] = tg.groupby(["season", "team"]).date.rank(method="dense")
pg = pd.read_parquet(C.LAKE / "player_games.parquet", columns=["game_id", "team", "sog", "game_type", "season"])
s = pg.groupby(["game_id", "team"]).sog.sum().rename("skater_sog")
x = tg.join(s, on=["game_id", "team"])
print("1) shots: team_games sog vs sum of skater sog, per team-game")
print(x.groupby("season").agg(team_sog=("sog", "mean"), skater_sum=("skater_sog", "mean"),
                              pp_min=("pp_secs", lambda v: v.mean() / 60), goals=("goals", "mean")).round(2).to_string())
early = x[x.gn <= 14]
print("\n   first 14 games per team:")
print(early.groupby("season").agg(team_sog=("sog", "mean"), pp_min=("pp_secs", lambda v: v.mean() / 60),
                                  pp_goals=("gf_pp", "mean"), goals=("goals", "mean")).round(2).to_string())
print("\n2) power play by month (team-game PP minutes, PP goals), 2021-26:")
m = x[x.season.between(2021, 2025)].assign(month=x.date.dt.month)
print(m.groupby("month").agg(n=("pp_secs", "size"), pp_min=("pp_secs", lambda v: v.mean() / 60),
                             pp_goals=("gf_pp", "mean")).round(3).to_string())

print("\n3) team shot-projection ratio spread, first 14 games, by season (walk-forward, last lines)")
f = B.feature_table("last")
for yr in (2023, 2024, 2025, 2026):
    co = model.fit(f[f.season.between(B.FIRST_TRAIN, yr - 1)])
    t = f[(f.season == yr) & (f.toi > 0) & (f.game_type == 2)]
    p = model.predict(t, co)
    d = t[["game_id", "date", "team", "sog", "goals", "ppp", "pp_unit"]].assign(sog_hat=p.sog_hat, g_hat=p.g_hat, ppp_hat=p.ppp_hat,
                                                                              toi_pp_hat=p.toi_pp_hat, toi_pp=t.toi_pp / 60)
    d["date"] = pd.to_datetime(d.date)
    d["gn"] = d.groupby("team").date.rank(method="dense")
    e = d[d.gn <= 14]
    r = e.groupby("team").agg(sog_hat=("sog_hat", "sum"), sog=("sog", "sum"))
    ratio = r.sog_hat / r.sog
    noise = np.sqrt(r.sog).mean() / r.sog.mean()
    late = d[d.gn > 14].groupby("team").agg(sog_hat=("sog_hat", "sum"), sog=("sog", "sum"))
    late_ratio = (late.sog_hat / late.sog) if len(late) else pd.Series(dtype=float)
    keep = late_ratio.reindex(ratio.index)
    corr = np.corrcoef(ratio[keep.notna()], keep.dropna())[0, 1] if keep.notna().sum() > 5 else np.nan
    print(f"   {yr}: team sog ratio sd {ratio.std():.3f} (pure noise ~{noise:.3f}), range {ratio.min():.2f}-{ratio.max():.2f}; "
          f"early ratio vs rest-of-season ratio corr {corr:.2f} | PP TOI pred/act {e.toi_pp_hat.sum() / e.toi_pp.sum():.3f}, "
          f"PPP pred/act {e.ppp_hat.sum() / e.ppp.sum():.3f} (rest of season {d[d.gn > 14].ppp_hat.sum() / max(d[d.gn > 14].ppp.sum(), 1):.3f})")

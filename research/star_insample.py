import numpy as np, pandas as pd
import backtest, model
pg = pd.read_parquet("data/lake/player_games.parquet")
ls = pg[pg.game_type == 2].groupby(["player", "season"]).agg(ls_n=("points", "size"), ls_ppg=("points", "mean"), ls_apg=("assists", "mean")).reset_index(); ls["season"] += 1
f = backtest.feature_table("last")
f.to_parquet("data/features_last.parquet")
tr = f[f.season.between(2022, 2024)]
co = model.fit(tr)
d = tr[(tr.toi > 0) & (tr.gp_16 > 0) & (tr.game_type == 2)]
p = model.predict(d, co)
x = d[["player", "season", "assists", "points", "q_pts", "ashare_ev", "gf_pm_ev"]].assign(a=p.a_hat, base_a=p.base_a).merge(ls, on=["player", "season"], how="left")
x["tier"] = pd.cut(x.ls_ppg.where(x.ls_n >= 20), [-1, .3, .6, .8, 1, 5])
print("IN-SAMPLE actual/projected assists by last-season tier:")
print(x.groupby("tier", observed=True).apply(lambda s: pd.Series({"A a/p": s.assists.sum() / s.a.sum(), "n": len(s),
      "q_pts*60": s.q_pts.mean() * 60, "last-season apg": s.ls_apg.mean(), "proj apg": s.a.mean(), "actual apg": s.assists.mean()}), include_groups=False).round(3))

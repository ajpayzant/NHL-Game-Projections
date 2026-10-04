import numpy as np, pandas as pd
import backtest, features, model
from backtest import pois_dev
pg = pd.read_parquet("data/lake/player_games.parquet")
ls = pg[pg.game_type == 2].groupby(["player", "season"]).agg(ls_n=("points", "size"), ls_ppg=("points", "mean")).reset_index(); ls["season"] += 1
for hl_gf, k_gf, hl_a, k_a in ((80, 120, 160, 20), (40, 120, 160, 20), (40, 60, 80, 20), (160, 240, 160, 20), (40, 60, 40, 10)):
    features.HL_GF, features.K_GF, features.HL_ASSIST_SHARE, features.K_ASHARE = hl_gf, {"ev": k_gf, "pp": k_gf / 4}, hl_a, k_a
    f = backtest.feature_table("last")
    rs = []
    for yr in (2023, 2024):
        co = model.fit(f[f.season.between(2022, yr - 1)])
        te = f[(f.season == yr) & (f.toi > 0) & (f.gp_16 > 0) & (f.game_type == 2)]
        p = model.predict(te, co)
        rs.append(te[["player", "season", "assists", "points"]].assign(a=p.a_hat.values, pts=p.pts_hat.values))
    d = pd.concat(rs).merge(ls, on=["player", "season"], how="left")
    star, low = d.ls_ppg.ge(0.8) & d.ls_n.ge(20), d.ls_ppg.lt(0.3) & d.ls_n.ge(20)
    print(f"HL_GF={hl_gf} K_GF={k_gf} HL_A={hl_a} K_A={k_a}: A dev {pois_dev(d.assists, d.a):.5f} PTS dev {pois_dev(d.points, d.pts):.5f} "
          f"stars A {d.assists[star].sum() / d.a[star].sum():.3f} depth A {d.assists[low].sum() / d.a[low].sum():.3f}", flush=True)

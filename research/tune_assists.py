"""Re-tune the assist pieces' shrinkage with role priors in place: validation = 2023-24
(fit on 2022-23), scored on deviance and on the star tier's actual/projected ratio."""
import itertools
import numpy as np, pandas as pd
import backtest, features, model
from backtest import pois_dev
pg = pd.read_parquet("data/lake/player_games.parquet")
ls = pg[pg.game_type == 2].groupby(["player", "season"]).agg(n=("points", "size"), ppg=("points", "mean")).reset_index()
ls["season"] += 1
for kgf, ksh, hl in ((120, 20, 40), (120, 20, 80), (120, 20, 160), (120, 40, 80)):
    features.K_GF = {"ev": kgf, "pp": kgf / 4}
    features.K_ASHARE = ksh
    features.HL_ASSIST_SHARE = hl
    f = backtest.feature_table("last")
    co = model.fit(f[f.season == 2022])
    te = f[(f.season == 2023) & (f.toi > 0) & (f.gp_16 > 0) & (f.game_type == 2)]
    p = model.predict(te, co)
    d = te[["player", "season", "assists", "points"]].assign(a=p.a_hat, pts=p.pts_hat).merge(ls, on=["player", "season"], how="left")
    star = d.ppg.ge(0.8) & d.n.ge(20)
    low = d.ppg.lt(0.3) & d.n.ge(20)
    print(f"K_GF={kgf:3d} K_ASHARE={ksh:2d} HL={hl}: A dev {pois_dev(d.assists, d.a):.5f}  PTS dev {pois_dev(d.points, d.pts):.5f}"
          f"  stars A a/p {d.assists[star].sum() / d.a[star].sum():.3f}  low A a/p {d.assists[low].sum() / d.a[low].sum():.3f}", flush=True)

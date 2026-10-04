"""Lighter regression: scale every shrinkage constant and rerun the walk-forward backtest."""
import numpy as np, pandas as pd
import backtest, features as F, model
from backtest import pois_dev
pg = pd.read_parquet("data/lake/player_games.parquet")
ls = pg[pg.game_type == 2].groupby(["player", "season"]).agg(ls_n=("points", "size"), ls_ppg=("points", "mean")).reset_index(); ls["season"] += 1
base = dict(K_SOG=dict(F.K_SOG), K_SHOOT=F.K_SHOOT, K_GF=dict(F.K_GF), K_ASHARE=F.K_ASHARE)
for scale in (1.0, 0.75, 0.5):
    F.K_SOG = {k: v * scale for k, v in base["K_SOG"].items()}; F.K_SHOOT = base["K_SHOOT"] * scale
    F.K_GF = {k: v * scale for k, v in base["K_GF"].items()}; F.K_ASHARE = base["K_ASHARE"] * scale
    f = backtest.feature_table("last")
    rows = []
    for yr in (2023, 2024, 2025):
        co = model.fit(f[f.season.between(2021, yr - 1)])
        te = f[(f.season == yr) & (f.toi > 0) & (f.gp_16 > 0) & (f.game_type == 2)]
        p = model.predict(te, co)
        rows.append(te[["player", "season", "sog", "goals", "assists", "points"]].assign(s=p.sog_hat.values, g=p.g_hat.values, a=p.a_hat.values, pt=p.pts_hat.values))
    d = pd.concat(rows).merge(ls, on=["player", "season"], how="left")
    star = d.ls_ppg.ge(0.9) & d.ls_n.ge(20); low = d.ls_ppg.lt(0.3) & d.ls_n.ge(20)
    r = lambda m, y, h: d[y][m].sum() / d[h][m].sum()
    print(f"regression x{scale:.2f}: dev SOG {pois_dev(d.sog, d.s):.4f} G {pois_dev(d.goals, d.g):.4f} A {pois_dev(d.assists, d.a):.4f} PTS {pois_dev(d.points, d.pt):.4f} | "
          f"stars a/p PTS {r(star, 'points', 'pt'):.3f} G {r(star, 'goals', 'g'):.3f} SOG {r(star, 'sog', 's'):.3f} | depth PTS {r(low, 'points', 'pt'):.3f}", flush=True)

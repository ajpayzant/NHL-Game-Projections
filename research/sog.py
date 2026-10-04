"""Shot rates: recency, shrinkage, strength split, opponent and rink. Uses ACTUAL ice time so
only the rate is being judged."""
import itertools
import numpy as np
import pandas as pd

import ew
from research.common import load, pois_dev, mae

pg, tg = load()
pg = pg[pg.toi > 0].reset_index(drop=True)
pg["one"] = 1.0
ST = ("ev", "pp", "sh")
cols = [f"sog_{s}" for s in ST] + [f"toi_{s}" for s in ST]
val = pg.season.eq(2022) & pg.game_type.eq(2)
test = pg.season.between(2023, 2025) & pg.game_type.eq(2)
prior = {(F, s): pg.loc[pg.F == F, f"sog_{s}"].sum() / (pg.loc[pg.F == F, f"toi_{s}"].sum() / 60)
         for F in (True, False) for s in ST}
print("prior sog/60:", {k: round(v, 2) for k, v in prior.items()})
y = pg.sog


def predict(e, k, split=True):
    mu = 0
    if split:
        for s in ST:
            pr = np.where(pg.F, prior[(True, s)], prior[(False, s)])
            ks = k if s == "ev" else k / 3
            rate = (e[f"sog_{s}"] + ks * pr) / (e[f"toi_{s}"] / 60 + ks)
            mu = mu + rate * pg[f"toi_{s}"] / 60
    else:
        pr = np.where(pg.F, prior[(True, "ev")], prior[(False, "ev")]) * 1.05
        tot_s = sum(e[f"sog_{s}"] for s in ST); tot_t = sum(e[f"toi_{s}"] for s in ST) / 60
        mu = (tot_s + k * pr) / (tot_t + k) * pg.toi / 60
    return mu


best = None
for hl, gap in itertools.product((10, 20, 40, 80), (0, 20, 60)):
    e = ew.pregame(pg, "player", cols, hl, gap)
    for k in (15, 30, 60, 120):
        d = pois_dev(y[val], predict(e, k)[val])
        if best is None or d < best[0]:
            best = (d, hl, gap, k)
    print(f"hl={hl} gap={gap}: best so far {best}", flush=True)
d, hl, gap, k = best
e = ew.pregame(pg, "player", cols, hl, gap)
mu = predict(e, k)
mu_tot = predict(e, k, split=False)
print(f"\nchosen hl={hl} gap={gap} k={k}")
print(f"test: split dev {pois_dev(y[test], mu[test]):.4f} MAE {mae(y[test], mu[test]):.3f} | "
      f"unsplit dev {pois_dev(y[test], mu_tot[test]):.4f}")
# baselines with actual TOI scaled: season-to-date sog per game (no TOI knowledge)
s2d = pg.groupby(["player", "season"]).sog.transform(lambda s: s.shift(1).expanding().mean())
l10 = pg.groupby("player").sog.transform(lambda s: s.shift(1).rolling(10, min_periods=1).mean())
m = test & s2d.notna() & l10.notna()
print(f"baselines (per game, no TOI): season-to-date dev {pois_dev(y[m], s2d[m].clip(0.05)):.4f} "
      f"MAE {mae(y[m], s2d[m]):.3f}; last-10 dev {pois_dev(y[m], l10[m].clip(0.05)):.4f}; model on same rows {pois_dev(y[m], mu[m]):.4f}")
pg["mu_sog"] = mu
pg[["game_id", "player", "mu_sog"]].to_parquet("data/research_sog_mu.parquet")

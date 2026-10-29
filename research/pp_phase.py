"""Season-stage offsets for team PP / PK minutes (model.PHASES): walk-forward with and without.
python -m research.pp_phase  -> research/pp_phase_out.txt"""
from pathlib import Path

import numpy as np
import pandas as pd

import backtest as B
import model

OUT = Path(__file__).with_name("pp_phase_out.txt")
f = B.feature_table("last")
rows, lines = [], []
for yr in (2023, 2024, 2025, 2026):
    train = f[f.season.between(B.FIRST_TRAIN, yr - 1)]
    test = f[(f.season == yr) & (f.toi > 0) & (f.game_type == 2)]
    for on in (False, True):
        model.USE_PHASES = on
        co = model.fit(train)
        p = model.predict(test, co)
        d = test[["team_gn", "toi", "toi_pp", "sog", "goals", "assists", "points", "ppp"]].assign(
            **{c: p[c] for c in ("toi_hat", "toi_pp_hat", "sog_hat", "g_hat", "a_hat", "pts_hat", "ppp_hat")})
        d["phase"] = pd.cut(d.team_gn, [0, 14, 30, 50, 90], labels=["1-14", "15-30", "31-50", "51-82"])
        for ph, g in list(d.groupby("phase", observed=True)) + [("all", d)]:
            rows.append({"season": yr, "phases": on, "games": ph, "n": len(g),
                         "PP TOI ratio": g.toi_pp_hat.sum() / (g.toi_pp.sum() / 60),
                         "PPP ratio": g.ppp_hat.sum() / g.ppp.sum(),
                         "TOI MAE": np.abs(g.toi_hat - g.toi / 60).mean(),
                         "PP TOI MAE": np.abs(g.toi_pp_hat - g.toi_pp / 60).mean(),
                         "PPP dev": B.pois_dev(g.ppp, g.ppp_hat), "PTS dev": B.pois_dev(g.points, g.pts_hat),
                         "SOG dev": B.pois_dev(g.sog, g.sog_hat), "G dev": B.pois_dev(g.goals, g.g_hat)})
    print(yr, "done", flush=True)
model.USE_PHASES = True
r = pd.DataFrame(rows)
pd.set_option("display.width", 220)
for yr in (2023, 2024, 2025, 2026):
    lines.append(f"\n{yr}:\n" + r[r.season == yr].drop(columns="season").round(4).to_string(index=False))
# pooled 2023-25 (full seasons) by stage, before -> after
pool = r[r.season < 2026].copy()
w = pool.assign(wn=pool.n)
agg = {}
for (on, ph), g in w.groupby(["phases", "games"]):
    agg[(on, ph)] = {k: np.average(g[k], weights=g.n) for k in ("PP TOI ratio", "PPP ratio", "TOI MAE", "PP TOI MAE", "PPP dev", "PTS dev", "SOG dev", "G dev")}
t = pd.DataFrame(agg).T
t.index.names = ["phases", "games"]
lines.append("\npooled 2023-26 full seasons (False = current model, True = with season stages):\n" + t.round(4).to_string())
OUT.write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))

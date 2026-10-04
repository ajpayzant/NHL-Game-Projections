"""Linemate effect when linemates changed; opponent goal suppression for assists."""
import numpy as np
import pandas as pd
import statsmodels.api as sm

import ew
from research.common import load, pois_dev

pg, tg = load()
pg = pg[pg.toi > 0].reset_index(drop=True)
pg = pg.merge(pd.read_parquet("data/research_ast.parquet"), on=["game_id", "player"])
# usual linemates: did tonight's linemates both appear among his last game's two?
pg = pg.sort_values(["date", "game_id"]).reset_index(drop=True)
prev = pg.groupby("player")[["lm_f1", "lm_f2"]].shift(1)
now = pg[["lm_f1", "lm_f2"]]
same = [len({a, b} & {c, d}) for a, b, c, d in zip(now.lm_f1, now.lm_f2, prev.lm_f1, prev.lm_f2)]
pg["n_same"] = same
# opponent: EW goals against per 60 at EV+PK and xGA, relative to league
evs = pg.groupby(["game_id", "team"]).toi_ev.sum().div(5).rename("ev_secs").reset_index()
tg = tg.merge(evs, on=["game_id", "team"])
tg["ga_all"] = tg.goals_against; tg["mins"] = (tg.ev_secs + tg.sh_secs + tg.pp_secs) / 60
tg["xga_all"] = tg.xga_ev + tg.xga_sh + tg.xga_pp
e = ew.pregame(tg, "team", ["ga_all", "xga_all", "mins"], 40, 10)
lg_ga = tg.ga_all.sum() / tg.mins.sum(); lg_x = tg.xga_all.sum() / tg.mins.sum()
tg["opp_ga"] = (e.ga_all + 180 * lg_ga) / (e.mins + 180) / lg_ga
tg["opp_xga"] = (e.xga_all + 180 * lg_x) / (e.mins + 180) / lg_x
pg = pg.merge(tg[["game_id", "team", "opp_ga", "opp_xga"]].rename(columns={"team": "opp"}), on=["game_id", "opp"])
d = pg[pg.game_type.eq(2) & pg.has_shifts & pg.F].copy()
X = sm.add_constant(pd.DataFrame({
    "mate": np.log((d.mate_q_pts / d.q_pts).fillna(1).clip(0.3, 3)),
    "mate_chg": np.log((d.mate_q_pts / d.q_pts).fillna(1).clip(0.3, 3)) * (d.n_same < 2),
    "opp_ga": np.log(d.opp_ga), "opp_xga": np.log(d.opp_xga), "home": d.is_home.astype(float)}))
tr = d.season.between(2021, 2022).to_numpy(); te = d.season.between(2023, 2025).to_numpy()
yy = d.assists.to_numpy(); off = np.log(d.mu_b.clip(1e-4)).to_numpy()
print(f"forwards; linemates same as last game: both {np.mean(d.n_same == 2):.1%}, one {np.mean(d.n_same == 1):.1%}, none {np.mean(d.n_same == 0):.1%}")
for cols in (["const"], ["const", "home"], ["const", "home", "opp_ga"], ["const", "home", "opp_xga"],
             ["const", "home", "opp_xga", "mate"], ["const", "home", "opp_xga", "mate", "mate_chg"]):
    m = sm.GLM(yy[tr], X.loc[tr, cols], family=sm.families.Poisson(), offset=off[tr]).fit()
    p = m.predict(X.loc[te, cols], offset=off[te])
    chg = (d.n_same < 2).to_numpy()[te]
    print(f"{'+'.join(cols):42s} test {pois_dev(yy[te], p):.5f}  (changed-linemate rows {pois_dev(yy[te][chg], p[chg]):.5f}) coefs {dict(zip(cols, np.round(m.params, 3)))}")

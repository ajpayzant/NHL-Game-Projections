"""Context on top of the base shot rate: opponent, rink, home, recent form, rest."""
import numpy as np
import pandas as pd
import statsmodels.api as sm

import ew
from research.common import load, pois_dev

pg, tg = load()
pg = pg[pg.toi > 0].reset_index(drop=True)
mu = pd.read_parquet("data/research_sog_mu.parquet")
pg = pg.merge(mu, on=["game_id", "player"])
# team EV seconds ~ skater EV seconds / 5
evs = pg.groupby(["game_id", "team"]).toi_ev.sum().div(5).rename("ev_secs").reset_index()
tg = tg.merge(evs, on=["game_id", "team"], how="left")
tg["one"] = 1.0
tg["tot_sog"] = tg.sog + tg.sog_against
for hl in (20, 82):
    e = ew.pregame(tg, "team", ["sa_ev", "ev_secs", "sa_pp", "sh_secs", "one", "sf_ev", "sog_against", "sog"], hl, 10)
    lg_ev = tg.sa_ev.sum() / tg.ev_secs.sum()
    lg_pk = tg.sa_pp.sum() / tg.sh_secs.sum()
    k = 3 * 3000  # ~3 games of league-average prior
    tg[f"opp_ev_{hl}"] = (e.sa_ev + k * lg_ev) / (e.ev_secs + k) / lg_ev  # what this team allows
    tg[f"opp_pk_{hl}"] = (e.sa_pp + 2000 * lg_pk) / (e.sh_secs + 2000) / lg_pk
# rink: total SOG in a team's home games vs its road games
tg = tg.sort_values(["date", "game_id"]).reset_index(drop=True)
home = tg[tg.is_home].copy(); away = tg[~tg.is_home].copy()
eh = ew.pregame(home.assign(one=1.0), "team", ["tot_sog", "one"], 160, 0)
ea = ew.pregame(away.assign(one=1.0), "team", ["tot_sog", "one"], 160, 0)
home["rink_h"] = (eh.tot_sog + 20 * 60) / (eh.one + 20)
away["rink_a"] = (ea.tot_sog + 20 * 60) / (ea.one + 20)
# latest road average for each team as of each home game
ra = away[["date", "team", "rink_a"]].sort_values("date")
home = pd.merge_asof(home.sort_values("date"), ra, on="date", by="team", direction="backward")
home["rink"] = home.rink_h / home.rink_a.fillna(60)
tg = tg.merge(home[["game_id", "rink"]], on="game_id", how="left")
# rest: days since the team's previous game
tg["rest"] = tg.groupby("team").date.diff().dt.days.clip(upper=4).fillna(4)
opp_cols = ["game_id", "team"] + [c for c in tg.columns if c.startswith("opp_")]
o = tg[opp_cols].rename(columns={"team": "opp"})
pg = pg.merge(o, on=["game_id", "opp"], how="left")
pg = pg.merge(tg[["game_id", "team", "rink", "rest"]], on=["game_id", "team"], how="left")
pg = pg.merge(tg[["game_id", "team", "rest"]].rename(columns={"team": "opp", "rest": "opp_rest"}), on=["game_id", "opp"])
# recent form: short-window shot rate vs the base
e5 = ew.pregame(pg.assign(t=pg.toi / 60), "player", ["sog", "t"], 5, 0)
pg["form"] = ((e5.sog + 3) / (e5.t + 3 / (pg.mu_sog / (pg.toi / 60)).clip(0.01)) / (pg.mu_sog / (pg.toi / 60))).clip(0.3, 3)

d = pg[pg.game_type.eq(2) & pg.rink.notna()].copy()
ev_frac = d.toi_ev / d.toi
X = pd.DataFrame({
    "log_opp": np.log(ev_frac * d.opp_ev_20 + (1 - ev_frac) * d.opp_pk_20),
    "log_opp82": np.log(ev_frac * d.opp_ev_82 + (1 - ev_frac) * d.opp_pk_82),
    "log_rink": np.log(d.rink),
    "home": d.is_home.astype(float),
    "log_form": np.log(d.form),
    "b2b": (d.rest == 1).astype(float),
    "opp_b2b": (d.opp_rest == 1).astype(float),
})
X = sm.add_constant(X)
train = d.season.between(2021, 2022).to_numpy(); test = d.season.between(2023, 2025).to_numpy()
y = d.sog.to_numpy(); off = np.log(d.mu_sog.clip(0.01)).to_numpy()
print(f"base only test dev {pois_dev(y[test], d.mu_sog[test]):.4f}")
for cols in (["const"], ["const", "log_opp"], ["const", "log_opp82"], ["const", "log_opp", "log_opp82"],
             ["const", "log_opp", "log_rink"], ["const", "log_opp", "log_rink", "home"],
             ["const", "log_opp", "log_rink", "home", "log_form"],
             ["const", "log_opp", "log_rink", "home", "log_form", "b2b", "opp_b2b"]):
    m = sm.GLM(y[train], X.loc[train, cols], family=sm.families.Poisson(), offset=off[train]).fit()
    p = m.predict(X.loc[test, cols], offset=off[test])
    print(f"{'+'.join(cols[1:]) or 'recal':45s} test dev {pois_dev(y[test], p):.4f}  coefs {dict(zip(cols, np.round(m.params, 3)))}")

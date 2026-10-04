"""Goals: shooting % vs shot quality (ixG) vs direct goal rate; the opposing goalie.
Uses actual ice time and the base shot model, so the finishing step is what is judged."""
import itertools
import numpy as np
import pandas as pd
import statsmodels.api as sm

import config as C
import ew
from research.common import load, pois_dev

pg, tg = load()
pg = pg[pg.toi > 0].reset_index(drop=True)
pg = pg.merge(pd.read_parquet("data/research_sog_mu.parquet"), on=["game_id", "player"])
ST = ("ev", "pp", "sh")
# goals excluding empty-net (they are scored at an empty net, sh% is about goalies)
for s in ST:
    pg[f"gn_{s}"] = pg[f"g_{s}"] - pg[f"eng_{s}"]
pg["g_all"] = pg.goals
pg["sog_all"] = sum(pg[f"sog_{s}"] for s in ST)
pg["ixg_all"] = sum(pg[f"ixg_{s}"] for s in ST)
pg["t"] = pg.toi / 60
train = pg.season.between(2021, 2022) & pg.game_type.eq(2)
val = pg.season.eq(2022) & pg.game_type.eq(2)
test = pg.season.between(2023, 2025) & pg.game_type.eq(2)
y = pg.goals.to_numpy()
lg_sh = {F: pg.loc[pg.F == F, "g_all"].sum() / pg.loc[pg.F == F, "sog_all"].sum() for F in (True, False)}
lg_g60 = {F: pg.loc[pg.F == F, "g_all"].sum() / pg.loc[pg.F == F, "t"].sum() for F in (True, False)}
lg_conv = {F: pg.loc[pg.F == F, "g_all"].sum() / pg.loc[pg.F == F, "ixg_all"].sum() for F in (True, False)}
print("league sh%", {k: round(v, 4) for k, v in lg_sh.items()}, "goals/ixg", {k: round(v, 3) for k, v in lg_conv.items()})
res = {}
for hl in (40, 80, 160, 320):
    e = ew.pregame(pg, "player", ["g_all", "sog_all", "ixg_all", "t"], hl, 20)
    for k in (50, 100, 200, 400, 800):
        F = pg.F.to_numpy()
        prior_sh = np.where(F, lg_sh[True], lg_sh[False])
        sh = (e.g_all + k * prior_sh) / (e.sog_all + k)
        res[("sh%", hl, k)] = pg.mu_sog * sh
        # ixG per shot (shot quality) shrunk, times finishing (goals / ixG) shrunk
        xq = (e.ixg_all + 30 * prior_sh) / (e.sog_all + 30)
        fin = (e.g_all + k * 0.1 * np.where(F, lg_conv[True], lg_conv[False])) / (e.ixg_all + k * 0.1)
        res[("ixg*fin", hl, k)] = pg.mu_sog * xq * fin
        res[("ixg only", hl, k)] = pg.mu_sog * (e.ixg_all + k * 0.2 * prior_sh) / (e.sog_all + k * 0.2) * np.where(F, lg_conv[True], lg_conv[False])
        kt = k / 10
        res[("g/60", hl, k)] = pg.t * (e.g_all + kt * np.where(F, lg_g60[True], lg_g60[False])) / (e.t + kt)
best = {}
for (kind, hl, k), mu in res.items():
    d = pois_dev(y[val], mu[val])
    if kind not in best or d < best[kind][0]:
        best[kind] = (d, hl, k)
for kind, (d, hl, k) in best.items():
    mu = res[(kind, hl, k)]
    print(f"{kind:10s} hl={hl:3d} k={k:3d}: val dev {d:.5f}  test dev {pois_dev(y[test], mu[test]):.5f}")
kind, (d, hl, k) = min(best.items(), key=lambda kv: kv[1][0])
pg["mu_g"] = res[(kind, hl, k)]
print("using", kind, hl, k)

# opposing goalie: shrunk GSAx per shot faced (EW), from the shots table
sh = pd.read_parquet(C.LAKE / "shots_xg.parquet", columns=["game_id", "date", "season", "goalie", "event", "xg_on", "target_net_empty"])
sh = sh[sh.event.isin(["goal", "sog"]) & ~sh.target_net_empty & sh.goalie.notna()]
gg = sh.groupby(["game_id", "goalie"]).agg(date=("date", "first"), season=("season", "first"),
     sa=("event", "size"), ga=("event", lambda s: (s == "goal").sum()), xga=("xg_on", "sum")).reset_index()
gg["date"] = pd.to_datetime(gg.date.astype(str)); gg = gg.sort_values(["date", "game_id"]).reset_index(drop=True)
e = ew.pregame(gg, "goalie", ["sa", "ga", "xga"], 60, 10)
gg["gsax_ps"] = (e.xga - e.ga) / (e.sa + 600)  # ~600 shots of league-average prior
# who faced this team's shots: the opposing goalie with most shots faced
opp_g = sh.merge(pd.read_parquet(C.LAKE / "shots_xg.parquet", columns=["game_id", "team"]).iloc[:0], how="left") if False else None
sg = pd.read_parquet(C.LAKE / "shots_xg.parquet", columns=["game_id", "team", "goalie", "event", "target_net_empty"])
sg = sg[sg.goalie.notna() & sg.event.isin(["goal", "sog"])]
main = sg.groupby(["game_id", "team", "goalie"]).size().reset_index(name="n").sort_values("n").groupby(["game_id", "team"]).tail(1)
main = main.merge(gg[["game_id", "goalie", "gsax_ps"]], on=["game_id", "goalie"], how="left")
pg = pg.merge(main[["game_id", "team", "gsax_ps"]], on=["game_id", "team"], how="left")
# opponent defence: xG against per shot (shot quality allowed)
tg["one"] = 1.0
et = ew.pregame(tg, "team", ["xga_ev", "sa_ev", "ga_ev"], 40, 10)
tg["opp_xq"] = ((et.xga_ev + 50 * 0.09) / (et.sa_ev + 500)) / 0.09
pg = pg.merge(tg[["game_id", "team", "opp_xq"]].rename(columns={"team": "opp"}), on=["game_id", "opp"], how="left")
d = pg[pg.game_type.eq(2)].copy()
X = sm.add_constant(pd.DataFrame({"gsax": d.gsax_ps.fillna(0) * 100, "log_xq": np.log(d.opp_xq.fillna(1)),
                                  "home": d.is_home.astype(float)}))
tr = d.season.between(2021, 2022).to_numpy(); te = d.season.between(2023, 2025).to_numpy()
off = np.log(d.mu_g.clip(1e-4)).to_numpy(); yy = d.goals.to_numpy()
print(f"base test dev {pois_dev(yy[te], d.mu_g[te]):.5f}")
for cols in (["const"], ["const", "gsax"], ["const", "gsax", "log_xq"], ["const", "gsax", "log_xq", "home"]):
    m = sm.GLM(yy[tr], X.loc[tr, cols], family=sm.families.Poisson(), offset=off[tr]).fit()
    print(f"{'+'.join(cols):30s} test dev {pois_dev(yy[te], m.predict(X.loc[te, cols], offset=off[te])):.5f} coefs {dict(zip(cols, np.round(m.params, 3)))}")

"""Assists: direct rate vs on-ice goals x assist share; does linemate quality add anything?
Actual ice time, so the rate step is what is judged."""
import numpy as np
import pandas as pd
import statsmodels.api as sm

import ew
from research.common import load, pois_dev

pg, tg = load()
pg = pg[pg.toi > 0].reset_index(drop=True)
ST = ("ev", "pp", "sh")
pg["t_ev"], pg["t_pp"], pg["t_sh"] = pg.toi_ev / 60, pg.toi_pp / 60, pg.toi_sh / 60
for s in ST:  # ice time with on-ice data available
    pg[f"ton_{s}"] = pg[f"t_{s}"].where(pg.has_shifts)
train = pg.season.between(2021, 2022) & pg.game_type.eq(2)
val = pg.season.eq(2022) & pg.game_type.eq(2)
test = pg.season.between(2023, 2025) & pg.game_type.eq(2)
y = pg.assists.to_numpy()
F = pg.F.to_numpy()
lg = lambda num, den, f: pg.loc[pg.F == f, num].sum() / pg.loc[pg.F == f, den].sum()

# (a) direct assist rate by strength
best = None
for hl in (40, 80, 160):
    e = ew.pregame(pg, "player", [f"a_{s}" for s in ST] + [f"t_{s}" for s in ST], hl, 20)
    for k in (30, 60, 120, 240):
        mu = 0
        for s in ST:
            pr = np.where(F, lg(f"a_{s}", f"t_{s}", True), lg(f"a_{s}", f"t_{s}", False))
            ks = k if s == "ev" else k / 4
            mu = mu + pg[f"t_{s}"] * (e[f"a_{s}"] + ks * pr) / (e[f"t_{s}"] + ks)
        d = pois_dev(y[val], mu[val])
        if best is None or d < best[0]:
            best = (d, hl, k, mu)
d, hl_a, k_a, mu_a = best
print(f"(a) direct rate hl={hl_a} k={k_a}: val {d:.5f} test {pois_dev(y[test], mu_a[test]):.5f}")

# (b) on-ice goals for x assist share, by strength
cols = [f"ogf_{s}" for s in ("ev", "pp")] + [f"oxgf_{s}" for s in ("ev", "pp")] + [f"ton_{s}" for s in ("ev", "pp")] + [f"a_{s}" for s in ST] + ["a_on_ev", "a_on_pp"]
pg["a_on_ev"] = pg.a_ev.where(pg.has_shifts); pg["a_on_pp"] = pg.a_pp.where(pg.has_shifts)
best = None
for hl_g in (20, 40, 80, 160):
    e = ew.pregame(pg, "player", cols, hl_g, 20)
    e_long = ew.pregame(pg, "player", ["a_on_ev", "a_on_pp", "ogf_ev", "ogf_pp"], 160, 20)
    for kg in (30, 60, 120):
        for ks in (5, 10, 20):
            mu = 0
            for s in ("ev", "pp"):
                lgf = lg(f"ogf_{s}", f"ton_{s}", True) if True else 0
                pr_gf = np.where(F, lg(f"ogf_{s}", f"ton_{s}", True), lg(f"ogf_{s}", f"ton_{s}", False))
                k1 = kg if s == "ev" else kg / 4
                gf60 = (e[f"ogf_{s}"] + k1 * pr_gf) / (e[f"ton_{s}"] + k1)
                pr_sh = np.where(F, lg(f"a_on_{s}", f"ogf_{s}", True), lg(f"a_on_{s}", f"ogf_{s}", False))
                share = (e_long[f"a_on_{s}"] + ks * pr_sh) / (e_long[f"ogf_{s}"] + ks)
                mu = mu + pg[f"t_{s}"] * gf60 * share
            mu = mu + pg.t_sh * np.where(F, lg("a_sh", "t_sh", True), lg("a_sh", "t_sh", False))
            dd = pois_dev(y[val], mu[val])
            if best is None or dd < best[0]:
                best = (dd, hl_g, kg, ks, mu)
d, hl_g, kg, ks, mu_b = best
print(f"(b) on-ice GF x share hl={hl_g} kgf={kg} kshare={ks}: val {d:.5f} test {pois_dev(y[test], mu_b[test]):.5f}")
pg["mu_a"], pg["mu_b"] = mu_a, mu_b

# linemate quality: tonight's linemates' (and last game's linemates') pregame on-ice EV GF/60 and points/60 vs the player's own
e = ew.pregame(pg.assign(p=pg.points), "player", ["ogf_ev", "ton_ev", "oxgf_ev", "p", "t_ev", "t_pp", "t_sh"], 80, 20)
pr = pg.ogf_ev.sum() / pg.ton_ev.sum()
pg["q_gf"] = (e.ogf_ev + 60 * pr) / (e.ton_ev + 60)
pg["q_xgf"] = (e.oxgf_ev + 60 * pg.oxgf_ev.sum() / pg.ton_ev.sum()) / (e.ton_ev + 60)
tt = e.t_ev + e.t_pp + e.t_sh
pg["q_pts"] = (e.p + 60 * pg.points.sum() / (pg.toi.sum() / 60)) / (tt + 60)
q = pg.set_index(["game_id", "player"])[["q_gf", "q_xgf", "q_pts"]]
def mates(c1, c2):
    out = {}
    for qc in ("q_gf", "q_xgf", "q_pts"):
        vals = []
        for c in (c1, c2):
            k = pd.MultiIndex.from_arrays([pg.game_id, pg[c].fillna(-1).astype(int)])
            vals.append(q[qc].reindex(k).to_numpy())
        out[qc] = np.nanmean(np.column_stack(vals), axis=1)
    return out
m_now = mates("lm_f1", "lm_f2")
# D partner for D: use lm_d1 and the forwards he was with most
m_now_d = mates("lm_d1", "lm_f1")
for qc in ("q_gf", "q_xgf", "q_pts"):
    pg[f"mate_{qc}"] = np.where(F, m_now[qc], m_now_d[qc])
d = pg[pg.game_type.eq(2) & pg.has_shifts].copy()
rel = lambda c: np.log((d[f"mate_{c}"] / d[c]).fillna(1).clip(0.3, 3))
team_gf = None
X = sm.add_constant(pd.DataFrame({"mate_gf": rel("q_gf"), "mate_xgf": rel("q_xgf"), "mate_pts": rel("q_pts"),
                                  "home": d.is_home.astype(float)}))
tr = d.season.between(2021, 2022).to_numpy(); te = d.season.between(2023, 2025).to_numpy()
yy = d.assists.to_numpy()
for base in ("mu_a", "mu_b"):
    off = np.log(d[base].clip(1e-4)).to_numpy()
    print(f"{base}: base test {pois_dev(yy[te], d[base][te]):.5f}")
    for cols in (["const"], ["const", "mate_gf"], ["const", "mate_xgf"], ["const", "mate_pts"], ["const", "mate_pts", "mate_xgf", "home"]):
        m = sm.GLM(yy[tr], X.loc[tr, cols], family=sm.families.Poisson(), offset=off[tr]).fit()
        print(f"   {'+'.join(cols):34s} test {pois_dev(yy[te], m.predict(X.loc[te, cols], offset=off[te])):.5f} coefs {dict(zip(cols, np.round(m.params, 3)))}")
pg[["game_id", "player", "mu_a", "mu_b", "mate_q_pts", "q_pts", "mate_q_xgf", "q_xgf"]].to_parquet("data/research_ast.parquet")

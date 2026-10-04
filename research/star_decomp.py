import numpy as np, pandas as pd
import backtest, model
pg = pd.read_parquet("data/lake/player_games.parquet")
ls = pg[pg.game_type == 2].groupby(["player", "season"]).agg(ls_n=("points", "size"), ls_ppg=("points", "mean")).reset_index(); ls["season"] += 1
f = backtest.feature_table("last")
co = model.fit(f[f.season.between(2022, 2023)])
te = f[(f.season.between(2024, 2025)) & (f.toi > 0) & (f.gp_16 > 0) & (f.game_type == 2) & f.toi_5v5.notna()]
p = model.predict(te, co)
mult = model.multipliers(te, p, co)["a"]
d = te.assign(**{c: p[c] for c in p.columns}).merge(ls, on=["player", "season"], how="left")
d.index = te.index
d["tier"] = pd.cut(d.ls_ppg.where(d.ls_n >= 20), [-1, .3, .6, .8, 1, 5])
g = d.groupby("tier", observed=True)
out = pd.DataFrame({
    "EV TOI a/p": g.t_ev.sum() / g.toi_ev_hat.sum(),
    "EV onGF/min a/p": g.apply(lambda s: s.ogf_ev.sum() / s.t_ev.sum() / ((s.gf_pm_ev * s.t_ev).sum() / s.t_ev.sum()), include_groups=False),
    "EV ashare a/p": g.apply(lambda s: s.a_ev.sum() / s.ogf_ev.sum() / ((s.ashare_ev * s.ogf_ev).sum() / s.ogf_ev.sum()), include_groups=False),
    "PP onGF/min a/p": g.apply(lambda s: s.ogf_pp.sum() / s.t_pp.sum() / ((s.gf_pm_pp * s.t_pp).sum() / s.t_pp.sum()), include_groups=False),
    "PP ashare a/p": g.apply(lambda s: s.a_pp.sum() / s.ogf_pp.sum() / ((s.ashare_pp * s.ogf_pp).sum() / s.ogf_pp.sum()), include_groups=False),
    "base A a/p": g.assists.sum() / g.base_a.sum(), "final A a/p": g.assists.sum() / g.a_hat.sum()})
print(out.round(3).to_string())
for c in mult.columns:
    print(c, mult.join(d.tier).groupby("tier", observed=True)[c].mean().round(3).to_dict())

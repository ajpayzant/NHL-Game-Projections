"""Leak-reduced oracle: tonight's trio/pair membership (what a line chart knows), ranked by
the members' PREGAME ice time rather than that night's, and PP/PK units ranked the same way."""
import numpy as np
import pandas as pd

pg = pd.read_parquet("data/research_toi.parquet")
pg = pg[pg.line.notna()].reset_index(drop=True)
# group id: the trio / pair this player belonged to tonight = sorted (player, lm...) of size 3/2
def grp_key(r):
    if r.F:
        mates = [r.player] + [int(x) for x in (r.lm_f1, r.lm_f2) if pd.notna(x)]
    else:
        mates = [r.player] + [int(x) for x in (r.lm_d1,) if pd.notna(x)]
    return tuple(sorted(mates))
pg["unit"] = [grp_key(r) for r in pg.itertuples()]
pg["pre_share"] = pg.ev_share_16.fillna(pg.groupby("grp").ev_share_16.transform("median"))
pg["unit_pre"] = pg.groupby(["game_id", "team", "unit"]).pre_share.transform("mean")
pg["line_pre"] = pg.groupby(["game_id", "team", "grp"]).unit_pre.rank(method="dense", ascending=False)
pg["line_pre"] = np.minimum(pg.line_pre, np.where(pg.F, 4, 3))
agree = (pg.line_pre == pg.line).mean()
print(f"pregame-ranked line equals that-night-ranked line in {agree:.1%} of rows")
print(pd.crosstab(pg.line[pg.F], pg.line_pre[pg.F], normalize="index").round(2))
# PP: rank by pregame PP share among players who got PP time tonight
pp = pg.pp_share_16.fillna(0)
r = pp.where(pg.toi_pp > 0).groupby([pg.game_id, pg.team]).rank(ascending=False, method="first")
pg["pp_pre"] = np.where(pg.toi_pp <= 0, 0, np.where(r <= 5, 1, np.where(r <= 10, 2, 0)))
print(f"PP unit agreement {(pg.pp_pre == pg.pp_unit).mean():.1%}")
pg[["line_pre", "pp_pre"]].to_parquet("data/research_slots.parquet")

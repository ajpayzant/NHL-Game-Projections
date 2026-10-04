"""Value of knowing tonight's line / PP unit / PK unit for ice time (oracle vs last game)."""
import numpy as np
import pandas as pd

import ew
from research.common import mae, rmse

pg = pd.read_parquet("data/research_toi.parquet")
pg = pg[pg.line.notna()].reset_index(drop=True)
pg["one"] = 1.0
# team usage of each slot: EW of the share a slot gets, per team (coach style)
specs = {"ev": ("line", "share_ev", "pool_ev"), "pp": ("pp_unit", "toi_pp", "pp_secs"), "sh": ("pk_unit", "toi_sh", "sh_secs")}
for st, (slot, num, den) in specs.items():
    key = pg.team.astype(str) + pg.grp + pg[slot].astype(int).astype(str)
    g = pg.assign(k=key).groupby(["k", "game_id"], sort=False).agg(
        n=(num, "sum" if st != "ev" else "mean"), d=(den, "first"), season=("season", "first"),
        date=("date", "first")).reset_index().sort_values(["date", "game_id"]).reset_index(drop=True)
    if st != "ev":
        cnt = pg.assign(k=key).groupby(["k", "game_id"]).size().rename("cnt").reset_index()
        g = g.merge(cnt, on=["k", "game_id"])
        g["n"] = g.n / g.cnt  # per-player share in this slot
    e = ew.pregame(g, "k", ["n", "d"], 20, 10)
    g["slot_share"] = e.n / e.d if st != "ev" else e.n / (ew.pregame(g.assign(o=1.0), "k", ["o"], 20, 10).o)
    m = dict(zip(zip(g.k, g.game_id), g.slot_share))
    pg[f"slot_{st}_actual"] = [m.get(x) for x in zip(key, pg.game_id)]
    pkey = pg.team.astype(str) + pg.grp + pg[f"prev_{slot}"].fillna(-1).astype(int).astype(str)
    pg[f"slot_{st}_prev"] = [m.get(x) for x in zip(pkey, pg.game_id)]

test_mask = pg.season.between(2023, 2025) & pg.game_type.eq(2) & pg.w_4.gt(0)
train_mask = pg.season.eq(2022) & pg.game_type.eq(2) & pg.w_4.gt(0)


def fit_eval(target, cols, label):
    d = pg[cols + [target]].copy()
    d[cols] = d[cols].fillna(d[cols].median())
    X = np.column_stack([np.ones(len(d)), d[cols].to_numpy()])
    fm = train_mask & d[target].notna()
    b = np.linalg.lstsq(X[fm], d.loc[fm, target], rcond=None)[0]
    p = X @ b
    return p, b


out = {}
for lab, suffix in (("last game's slots", "prev"), ("tonight's actual slots", "actual")):
    pred = 0
    for st, base, den in (("ev", "ev_share", "pool_ev"), ("pp", "pp_share", "pp_secs"), ("sh", "sh_share", "sh_secs")):
        cols = [f"{base}_4", f"{base}_16", f"slot_{st}_{suffix}"]
        tgt = {"ev": "share_ev", "pp": "toi_pp", "sh": "toi_sh"}[st]
        if st == "ev":
            p, b = fit_eval("share_ev", cols, lab)
            pred = pred + p * pg.pool_ev  # actual pool: isolates the share question
        else:
            pg["_t"] = pg[tgt] / pg[den].where(pg[den] > 0)
            p, b = fit_eval("_t", cols, lab)
            pred = pred + np.clip(p, 0, 1) * pg[den].fillna(0)
        print(f"  {lab} {st}: coefs {np.round(b, 3)}")
    out[lab] = pred
y = pg.toi[test_mask] / 60
for lab, p in out.items():
    print(f"{lab:24s}: MAE {mae(y, p[test_mask] / 60):.3f}  RMSE {rmse(y, p[test_mask] / 60):.3f} (team minutes known)")

"""How predictable is ice time, what drives it, and what do known lines add?"""
import numpy as np
import pandas as pd

import ew
from research.common import load, rmse, mae

pg, tg = load()
pg = pg[pg.toi > 0].reset_index(drop=True)
pg["one"] = 1.0
pg["share_ev"] = pg.toi_ev / pg.pool_ev
# previous game's deployment (what a 'last game' line chart shows)
pg = pg.sort_values(["date", "game_id"]).reset_index(drop=True)
for c in ("line", "pp_unit", "pk_unit"):
    pg[f"prev_{c}"] = pg.groupby("player")[c].shift(1)

res = []
for hl in (2, 4, 8, 16, 32):
    e = ew.pregame(pg, "player", ["one", "toi", "toi_ev", "toi_pp", "toi_sh", "pool_ev", "pp_secs", "sh_secs"], hl, 10)
    pg[f"toi_{hl}"] = e.toi / e.one
    pg[f"ev_share_{hl}"] = e.toi_ev / e.pool_ev
    pg[f"pp_share_{hl}"] = e.toi_pp / e.pp_secs.where(e.pp_secs > 0)
    pg[f"sh_share_{hl}"] = e.toi_sh / e.sh_secs.where(e.sh_secs > 0)
    pg[f"w_{hl}"] = e.one
pg["last10"] = pg.groupby("player").toi.transform(lambda s: s.shift(1).rolling(10, min_periods=1).mean())
pg["season_avg"] = pg.groupby(["player", "season"]).toi.transform(lambda s: s.shift(1).expanding().mean())

test = pg[(pg.season >= 2023) & (pg.season <= 2025) & pg.w_8.gt(0) & pg.game_type.eq(2)].copy()
print(f"test rows {len(test):,}")
y = test.toi / 60
for c in ["last10", "season_avg"] + [f"toi_{h}" for h in (2, 4, 8, 16, 32)]:
    m = test[c].notna()
    print(f"{c:12s} MAE {mae(y[m], test.loc[m, c] / 60):.3f}  RMSE {rmse(y[m], test.loc[m, c] / 60):.3f}")

# the oracle: if you knew the player's actual EV/PP/SH team minutes, how good is share x minutes?
for hl in (4, 8, 16):
    p = (test[f"ev_share_{hl}"] * test.pool_ev + test[f"pp_share_{hl}"].fillna(0) * test.pp_secs
         + test[f"sh_share_{hl}"].fillna(0) * test.sh_secs) / 60
    print(f"share x actual team minutes hl={hl}: MAE {mae(y, p):.3f}")

# slot changes: when last game's line differs from tonight's, how much does the miss grow?
t = test[test.line.notna() & test.prev_line.notna()]
same = t.line == t.prev_line
for lab, m in (("same line as last game", same), ("moved line", ~same)):
    print(f"{lab:24s} n={m.sum():,} ({m.mean():.1%})  MAE toi_8 {mae(t.toi[m] / 60, t.toi_8[m] / 60):.3f}")
pp_same = t.pp_unit == t.prev_pp_unit
print(f"PP unit changed in {1 - pp_same.mean():.1%} of games; MAE toi_8 same {mae(t.toi[pp_same]/60, t.toi_8[pp_same]/60):.3f} "
      f"changed {mae(t.toi[~pp_same]/60, t.toi_8[~pp_same]/60):.3f}")
# noise floor: game-to-game sd of TOI for the same player within a season around his season mean
within = pg[pg.game_type.eq(2)].groupby(["player", "season"]).toi.transform(lambda s: s - s.mean()) / 60
print(f"within-player-season sd of TOI {within.std():.3f} min, MAE {within.abs().mean():.3f}")
pg.to_parquet("data/research_toi.parquet")

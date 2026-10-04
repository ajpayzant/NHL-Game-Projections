import json, numpy as np, pandas as pd
import backtest, model
f = backtest.feature_table("last")
f.to_parquet("data/features_last.parquet")
for yr in (2023, 2024, 2025, 2026):
    co = model.fit(f[f.season.between(2022, yr - 1)])
    print(yr, {k: {kk: round(vv, 3) for kk, vv in co[k].items()} for k in ("sog", "g", "a")})
    print("   team", {k: {kk: round(vv, 3) for kk, vv in v.items()} for k, v in co["toi"]["team"].items()})
d = f[(f.toi > 0) & f.gp_16.gt(0) & (f.game_type == 2)]
print(d.groupby("season").apply(lambda s: pd.Series({"sog/min": s.sog.sum() / (s.toi.sum() / 60), "rate_ev": s.sog_pm_ev.mean(),
     "actual_ev_rate": s.sog_ev.sum() / s.t_ev.sum(), "g/sog": s.goals.sum() / s.sog.sum(), "sh_pct": s.sh_pct.mean()}), include_groups=False).round(4))

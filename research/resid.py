import numpy as np, pandas as pd
b = pd.read_parquet("data/backtest/backtest_last.parquet")
b = b[(b.gp_16 > 0) & (b.game_type == 2)]
b["toi_min"] = b.toi / 60
print("by season: actual / predicted")
print(b.groupby("season").apply(lambda s: pd.Series({"toi": s.toi_min.mean() / s.toi_hat.mean(), "sog": s.sog.mean() / s.sog_hat.mean(),
      "base_sog": s.sog.mean() / s.base_sog.mean(), "g": s.goals.mean() / s.g_hat.mean(), "a": s.assists.mean() / s.a_hat.mean()}), include_groups=False).round(3))
for c, y in (("sog_hat", "sog"), ("pts_hat", "points"), ("a_hat", "assists"), ("g_hat", "goals"), ("toi_hat", "toi_min")):
    q = pd.qcut(b[c], 10, duplicates="drop")
    t = b.groupby(q, observed=True).agg(pred=(c, "mean"), act=(y, "mean"))
    print(c, " ".join(f"{r.pred:.2f}:{r.act / r.pred:.2f}" for r in t.itertuples()))
print("sog ratio by season x month")
b["month"] = b.date.dt.month
print(b.groupby(["season", "month"]).apply(lambda s: s.sog.sum() / s.sog_hat.sum(), include_groups=False).unstack().round(3))

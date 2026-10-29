"""Are the early-season weaknesses noise that fades, or patterns that last? Walk-forward 2023-26
(data/backtest/backtest_last.parquet), split by the team's game number. python -m research.season_noise"""
import numpy as np
import pandas as pd

import config as C

b = pd.read_parquet(C.DATA / "backtest" / "backtest_last.parquet")
b = b[b.game_type == 2].copy()
pg = pd.read_parquet(C.LAKE / "player_games.parquet", columns=["game_id", "player", "toi_pp", "season", "game_type", "date"])
b = b.merge(pg[["game_id", "player", "toi_pp"]], on=["game_id", "player"], how="left")
b["date"] = pd.to_datetime(b.date)
b["gn"] = b.groupby(["season", "team"]).date.rank(method="dense")
b["phase"] = pd.cut(b.gn, [0, 14, 30, 50, 90], labels=["1-14", "15-30", "31-50", "51-82"])
prior = pg[pg.game_type == 2].groupby(["player", "season"]).size().rename("gp").reset_index()
prior["cum"] = prior.sort_values("season").groupby("player").gp.cumsum() - prior.gp     # NHL GP before that season
b = b.merge(prior[["player", "season", "cum"]], on=["player", "season"], how="left")
b["young"] = b.cum.fillna(0) < 82
out = []


def ratio(g, hat, act):
    return g[hat].sum() / max(g[act].sum(), 1e-9)


print("ratios are projected / actual (1.00 = unbiased); each cell pools 2023-24, 2024-25, 2025-26\n")
rows = []
for ph, g in b.groupby("phase", observed=True):
    y = g[g.young]
    s = g[g.pts_hat >= 0.9]
    rows.append({"games": ph, "n": len(g),
                 "PP TOI": ratio(g, "toi_pp_hat", "toi_pp") * 60, "PPP": ratio(g, "ppp_hat", "ppp"),
                 "<82GP PTS": ratio(y, "pts_hat", "points"), "<82GP TOI bias": (y.toi_hat - y.toi / 60).mean(),
                 "stars PTS": ratio(s, "pts_hat", "points"), "stars A": ratio(s, "a_hat", "assists"),
                 "all PTS": ratio(g, "pts_hat", "points"), "all SOG": ratio(g, "sog_hat", "sog")})
print(pd.DataFrame(rows).round(3).to_string(index=False))

print("\nby season, games 1-14 vs the rest:")
r2 = []
for (yr, early), g in b.groupby(["season", b.gn <= 14]):
    y = g[g.young]; s = g[g.pts_hat >= 0.9]
    r2.append({"season": yr, "games": "1-14" if early else "15+", "PP TOI": ratio(g, "toi_pp_hat", "toi_pp") * 60,
               "PPP": ratio(g, "ppp_hat", "ppp"), "<82GP PTS": ratio(y, "pts_hat", "points"),
               "stars PTS": ratio(s, "pts_hat", "points")})
print(pd.DataFrame(r2).round(3).to_string(index=False))

print("\nteam shots: spread of each team's projected/actual SOG ratio within a phase, vs pure noise,")
print("and whether a team's miss in one phase carries into the next (corr):")
for yr, g in b.groupby("season"):
    t = g.groupby(["team", "phase"], observed=True).agg(h=("sog_hat", "sum"), a=("sog", "sum")).reset_index()
    t["r"] = t.h / t.a
    w = t.pivot(index="team", columns="phase", values="r")
    noise = t.groupby("phase", observed=True).a.apply(lambda a: (np.sqrt(a) / a).mean())
    sd = w.std()
    cells = " | ".join(f"{ph}: sd {sd[ph]:.3f} vs noise {noise[ph]:.3f}" for ph in w.columns)
    corr = " ".join(f"{a}->{c} {w[a].corr(w[c]):+.2f}" for a, c in zip(w.columns[:-1], w.columns[1:]))
    print(f"  {yr}: {cells}\n        carry-over {corr}")

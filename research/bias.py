"""Does the model under-project stars or inflate depth players? Residual ratios by tier.
Tiers are defined from PREGAME information only (last season's points/game, career games),
because grouping by the actual outcome builds in regression bias."""
import numpy as np, pandas as pd
import config as C
b = pd.read_parquet(C.DATA / "backtest" / "backtest_last.parquet")
b = b[(b.game_type == 2) & (b.gp_16 > 0)].copy()
pg = pd.read_parquet(C.LAKE / "player_games.parquet")
ps = pg[pg.game_type == 2].groupby(["player", "season"]).agg(gp=("points", "size"), ppg=("points", "mean")).reset_index()
ps["season"] += 1
b = b.merge(ps.rename(columns={"gp": "ls_gp", "ppg": "ls_ppg"}), on=["player", "season"], how="left")
b["tier"] = pd.cut(b.ls_ppg.where(b.ls_gp >= 20), [-0.01, 0.2, 0.4, 0.6, 0.8, 1.0, 5], labels=["<0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0", "1.0+"]).astype(str)
b.loc[b.ls_ppg.isna() | (b.ls_gp < 20), "tier"] = "new / <20 GP"
car = pg.sort_values("date").groupby("player").cumcount()
b = b.merge(pg.assign(car=car)[["game_id", "player", "car"]], on=["game_id", "player"], how="left")
b["exp"] = pd.cut(b.car, [-1, 20, 82, 246, 10_000], labels=["<20 GP", "20-82", "82-246", "246+"])
b["toi_min"] = b.toi / 60
def tab(by):
    g = b.groupby(by, observed=True)
    return pd.DataFrame({"n": g.size(), "TOI a/p": g.toi_min.sum() / g.toi_hat.sum(), "SOG a/p": g.sog.sum() / g.sog_hat.sum(),
                         "G a/p": g.goals.sum() / g.g_hat.sum(), "A a/p": g.assists.sum() / g.a_hat.sum(),
                         "PTS a/p": g.points.sum() / g.pts_hat.sum(), "pts/gm": g.points.mean()}).round(3)
for by in ("tier", "pos", "exp", ["pos", "tier"]):
    print(f"\n--- actual / projected by {by} ---"); print(tab(by).to_string())
print("\n--- stars (last season 0.8+ pts/gm) and depth (<0.3) by season: actual / projected ---")
b["grpT"] = np.where(b.ls_ppg.ge(0.8) & b.ls_gp.ge(20), "star", np.where(b.ls_ppg.lt(0.3) & b.ls_gp.ge(20), "depth", "middle"))
print(b.groupby(["season", "grpT"]).apply(lambda s: pd.Series({"A": s.assists.sum() / s.a_hat.sum(), "G": s.goals.sum() / s.g_hat.sum(),
      "PTS": s.points.sum() / s.pts_hat.sum(), "SOG": s.sog.sum() / s.sog_hat.sum()}), include_groups=False).round(3).unstack().to_string())

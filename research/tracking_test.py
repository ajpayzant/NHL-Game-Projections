"""Test tracking.grade on a past day: a snapshot built from the backtest for that day, graded
with real historical odds and results.  python -m research.tracking_test 2026-01-15"""
import json
import sys

import pandas as pd

import config as C
import live
import model
import tracking as T
from research.oddsutil import NAME

day = sys.argv[1]
f = T._files(day)
tg = pd.read_parquet(C.LAKE / "team_games.parquet")
tg["team"], tg["opp"] = tg.team.astype(str), tg.opp.astype(str)
h = tg[tg.is_home & (pd.to_datetime(tg.date).dt.strftime("%Y-%m-%d") == day) & (tg.game_type == 2)]
games = pd.DataFrame({"game_id": h.game_id, "date": day, "start_utc": h.start_utc, "game_type": 2, "home": h.team,
                      "away": h.opp, "home_name": h.team.map(NAME), "away_name": h.opp.map(NAME)})
b = pd.read_parquet(C.DATA / "backtest" / "backtest_last.parquet")
sk = b[b.game_id.isin(games.game_id)][["game_id", "team", "opp", "player", "name", "g_hat", "a_hat", "pts_hat", "sog_hat", "ppp_hat"]].copy()
sk["team"], sk["opp"], sk["name"] = sk.team.astype(str), sk.opp.astype(str), sk["name"].astype(str)
gs = pd.read_parquet(C.ROOT.parent / "NHL_Goalie_Saves" / "data" / "backtest" / "preds_2025.parquet")
gs = gs[gs.game_id.isin(games.game_id)]
s = live.load_state()
full = dict(zip(s["known"].player.astype(int), s["known"].name))
for r in s["rosters"].values():
    full.update(dict(zip(r.player.astype(int), r.name)))
gl = pd.DataFrame({"game_id": gs.game_id, "team": gs.team.astype(str), "goalie_id": gs.goalie,
                   "goalie": [full.get(int(g), n) for g, n in zip(gs.goalie, gs.name)], "mean": gs.m_mean,
                   "p_over": [json.dumps({c[9:]: float(r[c]) for c in gs.columns if c.startswith("m_p_over_")})
                              for _, r in gs.iterrows()]})
T.TRACK.mkdir(parents=True, exist_ok=True)
games.to_parquet(f["games"], index=False); sk.to_parquet(f["skaters"], index=False); gl.to_parquet(f["goalies"], index=False)
f["meta"].write_text(json.dumps({"alpha": model.load()["alpha"], "built_at": "backtest", "snapshot_at": "backtest"}))
f["done"].unlink(missing_ok=True)
live_before = pd.read_parquet(T.RESULTS_LIVE) if T.RESULTS_LIVE.exists() else None
T.grade(day)
r = pd.read_parquet(T.RESULTS_LIVE)
r = r[r.date == day]
print(r.groupby(["snap", "kind"]).agg(bets=("bets", "sum"), wins=("wins", "sum"), pnl=("pnl", "sum")).round(2).to_string())
# leave no test rows behind in the live results
if live_before is None:
    T.RESULTS_LIVE.unlink()
else:
    live_before.to_parquet(T.RESULTS_LIVE, index=False)
for k in ("games", "skaters", "goalies", "meta", "done"):
    f[k].unlink(missing_ok=True)

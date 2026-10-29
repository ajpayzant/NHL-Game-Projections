"""Print every Trends table for the first slate day: python -m research.trends_check"""
import pandas as pd

import config as C
import live
import trends as T

pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
s = live.load_state()
day = sorted(s["games"].date.unique())[0]
deps = live.deployments(s, day, log=False)
p = live.project(s["slate"][s["slate"].date == pd.Timestamp(day)], deps, s)
h = pd.read_parquet(C.LAKE / "player_games.parquet")
h["ppp"] = h.g_pp + h.a_pp
tg = pd.read_parquet(C.LAKE / "team_games.parquet")
print(day, len(p), "skaters")
f = T.form(h, p, day).sort_values("heat")
cols = ["name", "team", "gp", "pts", "pts_gm", "base_pts", "sog_gm", "base_sog", "toi", "base_toi", "heat", "driver"]
print(f[cols].tail(8).round(2).to_string()); print(f[cols].head(8).round(2).to_string())
st = T.streaks(h, p, day)
print(st.groupby(["kind", "streak"]).size()); print(st.sort_values("games", ascending=False)[["name", "team", "streak", "games", "during"]].head(15).to_string())
m = T.matchups(h, p, day).sort_values("score")
print(m[["name", "opp", "gp_vs", "pts_gm_vs", "pts_gm", "score", "last"]].tail(6).round(2).to_string())
print(m[["name", "opp", "gp_vs", "pts_gm_vs", "pts_gm", "score"]].head(4).round(2).to_string())
gl = pd.DataFrame([{"goalie": d["goalie"]} for d in deps.values() if d.get("goalie")])
car = pd.read_parquet(T.CAREERS)
ms = T.milestones(car, h, p, gl, day, s['coefs'])
nm = dict(zip(s["known"].player, s["known"].name))
print(ms.assign(name=ms.player.map(nm)).sort_values("needs").to_string())
hr = T.hit_rates(h, p, day, "points", 1, 10)
print(hr[hr.gp >= 8].sort_values("rate").tail(5).to_string())
teams = sorted(set(p.team))
print(T.team_form(tg, teams, day, 10).round(2).to_string())
g0 = s["games"][s["games"].date == day].iloc[0]
print(T.head_to_head(tg, g0.home, g0.away, day)[["date", "team", "opp", "goals", "goals_against", "last_period_type"]])

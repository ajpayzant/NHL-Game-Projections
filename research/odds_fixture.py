"""Sample odds in The Odds API's response format, for testing the Edges tab without a key:
python -m research.odds_fixture  (writes data/odds/odds.parquet; a real fetch overwrites it)."""
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import live
import markets
import odds

rng = np.random.default_rng(7)
s = live.load_state()
day = sorted(s["games"].date.unique())[0]
deps = live.deployments(s, day, log=False)
p = live.project(s["slate"][s["slate"].date == pd.Timestamp(day)], deps, s)
games = s["games"][s["games"].date == day]


def am(q):  # probability -> american price
    return round(-100 * q / (1 - q)) if q >= 0.5 else round(100 * (1 - q) / q)


def two(q, vig=0.045, noise=0.03):
    q = float(np.clip(q + rng.normal(0, noise), 0.03, 0.97))
    return am(q * (1 + vig / 2)), am((1 - q) * (1 + vig / 2))


events = []
books = [("draftkings", "DraftKings"), ("fanduel", "FanDuel"), ("betmgm", "BetMGM"), ("williamhill_us", "Caesars")]
for g in games.itertuples():
    lam = p.groupby("team").g_hat.sum()
    P = markets.game_probs(np.array([lam[g.home]]), np.array([lam[g.away]]))
    ev = {"id": f"ev{g.game_id}", "commence_time": "2099-01-01T00:00:00Z", "home_team": g.home_name, "away_team": g.away_name,
          "bookmakers": []}
    tp = p[p.game_id == g.game_id]
    for k, title in books:
        h, a = two(P["home_win"][0])
        sh, sa = two(P["home_m15"][0] if lam[g.home] >= lam[g.away] else 1 - P["away_m15"][0])
        fav_home = lam[g.home] >= lam[g.away]
        o, u = two(P["over"][6.5][0])
        mk = [{"key": "h2h", "outcomes": [{"name": g.home_name, "price": h}, {"name": g.away_name, "price": a}]},
              {"key": "spreads", "outcomes": [{"name": g.home_name, "price": sh, "point": -1.5 if fav_home else 1.5},
                                              {"name": g.away_name, "price": sa, "point": 1.5 if fav_home else -1.5}]},
              {"key": "totals", "outcomes": [{"name": "Over", "price": o, "point": 6.5}, {"name": "Under", "price": u, "point": 6.5}]}]
        for m, col, line in (("player_points", "p_pts_1", 0.5), ("player_shots_on_goal", "p_sog_3", 2.5),
                             ("player_goals", "p_g_1", 0.5), ("player_assists", "p_a_1", 0.5)):
            outs = []
            for r in tp.nlargest(6, "pts_hat").itertuples():
                ov, un = two(getattr(r, col))
                outs += [{"name": "Over", "description": r.name, "price": ov, "point": line},
                         {"name": "Under", "description": r.name, "price": un, "point": line}]
            mk.append({"key": m, "outcomes": outs})
        ev["bookmakers"].append({"key": k, "title": title, "markets": mk})
    events.append(ev)
rows = [r for ev in events for r in odds._rows(ev, datetime.now(timezone.utc).isoformat(timespec="seconds"))]
odds.ODDS_DIR.mkdir(parents=True, exist_ok=True)
pd.DataFrame(rows).to_parquet(odds.ODDS_FILE, index=False)
print(f"{day}: {len(events)} games, {len(rows)} sample prices -> {odds.ODDS_FILE}")

# goalie saves for each team's likely starter, priced off the saves model
from saves import live as SL
S = SL.load_state()
nm = dict(zip(s["known"].player, s["known"].name))
for r in s["rosters"].values():
    nm.update(dict(zip(r.player.astype(int), r.name)))
add = []
for ev, g in zip(events, games.itertuples()):
    for team in (g.home, g.away):
        d = deps.get(team, {})
        pr = SL.starter_probs(S["cand"][(S["cand"].game_id == g.game_id) & (S["cand"].team == team)],
                              {team: {"goalie": d.get("goalie"), "goalie_status": d.get("goalie_status", ""), "source": d.get("source", "")}})
        proj = SL.project_team(S, g.game_id, team, pr, seed=g.game_id % 100000)
        if not proj:
            continue
        top = proj[0]
        line = float(np.floor(top["median"])) + 0.5
        for bk in ev["bookmakers"]:
            ov, un = two(top[f"p_over_{line}"])
            add.append((bk, {"name": "Over", "description": nm.get(top["goalie"], top["name"]), "price": ov, "point": line}))
            add.append((bk, {"name": "Under", "description": nm.get(top["goalie"], top["name"]), "price": un, "point": line}))
for bk, out in add:
    m = next((m for m in bk["markets"] if m["key"] == "player_total_saves"), None)
    if m is None:
        m = {"key": "player_total_saves", "outcomes": []}
        bk["markets"].append(m)
    m["outcomes"].append(out)
rows = [r for ev in events for r in odds._rows(ev, datetime.now(timezone.utc).isoformat(timespec="seconds"))]
pd.DataFrame(rows).to_parquet(odds.ODDS_FILE, index=False)
print(f"with saves: {len(rows)} sample prices")

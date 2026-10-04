"""How hard does the model regress players? For each rating: the weight his own history gets
(own exposure / (own exposure + prior exposure)), going into his next game, by player group;
and stars' model rates next to their raw last-season and career rates."""
import numpy as np, pandas as pd
import config as C, ew, features as F
pg = pd.read_parquet(C.LAKE / "player_games.parquet")
pg = pg.sort_values(["date", "game_id"]).reset_index(drop=True)
pg["t_ev"], pg["t_pp"] = pg.toi_ev / 60, pg.toi_pp / 60
pg["sog_all"] = pg.sog_ev + pg.sog_pp + pg.sog_sh
pg["ton_ev"] = pg.t_ev.where(pg.toi_5v5.notna())
specs = [("Shots/min EV", "t_ev", F.HL_RATE, F.K_SOG["ev"]), ("Shots/min PP", "t_pp", F.HL_RATE, F.K_SOG["pp"]),
         ("Shooting %", "sog_all", F.HL_SHOOT, F.K_SHOOT), ("On-ice goals/min EV", "ton_ev", F.HL_GF, F.K_GF["ev"]),
         ("Assist share EV", "ogf_ev", F.HL_ASSIST_SHARE, F.K_ASHARE)]
lat = {name: ew.latest(pg, "player", [den], hl, F.GAP, next_season=C.CURRENT_SEASON)[den] for name, den, hl, k in specs}
w = pd.DataFrame({name: lat[name] / (lat[name] + k) for name, den, hl, k in specs})
reg = pg[(pg.season == C.CURRENT_SEASON - 1) & (pg.game_type == 2)].groupby("player").agg(gp=("points", "size"), ppg=("points", "mean"))
career = pg.groupby("player").size().rename("career")
d = w.join(reg).join(career).dropna(subset=["gp"])
d = d[d.gp >= 10]
d["group"] = np.select([d.ppg >= 0.9, d.ppg >= 0.5, d.career < 120], ["Stars (0.9+ pts/gm)", "Top-6 / top-4 (0.5-0.9)", "Young / new (<120 career GP)"], "Regulars (<0.5)")
print("Weight on the player's OWN history (rest = role baseline), median by group, going into 2026-27:")
print(d.groupby("group")[[s[0] for s in specs]].median().round(2).to_string())
print("\nn per group:", d.group.value_counts().to_dict())

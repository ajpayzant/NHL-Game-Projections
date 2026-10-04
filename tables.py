"""Game-level tables by strength, built from the lake.

    shots_xg      the shots table with xg / xg_on scored walk-forward (xg.py)
    player_games  one row per skater-game: ice time, individual shots / goals / assists and
                  on-ice goals / shots / xG for and against, each split ev / pp / sh;
                  plus his deployment that night (line, pair, PP and PK unit)
    team_games    one row per team-game: the same counts for the team, PP / SH seconds

Individual stats are counted from the play-by-play, so a player's ev + pp + sh shots equal
his box-score shots except for the odd scorer correction. On-ice counts need the shift chart.

    python tables.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config as C
import parse
import xg

STRENGTHS = ("ev", "pp", "sh")


def build_shots() -> pd.DataFrame:
    shots = parse.load("shots")
    shots = xg.score(shots)
    shots.to_parquet(C.LAKE / "shots_xg.parquet", index=False)
    return shots


def update_shots() -> pd.DataFrame:
    """Daily path: keep the scored history, score only the season in progress with models
    fitted on every earlier season (fitted once per season, then cached)."""
    old = pd.read_parquet(C.LAKE / "shots_xg.parquet")
    past = old[old.season < C.CURRENT_SEASON]
    cur = parse.load("shots", [C.CURRENT_SEASON])
    tag = f"_{C.CURRENT_SEASON}"
    try:
        models = {k: xg.lgb.Booster(model_file=str(C.MODELS / f"{k}{tag}.txt")) for k in ("xg", "xg_on")}
    except Exception:
        models = xg.fit_current(past, tag)
    shots = pd.concat([past, xg.score_with(cur, models)], ignore_index=True)
    for c in parse.CATEGORICAL:
        if c in shots and shots[c].dtype != "category":
            shots[c] = shots[c].astype("category")
    shots.to_parquet(C.LAKE / "shots_xg.parquet", index=False)
    return shots


def _individual(shots: pd.DataFrame) -> pd.DataFrame:
    """Per player-game: sog / goals / assists / ixg / attempts by strength."""
    s = shots[~shots.target_net_empty | (shots.event == "goal")]  # empty-net goals still count
    on_net = s.event.isin(["goal", "sog"])
    base = pd.DataFrame({"game_id": s.game_id, "player": s.shooter.astype("Int64"),
                         "strength": s.strength.astype(str),
                         "sog": on_net.astype(int), "g": (s.event == "goal").astype(int),
                         "ixg": s.xg.fillna(0.0), "icf": 1,
                         "eng": ((s.event == "goal") & s.target_net_empty).astype(int)})
    goals = s[s.event == "goal"]
    ast = pd.concat([pd.DataFrame({"game_id": goals.game_id, "player": goals[c],
                                   "strength": goals.strength.astype(str),
                                   "a": 1, "a1": int(c == "a1")})
                     for c in ("a1", "a2")]).dropna(subset=["player"])
    parts = [base.groupby(["game_id", "player", "strength"])[["sog", "g", "ixg", "icf", "eng"]].sum(),
             ast.groupby(["game_id", "player", "strength"])[["a", "a1"]].sum()]
    out = pd.concat(parts, axis=1).fillna(0).unstack("strength", fill_value=0)
    out.columns = [f"{k}_{st}" for k, st in out.columns]
    return out.reset_index()


def _on_ice(shots: pd.DataFrame) -> pd.DataFrame:
    """Per player-game: goals / shots on goal / xG for and against while he was on the ice.

    Strength is from the player's own side, so a goal he gives up on the PK is 'ga_sh'."""
    s = shots[shots.f1.notna()]
    flip = {"ev": "ev", "pp": "sh", "sh": "pp"}
    vals = pd.DataFrame({"g": (s.event == "goal").astype(int),
                         "s": s.event.isin(["goal", "sog"]).astype(int),
                         "xg": s.xg.fillna(0.0)})
    frames = []
    for cols, side in ((parse.ON_FOR, "f"), (parse.ON_AGAINST, "a")):
        st = s.strength.astype(str) if side == "f" else s.strength.astype(str).map(flip)
        for c in cols:
            m = s[c].notna()
            frames.append(pd.DataFrame({"game_id": s.game_id[m], "player": s[c][m],
                                        "strength": st[m], "side": side,
                                        **{k: v[m] for k, v in vals.items()}}))
    d = pd.concat(frames, ignore_index=True)
    agg = d.groupby(["game_id", "player", "side", "strength"])[["g", "s", "xg"]].sum()
    agg = agg.unstack(["side", "strength"], fill_value=0)
    agg.columns = [f"o{k}{side}_{st}" for k, side, st in agg.columns]  # e.g. ogf_ev, osa_pp, oxga_sh
    return agg.reset_index()


def deployment(pg: pd.DataFrame) -> pd.DataFrame:
    """Line / pair / unit numbers for each player-game, from that night's ice time.

    Forwards: trios built from mutual 5v5 linemates, ranked 1-4 by the trio's mean EV ice
    time (a 13th forward joins the trio he spent most time with). Defence: pairs ranked 1-3
    the same way. PP1 / PP2 are the five most and next five most PP minutes (0 = no PP time);
    PK1 / PK2 the same with four."""
    out = []
    for (gid, team), g in pg.groupby(["game_id", "team"], observed=True, sort=False):
        line = {}
        for is_d, size, mate_cols in ((False, 3, ["lm_f1", "lm_f2"]), (True, 2, ["lm_d1"])):
            grp = g[(g.pos == "D") == is_d]
            ids = set(grp.player)
            toi = dict(zip(grp.player, grp.toi_ev))
            groups: list[list[int]] = []
            assigned: dict[int, int] = {}
            for p in grp.sort_values("toi_ev", ascending=False).player:
                if p in assigned:
                    continue
                row = grp[grp.player == p].iloc[0]
                mates = [int(row[c]) for c in mate_cols if pd.notna(row[c]) and int(row[c]) in ids
                         and int(row[c]) not in assigned]
                members = [p, *mates][:size]
                if len(members) == 1 and groups:  # leftover: join the group he played with most
                    tgt = next((assigned[int(row[c])] for c in mate_cols
                                if pd.notna(row[c]) and int(row[c]) in assigned), len(groups) - 1)
                    groups[tgt].append(p)
                    assigned[p] = tgt
                    continue
                for m in members:
                    assigned[m] = len(groups)
                groups.append(members)
            order = sorted(range(len(groups)), key=lambda i: -np.mean([toi[m] for m in groups[i]]))
            rank = {gi: min(r + 1, 4 if size == 3 else 3) for r, gi in enumerate(order)}
            line.update({p: rank[gi] for p, gi in assigned.items()})
        pp_rank = g.toi_pp.rank(ascending=False, method="first")
        sh_rank = g.toi_sh.rank(ascending=False, method="first")
        out.append(pd.DataFrame({
            "game_id": gid, "player": g.player.values,
            "line": g.player.map(line).values,
            "pp_unit": np.where(g.toi_pp <= 0, 0, np.where(pp_rank <= 5, 1, np.where(pp_rank <= 10, 2, 0))),
            "pk_unit": np.where(g.toi_sh <= 0, 0, np.where(sh_rank <= 4, 1, np.where(sh_rank <= 8, 2, 0))),
        }))
    return pd.concat(out, ignore_index=True)


def build_player_games(shots: pd.DataFrame) -> pd.DataFrame:
    sg = parse.load("skater_games")
    sg["pos"] = sg.pos.astype(str)
    # official ice time where the report has it, else the shift chart's split scaled to the
    # box-score total, else everything as EV
    split = sg[["shift_ev", "shift_pp", "shift_sh"]]
    scale = sg.box_toi / split.sum(axis=1).where(split.sum(axis=1) > 0)
    for c in ("ev", "pp", "sh"):
        sg[f"toi_{c}"] = sg[f"toi_{c}"].fillna((sg[f"shift_{c}"] * scale).round())
    for c in ("toi", "toi_ev", "toi_pp", "toi_sh"):
        sg[c] = sg[c].fillna(sg.box_toi if c == "toi" else 0)
    sg["toi_ev"] = sg.toi_ev.where(sg.toi_ev + sg.toi_pp + sg.toi_sh > 0, sg.box_toi)
    pg = sg.merge(_individual(shots), on=["game_id", "player"], how="left")
    pg = pg.merge(_on_ice(shots), on=["game_id", "player"], how="left")
    count_cols = [c for c in pg.columns if any(c.endswith(f"_{s}") for s in STRENGTHS)
                  and c.split("_")[0] not in ("toi",)]
    has_shifts = pg.toi_5v5.notna()
    for c in count_cols:
        pg[c] = pg[c].fillna(0)
        if c[0] == "o":
            pg.loc[~has_shifts, c] = np.nan  # no shift chart: on-ice unknown, not zero
    pg = pg.merge(deployment(pg), on=["game_id", "player"], how="left")
    pg.loc[~has_shifts, "line"] = np.nan
    pg["date"] = pd.to_datetime(pg.date.astype(str))
    pg = pg.sort_values(["date", "game_id", "team", "player"]).reset_index(drop=True)
    pg.to_parquet(C.LAKE / "player_games.parquet", index=False)
    return pg


def build_team_games(shots: pd.DataFrame, pg: pd.DataFrame) -> pd.DataFrame:
    tg = parse.load("team_games")
    tg["team"], tg["opp"] = tg.team.astype(str), tg.opp.astype(str)
    # PP / SH time from the official skater ice time (5 skaters on a PP, 4 on a PK), so team
    # time and player time come from the same source, and games without a shift chart work
    st = pg.groupby(["game_id", "team"], observed=True)[["toi_pp", "toi_sh"]].sum().reset_index()
    st["team"] = st.team.astype(str)
    tg = tg.drop(columns=["pp_secs", "sh_secs"]).merge(st, on=["game_id", "team"], how="left")
    tg["pp_secs"] = tg.pop("toi_pp") / 5
    tg["sh_secs"] = tg.pop("toi_sh") / 4
    s = shots
    flags = pd.DataFrame({"game_id": s.game_id, "team": s.team.astype(str),
                          "strength": s.strength.astype(str),
                          "gf": (s.event == "goal").astype(int),
                          "sf": s.event.isin(["goal", "sog"]).astype(int),
                          "xgf": s.xg.fillna(0.0), "cf": 1})
    agg = flags.groupby(["game_id", "team", "strength"])[["gf", "sf", "xgf", "cf"]].sum()
    agg = agg.unstack("strength", fill_value=0)
    agg.columns = [f"{k}_{st}" for k, st in agg.columns]
    agg = agg.reset_index()
    tg = tg.merge(agg, on=["game_id", "team"], how="left")
    flip = {"ev": "ev", "pp": "sh", "sh": "pp"}
    # gf_pp for the other side is this team's ga_sh
    against = agg.rename(columns={"team": "opp", **{c: c[:-4] + "a_" + flip[c[-2:]]
                                                    for c in agg.columns if c not in ("game_id", "team")}})
    tg = tg.merge(against, on=["game_id", "opp"], how="left")
    tg["date"] = pd.to_datetime(tg.date.astype(str))
    tg = tg.sort_values(["date", "game_id", "team"]).reset_index(drop=True)
    tg.to_parquet(C.LAKE / "team_games.parquet", index=False)
    return tg


def build(rescore: bool = True, daily: bool = False) -> None:
    if daily:
        shots = update_shots()
    else:
        shots = build_shots() if rescore else pd.read_parquet(C.LAKE / "shots_xg.parquet")
    pg = build_player_games(shots)
    tg = build_team_games(shots, pg)
    print(f"player_games {len(pg):,}  team_games {len(tg):,}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-rescore", action="store_true", help="reuse shots_xg.parquet")
    build(not ap.parse_args().no_rescore)

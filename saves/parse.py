"""Turn raw game JSON into the four lake tables everything else reads.

    shots        one row per shot attempt (goal / on goal / missed / blocked), regulation + OT
    team_games   one row per team per game: shots, attempts, goals, penalties, by strength
    goalie_games one row per goalie who dressed: starter flag, TOI, shots faced, saves
    skater_games one row per skater who dressed: position, TOI, shots on goal (lineup strength)

Shootout attempts are dropped everywhere: they are not saves in the box score and not the
thing a saves projection is about.

    python parse.py            # rebuild every season found under data/raw
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from . import fetch

SHOT_EVENTS = {"goal": "goal", "shot-on-goal": "sog", "missed-shot": "miss", "blocked-shot": "block"}
NET_X = 89.0


def _secs(mmss: str) -> int:
    m, s = mmss.split(":")
    return int(m) * 60 + int(s)


def _strength(code: str, home_shooter: bool) -> tuple[int, int, bool, bool]:
    """situationCode is <away goalie><away skaters><home skaters><home goalie>.

    Returns (skaters for, skaters against, own net empty, target net empty) from the
    shooter's point of view."""
    if not code or len(code) != 4:
        return 5, 5, False, False
    ag, ask, hsk, hg = (int(c) for c in code)
    if home_shooter:
        return hsk, ask, hg == 0, ag == 0
    return ask, hsk, ag == 0, hg == 0


def parse_game(year: int, gid: int) -> tuple[list[dict], list[dict], list[dict], list[dict]] | None:
    try:
        pbp = fetch.read_raw(year, gid, "pbp")
        box = fetch.read_raw(year, gid, "box")
    except FileNotFoundError:
        return None
    home, away = pbp["homeTeam"], pbp["awayTeam"]
    team_of = {r["playerId"]: r["teamId"] for r in pbp["rosterSpots"]}
    abbrev = {home["id"]: home["abbrev"], away["id"]: away["abbrev"]}
    date = pbp["gameDate"]
    game_type = pbp["gameType"]
    last_type = (pbp.get("gameOutcome") or {}).get("lastPeriodType", "REG")

    shots: list[dict] = []
    pens = {home["id"]: [0, 0], away["id"]: [0, 0]}  # [minors+majors called, penalty minutes]
    score = {home["id"]: 0, away["id"]: 0}
    prev = None  # previous event, for rebound / rush flags
    game_end_secs = 0
    for p in pbp["plays"]:
        pd_ = p["periodDescriptor"]
        ptype = pd_["periodType"]
        if ptype == "SO":
            break
        t = (pd_["number"] - 1) * 1200 + _secs(p.get("timeInPeriod", "00:00"))
        game_end_secs = max(game_end_secs, t)
        kind = p["typeDescKey"]
        det = p.get("details", {}) or {}
        if kind == "penalty":
            owner = det.get("eventOwnerTeamId")
            if owner in pens and det.get("typeCode") in ("MIN", "MAJ", "BEN"):
                pens[owner][0] += 1
            if owner in pens:
                pens[owner][1] += det.get("duration", 0) or 0
        if kind in SHOT_EVENTS:
            shooter = det.get("shootingPlayerId") or det.get("scoringPlayerId")
            team = team_of.get(shooter, det.get("eventOwnerTeamId"))
            if team not in abbrev:
                prev = None
                continue
            opp = away["id"] if team == home["id"] else home["id"]
            is_home = team == home["id"]
            sk_for, sk_against, own_empty, target_empty = _strength(p.get("situationCode", ""), is_home)
            x, y = det.get("xCoord"), det.get("yCoord")
            side = p.get("homeTeamDefendingSide")
            if x is not None and y is not None:
                if side in ("left", "right"):
                    # a shooter attacks the net his opponent defends
                    opp_side = ("right" if side == "left" else "left") if is_home else side
                    net_x = NET_X if opp_side == "right" else -NET_X
                else:
                    net_x = NET_X if x >= 0 else -NET_X
                dx, dy = abs(net_x - x), y
                dist = math.hypot(dx, dy)
                angle = math.degrees(math.atan2(abs(dy), dx)) if dx > 0 else 90.0
                behind = (x > NET_X) if net_x > 0 else (x < -NET_X)
            else:
                dist = angle = np.nan
                behind = False
            since_prev = t - prev["t"] if prev else 999
            rebound = bool(prev and prev["shot"] and prev["team"] == team and since_prev <= 3)
            rush = bool(prev and since_prev <= 4 and prev["zone"] in ("N", "D")
                        and prev["kind"] not in ("faceoff",))
            shots.append({
                "game_id": gid, "season": year, "game_type": game_type, "date": date,
                "period": pd_["number"], "period_type": ptype, "t": t,
                "event": SHOT_EVENTS[kind], "team": abbrev[team], "opp": abbrev[opp],
                "is_home": is_home, "shooter": shooter,
                "goalie": det.get("goalieInNetId"),
                "shot_type": det.get("shotType"), "x": x, "y": y,
                "dist": dist, "angle": angle, "behind_net": behind,
                "zone": det.get("zoneCode"),
                "sk_for": sk_for, "sk_against": sk_against,
                "own_net_empty": own_empty, "target_net_empty": target_empty,
                "score_diff": score[team] - score[opp],
                "rebound": rebound, "rush": rush,
                "teammate_block": det.get("reason") == "teammate-blocked",
            })
            if kind == "goal":
                score[team] += 1
        prev = {"t": t, "team": team_of.get(det.get("eventOwnerTeamId"), det.get("eventOwnerTeamId")),
                "shot": kind in SHOT_EVENTS, "zone": det.get("zoneCode"), "kind": kind}
        if kind in SHOT_EVENTS:
            prev["team"] = team

    game_secs = max(3600, game_end_secs)
    teams = []
    for tm, opp, is_home in ((home, away, True), (away, home, False)):
        teams.append({
            "game_id": gid, "season": year, "game_type": game_type, "date": date,
            "team": tm["abbrev"], "opp": opp["abbrev"], "is_home": is_home,
            "goals": tm.get("score"), "goals_against": opp.get("score"),
            "sog_box": tm.get("sog"), "sog_against_box": opp.get("sog"),
            "pen_taken": pens[tm["id"]][0], "pim": pens[tm["id"]][1],
            "pen_drawn": pens[opp["id"]][0],
            "last_period_type": last_type, "game_secs": game_secs,
            "start_utc": pbp.get("startTimeUTC"),
        })

    goalies = []
    for side, tm, opp in (("homeTeam", home, away), ("awayTeam", away, home)):
        for g in box["playerByGameStats"][side].get("goalies", []):
            es = (g.get("evenStrengthShotsAgainst") or "0/0").split("/")
            pp = (g.get("powerPlayShotsAgainst") or "0/0").split("/")
            sh = (g.get("shorthandedShotsAgainst") or "0/0").split("/")
            goalies.append({
                "game_id": gid, "season": year, "game_type": game_type, "date": date,
                "team": tm["abbrev"], "opp": opp["abbrev"], "is_home": side == "homeTeam",
                "goalie": g["playerId"], "name": g["name"]["default"],
                "starter": bool(g.get("starter")), "toi": _secs(g.get("toi") or "00:00"),
                "shots_against": g.get("shotsAgainst", 0), "saves": g.get("saves", 0),
                "goals_against": g.get("goalsAgainst", 0),
                # the box's "PP shots against" are shots faced while HIS team was shorthanded
                "es_sa": int(es[1]), "es_sv": int(es[0]),
                "pk_sa": int(pp[1]), "pk_sv": int(pp[0]),
                "sh_sa": int(sh[1]), "sh_sv": int(sh[0]),
                "decision": g.get("decision"), "game_secs": game_secs,
            })
    skaters = []
    for side, tm in (("homeTeam", home), ("awayTeam", away)):
        for pos in ("forwards", "defense"):
            for p in box["playerByGameStats"][side].get(pos, []):
                skaters.append({"game_id": gid, "team": tm["abbrev"], "player": p["playerId"],
                                "pos": pos[0].upper(), "toi": _secs(p.get("toi") or "00:00"),
                                "sog": p.get("sog", 0) or 0})
    return shots, teams, goalies, skaters


CATEGORICAL = ("date", "period_type", "event", "team", "opp", "shot_type", "zone",
               "last_period_type", "decision", "start_utc", "name")


def _frame(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    for c in CATEGORICAL:
        if c in df:
            df[c] = df[c].astype("category")
    return df


def build_season(year: int) -> dict[str, pd.DataFrame]:
    """Parse one season and write data/lake/<table>_<year>.parquet."""
    gids = sorted({int(f.name.split("_")[0]) for f in (C.RAW / str(year)).glob("*_box.json.gz")})
    S, T, G, K = [], [], [], []
    for gid in gids:  # serial: this machine has no memory to spare for a process pool
        r = parse_game(year, gid)
        if r:
            S += r[0]; T += r[1]; G += r[2]; K += r[3]
    out = {"shots": _frame(S), "team_games": _frame(T), "goalie_games": _frame(G),
           "skater_games": _frame(K)}
    del S, T, G, K
    _finish_team_games(out)
    C.LAKE.mkdir(parents=True, exist_ok=True)
    for name, df in out.items():
        df.to_parquet(C.LAKE / f"{name}_{year}.parquet", index=False)
    print(f"{year}: {len(out['team_games']) // 2:,} games, {len(out['shots']):,} shots", flush=True)
    return out


def load(name: str, years: list[int] | None = None) -> pd.DataFrame:
    """Concatenate a lake table over seasons; strings come back as categories."""
    # season files only: derived tables such as team_games_xg share the prefix
    files = sorted(C.LAKE.glob(f"{name}_[0-9][0-9][0-9][0-9].parquet"))
    if years is not None:
        files = [f for f in files if int(f.stem.rsplit("_", 1)[1]) in years]
    frames = [pd.read_parquet(f) for f in files]
    df = pd.concat(frames, ignore_index=True)
    for c in CATEGORICAL:
        if c in df and df[c].dtype != "category":
            df[c] = df[c].astype("category")
    return df


def build(years: list[int] | None = None) -> None:
    years = years or sorted(int(p.name) for p in C.RAW.iterdir() if p.is_dir())
    for yr in years:
        build_season(yr)


def _finish_team_games(out: dict[str, pd.DataFrame]) -> None:
    """Add shot / attempt counts by strength to team_games, from the shots table."""
    s, tg = out["shots"], out["team_games"]
    if s.empty:
        return
    on_net = s.event.isin(["goal", "sog"])
    ev = (s.sk_for == s.sk_against) & ~s.own_net_empty & ~s.target_net_empty
    pp = s.sk_for > s.sk_against
    pk = s.sk_for < s.sk_against
    faced = on_net & ~s.target_net_empty  # shots a goalie could actually face
    flags = pd.DataFrame({
        "game_id": s.game_id, "team": s.team.astype(str),
        "sf": on_net, "cf": True, "ff": s.event != "block",
        "sf_faced": faced, "sf_ev": faced & ev, "sf_pp": faced & pp, "sf_pk": faced & pk,
        "cf_ev": ev, "gf_faced": (s.event == "goal") & ~s.target_net_empty,
    })
    agg = flags.groupby(["game_id", "team"]).sum().reset_index()
    tg = tg.assign(team=tg.team.astype(str), opp=tg.opp.astype(str))
    tg2 = tg.merge(agg, on=["game_id", "team"], how="left")
    # sf -> sa, ff -> fa, sf_ev -> sa_ev: the second letter is always the for/against one
    against = agg.rename(columns={"team": "opp", **{c: c[0] + "a" + c[2:] for c in agg.columns
                                                     if c not in ("game_id", "team")}})
    tg2 = tg2.merge(against, on=["game_id", "opp"], how="left")
    for c in ("team", "opp"):
        tg2[c] = tg2[c].astype("category")
    out["team_games"] = tg2


if __name__ == "__main__":
    build()

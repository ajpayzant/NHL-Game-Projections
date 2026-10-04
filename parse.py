"""Turn raw game JSON into the lake tables everything else reads.

    shots        one row per shot attempt (regulation + OT) with the shooter, the assisters,
                 the strength from the shooter's side, and the skaters on the ice for each side
    skater_games one row per skater who dressed: box-score stats, official EV / PP / SH ice
                 time, and who he played with (5v5 linemates from the shift chart)
    team_games   one row per team per game: goals, shots, penalties, PP / SH seconds

Strength is the man advantage after taking a pulled goalie out: 6 skaters with an empty net
against 5 is even strength (the NHL counts that time as EV too), 5 against 4 is a power play.

    python parse.py            # rebuild every season found under data/raw
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

import config as C
import fetch

SHOT_EVENTS = {"goal": "goal", "shot-on-goal": "sog", "missed-shot": "miss", "blocked-shot": "block"}
NET_X = 89.0
MAX_ON = 6  # skaters on the ice for one side, at most


def _secs(mmss: str | None) -> int:
    if not mmss:
        return 0
    m, s = mmss.split(":")
    return int(m) * 60 + int(s)


def _situation(code: str, home_side: bool) -> tuple[int, int, bool, bool]:
    """situationCode is <away goalie><away skaters><home skaters><home goalie>.

    Returns (skaters for, skaters against, own net empty, target net empty) from one side."""
    if not code or len(code) != 4:
        return 5, 5, False, False
    ag, ask, hsk, hg = (int(c) for c in code)
    if home_side:
        return hsk, ask, hg == 0, ag == 0
    return ask, hsk, ag == 0, hg == 0


def strength(sk_for: int, sk_against: int, own_empty: bool, target_empty: bool) -> str:
    """'ev', 'pp' or 'sh', counting an extra attacker as no advantage."""
    f = sk_for - int(own_empty)
    a = sk_against - int(target_empty)
    return "ev" if f == a else ("pp" if f > a else "sh")


class OnIce:
    """Who was on the ice each second, from the shift chart.

    A shift covers seconds [start, end). An event at second t is credited to the players on
    for second t-1 (a goal ends shifts at t; a faceoff at t=0 of a period uses second 0)."""

    def __init__(self, shifts: list[dict], periods: dict[int, int]):
        rows = [s for s in shifts if s.get("typeCode") == 517 and s.get("period") in periods]
        self.players = sorted({s["playerId"] for s in rows})
        idx = {p: i for i, p in enumerate(self.players)}
        self.team = {s["playerId"]: s["teamId"] for s in rows}
        self.n = max([periods[s["period"]] + _secs(s.get("endTime")) for s in rows] + [3600]) + 1
        self.m = np.zeros((len(self.players), self.n), dtype=bool)
        for s in rows:
            off = periods[s["period"]]
            a, b = off + _secs(s.get("startTime")), off + _secs(s.get("endTime"))
            if b > a:
                self.m[idx[s["playerId"]], a:b] = True
        self.idx = idx

    def at(self, t: int, period_start: int) -> np.ndarray:
        """Indices of players on the ice for the second credited with an event at t."""
        sec = t - 1 if t > period_start else t
        return np.flatnonzero(self.m[:, min(sec, self.n - 1)])


def _period_offsets(pbp: dict) -> dict[int, int]:
    """Seconds elapsed at the start of each period; shootouts are dropped."""
    out, t = {}, 0
    reg_len, ot_len = 1200, (300 if pbp.get("gameType") == 2 else 1200)
    for p in range(1, 12):
        out[p] = t
        t += reg_len if p <= 3 else ot_len
    if pbp.get("gameType") == 2:
        out = {p: o for p, o in out.items() if p <= 4}
    return out


def parse_game(year: int, gid: int) -> dict[str, list[dict]] | None:
    try:
        pbp = fetch.read_raw(year, gid, "pbp")
        box = fetch.read_raw(year, gid, "box")
    except FileNotFoundError:
        return None
    try:
        shifts = fetch.read_raw(year, gid, "shifts")["data"]
    except FileNotFoundError:
        shifts = []
    home, away = pbp["homeTeam"], pbp["awayTeam"]
    abbrev = {home["id"]: home["abbrev"], away["id"]: away["abbrev"]}
    other = {home["id"]: away["id"], away["id"]: home["id"]}
    date, game_type = pbp["gameDate"], pbp["gameType"]
    team_of = {r["playerId"]: r["teamId"] for r in pbp["rosterSpots"]}
    goalie_ids = {r["playerId"] for r in pbp["rosterSpots"] if r.get("positionCode") == "G"}
    names = {r["playerId"]: f'{r["firstName"]["default"]} {r["lastName"]["default"]}'
             for r in pbp["rosterSpots"]}
    offsets = _period_offsets(pbp)
    ice = OnIce(shifts, offsets) if shifts else None

    shots: list[dict] = []
    extra: dict[int, dict] = {}  # per-player counts: zone faceoffs on ice, penalties drawn / taken
    pens = {home["id"]: 0, away["id"]: 0}  # minors + majors called against each team
    score = {home["id"]: 0, away["id"]: 0}
    prev = None
    for p in pbp["plays"]:
        pdsc = p["periodDescriptor"]
        if pdsc["periodType"] == "SO":
            break
        per = pdsc["number"]
        if per not in offsets:
            continue
        t = offsets[per] + _secs(p.get("timeInPeriod"))
        kind = p["typeDescKey"]
        det = p.get("details", {}) or {}
        if kind == "penalty" and det.get("typeCode") in ("MIN", "MAJ", "BEN"):
            owner = det.get("eventOwnerTeamId")
            if owner in pens:
                pens[owner] += 1
            for pid, k in ((det.get("committedByPlayerId"), "pen_taken"), (det.get("drawnByPlayerId"), "pen_drawn")):
                if pid:
                    extra.setdefault(pid, {}).setdefault(k, 0)
                    extra[pid][k] += 1
        if kind == "faceoff" and ice is not None and det.get("zoneCode") in ("O", "D", "N"):
            # zone from the winner's side; on-ice players are those on for second t itself
            owner, z = det.get("eventOwnerTeamId"), det["zoneCode"]
            for i in np.flatnonzero(ice.m[:, min(t, ice.n - 1)]):
                q = ice.players[i]
                if q in goalie_ids:
                    continue
                zz = z if ice.team.get(q) == owner else {"O": "D", "D": "O", "N": "N"}[z]
                extra.setdefault(q, {}).setdefault(f"fo_{zz.lower()}z", 0)
                extra[q][f"fo_{zz.lower()}z"] += 1
        if kind in SHOT_EVENTS:
            shooter = det.get("shootingPlayerId") or det.get("scoringPlayerId")
            team = team_of.get(shooter, det.get("eventOwnerTeamId"))
            if team not in abbrev:
                prev = None
                continue
            opp = other[team]
            is_home = team == home["id"]
            sk_for, sk_against, own_empty, target_empty = _situation(p.get("situationCode", ""), is_home)
            x, y = det.get("xCoord"), det.get("yCoord")
            side = p.get("homeTeamDefendingSide")
            if x is not None and y is not None:
                if side in ("left", "right"):
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
            row = {
                "game_id": gid, "season": year, "game_type": game_type, "date": date,
                "period": per, "is_ot": per > 3, "t": t,
                "event": SHOT_EVENTS[kind], "team": abbrev[team], "opp": abbrev[opp],
                "is_home": is_home, "shooter": shooter,
                "a1": det.get("assist1PlayerId") if kind == "goal" else None,
                "a2": det.get("assist2PlayerId") if kind == "goal" else None,
                "goalie": det.get("goalieInNetId"),
                "shot_type": det.get("shotType"), "x": x, "y": y,
                "dist": dist, "angle": angle, "behind_net": behind,
                "sk_for": sk_for, "sk_against": sk_against,
                "own_net_empty": own_empty, "target_net_empty": target_empty,
                "strength": strength(sk_for, sk_against, own_empty, target_empty),
                "score_diff": score[team] - score[opp],
                "rebound": bool(prev and prev["shot"] and prev["team"] == team and since_prev <= 3),
                "rush": bool(prev and since_prev <= 4 and prev["zone"] in ("N", "D")
                             and prev["kind"] != "faceoff"),
            }
            if ice is not None:
                on = [ice.players[i] for i in ice.at(t, offsets[per])]
                f = [q for q in on if ice.team.get(q) == team and q not in goalie_ids][:MAX_ON]
                a = [q for q in on if ice.team.get(q) == opp and q not in goalie_ids][:MAX_ON]
                for i in range(MAX_ON):
                    row[f"f{i + 1}"] = f[i] if i < len(f) else None
                    row[f"op{i + 1}"] = a[i] if i < len(a) else None
            shots.append(row)
            if kind == "goal":
                score[team] += 1
        owner = det.get("eventOwnerTeamId")
        prev = {"t": t, "team": team if kind in SHOT_EVENTS else owner,
                "shot": kind in SHOT_EVENTS, "zone": det.get("zoneCode"), "kind": kind}

    # strength seconds and 5v5 linemates from the shift chart
    pp_secs = {home["id"]: 0, away["id"]: 0}
    linemates: dict[int, dict] = {}
    if ice is not None and len(ice.players):
        team_arr = np.array([ice.team[q] for q in ice.players])
        is_g = np.array([q in goalie_ids for q in ice.players])
        n_sk = {tid: ice.m[(team_arr == tid) & ~is_g].sum(0) for tid in abbrev}
        g_in = {tid: ice.m[(team_arr == tid) & is_g].any(0) for tid in abbrev}
        played = n_sk[home["id"]] + n_sk[away["id"]] > 0
        adv = {}
        for tid in abbrev:
            f = n_sk[tid] - (~g_in[tid]).astype(int)
            a = n_sk[other[tid]] - (~g_in[other[tid]]).astype(int)
            pp_secs[tid] = int(((f > a) & played).sum())
            adv[tid] = np.sign(f - a)
        five = (n_sk[home["id"]] == 5) & (n_sk[away["id"]] == 5) & g_in[home["id"]] & g_in[away["id"]]
        mm = ice.m[:, five].astype(np.int32)
        shared = mm @ mm.T  # 5v5 seconds each pair spent on the ice together
        pos = _positions(box)
        for i, q in enumerate(ice.players):
            if q in goalie_ids or q not in pos:
                continue
            mates = [(shared[i, j], ice.players[j]) for j in range(len(ice.players))
                     if j != i and team_arr[j] == team_arr[i] and not is_g[j]]
            fw = sorted([m for m in mates if pos.get(m[1]) != "D"], reverse=True)
            dm = sorted([m for m in mates if pos.get(m[1]) == "D"], reverse=True)
            on = ice.m[i]
            a = adv.get(team_arr[i], np.zeros(ice.n, dtype=int))  # shift-chart team errors
            linemates[q] = {"toi_5v5": int(shared[i, i]),
                            # strength split from the shift chart, for games the official
                            # time-on-ice report has not reached yet
                            "shift_ev": int((on & (a == 0)).sum()), "shift_pp": int((on & (a > 0)).sum()),
                            "shift_sh": int((on & (a < 0)).sum()),
                            "lm_f1": fw[0][1] if fw and fw[0][0] > 0 else None,
                            "lm_f2": fw[1][1] if len(fw) > 1 and fw[1][0] > 0 else None,
                            "lm_d1": dm[0][1] if dm and dm[0][0] > 0 else None,
                            "lm_d2": dm[1][1] if len(dm) > 1 and dm[1][0] > 0 else None}

    last_type = (pbp.get("gameOutcome") or {}).get("lastPeriodType", "REG")
    teams = []
    for tm, opp, is_home in ((home, away, True), (away, home, False)):
        teams.append({
            "game_id": gid, "season": year, "game_type": game_type, "date": date,
            "team": tm["abbrev"], "opp": opp["abbrev"], "is_home": is_home,
            "goals": tm.get("score"), "goals_against": opp.get("score"),
            "sog": tm.get("sog"), "sog_against": opp.get("sog"),
            "pen_taken": pens[tm["id"]], "pen_drawn": pens[opp["id"]],
            "pp_secs": pp_secs[tm["id"]], "sh_secs": pp_secs[opp["id"]],
            "has_shifts": ice is not None, "last_period_type": last_type,
            "start_utc": pbp.get("startTimeUTC"),
        })

    skaters = []
    for side, tm, opp in (("homeTeam", home, away), ("awayTeam", away, home)):
        for grp in ("forwards", "defense"):
            for p in box["playerByGameStats"][side].get(grp, []):
                pid = p["playerId"]
                skaters.append({
                    "game_id": gid, "season": year, "game_type": game_type, "date": date,
                    "team": tm["abbrev"], "opp": opp["abbrev"], "is_home": side == "homeTeam",
                    "player": pid, "name": names.get(pid, p["name"]["default"]),
                    "pos": p.get("position", grp[0].upper()),
                    "goals": p.get("goals", 0) or 0, "assists": p.get("assists", 0) or 0,
                    "points": p.get("points", 0) or 0, "sog": p.get("sog", 0) or 0,
                    "ppg": p.get("powerPlayGoals", 0) or 0, "pim": p.get("pim", 0) or 0,
                    "box_toi": _secs(p.get("toi")),
                    "hits": p.get("hits", 0) or 0, "blocks": p.get("blockedShots", 0) or 0,
                    "takeaways": p.get("takeaways", 0) or 0, "giveaways": p.get("giveaways", 0) or 0,
                    **{k: (extra.get(pid) or {}).get(k, 0) for k in
                       ("fo_oz", "fo_dz", "fo_nz", "pen_taken", "pen_drawn")},
                    **linemates.get(pid, {}),
                })
    return {"shots": shots, "team_games": teams, "skater_games": skaters}


def _positions(box: dict) -> dict[int, str]:
    out = {}
    for side in ("homeTeam", "awayTeam"):
        for grp in ("forwards", "defense"):
            for p in box["playerByGameStats"][side].get(grp, []):
                out[p["playerId"]] = "D" if grp == "defense" else "F"
    return out


CATEGORICAL = ("date", "event", "team", "opp", "shot_type", "strength", "last_period_type",
               "start_utc", "name", "pos")


ON_FOR = [f"f{i}" for i in range(1, MAX_ON + 1)]       # shooting side's skaters on the ice
ON_AGAINST = [f"op{i}" for i in range(1, MAX_ON + 1)]  # defending side's skaters
ID_COLS = {"a1", "a2", "goalie", *ON_FOR, *ON_AGAINST}


def _frame(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    for c in CATEGORICAL:
        if c in df:
            df[c] = df[c].astype("category")
    for c in df.columns:  # nullable player-id columns
        if c in ("a1", "a2", "goalie") or c.startswith(("lm_", "f", "x")) and c[1:].isdigit() \
                or c.startswith("lm_"):
            df[c] = df[c].astype("Int64")
    return df


def build_season(year: int) -> None:
    """Parse one season and write data/lake/<table>_<year>.parquet."""
    gids = sorted({int(f.name.split("_")[0]) for f in (C.RAW / str(year)).glob("*_box.json.gz")})
    acc: dict[str, list[dict]] = {"shots": [], "team_games": [], "skater_games": []}
    for gid in gids:  # serial: this machine has no memory to spare for a process pool
        r = parse_game(year, gid)
        if r:
            for k, v in r.items():
                acc[k] += v
    out = {k: _frame(v) for k, v in acc.items()}
    del acc
    toi_file = C.LAKE / f"toi_{year}.parquet"
    if toi_file.exists():
        toi = pd.read_parquet(toi_file).drop(columns="toi_ot")
        out["skater_games"] = out["skater_games"].merge(toi, on=["game_id", "player"], how="left")
    C.LAKE.mkdir(parents=True, exist_ok=True)
    for name, df in out.items():
        df.to_parquet(C.LAKE / f"{name}_{year}.parquet", index=False)
    sg = out["skater_games"]
    print(f"{year}: {len(out['team_games']) // 2:,} games, {len(out['shots']):,} shots, "
          f"{len(sg):,} skater-games ({sg['toi_ev'].notna().mean():.1%} with official TOI, "
          f"{sg['toi_5v5'].notna().mean():.1%} with shifts)", flush=True)


def load(name: str, years: list[int] | None = None) -> pd.DataFrame:
    """Concatenate a lake table over seasons; strings come back as categories."""
    files = sorted(C.LAKE.glob(f"{name}_[0-9][0-9][0-9][0-9].parquet"))
    if years is not None:
        files = [f for f in files if int(f.stem.rsplit("_", 1)[1]) in years]
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    for c in CATEGORICAL:
        if c in df and df[c].dtype != "category":
            df[c] = df[c].astype("category")
    return df


def build(years: list[int] | None = None) -> None:
    years = years or sorted(int(p.name) for p in C.RAW.iterdir() if p.is_dir())
    for yr in years:
        build_season(yr)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, action="append")
    build(ap.parse_args().season)

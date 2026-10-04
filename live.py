"""The live slate: upcoming games -> lineups and lines -> each skater's projection.

    python live.py                # fetch finished games, rebuild the lake, refit, cache states
    python live.py --no-fetch     # refit and re-cache from the lake as it is

The heavy part (running sums over every past game, the fit) is cached per slate date in
data/live. Deployment (who dresses, line, PP unit, PK unit, linemates) and the opposing
goalie are applied on top at view time by `project`, so Daily Faceoff updates and manual
overrides take effect instantly, without a refit.

Deployment for a team comes from, in order: a manual override, Daily Faceoff's line chart,
the team's last game. Starting goalies: override, Daily Faceoff's starting-goalie page, the
line chart's G1, the goalie who started the team's last game.
"""
from __future__ import annotations

import argparse
import json
import pickle
import re
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

import config as C
import features
import fetch
import model
import parse
import store
import tables

LIVE = C.DATA / "live"
HORIZON_DAYS = 3
FIRST_TRAIN = 2021


# ---------------------------------------------------------------- sources

def schedule(start: date | None = None, days: int = HORIZON_DAYS) -> pd.DataFrame:
    """Games not yet final from `start` for `days` days."""
    start = start or datetime.now(ZoneInfo("America/New_York")).date()  # the NHL's day, wherever this runs
    end = start + timedelta(days=days)
    rows, day = [], start
    while day < end:
        d = fetch.get_json(f"{C.NHL_WEB}/schedule/{day.isoformat()}")
        for wk in d.get("gameWeek", []):
            for g in wk["games"]:
                if g["gameType"] not in C.GAME_TYPES or not (start.isoformat() <= wk["date"] < end.isoformat()):
                    continue
                rows.append({"game_id": g["id"], "date": wk["date"], "start_utc": g["startTimeUTC"],
                             "game_type": g["gameType"], "state": g["gameState"],
                             "home": g["homeTeam"]["abbrev"], "away": g["awayTeam"]["abbrev"],
                             "home_name": _team_name(g["homeTeam"]), "away_name": _team_name(g["awayTeam"])})
        day += timedelta(days=7)
    out = pd.DataFrame(rows).drop_duplicates("game_id") if rows else pd.DataFrame()
    return out[~out.state.isin(["OFF", "FINAL"])].reset_index(drop=True) if len(out) else out


def _team_name(t: dict) -> str:
    return f"{t['placeName']['default']} {t['commonName']['default']}"


def key(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z]", "", s.lower())


def roster(team: str) -> pd.DataFrame:
    """Today's NHL roster: id, name, position, sweater number."""
    try:
        r = fetch.get_json(f"{C.NHL_WEB}/roster/{team}/current")
    except Exception:
        return pd.DataFrame(columns=["player", "name", "pos", "number", "team"])
    rows = [{"player": p["id"], "name": f"{p['firstName']['default']} {p['lastName']['default']}",
             "last": p["lastName"]["default"], "pos": p.get("positionCode"), "number": p.get("sweaterNumber"),
             "shoots": p.get("shootsCatches"), "headshot": p.get("headshot"), "birth": p.get("birthDate"),
             "team": team}
            for grp in ("forwards", "defensemen", "goalies") for p in r.get(grp, [])]
    return pd.DataFrame(rows)


def _dfo_page(url: str) -> dict:
    html = requests.get(url, headers={"User-Agent": C.BROWSER_UA}, timeout=C.REQUEST_TIMEOUT).text
    m = re.search(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', html, re.S)
    return json.loads(m.group(1))["props"]["pageProps"] if m else {}


def dfo_slugs() -> dict[str, str]:
    """Daily Faceoff team slug by normalised full team name."""
    pp = _dfo_page(C.DFO_LINES.format(slug="toronto-maple-leafs"))
    return {key(t["name"]): t["slug"] for t in pp.get("sortedTeams", [])}


def dfo_lines(slug: str) -> tuple[pd.DataFrame, dict]:
    """One team's Daily Faceoff line chart: a row per (player, group), and page metadata
    (sourceName says what the chart is based on, e.g. 'Last Game (2026-10-02)')."""
    c = (_dfo_page(C.DFO_LINES.format(slug=slug)).get("combinations") or {})
    rows = [{"dfo_name": p["name"], "number": p.get("jerseyNumber"), "group": p.get("groupIdentifier"),
             "slot": p.get("positionIdentifier"), "injury": p.get("injuryStatus"),
             "gtd": bool(p.get("gameTimeDecision"))} for p in c.get("players", [])]
    meta = {"team": c.get("teamAbbreviation"), "source": c.get("sourceName"), "updated": c.get("updatedAt"),
            "team_name": c.get("teamName")}
    return pd.DataFrame(rows), meta


def dfo_starters(day: str) -> pd.DataFrame:
    """Daily Faceoff's starting-goalie reports for a date, one row per team."""
    games = _dfo_page(C.DFO_STARTERS.format(date=day)).get("data") or []
    rows = []
    for g in games:
        for side in ("home", "away"):
            rows.append({"date": day, "team_name": g.get(f"{side}TeamName"),
                         "dfo_goalie": g.get(f"{side}GoalieName"),
                         "status": g.get(f"{side}NewsStrengthName") or "Unconfirmed"})
    return pd.DataFrame(rows)


def log_snapshot(kind: str, df: pd.DataFrame) -> None:
    """Append what Daily Faceoff said, with a timestamp. These snapshots are the only record
    of PREGAME line charts, which is what the line-slot weights should eventually be fitted on."""
    if df.empty:
        return
    C.DFO_DIR.mkdir(parents=True, exist_ok=True)
    path = C.DFO_DIR / f"{kind}_snapshots.parquet"
    snap = df.assign(fetched_at=datetime.now(timezone.utc).isoformat(timespec="seconds")).astype(str)
    if path.exists():
        snap = pd.concat([pd.read_parquet(path), snap], ignore_index=True)
    snap.to_parquet(path, index=False)


# ---------------------------------------------------------------- deployments
# A deployment is {"skaters": {player_id: {"line": n, "pp": u, "pk": u}}, "goalie": id,
# "source": str}. Forwards on the same line are linemates; defence on the same pair partners.

SUFFIXES = {"jr", "sr", "ii", "iii", "iv"}


def _surname(name: str) -> str:
    parts = [p for p in str(name).replace(".", " ").split() if key(p) not in SUFFIXES]
    return key(parts[-1]) if parts else ""


def _match(names: pd.DataFrame, roster_pool: pd.DataFrame, known: pd.DataFrame | None = None) -> pd.Series:
    """NHL player id for each Daily Faceoff row. Today's roster first (full name, surname +
    sweater number, unique surname), then every player seen before by full name."""
    pools = [roster_pool] + ([known] if known is not None else [])
    out = []
    for n, num in zip(names.dfo_name, names.number):
        k, last = key(n), _surname(n)
        pid = None
        for i, pool in enumerate(pools):
            cands = pool[pool.name.map(key) == k]
            if len(cands) > 1 and pd.notna(num):  # two players with one name: the number decides
                cands = cands[cands.number == num]
            pid = cands.player.iloc[0] if len(cands) == 1 else None
            if pid is None and i == 0:
                sur = pool.name.map(_surname)
                hit = pool[(sur == last) & (pool.number == num)] if pd.notna(num) else pool.iloc[:0]
                if len(hit) == 1:
                    pid = hit.player.iloc[0]
                elif (sur == last).sum() == 1:
                    pid = pool.player[sur == last].iloc[0]
            if pid is not None:
                break
        out.append(pid)
    return pd.Series(out, index=names.index, dtype="Int64")


def deployment_from_dfo(lines: pd.DataFrame, pool: pd.DataFrame, known: pd.DataFrame | None = None) -> dict:
    lines = lines.assign(player=_match(lines, pool, known)).dropna(subset=["player"])
    sk: dict[int, dict] = {}
    for r in lines.itertuples():
        g = str(r.group)
        if re.fullmatch(r"[fd]\d", g):
            sk.setdefault(int(r.player), {"line": None, "pp": 0, "pk": 0})
            sk[int(r.player)].update(line=int(g[1]), spot=str(r.slot).upper())
    for r in lines.itertuples():
        g = str(r.group)
        if int(r.player) in sk and re.fullmatch(r"(pp|pk)\d", g):
            sk[int(r.player)][g[:2]] = int(g[2])
    goalies = lines[lines.group == "g"].sort_values("slot")
    return {"skaters": sk, "goalie": int(goalies.player.iloc[0]) if len(goalies) else None,
            "backup": int(goalies.player.iloc[1]) if len(goalies) > 1 else None,
            "source": "Daily Faceoff"}


def deployment_from_last_game(team: str, pg: pd.DataFrame, starters: pd.DataFrame) -> dict:
    last_gid = pg.loc[pg.team == team, "game_id"].max()
    g = pg[(pg.game_id == last_gid) & (pg.team == team)]
    sk = {int(r.player): {"line": int(r.line) if pd.notna(r.line) else (4 if r.pos != "D" else 3),
                          "pp": int(r.pp_unit), "pk": int(r.pk_unit)} for r in g.itertuples()}
    st = starters[(starters.game_id == last_gid) & (starters.team == team)]
    return {"skaters": sk, "goalie": int(st.goalie.iloc[0]) if len(st) else None,
            "source": f"last game ({pd.Timestamp(g.date.iloc[0]).date()})" if len(g) else "none"}


def load_overrides() -> dict:
    """Manual overrides keyed 'date:team' (a secret gist on Streamlit Cloud, else a local file)."""
    return store.load()


def save_override(day: str, team: str, dep: dict | None) -> None:
    """dep=None clears the override for that team on that date."""
    o = load_overrides()
    k = f"{day}:{team}"
    if dep is None:
        o.pop(k, None)
    else:
        o[k] = {"skaters": {str(p): v for p, v in dep["skaters"].items()}, "goalie": dep.get("goalie"),
                "backup": dep.get("backup"), "source": "manual override"}
    store.save(o)


def _from_json(d: dict) -> dict:
    return {"skaters": {int(p): v for p, v in d["skaters"].items()}, "goalie": d.get("goalie"),
            "backup": d.get("backup"), "source": d.get("source", "manual override")}


def deployments(state: dict, day: str, use_dfo: bool = True, log: bool = True,
                overrides: bool = True) -> dict[str, dict]:
    """Deployment for every team playing on `day`: override > Daily Faceoff > last game."""
    games = state["games"][state["games"].date == day]
    teams = sorted(set(games.home) | set(games.away))
    names = dict(zip(games.home, games.home_name)) | dict(zip(games.away, games.away_name))
    out: dict[str, dict] = {t: deployment_from_last_game(t, state["history"], state["starters"]) for t in teams}
    meta = {}
    if use_dfo:
        try:
            slugs = dfo_slugs()
            starters = dfo_starters(day)
        except Exception:
            slugs, starters = {}, pd.DataFrame()
        line_rows = []

        def get(t):
            slug = slugs.get(key(names[t]))
            try:
                return t, (dfo_lines(slug) if slug else None)
            except Exception:
                return t, None

        with ThreadPoolExecutor(6) as pool_ex:
            charts = dict(pool_ex.map(get, teams))
        for t in teams:
            if charts.get(t) is None:
                continue
            lines, m = charts[t]
            pool = state["rosters"].get(t)
            if pool is None or lines.empty:
                continue
            dep = deployment_from_dfo(lines, pool, state["known"])
            if len(dep["skaters"]) >= 16:
                dep["goalie"] = dep["goalie"] or out[t]["goalie"]
                out[t] = dep
            meta[t] = m
            line_rows.append(lines.assign(team=t, date=day, source=m["source"], updated=m["updated"]))
        if len(starters):
            for t in teams:
                s = starters[starters.team_name.map(key) == key(names[t])]
                if len(s) and s.dfo_goalie.iloc[0]:
                    pool = state["rosters"].get(t)
                    gid = _match(pd.DataFrame({"dfo_name": [s.dfo_goalie.iloc[0]], "number": [None]}), pool).iloc[0] \
                        if pool is not None else None
                    if pd.notna(gid):
                        out[t]["goalie"] = int(gid)
                        out[t]["goalie_status"] = s.status.iloc[0]
        if log:
            log_snapshot("lines", pd.concat(line_rows, ignore_index=True) if line_rows else pd.DataFrame())
            log_snapshot("starters", starters)
    for k, v in (load_overrides() if overrides else {}).items():
        d, t = k.split(":")
        if d == day and t in out:
            out[t] = _from_json(v)
    for t in out:
        out[t]["dfo"] = meta.get(t)
    return out


# ---------------------------------------------------------------- projection

F_SPOTS, D_SPOTS = ("LW", "C", "RW"), ("LD", "RD")
POS_SPOT = {"L": "LW", "C": "C", "R": "RW"}


def assign_spots(dep: dict, info: pd.DataFrame) -> dict[tuple[str, int, str], int]:
    """Depth chart {(F|D, line, spot): player}. Daily Faceoff's own spots are kept; anyone
    without one goes to his natural spot (position, or the side he shoots for defence) if
    free, else the first free spot on his line. `info` has player, pos, shoots."""
    pos = dict(zip(info.player, info.pos))
    shoots = dict(zip(info.player, info.get("shoots", pd.Series(dtype=str))))
    chart: dict[tuple[str, int, str], int] = {}
    later = []
    for p, v in dep["skaters"].items():
        grp = "D" if pos.get(p) == "D" else "F"
        line = int(v.get("line") or (3 if grp == "D" else 4))
        spot = v.get("spot")
        if spot in (F_SPOTS if grp == "F" else D_SPOTS) and (grp, line, spot) not in chart:
            chart[(grp, line, spot)] = p
        else:
            later.append((p, grp, line))
    for p, grp, line in later:
        spots = F_SPOTS if grp == "F" else D_SPOTS
        nat = POS_SPOT.get(pos.get(p)) if grp == "F" else ("LD" if shoots.get(p) == "L" else "RD")
        order = [nat] + [s for s in spots if s != nat]
        lines = [line] + [l for l in range(1, 5 if grp == "F" else 4) if l != line]
        for l in lines:
            free = [s for s in order if (grp, l, s) not in chart]
            if free:
                chart[(grp, l, free[0])] = p
                break
    return chart


def slot_weight(dep: dict) -> float:
    """How much to trust the line slots: 0 when they are just the last game's (a team's last
    game, or a Daily Faceoff chart still marked 'Last Game'), 0.5 when someone has updated them
    for tonight (a Daily Faceoff beat reporter, or a manual override)."""
    if dep.get("source", "").startswith("last game"):
        return 0.0
    dfo = dep.get("dfo") or {}
    if dep.get("source") == "Daily Faceoff" and str(dfo.get("source", "")).lower().startswith("last game"):
        return 0.0
    return 0.5


def apply_deployment(slate: pd.DataFrame, deps: dict[str, dict], state: dict) -> pd.DataFrame:
    """Set who dresses, slots, linemates and opposing goalies on the cached pregame states."""
    s = slate.copy()
    s["dressed"] = False
    for c in ("given_line", "given_pp", "given_pk", "given_m1", "given_m2", "given_d1", "toi_target"):
        s[c] = np.nan
    for team, dep in deps.items():
        sk = dep["skaters"]
        rows = s.index[(s.team == team) & s.player.isin(list(sk))]
        s.loc[rows, "slot_w"] = slot_weight(dep)
        for i in rows:
            p = int(s.at[i, "player"])
            s.at[i, "dressed"] = True
            s.at[i, "given_line"] = sk[p].get("line") or (3 if s.at[i, "grp"] == "D" else 4)
            s.at[i, "given_pp"] = sk[p].get("pp") or 0
            s.at[i, "given_pk"] = sk[p].get("pk") or 0
            if sk[p].get("toi"):
                s.at[i, "toi_target"] = float(sk[p]["toi"])
        for grp in ("F", "D"):
            members = s.loc[rows][s.loc[rows].grp == grp]
            for ln, g in members.groupby("given_line"):
                ids = list(g.player)
                for i, p in zip(g.index, ids):
                    mates = [q for q in ids if q != p]
                    if grp == "F":
                        s.at[i, "given_m1"] = mates[0] if mates else np.nan
                        s.at[i, "given_m2"] = mates[1] if len(mates) > 1 else np.nan
                    else:
                        s.at[i, "given_d1"] = mates[0] if mates else np.nan
    s = s[s.dressed].copy()
    lk = state["slots"]
    s["slot_ev"] = features.slot_value(lk, "ev", s.team, s.grp, s.given_line)
    s["slot_pp"] = features.slot_value(lk, "pp", s.team, s.grp, s.given_pp)
    s["slot_sh"] = features.slot_value(lk, "sh", s.team, s.grp, s.given_pk)
    s["mate"] = features._mates(s.reset_index(drop=True)).to_numpy()
    gsax = state["goalies"]
    goalie_of = {t: d.get("goalie") for t, d in deps.items()}
    s["opp_goalie"] = s.opp.map(goalie_of)
    s["opp_gsax"] = s.opp_goalie.map(gsax).fillna(0.0)
    return s


def project(slate: pd.DataFrame, deps: dict[str, dict], state: dict) -> pd.DataFrame:
    s = apply_deployment(slate, deps, state)
    if s.empty:
        return s
    co = state["coefs"]
    p = model.predict(s, co)
    p = pd.concat([p, model.probabilities(p, co)], axis=1)
    return pd.concat([s.drop(columns=[c for c in p.columns if c in s.columns]), p], axis=1)


# ---------------------------------------------------------------- build

def refresh_lake(verbose: bool = True) -> None:
    fetch.fetch_season(C.CURRENT_SEASON, verbose)
    fetch.fetch_toi(C.CURRENT_SEASON)
    parse.build_season(C.CURRENT_SEASON)
    tables.build(daily=True)
    fetch.fetch_bio()  # ages for anyone new


def _pending(games: pd.DataFrame, rosters: dict[str, pd.DataFrame], pg: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    last_pos = pg.drop_duplicates("player", keep="last").set_index("player").pos.astype(str)
    prow, trow = [], []
    for g in games.itertuples():
        season = g.game_id // 1_000_000
        for team, opp, home in ((g.home, g.away, True), (g.away, g.home, False)):
            trow.append({"game_id": g.game_id, "season": season, "game_type": g.game_type,
                         "date": pd.Timestamp(g.date), "team": team, "opp": opp, "is_home": home})
            r = rosters.get(team, pd.DataFrame())
            for p in r[r.pos != "G"].itertuples():
                prow.append({"game_id": g.game_id, "season": season, "game_type": g.game_type,
                             "date": pd.Timestamp(g.date), "team": team, "opp": opp, "is_home": home,
                             "player": p.player, "name": p.name, "birth": p.birth,
                             "pos": last_pos.get(p.player, p.pos)})
    return (pd.DataFrame(prow).assign(pending=True), pd.DataFrame(trow).assign(pending=True))


def _starters(shots: pd.DataFrame) -> pd.DataFrame:
    s = shots[shots.event.isin(["goal", "sog"]) & shots.goalie.notna()]
    n = s.groupby(["game_id", "opp", "goalie"], observed=True).size().reset_index(name="n")
    n = n.sort_values("n").groupby(["game_id", "opp"]).tail(1).rename(columns={"opp": "team"})
    n["team"] = n.team.astype(str)
    return n[["game_id", "team", "goalie"]]


def build_state(fetch_new: bool = True, verbose: bool = True) -> dict:
    if fetch_new:
        refresh_lake(verbose)
    pg, tg = features.load_history()
    shots = pd.read_parquet(C.LAKE / "shots_xg.parquet",
                            columns=["game_id", "date", "season", "opp", "goalie", "event", "xg_on", "target_net_empty"])
    games = schedule()
    teams = sorted(set(games.home) | set(games.away)) if len(games) else []
    rosters = {t: roster(t) for t in teams}
    ppend, tpend = _pending(games, rosters, pg) if len(games) else (pd.DataFrame(), pd.DataFrame())
    f = features.build(pd.concat([pg, ppend], ignore_index=True), pd.concat([tg, tpend], ignore_index=True), shots)
    hist = f[~f.pending]
    co = model.fit(hist[hist.season >= FIRST_TRAIN])
    model.save(co)
    names = pg.drop_duplicates("player", keep="last")[["player", "name", "team"]]
    known = names.assign(name=names.name.astype(str), last=names.name.astype(str).str.split(" ", n=1).str[-1],
                         number=None, pos=None)
    state = {"built_at": datetime.now().isoformat(timespec="minutes"), "games": games,
             "slate": f[f.pending].reset_index(drop=True), "coefs": co,
             "slots": features.slot_lookup(f), "goalies": features.goalie_latest(shots),
             "rosters": rosters, "known": known,
             "history": pg[pg.season >= C.CURRENT_SEASON - 1], "starters": _starters(shots)}
    LIVE.mkdir(parents=True, exist_ok=True)
    with open(LIVE / "state.pkl", "wb") as fh:
        pickle.dump(state, fh)
    if verbose:
        print(f"state built: {len(games)} games, {len(state['slate'])} roster skaters, "
              f"coefs fitted on {len(hist[hist.season >= FIRST_TRAIN]):,} player-games")
    return state


def load_state() -> dict | None:
    p = LIVE / "state.pkl"
    if not p.exists():
        return None
    with open(p, "rb") as fh:
        return pickle.load(fh)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-fetch", action="store_true")
    args = ap.parse_args()
    state = build_state(fetch_new=not args.no_fetch)
    for day in sorted(state["games"].date.unique())[:1]:
        deps = deployments(state, day)
        proj = project(state["slate"][state["slate"].date == pd.Timestamp(day)], deps, state)
        cols = ["name", "team", "opp", "given_line", "given_pp", "toi_hat", "sog_hat", "g_hat", "a_hat", "pts_hat", "p_pts_1"]
        print(f"\n{day}: top projected points")
        print(proj.sort_values("pts_hat", ascending=False)[cols].head(15).round(2).to_string(index=False))
        for t, d in deps.items():
            print(f"  {t}: {len(d['skaters'])} skaters from {d['source']}"
                  f"{' / ' + d['dfo']['source'] if d.get('dfo') else ''}; goalie {d.get('goalie')} {d.get('goalie_status', '')}")


if __name__ == "__main__":
    main()

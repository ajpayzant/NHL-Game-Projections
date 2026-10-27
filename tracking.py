"""Results tracking for the Results tab: the model's final projections vs the betting lines.

    python tracking.py snapshot    # store today's final projections (no API credits)
    python tracking.py grade       # grade every finished day: opening + closing lines, results
    python tracking.py backtest    # results_backtest.parquet from the 2024-26 research bets

The refresh workflow runs snapshot and grade after live.py. Snapshot keeps the last
projection of each game before it starts (a later refresh replaces games not yet started).
Grade pulls, from The Odds API's historical data, the opening (10 AM ET) and closing (10 minutes
before puck drop) prices of DraftKings, FanDuel, BetMGM, Caesars and Fanatics for every game
line and prop, prices each with the stored projection exactly as the Edges tab does, and
grades it with the final result. About 120 credits per game.

Only aggregates are kept (bets, wins, profit and log-loss sums by date, opening / closing price,
market, side and edge bucket): no individual prices, since these files are published with the
app's data. Profit is per 1-unit bet at the best of the five books' prices.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

import config as C
import edges
import live
from saves import live as saves_live

TRACK = C.DATA / "tracking"
RESULTS_LIVE = TRACK / "results_live.parquet"
RESULTS_BACKTEST = TRACK / "results_backtest.parquet"
HIST = "https://api.the-odds-api.com/v4/historical/sports/icehockey_nhl"
# The Odds API (the Edges tab now uses OpticOdds, odds.py): its book keys, markets and key
HIST_BOOKS = {"draftkings": "DraftKings", "fanduel": "FanDuel", "betmgm": "BetMGM", "williamhill_us": "Caesars",
              "fanatics": "Fanatics"}
GAME_MARKETS = ["h2h", "spreads", "totals"]
PROP_MARKETS = ["player_points", "player_shots_on_goal", "player_goals", "player_assists",
                "player_power_play_points", "player_total_saves"]


def api_key() -> str | None:
    """The Odds API key: ODDS_API_KEY in the environment (the workflow's repo secret)."""
    return os.environ.get("ODDS_API_KEY")


def _rows(ev: dict, fetched_at: str) -> list[dict]:
    """A The Odds API event in odds.py's row format."""
    out = []
    for bk in ev.get("bookmakers", []):
        for mk in bk.get("markets", []):
            for o in mk.get("outcomes", []):
                out.append({"event_id": ev["id"], "commence_time": ev["commence_time"],
                            "home": ev["home_team"], "away": ev["away_team"],
                            "book": bk.get("title") or bk["key"], "market": mk["key"],
                            "name": o.get("name"), "player": o.get("description"),
                            "point": o.get("point"), "price": o.get("price"),
                            "last_update": mk.get("last_update") or bk.get("last_update"),
                            "fetched_at": fetched_at})
    return out
OPEN_UTC = "14:00:00Z"           # 10 AM ET on game day
CLOSE_BEFORE = timedelta(minutes=10)
BUCKETS = [(-1.0, 0.0, "< 0"), (0.0, 0.02, "0-2%"), (0.02, 0.05, "2-5%"), (0.05, 0.08, "5-8%"),
           (0.08, 0.12, "8-12%"), (0.12, 1.0, "12%+")]
PRIMARY = {"Over", "Favourite", "Favourite -"}   # one side per bet, for accuracy (log loss)
ET = ZoneInfo("America/New_York")


def _today() -> str:
    return datetime.now(ET).date().isoformat()


def _files(day: str) -> dict:
    return {k: TRACK / f"snap_{day}_{k}.parquet" for k in ("games", "skaters", "goalies")} | \
        {"meta": TRACK / f"snap_{day}_meta.json", "done": TRACK / f"graded_{day}.done"}


# ---------------------------------------------------------------- snapshot

def snapshot(state: dict | None = None, day: str | None = None, verbose: bool = True) -> None:
    """Store the day's projections; games that have started keep their earlier snapshot."""
    state = state or live.load_state()
    day = day or _today()
    games = state["games"][state["games"].date == day]
    if games.empty:
        if verbose:
            print(f"tracking: no games on {day}")
        return
    deps = live.deployments(state, day, log=False)
    p = live.project(state["slate"][state["slate"].date == pd.Timestamp(day)], deps, state)
    sk = p[["game_id", "team", "opp", "player", "name", "g_hat", "a_hat", "pts_hat", "sog_hat", "ppp_hat"]].copy()
    names = {int(a): n for r in state["rosters"].values() if r is not None and len(r)
             for a, n in zip(r.player, r.name)}
    rows = []
    S = saves_live.load_state()
    if S is not None:
        for g in games.itertuples():
            for team in (g.away, g.home):
                d = deps.get(team, {})
                cand = S["cand"][(S["cand"].game_id == g.game_id) & (S["cand"].team == team)]
                pr = saves_live.starter_probs(cand, {team: {"goalie": d.get("goalie"), "source": d.get("source", ""),
                                                            "goalie_status": d.get("goalie_status", "")}})
                for q in saves_live.project_team(S, int(g.game_id), team, pr, seed=int(g.game_id) % 100000):
                    rows.append({"game_id": int(g.game_id), "team": team, "goalie_id": q["goalie"],
                                 "goalie": names.get(q["goalie"], q["name"]), "mean": q["mean"],
                                 "p_over": json.dumps({c[7:]: v for c, v in q.items() if c.startswith("p_over_")})})
    gl = pd.DataFrame(rows)
    f = _files(day)
    TRACK.mkdir(parents=True, exist_ok=True)
    now = pd.Timestamp.now(tz="UTC")
    started = set(games.game_id[pd.to_datetime(games.start_utc, utc=True) <= now])
    for k, new in (("games", games), ("skaters", sk), ("goalies", gl)):
        if f[k].exists() and started:   # keep what was projected before those games began
            old = pd.read_parquet(f[k])
            new = pd.concat([old[old.game_id.isin(started)], new[~new.game_id.isin(started)]], ignore_index=True)
        new.to_parquet(f[k], index=False)
    f["meta"].write_text(json.dumps({"alpha": state["coefs"]["alpha"], "built_at": state["built_at"],
                                     "snapshot_at": now.isoformat(timespec="minutes")}))
    if verbose:
        print(f"tracking: snapshot {day}: {len(games)} games, {len(sk)} skaters, {len(gl)} goalies "
              f"({len(started)} already started, kept)")


# ---------------------------------------------------------------- grading

def _hist(path: str, key: str, **params) -> dict:
    r = requests.get(HIST + path, params={"apiKey": key, "bookmakers": ",".join(HIST_BOOKS),
                                          "oddsFormat": "american", **params}, timeout=60)
    r.raise_for_status()
    return r.json()


def _odds_for(day: str, games: pd.DataFrame, key: str) -> dict[str, pd.DataFrame]:
    """{'open': rows, 'close': rows} in odds.py's row format, for the day's games."""
    stamp = f"{day}T{OPEN_UTC}"
    out = {"open": [], "close": []}
    events = _hist("/events", key, date=stamp).get("data", [])
    names = {live.key(h) + "|" + live.key(a) for h, a in zip(games.home_name, games.away_name)}
    events = [e for e in events if live.key(e["home_team"]) + "|" + live.key(e["away_team"]) in names]
    snap = _hist("/odds", key, date=stamp, markets=",".join(GAME_MARKETS)).get("data", [])
    out["open"] += [r for ev in snap for r in _rows(ev, stamp)]
    for start, grp in pd.Series(events).groupby([e["commence_time"] for e in events]):
        close = (pd.Timestamp(start) - CLOSE_BEFORE).strftime("%Y-%m-%dT%H:%M:%SZ")
        ids = {e["id"] for e in grp}
        snap = _hist("/odds", key, date=close, markets=",".join(GAME_MARKETS)).get("data", [])
        out["close"] += [r for ev in snap if ev["id"] in ids for r in _rows(ev, close)]
        for e in grp:
            for tag, when in (("open", stamp), ("close", close)):
                try:
                    ev = _hist(f"/events/{e['id']}/odds", key, date=when, markets=",".join(PROP_MARKETS)).get("data")
                except requests.HTTPError:
                    continue
                if ev:
                    out[tag] += _rows(ev, when)
    return {k: pd.DataFrame(v) for k, v in out.items()}


def _bets(o: pd.DataFrame, games: pd.DataFrame, sk: pd.DataFrame, gl: pd.DataFrame, alpha: dict) -> pd.DataFrame:
    """One row per bet side: the model's probability, the books' consensus no-vig, best price."""
    if o.empty:
        return pd.DataFrame()
    o = o[o.book.isin(HIST_BOOKS.values())]
    df, _ = edges.compare(o, games, sk, alpha, gl, now=pd.Timestamp("1970-01-01", tz="UTC"))
    if df.empty:
        return df
    df = df[df.novig.notna()]
    keys = ["game_id", "market", "name", "player", "point"]
    g = df.groupby(keys, dropna=False)
    b = g.agg(p_model=("p_model", "first"), p_mkt=("novig", "mean"), dec=("dec", "max"), kind=("kind", "first"),
              subject=("subject", "first"), pid=("pid", "first")).reset_index()
    b["side"] = np.where(b.market.isin(["totals"]) | b.market.str.startswith("player_"), b.name,
                         np.where(b.p_mkt >= 0.5, "Favourite", "Underdog"))
    b.loc[b.market == "spreads", "side"] = np.where(b.point[b.market == "spreads"] < 0, "Favourite -", "Underdog +")
    return b


def _outcomes(b: pd.DataFrame, games: pd.DataFrame) -> pd.Series:
    """1 = won, 0 = lost, NaN = push / void / not played."""
    tg = pd.read_parquet(C.LAKE / "team_games.parquet", columns=["game_id", "team", "goals", "goals_against"])
    tg["team"] = tg.team.astype(str)
    score = {(g, t): (gf, ga) for g, t, gf, ga in zip(tg.game_id, tg.team, tg.goals, tg.goals_against)}
    pg = pd.read_parquet(C.LAKE / "player_games.parquet",
                         columns=["game_id", "player", "toi", "sog", "points", "goals", "assists", "g_pp", "a_pp"])
    pg = pg[pg.toi > 0].assign(ppp=lambda d: d.g_pp + d.a_pp).set_index(["game_id", "player"])
    gk = pd.read_parquet(C.DATA / "saves" / "lake" / "goalies.parquet", columns=["game_id", "goalie", "starter", "saves"])
    gk = gk[gk.starter].set_index(["game_id", "goalie"]).saves
    gm = games.set_index("game_id")
    stat = {"player_shots_on_goal": "sog", "player_points": "points", "player_goals": "goals",
            "player_assists": "assists", "player_power_play_points": "ppp"}
    out = []
    for r in b.itertuples(index=False):
        g = gm.loc[r.game_id] if r.game_id in gm.index else None
        over = r.name == "Over"
        if r.market in ("h2h", "spreads", "totals"):
            if g is None or (r.game_id, g.home) not in score:
                out.append(np.nan); continue
            gf, ga = score[(r.game_id, g.home)]
            if r.market == "totals":
                t = gf + ga
                out.append(np.nan if t == r.point else float((t > r.point) == over)); continue
            home = live.key(r.name) == live.key(g.home_name)
            margin = (gf - ga) if home else (ga - gf)
            v = margin if r.market == "h2h" else margin + r.point
            out.append(np.nan if v == 0 else float(v > 0)); continue
        if r.market == "player_total_saves":
            v = gk.get((r.game_id, r.pid), np.nan) if pd.notna(r.pid) else np.nan
        else:
            v = pg[stat[r.market]].get((r.game_id, r.pid), np.nan) if pd.notna(r.pid) else np.nan
        out.append(np.nan if not np.isfinite(v) or v == r.point else float((v > r.point) == over))
    return pd.Series(out, index=b.index)


def aggregate(b: pd.DataFrame, source: str) -> pd.DataFrame:
    """Sums by date, snapshot, market, side and edge bucket (no individual prices)."""
    b = b.dropna(subset=["won", "p_model", "p_mkt", "dec"]).copy()
    b["edge"] = b.p_model - b.p_mkt
    b["bucket"] = pd.cut(b.edge, [x[0] for x in BUCKETS] + [1.0], labels=[x[2] for x in BUCKETS], right=False)
    b["pnl"] = np.where(b.won == 1, b.dec - 1, -1.0)
    pm, pk = np.clip(b.p_model, 1e-6, 1 - 1e-6), np.clip(b.p_mkt, 1e-6, 1 - 1e-6)
    b["ll_model"] = -(b.won * np.log(pm) + (1 - b.won) * np.log(1 - pm))
    b["ll_mkt"] = -(b.won * np.log(pk) + (1 - b.won) * np.log(1 - pk))
    b["primary"] = b.side.isin(PRIMARY)
    a = b.groupby(["date", "snap", "kind", "side", "bucket", "primary"], observed=True).agg(
        bets=("won", "size"), wins=("won", "sum"), pnl=("pnl", "sum"), ll_model=("ll_model", "sum"),
        ll_mkt=("ll_mkt", "sum")).reset_index()
    a["bucket"] = a.bucket.astype(str)
    return a.assign(source=source)


def grade(day: str | None = None, verbose: bool = True) -> None:
    """Grade every snapshot day before today that is not graded yet (or just `day`)."""
    key = api_key()
    if not key:
        if verbose:
            print("tracking: no ODDS_API_KEY, grading skipped")
        return
    days = [day] if day else sorted({f.name[5:15] for f in TRACK.glob("snap_*_games.parquet")})
    for d in days:
        f = _files(d)
        if f["done"].exists() or d >= _today() or not f["games"].exists():
            continue
        games, sk = pd.read_parquet(f["games"]), pd.read_parquet(f["skaters"])
        gl = pd.read_parquet(f["goalies"]) if f["goalies"].exists() else pd.DataFrame()
        if len(gl):
            gl["p_over"] = gl.p_over.map(lambda s: {float(k): v for k, v in json.loads(s).items()})
        alpha = json.loads(f["meta"].read_text())["alpha"]
        o = _odds_for(d, games, key)
        parts = []
        for snap, rows in o.items():
            b = _bets(rows, games, sk, gl, alpha)
            if len(b):
                b["won"] = _outcomes(b, games)
                parts.append(b.assign(snap=snap, date=d))
        if not parts:
            continue
        agg = aggregate(pd.concat(parts, ignore_index=True), "live")
        old = pd.read_parquet(RESULTS_LIVE) if RESULTS_LIVE.exists() else pd.DataFrame()
        if len(old):
            old = old[old.date != d]
        pd.concat([old, agg], ignore_index=True).to_parquet(RESULTS_LIVE, index=False)
        f["done"].write_text(datetime.now(timezone.utc).isoformat(timespec="minutes"))
        if verbose:
            print(f"tracking: graded {d}: {int(agg.bets.sum()):,} bet sides, profit {agg.pnl.sum():+.1f} units")


def backtest() -> None:
    """The 2024-26 research bets (research.market_edges) as the tab's backtest view."""
    A = pd.read_parquet(C.DATA / "odds_hist" / "bets.parquet")
    tg = pd.read_parquet(C.LAKE / "team_games.parquet", columns=["game_id", "date"]).drop_duplicates("game_id")
    A = A.merge(tg.assign(date=pd.to_datetime(tg.date).dt.strftime("%Y-%m-%d")), on="game_id", how="left")
    A["dec"] = A.best.map(lambda a: 1 + a / 100 if a > 0 else 1 + 100 / abs(a))
    TRACK.mkdir(parents=True, exist_ok=True)
    aggregate(A, "backtest").to_parquet(RESULTS_BACKTEST, index=False)
    print(f"tracking: backtest results from {len(A):,} bet sides -> {RESULTS_BACKTEST.name}")


if __name__ == "__main__":
    try:
        {"snapshot": snapshot, "grade": grade, "backtest": backtest}[sys.argv[1]]()
    except Exception as e:   # never fail the refresh over tracking
        print(f"tracking: {sys.argv[1]} failed: {e!r}", file=sys.stderr)

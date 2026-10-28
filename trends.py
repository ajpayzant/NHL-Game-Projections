"""Streaks, form, matchups, milestones and hit rates for the Trends tab.

    python trends.py careers     # official career totals for every rostered player -> data/live/careers.parquet

Context only: none of this feeds a projection (the research found recent form adds nothing once
TOI and the player's longer-run rates are known), so each table shows tonight's projection next
to the trend. Everything is computed from games before the selected date.

Career milestones need the NHL's own career totals (our game history starts in 2018-19), so the
refresh stores them from /player/{id}/landing; the rest comes from player_games / team_games /
the goalie game log.
"""
from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import numpy as np
import pandas as pd

import config as C
import fetch
import model

CAREERS = C.DATA / "live" / "careers.parquet"
MIN_BASE_GP = 20        # games behind a player's usual rate before he can be hot or cold
FORM_N = 10             # the "recent" window for hot / cold


# ---------------------------------------------------------------- career totals (refresh)

def _career(pid: int) -> dict | None:
    try:
        d = fetch.get_json(f"{C.NHL_WEB}/player/{int(pid)}/landing")
    except Exception:
        return None
    t = (d.get("careerTotals") or {}).get("regularSeason") or {}
    reg = [g["gameDate"] for g in d.get("last5Games") or [] if g.get("gameTypeId") == 2]
    row = {"player": int(pid), "pos": d.get("position"), "gp": t.get("gamesPlayed", 0),
           "through": max(reg) if reg else None}   # the totals include his games up to this date
    if d.get("position") == "G":
        row |= {"wins": t.get("wins", 0), "shutouts": t.get("shutouts", 0)}
    else:
        row |= {"goals": t.get("goals", 0), "assists": t.get("assists", 0), "points": t.get("points", 0),
                "ppp": t.get("powerPlayPoints", 0)}
    return row


def refresh_careers(state: dict, verbose: bool = True) -> pd.DataFrame:
    """NHL regular-season career totals for everyone on a roster of a team with an upcoming game."""
    pids = sorted({int(p) for r in state["rosters"].values() if r is not None and len(r) for p in r.player})
    with ThreadPoolExecutor(8) as ex:
        rows = [r for r in ex.map(_career, pids) if r]
    df = pd.DataFrame(rows).assign(fetched_at=datetime.now().isoformat(timespec="minutes"))
    CAREERS.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CAREERS, index=False)
    if verbose:
        print(f"careers: {len(df)} of {len(pids)} players")
    return df


# ---------------------------------------------------------------- helpers

def _run(x: np.ndarray) -> int:
    """Length of the run of True at the end of x."""
    n = 0
    for v in x[::-1]:
        if not v:
            break
        n += 1
    return n


def season_games(h: pd.DataFrame, players, day: str) -> pd.DataFrame:
    """This season's games before `day` (regular season and playoffs, as streaks count), oldest first."""
    h = h[h.player.isin(players) & (h.date < pd.Timestamp(day)) & (h.season == C.CURRENT_SEASON)]
    return h.sort_values(["player", "date"])


def baseline(h: pd.DataFrame, players, day: str) -> pd.DataFrame:
    """Each player's usual per-game rates: this season and last season's regular season, before
    his last FORM_N games (so the recent window is compared with something it is not part of)."""
    h = h[h.player.isin(players) & (h.date < pd.Timestamp(day)) & (h.season >= C.CURRENT_SEASON - 1)
          & (h.game_type == 2)].sort_values(["player", "date"])
    rank = h.groupby("player").cumcount(ascending=False)
    b = h[rank >= FORM_N].groupby("player")
    out = pd.DataFrame({"base_gp": b.size(), "base_pts": b.points.mean(), "base_g": b.goals.mean(),
                        "base_a": b.assists.mean(), "base_sog": b.sog.mean(), "base_toi": b.toi.mean() / 60,
                        "base_shp": b.goals.sum() / b.sog.sum().replace(0, np.nan)})
    return out


# ---------------------------------------------------------------- skaters

def form(h: pd.DataFrame, proj: pd.DataFrame, day: str) -> pd.DataFrame:
    """Last FORM_N games against the player's usual rates. Heat is the Poisson z-score of points
    and shots in the window (0 = exactly as usual), averaged; Driver says what is behind it."""
    players = proj.player.unique()
    h = h[h.player.isin(players) & (h.date < pd.Timestamp(day)) & (h.season >= C.CURRENT_SEASON - 1)]
    recent = h.sort_values("date").groupby("player").tail(FORM_N)
    # a window that reaches back into last season only counts if he played this season
    cur = h[h.season == C.CURRENT_SEASON].groupby("player").size()
    g = recent.groupby("player")
    f = pd.DataFrame({"gp": g.size(), "pts": g.points.sum(), "g": g.goals.sum(), "a": g.assists.sum(),
                      "sog": g.sog.sum(), "toi": g.toi.mean() / 60})
    f = f.join(baseline(h, players, day), how="inner")
    f = f[(f.base_gp >= MIN_BASE_GP) & (f.gp >= 5) & (cur.reindex(f.index).fillna(0) >= 3)]
    if f.empty:
        return f
    exp_pts, exp_sog = f.gp * f.base_pts, f.gp * f.base_sog
    z_pts = (f.pts - exp_pts) / np.sqrt(exp_pts.clip(lower=0.5))
    z_sog = (f.sog - exp_sog) / np.sqrt(exp_sog.clip(lower=0.5))
    f["heat"] = (z_pts + z_sog) / 2
    f["pts_gm"], f["sog_gm"] = f.pts / f.gp, f.sog / f.gp
    f["shp"] = f.g / f.sog.replace(0, np.nan)
    f["driver"] = [_driver(r) for r in f.itertuples()]
    return f.join(proj.set_index("player")[["name", "team", "opp", "is_home", "pts_hat", "sog_hat", "p_pts_1"]])


def _driver(r) -> str:
    """The main reason a window differs from usual: ice time, shot volume or finishing."""
    bits = []
    d_toi = r.toi - r.base_toi
    if abs(d_toi) >= 1.5:
        bits.append(f"TOI {r.toi:.1f} ({d_toi:+.1f} min)")
    d_sog = r.sog / r.gp - r.base_sog
    if abs(d_sog) >= max(0.5, 0.25 * r.base_sog):
        bits.append(f"{r.sog / r.gp:.1f} shots/gm (usually {r.base_sog:.1f})")
    if r.sog >= 10 and np.isfinite(r.base_shp) and abs(r.g / r.sog - r.base_shp) >= 0.06:
        bits.append(f"shooting {r.g / r.sog:.0%} (usually {r.base_shp:.0%})")
    return " · ".join(bits) or "same role and volume"


STREAKS = [  # (label, column, threshold, minimum length shown)
    ("Point streak", "points", 1, 3), ("Goal streak", "goals", 1, 2), ("Assist streak", "assists", 1, 3),
    ("Multi-point streak", "points", 2, 2), ("3+ shots streak", "sog", 3, 4)]
DROUGHTS = [  # (label, column, his usual per-game rate column, the least usual rate that makes it news)
    ("Goal drought", "goals", "base_g", 0.30), ("Point drought", "points", "base_pts", 0.65),
    ("Shot slump (under 2)", "sog", "base_sog", 3.0)]


def streaks(h: pd.DataFrame, proj: pd.DataFrame, day: str) -> pd.DataFrame:
    """Active runs, counted in consecutive games played this season. A drought is listed only for
    a player who usually produces, with the number he would normally have had in that span."""
    players = proj.player.unique()
    s = season_games(h, players, day)
    base = baseline(h, players, day)
    meta = proj.set_index("player")
    rows = []
    for pid, g in s.groupby("player"):
        if pid not in meta.index:
            continue
        for label, col, k, need in STREAKS:
            n = _run(g[col].to_numpy() >= k)
            if n >= need:
                w = g.tail(n)
                rows.append({"player": pid, "kind": "Hot", "streak": label, "games": n,
                             "during": f"{int(w.goals.sum())} G, {int(w.assists.sum())} A, {int(w.sog.sum())} SOG",
                             "since": w.date.iloc[0]})
        b = base.loc[pid] if pid in base.index else None
        if b is None or b.base_gp < MIN_BASE_GP:
            continue
        for label, col, bcol, least in DROUGHTS:
            if b[bcol] < least:
                continue
            n = _run(g[col].to_numpy() < (2 if col == "sog" else 1))
            if n >= 3 and (col == "sog" or b[bcol] * n >= 2.0):
                w = g.tail(n)
                during = (f"{w.sog.mean():.1f} shots/gm, usually {b[bcol]:.1f}" if col == "sog" else
                          f"{int(w.sog.sum())} SOG, {w.toi.mean() / 60:.1f} TOI/gm · "
                          f"usually {b[bcol] * n:.1f} {col} in {n} games")
                rows.append({"player": pid, "kind": "Cold", "streak": label, "games": n, "during": during,
                             "since": w.date.iloc[0]})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    # one row per player and kind: "Point streak 7 · Assist streak 7", the longest first
    out = out.sort_values("games", ascending=False)
    out = out.assign(streak=out.streak + " " + out.games.astype(str)).groupby(["player", "kind"], sort=False).agg(
        streak=("streak", " · ".join), games=("games", "max"), during=("during", "first"),
        since=("since", "min")).reset_index()
    return out.join(meta[["name", "team", "opp", "is_home", "pts_hat", "g_hat", "sog_hat", "p_pts_1", "p_g_1", "p_a_1",
                          "p_pts_2", "p_sog_3", "p_sog_2"]], on="player")


def matchups(h: pd.DataFrame, proj: pd.DataFrame, day: str, min_gp: int = 5) -> pd.DataFrame:
    """Career (since 2018-19) points and shots per game against tonight's opponent, next to the
    same player's rate against everyone else over the same seasons."""
    p = proj[["player", "name", "team", "opp", "is_home", "pts_hat", "sog_hat", "p_pts_1"]]
    h = h[h.player.isin(p.player) & (h.date < pd.Timestamp(day))]
    m = h.merge(p[["player", "opp"]].rename(columns={"opp": "tonight"}), on="player")
    vs = m[m.opp == m.tonight].groupby("player")
    rest = m[m.opp != m.tonight].groupby("player")
    out = pd.DataFrame({"gp_vs": vs.size(), "pts_vs": vs.points.sum(), "g_vs": vs.goals.sum(),
                        "pts_gm_vs": vs.points.mean(), "sog_gm_vs": vs.sog.mean(),
                        "pts_gm": rest.points.mean(), "sog_gm": rest.sog.mean(), "gp_other": rest.size()})
    out = out[(out.gp_vs >= min_gp) & (out.gp_other >= 40)]
    last = m[m.opp == m.tonight].sort_values("date").groupby("player").tail(1).set_index("player")
    out["last"] = last.apply(lambda r: f"{r.date:%b %d, %Y}: {int(r.goals)} G, {int(r.assists)} A, {int(r.sog)} SOG",
                             axis=1).reindex(out.index)
    out["diff"] = out.pts_gm_vs - out.pts_gm
    # shrink the gap toward 0 by sample size so 5 lucky games don't top the list
    out["score"] = out["diff"] * out.gp_vs / (out.gp_vs + 15)
    return out.join(p.set_index("player"))


def goalie_form(gg: pd.DataFrame, goalies, day: str, n: int = 5) -> pd.DataFrame:
    """Each goalie's last n starts against his season (and last season): record, save %, GAA."""
    g = gg[gg.goalie.isin(goalies) & (gg.date < pd.Timestamp(day)) & gg.starter
           & (gg.season >= C.CURRENT_SEASON - 1)].sort_values("date")
    rows = []
    for gid, x in g.groupby("goalie"):
        last, cur = x.tail(n), x[x.season == C.CURRENT_SEASON]
        prev = x[(x.season == C.CURRENT_SEASON - 1) & (x.game_type == 2)]
        sv = lambda d: d.saves.sum() / d.shots_against.sum() if d.shots_against.sum() else np.nan
        dec = last.decision.fillna("")
        rows.append({"goalie": gid, "starts": len(last), "record": f"{(dec == 'W').sum()}-{(dec == 'L').sum()}-{(dec == 'O').sum()}",
                     "sv_last": sv(last), "sv_season": sv(cur), "sv_prev": sv(prev), "gaa_last": last.goals_against.mean(),
                     "saves_last": last.saves.mean(), "season_starts": len(cur),
                     "streak": f"{dec.iloc[-1]}{_run((dec == dec.iloc[-1]).to_numpy())}" if len(dec) and dec.iloc[-1] else ""})
    out = pd.DataFrame(rows)
    if len(out):   # vs this season, or last season before 5 starts
        base = np.where(out.season_starts >= 5, out.sv_season, out.sv_prev)
        out["sv_diff"] = out.sv_last - base
    return out


# ---------------------------------------------------------------- milestones

CAREER_STEPS = {"goals": (50, 3), "assists": (100, 3), "points": (100, 3), "gp": (100, 1),
                "wins": (50, 2), "shutouts": (10, 1)}     # (round number every, within reach)
SEASON_STEPS = {"goals": ([10, 20, 30, 40, 50, 60, 70], 2), "points": ([25, 50, 75, 100, 125], 3),
                "assists": ([25, 50, 75, 100], 2)}
WORD = {"goals": "goals", "assists": "assists", "points": "points", "gp": "games", "wins": "wins", "shutouts": "shutouts"}


def milestones(careers: pd.DataFrame | None, h: pd.DataFrame, proj: pd.DataFrame, goalies: pd.DataFrame,
               day: str, alpha: dict) -> pd.DataFrame:
    """Round numbers within reach tonight: career (NHL regular-season totals from the refresh, plus
    any regular-season games after the last one they include) and this season, and first NHL
    goals / points. Each row has the model's chance of getting there tonight where it has one."""
    rows = []
    meta = proj.set_index("player")
    if careers is not None and len(careers):
        cr = careers.set_index("player")
        through = pd.to_datetime(cr.through).reindex(h.player).to_numpy()
        later = h[(h.date.to_numpy() > through) & (h.date < pd.Timestamp(day)) & (h.game_type == 2)].groupby("player")
        add = pd.DataFrame({"gp": later.size(), "goals": later.goals.sum(), "assists": later.assists.sum(),
                            "points": later.points.sum()})
        for pid in meta.index.intersection(cr.index):
            c = cr.loc[pid]
            extra = add.loc[pid] if pid in add.index else None
            for stat in ("goals", "assists", "points", "gp"):
                v = int(c[stat]) + (int(extra[stat]) if extra is not None else 0)
                rows += _near(pid, "Career", stat, v, *CAREER_STEPS[stat])
                if stat in ("goals", "points") and v == 0 and c.gp >= 1:
                    rows.append({"player": pid, "scope": "Career", "stat": stat, "now": 0, "target": 1, "needs": 1,
                                 "milestone": f"First NHL {stat[:-1]}"})
        gm = goalies.set_index("goalie") if len(goalies) else pd.DataFrame()
        for gid in gm.index.intersection(cr.index):
            c = cr.loc[gid]
            for stat in ("wins", "shutouts", "gp"):
                if pd.notna(c.get(stat)):
                    rows += _near(gid, "Career", stat, int(c[stat]), *CAREER_STEPS[stat])
    s = season_games(h, meta.index, day)
    s = s[s.game_type == 2].groupby("player")[["goals", "assists", "points"]].sum()
    for pid, r in s.iterrows():
        for stat, (steps, reach) in SEASON_STEPS.items():
            v = int(r[stat])
            nxt = next((x for x in steps if x > v), None)
            if nxt is not None and nxt - v <= reach:
                rows.append({"player": pid, "scope": "Season", "stat": stat, "now": v, "target": nxt, "needs": nxt - v,
                             "milestone": f"{nxt} {WORD[stat]} this season"})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    hat = {"goals": ("g_hat", "g"), "points": ("pts_hat", "pts"), "assists": ("a_hat", "a")}
    out["p_tonight"] = [float(model.prob_at_least([meta.at[pid, hat[st][0]]], alpha[hat[st][1]], k)[0])
                        if st in hat and pid in meta.index else np.nan
                        for pid, st, k in zip(out.player, out.stat, out.needs)]
    return out


def _near(pid: int, scope: str, stat: str, v: int, every: int, reach: int) -> list[dict]:
    target = (v // every + 1) * every
    if target < every or target - v > reach:   # 100 goals is news, 50 assists is not
        return []
    if stat in ("assists", "points", "gp") and target < 100:
        return []
    return [{"player": pid, "scope": scope, "stat": stat, "now": v, "target": target, "needs": target - v,
             "milestone": f"{target:,} career {WORD[stat]}" if stat != "gp" else f"{target:,}th NHL game"}]


# ---------------------------------------------------------------- hit rates

def hit_rates(h: pd.DataFrame, proj: pd.DataFrame, day: str, col: str, k: int, n: int | None) -> pd.DataFrame:
    """How often each player reached k of `col` in his last n games (n None = this season)."""
    players = proj.player.unique()
    g = h[h.player.isin(players) & (h.date < pd.Timestamp(day))].sort_values("date")
    win = g[g.season == C.CURRENT_SEASON] if n is None else g[g.season >= C.CURRENT_SEASON - 1].groupby("player").tail(n)
    season = g[g.season == C.CURRENT_SEASON]
    out = pd.DataFrame({"gp": win.groupby("player").size(), "hits": (win[col] >= k).groupby(win.player).sum(),
                        "season_rate": (season[col] >= k).groupby(season.player).mean()})
    out["rate"] = out.hits / out.gp
    # the current run of games at or over the line
    out["run"] = win.groupby("player")[col].apply(lambda x: _run(x.to_numpy() >= k))
    return out.join(proj.set_index("player")[["name", "team", "opp", "is_home"]])


# ---------------------------------------------------------------- teams

def team_form(tg: pd.DataFrame, teams, day: str, n: int | None) -> pd.DataFrame:
    """Record, streak, scoring and how often each team cleared common lines, last n games
    (n None = this season)."""
    t = tg[tg.team.isin(teams) & (tg.date < pd.Timestamp(day)) & (tg.season == C.CURRENT_SEASON)].sort_values("date")
    if n:
        t = t.groupby("team").tail(n)
    t = t.assign(win=t.goals > t.goals_against, ot=t.last_period_type.isin(["OT", "SO"]),
                 total=t.goals + t.goals_against, margin=t.goals - t.goals_against)
    rows = []
    for team, g in t.groupby("team"):
        w, otl = int(g.win.sum()), int((~g.win & g.ot).sum())
        res = np.where(g.win, "W", np.where(g.ot, "OT", "L"))
        streak = f"{res[-1]}{_run(res == res[-1])}"   # W3, L2, OT1
        rows.append({"team": team, "gp": len(g), "record": f"{w}-{len(g) - w - otl}-{otl}", "streak": streak,
                     "gf": g.goals.mean(), "ga": g.goals_against.mean(), "sf": g.sog.mean(), "sa": g.sog_against.mean(),
                     "ppg": g.gf_pp.mean(), "o55": (g.total > 5.5).mean(), "o65": (g.total > 6.5).mean(),
                     "gf3": (g.goals >= 3).mean(), "ga2": (g.goals_against <= 2).mean(),
                     "cover": (g.margin >= 2).mean(), "ot_rate": g.ot.mean(), "last5": "".join(
                         {"W": "W", "L": "L", "OT": "O"}[x] for x in res[-5:])})
    return pd.DataFrame(rows).set_index("team") if rows else pd.DataFrame()


def head_to_head(tg: pd.DataFrame, home: str, away: str, day: str, n: int = 5) -> pd.DataFrame:
    """The two teams' last n meetings, from the home team's side."""
    m = tg[(tg.team == home) & (tg.opp == away) & (tg.date < pd.Timestamp(day))].sort_values("date").tail(n)
    return m.iloc[::-1]


if __name__ == "__main__":
    if sys.argv[1:] == ["careers"]:
        import live
        refresh_careers(live.load_state())

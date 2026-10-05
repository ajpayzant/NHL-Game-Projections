"""The goalie half of the live slate: upcoming games -> who starts -> each candidate's saves.

Called by the top-level live.py after the skater state is built (same games, same day):

    saves.live.refresh_lake()       # parse this season's raw games (already fetched), rebuild tables
    saves.live.build_state(games)   # refit, cache data/saves/models/live.pkl

Who starts is taken from the skater tool's deployments (Lines & goalies page): a manual
override pins the goalie, a Daily Faceoff starting-goalie report moves the starter model's
odds by DFO_LR, a Daily Faceoff line chart's G1 moves them a little, and a team with neither
falls back to the starter model alone. The saves model is fitted exactly as in the backtest:
on every season before this one, with the online corrections replayed over this season.
"""
from __future__ import annotations

import pickle
from datetime import datetime

import numpy as np
import pandas as pd

from . import config as C
from . import features as F
from . import fetch
from . import goalie_model as GM
from . import parse
from . import pipeline
from . import starters as ST
from .model import SavesModel

STATE = C.MODELS / "live.pkl"
# How much a Daily Faceoff report moves the odds on the named goalie (likelihood ratios).
# Starting values, not yet fitted: the starter snapshot log in data/dfo is there to calibrate them.
DFO_LR = {"Confirmed": 200.0, "Likely": 8.0, "Expected": 8.0, "Unconfirmed": 2.0}
LINE_CHART_LR = 2.0     # the goalie listed first on a Daily Faceoff line chart, with no report
MIN_SHOWN_P = 0.03


# ---------------------------------------------------------------- build

def refresh_lake(verbose: bool = True) -> None:
    """Parse this season's completed games (the skater refresh fetched them) and rebuild the
    modelling tables."""
    if (C.RAW / str(C.CURRENT_SEASON)).exists():
        parse.build_season(C.CURRENT_SEASON)
    pipeline.build(verbose=verbose)


def _team_name(t: dict) -> str:
    return f"{t['placeName']['default']} {t['commonName']['default']}"


def roster_goalies(team: str) -> list[dict]:
    try:
        r = fetch.get_json(f"{C.NHL_WEB}/roster/{team}/current")
    except Exception:
        return []
    return [{"goalie": g["id"], "full": f"{g['firstName']['default']} {g['lastName']['default']}",
             "name": f"{g['firstName']['default'][0]}. {g['lastName']['default']}"}
            for g in r.get("goalies", [])]


def _future_rows(games: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for g in games.itertuples(index=False):
        for team, opp, home in ((g.home, g.away, True), (g.away, g.home, False)):
            rows.append({"game_id": g.game_id, "season": g.game_id // 1_000_000,
                         "game_type": g.game_type, "date": g.date, "team": team, "opp": opp,
                         "is_home": home, "start_utc": g.start_utc})
    return pd.DataFrame(rows)


def build_state(games: pd.DataFrame, verbose: bool = True) -> dict:
    d = pipeline.load()
    tg = F._plain(d["team_games_xg"])
    fut = _future_rows(games) if len(games) else pd.DataFrame(columns=["game_id"])
    fut = fut[~fut.game_id.isin(tg.game_id)]
    M_all = F.build_matrix(pd.concat([tg, fut], ignore_index=True), parse.load("skater_games"))
    M_all = M_all.merge(tg[["game_id", "team", "xga_on"]], on=["game_id", "team"], how="left")
    G = F._plain(d["goalies"])
    P = d["pulls"].assign(season=d["pulls"].game_id // 1_000_000)
    cur = C.CURRENT_SEASON
    done = M_all.sa_faced.notna()

    # saves model: fitted as in the backtest, then the online corrections over this season
    model = SavesModel().fit(M_all[done & (M_all.season < cur)], G[G.season < cur], P[P.season < cur])
    played = M_all[done & (M_all.season == cur)]
    if len(played):
        played = pd.concat([played, model.team_rows(played)], axis=1)
        gs = G[G.season == cur]
        for day, tm in played.groupby("date", sort=True):
            gd = gs[(gs.date == day) & (gs.pbp_sa > 0)]
            model.online.update(tm.mu_raw.values, tm.sa_faced.values.astype(float),
                                float(gd.pbp_ga.sum()), float(gd.xg_on_faced.sum()), float(gd.pbp_sa.sum()))
    slate = M_all[M_all.game_id.isin(fut.game_id)].copy()
    if len(slate):
        slate = pd.concat([slate, model.team_rows(slate)], axis=1)
        slate["mu"] = slate.mu_raw * (1 + model.online.shot_bias)

    # starter model: fitted on every completed team-game; upcoming games use today's rosters
    rosters = {t: roster_goalies(t) for t in sorted(set(fut.team))} if len(fut) else {}
    names = dict(zip(G.goalie, G.name))
    for r in rosters.values():
        for g in r:
            names.setdefault(g["goalie"], g["name"])
    cand = ST.candidate_rows(ST.schedule_from(M_all), G,
                             rosters={t: {g["goalie"] for g in r} for t, r in rosters.items()},
                             names=names)
    sm = ST.StarterModel().fit(cand[cand.started.notna()])
    cand = cand[cand.game_id.isin(fut.game_id)].copy()
    hist_path = C.LAKE / "goalie_history.parquet"
    hist = pd.read_parquet(hist_path) if hist_path.exists() else None
    talent = GM.current_talent(G, hist, [g["goalie"] for r in rosters.values() for g in r])
    # candidate talent at view time is the current state, which includes his latest game
    cand["talent"] = 1000 * cand.goalie.map(talent.g_talent).fillna(0.0)
    cand["p_model"] = sm.predict(cand) if len(cand) else []
    state = {"built_at": datetime.now().isoformat(timespec="minutes"), "games": games,
             "slate": slate, "cand": cand, "model": model, "starter_model": sm,
             "talent": talent, "names": names, "rosters": rosters,
             "last_game_date": str(tg.date.max())}
    C.MODELS.mkdir(parents=True, exist_ok=True)
    with open(STATE, "wb") as f:
        pickle.dump(state, f)
    if verbose:
        print(f"saves state: {len(games)} games, {len(cand)} candidate rows, "
              f"shot_bias {model.online.shot_bias:+.3f}, goal_ratio {model.online.goal_ratio:.3f}")
    return state


def load_state() -> dict | None:
    if not STATE.exists():
        return None
    with open(STATE, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------- view-time

def starter_probs(cand: pd.DataFrame, deps: dict[str, dict]) -> pd.DataFrame:
    """Starter model probability, moved by the skater tool's deployment for each team:
    override > Daily Faceoff starting-goalie report > Daily Faceoff line chart > model only."""
    c = cand.copy()
    c["p"] = c.p_model
    c["status"] = ""
    for (gid, team), idx in c.groupby(["game_id", "team"]).groups.items():
        d = deps.get(team)
        g = d.get("goalie") if d else None
        if g is None:
            continue
        named = (c.loc[idx, "goalie"] == int(g)).to_numpy()
        if not named.any():
            continue
        if d.get("source") == "manual override":
            c.loc[idx, "p"] = named.astype(float)
            c.loc[idx, "status"] = np.where(named, "Override", "")
            continue
        if d.get("goalie_status"):
            lr, status = DFO_LR.get(d["goalie_status"], 2.0), d["goalie_status"]
        elif d.get("source") == "Daily Faceoff":
            lr, status = LINE_CHART_LR, "DFO lines"
        else:
            continue
        odds = c.loc[idx, "p"].to_numpy() * np.where(named, lr, 1.0)
        c.loc[idx, "p"] = odds / odds.sum()
        c.loc[idx, "status"] = np.where(named, status, "")
    return c


def project_team(state: dict, game_id: int, team: str, probs: pd.DataFrame, seed: int = 0) -> list[dict]:
    """Saves distribution for every plausible starter of one team-game (always the top two:
    starter vs backup), most likely first. Backup = the most likely other goalie."""
    model: SavesModel = state["model"]
    rows = state["slate"][(state["slate"].game_id == game_id) & (state["slate"].team == team)]
    if rows.empty:
        return []
    row = rows.iloc[0]
    c = probs[(probs.game_id == game_id) & (probs.team == team)].sort_values("p", ascending=False)
    tal = state["talent"].g_talent
    out = []
    for i, r in enumerate(c.itertuples(index=False)):
        if i >= 2 and r.p < MIN_SHOWN_P:
            continue
        others = c[c.goalie != r.goalie]
        backup = int(others.goalie.iloc[0]) if len(others) else None
        bt = float(tal.get(backup, -0.004)) if backup is not None else -0.004
        p = model.project(float(row.mu), float(row.matchup_xg), float(tal.get(r.goalie, -0.004)), bt, seed=seed)
        p.update({"goalie": int(r.goalie), "name": r.name, "p_start": float(r.p),
                  "p_model": float(r.p_model), "status": r.status, "backup": backup,
                  "talent": float(tal.get(r.goalie, np.nan))})
        out.append(p)
    return out

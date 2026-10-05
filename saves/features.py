"""Pregame features for every team-game, built strictly from earlier games.

Everything here is a *state* that a team carries into a game: exponentially weighted rates at
several half-lives (form vs. true level), carried across seasons but shrunk toward the league
at each new season because rosters turn over. A row's features never include that game.

Row = one team in one game, from the point of view of that team's goalie:
    d_*    the team's own defence (what its goalies face)      -- "own style and quality"
    o_*    the opponent's offence (what it throws at the net)  -- "opponent style and quality"
    q_*    team quality for both sides, which drives score effects
    *_lu_* lineup strength from player-level states (who dressed last game, see lineups())
    rest, venue and schedule context
Target columns: sa_faced (shots on goal at a goalie in net), ga_faced, game_secs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C

HALFLIVES = {"s": 8, "l": 20}     # games; short = current form, long = level (tuned walk-forward)
SEASON_KEEP = 0.70                # share of a team's deviation from league kept over a summer
PER60 = ["sf_faced", "cf", "ff", "xgf", "sf_pp", "sf_ev", "gf_faced",
         "sa_faced", "ca", "fa", "xga", "sa_pp", "sa_ev", "ga_faced"]
PER_GAME = ["pen_taken", "pen_drawn"]


def add_xg_to_games(team_games: pd.DataFrame, goalie_games: pd.DataFrame,
                    shots: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sum shot-level xG into team-game and goalie-game rows."""
    team_games, goalie_games, shots = (_plain(team_games), _plain(goalie_games), _plain(shots))
    at = shots[~shots.target_net_empty]
    tf = at.groupby(["game_id", "team"]).agg(xgf=("xg", "sum"), xgf_on=("xg_on", "sum")).reset_index()
    tg = team_games.drop(columns=[c for c in ("xgf", "xga", "xgf_on", "xga_on") if c in team_games])
    tg = tg.merge(tf, on=["game_id", "team"], how="left")
    tg = tg.merge(tf.rename(columns={"team": "opp", "xgf": "xga", "xgf_on": "xga_on"}),
                  on=["game_id", "opp"], how="left")
    for c in ("xgf", "xga", "xgf_on", "xga_on"):
        tg[c] = tg[c].fillna(0.0)

    on_net = at[at.event.isin(["goal", "sog"]) & at.goalie.notna()]
    gf = on_net.groupby(["game_id", "goalie"]).agg(
        pbp_sa=("event", "size"), pbp_ga=("event", lambda e: (e == "goal").sum()),
        xg_on_faced=("xg_on", "sum")).reset_index()
    gf["goalie"] = gf.goalie.astype("int64")
    gg = goalie_games.drop(columns=[c for c in ("pbp_sa", "pbp_ga", "xg_on_faced") if c in goalie_games])
    gg = gg.merge(gf, on=["game_id", "goalie"], how="left")
    for c in ("pbp_sa", "pbp_ga", "xg_on_faced"):
        gg[c] = gg[c].fillna(0)
    return tg, gg


def _plain(df: pd.DataFrame) -> pd.DataFrame:
    """Categories back to strings: seasons carry different category sets, and a merge on
    mismatched categoricals silently drops rows."""
    cats = [c for c in df.columns if isinstance(df[c].dtype, pd.CategoricalDtype)]
    return df.astype({c: str for c in cats}) if cats else df


def _ew_pregame(values: np.ndarray, groups: np.ndarray, seasons: np.ndarray,
                league: np.ndarray, halflife: float, keep: float) -> np.ndarray:
    """Pregame EW mean per group, rows already sorted by (group, date).

    `league` is the league mean for the row's season, the value a team is shrunk toward at a
    season boundary and the value used before a team has any history."""
    a = 1 - 0.5 ** (1 / halflife)
    out = np.empty(len(values))
    state = np.nan
    prev_g = prev_s = None
    for i in range(len(values)):
        g, s = groups[i], seasons[i]
        if g != prev_g:
            state = league[i]
        elif s != prev_s:
            state = league[i] + keep * (state - league[i])
        out[i] = state
        v = values[i]
        if not np.isnan(v):
            state = state + a * (v - state)
        prev_g, prev_s = g, s
    return out


def _league_means(tg: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """League mean of each column by season, using the PREVIOUS season for the current one's
    start so a season's own games never leak into its prior."""
    m = tg[tg.game_type == 2].groupby("season")[cols].mean()
    shifted = m.shift(1)
    shifted.iloc[0] = m.iloc[0]
    return shifted


def team_states(tg: pd.DataFrame) -> pd.DataFrame:
    """Pregame team states, one row per team-game (same index as tg)."""
    tg = tg.sort_values(["team", "date", "game_id"]).copy()
    scale = 3600.0 / tg.game_secs.clip(lower=3600)
    raw = pd.DataFrame(index=tg.index)
    for c in PER60:
        raw[c] = tg[c] * scale
    for c in PER_GAME:
        raw[c] = tg[c].astype(float)
    raw["gd"] = (tg.goals - tg.goals_against).astype(float)
    raw["xg_share"] = tg.xgf / (tg.xgf + tg.xga).replace(0, np.nan)
    raw["block_rate"] = 1 - tg.fa / tg.ca.replace(0, np.nan)      # share of opp attempts we block
    raw["opp_miss_rate"] = 1 - tg.sa / tg.fa.replace(0, np.nan)   # share of opp unblocked that miss
    raw["own_block_rate"] = 1 - tg.ff / tg.cf.replace(0, np.nan)  # share of our attempts blocked
    raw["own_miss_rate"] = 1 - tg.sf / tg.ff.replace(0, np.nan)
    raw["xg_per_sa"] = tg.xga_on / tg.sa_faced.replace(0, np.nan)  # quality of shots we allow on net
    raw["xg_per_sf"] = tg.xgf_on / tg.sf_faced.replace(0, np.nan)

    lg = _league_means(pd.concat([tg[["season", "game_type"]], raw], axis=1), list(raw.columns))
    league = lg.reindex(tg.season.values)
    out = pd.DataFrame(index=tg.index)
    g, s = tg.team.values, tg.season.values
    for c in raw.columns:
        for tag, hl in HALFLIVES.items():
            out[f"{c}_{tag}"] = _ew_pregame(raw[c].values, g, s, league[c].values, hl, SEASON_KEEP)
    out["gp_season"] = tg.groupby(["team", "season"]).cumcount()
    return out.sort_index()


def opponent_adjusted(tg: pd.DataFrame, st: pd.DataFrame) -> pd.DataFrame:
    """Shots faced / generated relative to what the opponent usually does, then EW'd.

    A defence that allowed 30 shots to a team that averages 36 did well; the raw rate
    cannot see that. Returns pregame EW residuals keyed like tg."""
    tg = tg.copy()
    scale = 3600.0 / tg.game_secs.clip(lower=3600)
    opp_state = st[["sf_faced_l", "sa_faced_l"]].copy()
    opp_state["game_id"], opp_state["team"] = tg.game_id, tg.team
    m = tg[["game_id", "opp"]].merge(opp_state.rename(columns={"team": "opp"}),
                                     on=["game_id", "opp"], how="left")
    m.index = tg.index
    raw = pd.DataFrame(index=tg.index)
    raw["adj_sa"] = tg.sa_faced * scale - m.sf_faced_l     # we allowed vs what they usually get
    raw["adj_sf"] = tg.sf_faced * scale - m.sa_faced_l     # we got vs what they usually allow
    order = tg.sort_values(["team", "date", "game_id"]).index
    out = pd.DataFrame(index=order)
    zero = np.zeros(len(order))
    for c in raw.columns:
        for tag, hl in HALFLIVES.items():
            out[f"{c}_{tag}"] = _ew_pregame(raw.loc[order, c].values, tg.loc[order, "team"].values,
                                            tg.loc[order, "season"].values, zero, hl, SEASON_KEEP)
    return out.sort_index()


def venue_effect(tg: pd.DataFrame, halflife: float = 60) -> pd.Series:
    """Rink shot-counting bias: total SOG in a rink's games minus what its two teams produce
    elsewhere. Scorers differ, and this is the part of a shot total that is the rink, not the
    teams. Pregame, one value per team-game (the rink of that game)."""
    tg = tg.sort_values(["date", "game_id"]).copy()
    tot = (tg.sf + tg.sa) * 3600.0 / tg.game_secs.clip(lower=3600)
    tg["tot"] = tot
    rink = np.where(tg.is_home, tg.team, tg.opp)
    tg["rink"] = rink
    # each team's road total, EW, pregame
    road = tg[~tg.is_home].sort_values(["team", "date"])
    road_ew = road.groupby("team").tot.transform(lambda v: v.shift(1).ewm(halflife=30).mean())
    tg["road_ew"] = road_ew.reindex(tg.index)
    tg["road_ew"] = tg.groupby("team").road_ew.transform(lambda v: v.ffill())
    games = tg[tg.is_home][["game_id", "date", "team", "opp", "tot", "road_ew"]]
    opp_road = tg[~tg.is_home][["game_id", "road_ew"]].rename(columns={"road_ew": "opp_road_ew"})
    games = games.merge(opp_road, on="game_id")
    games["excess"] = games.tot - (games.road_ew + games.opp_road_ew) / 2
    games = games.sort_values(["team", "date"])
    games["venue"] = games.groupby("team").excess.transform(
        lambda v: v.shift(1).ewm(halflife=halflife, min_periods=5).mean())
    v = tg[["game_id"]].merge(games[["game_id", "venue"]], on="game_id", how="left")
    v.index = tg.index
    return v.venue.fillna(0.0).sort_index()


def rest(tg: pd.DataFrame) -> pd.DataFrame:
    tg = tg.sort_values(["team", "date", "game_id"])
    d = pd.to_datetime(tg.date)
    days = d.groupby(tg.team).diff().dt.days
    out = pd.DataFrame(index=tg.index)
    out["rest_days"] = days.clip(upper=5).fillna(5)
    out["b2b"] = (days == 1).astype(int)
    # 3 games in 4 nights
    d_prev2 = d.groupby(tg.team).shift(2)
    out["three_in_four"] = ((d - d_prev2).dt.days <= 3).astype(int)
    return out.sort_index()


def league_trend(tg: pd.DataFrame, col: str = "sa_faced", halflife: float = 400.0) -> pd.Series:
    """League-wide rate of `col` per 60 over recent team-games, as of the morning of each
    date (a day's own games excluded). Shot rates and save rates both drift year to year."""
    v = tg[col] * 3600.0 / tg.game_secs.clip(lower=3600)
    daily = pd.DataFrame({"date": tg.date, "v": v}).dropna().groupby("date").v.agg(["sum", "count"])
    dates = sorted(set(tg.date))
    state, out = np.nan, {}
    for d in dates:
        out[d] = state
        if d in daily.index:
            s_, n_ = daily.loc[d, "sum"], daily.loc[d, "count"]
            if np.isnan(state):
                state = s_ / n_
            else:
                w = 1 - 0.5 ** (n_ / halflife)
                state = state + w * (s_ / n_ - state)
    first = next(x for x in out.values() if not np.isnan(x)) if any(
        not np.isnan(x) for x in out.values()) else np.nan
    return tg.date.map(out).fillna(first)


LINEUP_HALFLIFE = 15              # games played, per player
LINEUP_PRIOR = {"F": (12.0, 1.0), "D": (17.0, 0.7)}   # (toi minutes, sog) for a player's first game
LINEUP_COLS = ["lu_sog", "lu_d_toi", "lu_d_vet_toi", "lu_rookies"]


def lineups(skaters: pd.DataFrame, tg: pd.DataFrame) -> pd.DataFrame:
    """Lineup strength of each played team-game, from the skaters who dressed.

    Each player carries a pregame EW of his ice time and shots on goal, so a lineup's expected
    shots and its experienced-defence ice time move with injuries, scratches, call-ups and
    trades before the team's own rates can. The *_dev columns are tonight's lineup against the
    team's recent lineups. Pregame use takes the team's PREVIOUS game (lineup_features), which is
    what is known before puck drop; the walk-forward test showed about half the gain of knowing
    the actual lineup survives that."""
    sk = _plain(skaters)
    sk = sk[sk.toi > 0].merge(_plain(tg)[["game_id", "team", "date"]], on=["game_id", "team"])
    sk = sk.sort_values(["player", "date", "game_id"]).reset_index(drop=True)
    a = 1 - 0.5 ** (1 / LINEUP_HALFLIFE)
    pl, pos = sk.player.values, sk.pos.values
    toi, sog = sk.toi.values / 60.0, sk.sog.values.astype(float)
    toi_ew, sog_ew, n_prev = np.empty(len(sk)), np.empty(len(sk)), np.empty(len(sk))
    cur, t_s, s_s, n = None, 0.0, 0.0, 0
    for i in range(len(sk)):
        if pl[i] != cur:
            cur, n = pl[i], 0
            t_s, s_s = LINEUP_PRIOR.get(pos[i], LINEUP_PRIOR["F"])
        toi_ew[i], sog_ew[i], n_prev[i] = t_s, s_s, n
        t_s += a * (toi[i] - t_s); s_s += a * (sog[i] - s_s); n += 1
    sk = sk.assign(toi_ew=toi_ew, sog_ew=sog_ew, n_prev=n_prev)
    is_d = sk.pos == "D"
    lu = pd.DataFrame({
        "lu_sog": sk.groupby(["game_id", "team"]).sog_ew.sum(),
        "lu_d_toi": sk[is_d].groupby(["game_id", "team"]).toi_ew.sum(),
        "lu_d_vet_toi": sk[is_d & (sk.n_prev >= 100)].groupby(["game_id", "team"]).toi_ew.sum(),
        "lu_rookies": sk[sk.n_prev < 10].groupby(["game_id", "team"]).size(),
    }).fillna({"lu_d_vet_toi": 0.0, "lu_rookies": 0.0}).reset_index()
    lu = lu.merge(sk[["game_id", "team", "date"]].drop_duplicates(), on=["game_id", "team"])
    lu = lu.sort_values(["team", "date", "game_id"]).reset_index(drop=True)
    for c in LINEUP_COLS:
        base = lu.groupby("team")[c].transform(lambda v: v.shift(1).ewm(halflife=10, min_periods=1).mean())
        lu[f"{c}_dev"] = (lu[c] - base).fillna(0.0)
    return lu


def lineup_features(rows: pd.DataFrame, lu: pd.DataFrame) -> pd.DataFrame:
    """For each (game_id, team, date) row: the lineup of that team's last game before the date."""
    r = rows[["game_id", "team", "date"]].reset_index()
    r["_d"] = pd.to_datetime(r.date)
    l = lu.assign(_d=pd.to_datetime(lu.date)).drop(columns=["game_id", "date"]).sort_values("_d")
    out = pd.merge_asof(r.sort_values("_d"), l, on="_d", by="team", allow_exact_matches=False)
    out = out.set_index("index").sort_index()
    cols = [c for c in lu.columns if c.startswith("lu_")]
    return out[cols].fillna(lu[cols].median())


FEATURE_PREFIX = ("d_", "o_", "q_", "ctx_")


def build_matrix(tg: pd.DataFrame, skaters: pd.DataFrame | None = None) -> pd.DataFrame:
    """Join each team-game to its own defence state and the opponent's offence state."""
    tg = _plain(tg).reset_index(drop=True)
    st = team_states(tg)
    adj = opponent_adjusted(tg, st)
    st = pd.concat([st, adj], axis=1)
    rs = rest(tg)
    venue = venue_effect(tg)

    own = st.add_prefix("own_")
    own[["game_id", "team"]] = tg[["game_id", "team"]]
    opp = st.add_prefix("opp_")
    opp[["game_id", "opp"]] = tg[["game_id", "team"]]
    m = tg.merge(own, on=["game_id", "team"]).merge(opp, on=["game_id", "opp"])
    rs_own = rs.add_prefix("own_"); rs_own[["game_id", "team"]] = tg[["game_id", "team"]]
    rs_opp = rs.add_prefix("opp_"); rs_opp[["game_id", "opp"]] = tg[["game_id", "team"]]
    m = m.merge(rs_own, on=["game_id", "team"]).merge(rs_opp, on=["game_id", "opp"])
    vv = pd.DataFrame({"game_id": tg.game_id, "team": tg.team, "venue": venue.values})
    m = m.merge(vv, on=["game_id", "team"])

    X = pd.DataFrame(index=m.index)
    for tag in HALFLIVES:
        # what our goalie faces: our defence ...
        for c in ("sa_faced", "ca", "fa", "xga", "sa_pp", "sa_ev", "pen_taken", "block_rate",
                  "opp_miss_rate", "xg_per_sa", "adj_sa"):
            X[f"d_{c}_{tag}"] = m[f"own_{c}_{tag}"]
        # ... against their offence
        for c in ("sf_faced", "cf", "ff", "xgf", "sf_pp", "sf_ev", "pen_drawn",
                  "own_block_rate", "own_miss_rate", "xg_per_sf", "adj_sf"):
            X[f"o_{c}_{tag}"] = m[f"opp_{c}_{tag}"]
        # quality and possession on both sides -> score effects
        for c in ("gd", "xg_share"):
            X[f"q_own_{c}_{tag}"] = m[f"own_{c}_{tag}"]
            X[f"q_opp_{c}_{tag}"] = m[f"opp_{c}_{tag}"]
        # the other half of the matchup: our offence keeps the puck away
        X[f"q_own_cf_{tag}"] = m[f"own_cf_{tag}"]
        X[f"q_opp_ca_{tag}"] = m[f"opp_ca_{tag}"]
    X["ctx_home"] = m.is_home.astype(int)
    X["ctx_venue"] = m.venue
    X["ctx_own_rest"] = m.own_rest_days
    X["ctx_opp_rest"] = m.opp_rest_days
    X["ctx_own_b2b"] = m.own_b2b
    X["ctx_opp_b2b"] = m.opp_b2b
    X["ctx_own_3in4"] = m.own_three_in_four
    X["ctx_opp_3in4"] = m.opp_three_in_four
    X["ctx_gp"] = m.own_gp_season
    X["ctx_playoff"] = (m.game_type == 3).astype(int)
    if skaters is not None:
        lu = lineups(skaters, tg[tg.game_secs.notna()])
        own_lu = lineup_features(m[["game_id", "team", "date"]], lu)
        opp_lu = lineup_features(m[["game_id", "opp", "date"]].rename(columns={"opp": "team"}), lu)
        for c in own_lu.columns:
            X[f"d_{c}"] = own_lu[c].values
            X[f"o_{c}"] = opp_lu[c].values
    X["trend_league_sa"] = league_trend(m).values  # diagnostic only: not a model input
    keep = ["game_id", "season", "game_type", "date", "team", "opp", "is_home",
            "sa_faced", "ga_faced", "sa", "game_secs", "last_period_type"]
    return pd.concat([m[keep], X], axis=1)

"""Pregame features for every skater-game, from history only.

One code path serves the backtest and live projections: upcoming games are appended to the
history as pending rows (no outcomes), and every running sum is taken over EARLIER rows, so a
pending row sees exactly what a finished game saw the morning it was played.

Each row carries the deployment the projection should assume ("given"): line, PP unit, PK
unit and linemates. In the backtest that is the player's previous game (what a line chart
shows the morning after); live it is Daily Faceoff or a manual override.

Half-lives (in games) and shrinkage were chosen on 2022-23 and are documented in README:
    ice-time shares   4 and 16 games      role changes show up fast
    shot rates        80 games            a stable skill
    on-ice goals      80 games
    assist share      160 games
    shooting %        320 games, 200 shots of league-average prior
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config as C
import ew

ST = ("ev", "pp", "sh")
GAP = 20                # a summer counts as this many games (rates)
SHARE_GAP = 10          # ...and this many for ice-time shares
HL_SHARE = (4, 16)
HL_RATE = 80
HL_GF = 80
HL_ASSIST_SHARE = 160
HL_SHOOT = 320
HL_TEAM = 20
HL_SLOT = 20
HL_GOALIE = 60
HL_LEAGUE = 400         # team-games (~1/6 of a season) for the league scoring environment

# shrinkage: minutes (or shots / goals) of league-average prior added to each player's history
K_SOG = {"ev": 120, "pp": 40, "sh": 40}
K_SHOOT = 200           # shots
K_GF = {"ev": 120, "pp": 30}
K_ASHARE = 20           # on-ice goals
K_MATE = 60             # minutes, for linemate quality
K_GOALIE = 600          # shots faced


def load_history() -> tuple[pd.DataFrame, pd.DataFrame]:
    pg = pd.read_parquet(C.LAKE / "player_games.parquet")
    tg = pd.read_parquet(C.LAKE / "team_games.parquet")
    return pg, tg


def _prep_players(pg: pd.DataFrame, tg: pd.DataFrame) -> pd.DataFrame:
    pg = pg.copy()
    pg["team"], pg["opp"], pg["pos"] = pg.team.astype(str), pg.opp.astype(str), pg.pos.astype(str)
    pg["F"] = pg.pos != "D"
    pg["grp"] = np.where(pg.F, "F", "D")
    pg = pg.merge(tg[["game_id", "team", "pp_secs", "sh_secs"]], on=["game_id", "team"], how="left")
    pg["pool_ev"] = pg.groupby(["game_id", "team", "grp"]).toi_ev.transform("sum")
    pg["share_ev"] = pg.toi_ev / pg.pool_ev
    for s in ST:
        pg[f"t_{s}"] = pg[f"toi_{s}"] / 60
    shifts = pg.toi_5v5.notna() | pg.pending
    for s in ("ev", "pp"):
        pg[f"ton_{s}"] = pg[f"t_{s}"].where(shifts)
        pg[f"a_on_{s}"] = pg[f"a_{s}"].where(shifts)
    pg["ppp"] = pg.g_pp + pg.a_pp                # power-play points (goals + assists on the PP)
    pg["ppp_on"] = pg.ppp.where(shifts)
    pg["sog_all"] = sum(pg[f"sog_{s}"] for s in ST)
    pg["one"] = 1.0
    return pg


def _add_age(pg: pd.DataFrame) -> pd.DataFrame:
    """Age on game day, from data/lake/bio.parquet (NHL player pages) or a pending row's own
    'birth' column (today's roster). Unknown: 27."""
    bio_file = C.LAKE / "bio.parquet"
    birth = pd.Series(pd.NaT, index=pg.index, dtype="datetime64[ns]")
    if "birth" in pg:
        birth = pd.to_datetime(pg.birth, errors="coerce")
    if bio_file.exists():
        bio = pd.read_parquet(bio_file, columns=["player", "birth"]).dropna().drop_duplicates("player")
        birth = birth.fillna(pg.player.map(pd.to_datetime(bio.set_index("player").birth)))
    pg["age"] = ((pg.date - birth).dt.days / 365.25).fillna(27.0)
    return pg


def _given_deployment(pg: pd.DataFrame) -> pd.DataFrame:
    """Fill given_* from the player's previous game wherever it was not supplied."""
    prev = pg.groupby("player")[["line", "pp_unit", "pk_unit", "lm_f1", "lm_f2", "lm_d1"]].shift(1)
    for c, src in (("given_line", "line"), ("given_pp", "pp_unit"), ("given_pk", "pk_unit"),
                   ("given_m1", "lm_f1"), ("given_m2", "lm_f2"), ("given_d1", "lm_d1")):
        if c not in pg:
            pg[c] = np.nan
        pg[c] = pg[c].astype(float).fillna(prev[src].astype(float))
    pg["given_line"] = pg.given_line.fillna(pd.Series(np.where(pg.F, 4, 3), index=pg.index))  # unknown: bottom
    pg["given_pp"] = pg.given_pp.fillna(0)
    pg["given_pk"] = pg.given_pk.fillna(0)
    return pg


GIVEN = {"ev": "given_line", "pp": "given_pp", "sh": "given_pk"}
ACTUAL = {"ev": "line", "pp": "pp_unit", "sh": "pk_unit"}
SLOT_SPECS = {"ev": ("line", "share_ev", None), "pp": ("pp_unit", "toi_pp", "pp_secs"),
              "sh": ("pk_unit", "toi_sh", "sh_secs")}


def _slot_history(pg: pd.DataFrame, st: str) -> tuple[pd.DataFrame, pd.Series]:
    """Each team's EW share for each slot after every game it used that slot (key is
    team + F/D + slot number, e.g. 'TORF1'), and the league average share per slot."""
    slot, num, den = SLOT_SPECS[st]
    hist = pg[~pg.pending & pg[slot].notna()]
    key = hist.team + hist.grp + hist[slot].astype(int).astype(str)
    v = hist[num] if den is None else hist[num] / hist[den].where(hist[den] > 0)
    g = (pd.DataFrame({"k": key, "game_id": hist.game_id, "date": hist.date, "season": hist.season, "v": v})
         .dropna().groupby(["k", "game_id"], sort=False)
         .agg(v=("v", "mean"), date=("date", "first"), season=("season", "first")).reset_index()
         .sort_values(["date", "game_id"]).reset_index(drop=True))
    g["one"] = 1.0
    lg = g.assign(slot=g.k.str[3:]).groupby("slot").v.mean()
    e = ew.pregame(g, "k", ["v", "one"], HL_SLOT, SHARE_GAP)
    decay = 0.5 ** (1 / HL_SLOT)
    after = (e.v + g.v) * decay, (e.one + 1) * decay
    out = pd.DataFrame({"k": g.k, "date": g.date,
                        "after": (after[0] + 2 * g.k.str[3:].map(lg)) / (after[1] + 2)})
    return out.sort_values("date"), lg


def slot_lookup(pg: pd.DataFrame) -> dict[str, tuple[pd.Series, pd.Series]]:
    """Latest share per slot key, plus the league fallback, for applying line overrides."""
    out = {}
    for st in SLOT_SPECS:
        after, lg = _slot_history(pg, st)
        out[st] = (after.groupby("k").after.last(), lg)
    return out


def slot_value(lookup: dict, st: str, team: pd.Series, grp: pd.Series, slot: pd.Series) -> pd.Series:
    latest, lg = lookup[st]
    s = slot.fillna(-1).astype(int).astype(str)
    return (team + grp + s).map(latest).fillna((grp + s).map(lg)).fillna(0.0)


def _slot_shares(pg: pd.DataFrame, slots: dict[str, str] = GIVEN, prefix: str = "slot") -> pd.DataFrame:
    """How much a team gives the slot each player is given: its EW average share for line k /
    PP unit / PK unit as of the game before. Falls back to the league average for the slot."""
    for st in SLOT_SPECS:
        after, lg = _slot_history(pg, st)
        given = pg[slots[st]].fillna(-1).astype(int).astype(str)
        gkey = pg.team + pg.grp + given
        vals = pd.merge_asof(
            pd.DataFrame({"k": gkey, "date": pg.date, "i": np.arange(len(pg))}).sort_values("date"),
            after, on="date", by="k", direction="backward", allow_exact_matches=False)
        out = pd.Series(np.nan, index=pg.index)
        out.iloc[vals.i.to_numpy()] = vals.after.to_numpy()
        pg[f"{prefix}_{st}"] = out.fillna((pg.grp + given).map(lg)).fillna(0.0)
    return pg


def league_env(pg: pd.DataFrame, num: str, den: str) -> pd.Series:
    """League rate num/den for each row's position group (F / D), from games BEFORE the row's
    date, exponentially weighted with a half-life of HL_LEAGUE team-games."""
    hist = pg[~pg.pending & pg[num].notna() & pg[den].notna()]
    daily = hist.groupby(["grp", "date"])[[num, den]].sum().reset_index()
    games = hist.groupby("date").game_id.nunique()
    out = []
    for grp, g in daily.groupby("grp"):
        g = g.sort_values("date")
        decay = 0.5 ** (2 * games.reindex(g.date).to_numpy() / HL_LEAGUE)
        n = d = 0.0
        after_n, after_d = [], []
        for x_n, x_d, dec in zip(g[num].to_numpy(), g[den].to_numpy(), decay):
            n, d = n * dec + x_n, d * dec + x_d
            after_n.append(n); after_d.append(d)
        out.append(pd.DataFrame({"grp": grp, "date": g.date.to_numpy(),
                                 "env": np.divide(after_n, after_d, out=np.full(len(after_d), np.nan),
                                                  where=np.array(after_d) > 0)}))
    after = pd.concat(out).sort_values("date")
    q = pd.DataFrame({"grp": pg.grp, "date": pg.date, "i": np.arange(len(pg))}).sort_values("date")
    m = pd.merge_asof(q, after, on="date", by="grp", direction="backward", allow_exact_matches=False)
    env = pd.Series(np.nan, index=pg.index)
    env.iloc[m.i.to_numpy()] = m.env.to_numpy()
    overall = hist.groupby("grp")[num].sum() / hist.groupby("grp")[den].sum()
    return env.fillna(pg.grp.map(overall))


def _player_states(pg: pd.DataFrame) -> pd.DataFrame:
    out = {}
    for hl in HL_SHARE:
        e = ew.pregame(pg, "player", ["one", "toi_ev", "pool_ev", "toi_pp", "pp_secs", "toi_sh", "sh_secs"], hl, SHARE_GAP)
        out[f"gp_{hl}"] = e.one
        out[f"ev_share_{hl}"] = e.toi_ev / e.pool_ev.where(e.pool_ev > 0)
        out[f"pp_share_{hl}"] = e.toi_pp / e.pp_secs.where(e.pp_secs > 0)
        out[f"sh_share_{hl}"] = e.toi_sh / e.sh_secs.where(e.sh_secs > 0)
    # Rates are relative to the league: each game's events are compared with what a league-
    # average forward (or defenceman) would have produced in the same minutes at that point in
    # time, so a player's history is not dragged by league-wide drift (shots per minute fell
    # ~12% from 2021 to 2025 while shooting % rose). Projection = relative rate x league now.
    #
    # Thin histories are pulled toward a ROLE baseline, not the position average: what a player
    # with his ice-time share, PP / PK share, position and NHL experience typically produces
    # (a Poisson GLM fitted on ROLE_FIT_SEASONS). Pulling a call-up toward the average skater
    # inflated depth players by ~17%; pulling a star toward it cost stars ~3-4%.
    role = _role_matrix(pg, out)

    def relative(num: str, den: str, hl: float, k: float) -> pd.Series:
        env = league_env(pg, num, den)
        x = pg[den] * env
        r0 = _role_prior(pg, role, pg[num], x)
        tmp = pd.DataFrame({"player": pg.player, "season": pg.season, "n": pg[num], "x": x}, index=pg.index)
        e = ew.pregame(tmp, "player", ["n", "x"], hl, GAP)
        kk = k * env  # k units of exposure at the league rate
        return (e.n + kk * r0) / (e.x + kk) * env

    for s in ST:
        out[f"sog_pm_{s}"] = relative(f"sog_{s}", f"t_{s}", HL_RATE, K_SOG[s])
    for s in ("ev", "pp"):
        out[f"gf_pm_{s}"] = relative(f"ogf_{s}", f"ton_{s}", HL_GF, K_GF[s])
        out[f"ashare_{s}"] = relative(f"a_on_{s}", f"ogf_{s}", HL_ASSIST_SHARE, K_ASHARE)
    # PP 'IPP': his share of his team's on-ice PP goals that he scores or assists on
    out["ipp_pp"] = relative("ppp_on", "ogf_pp", HL_ASSIST_SHARE, K_ASHARE)
    out["a_pm_sh"] = league_env(pg, "a_sh", "t_sh")
    out["sh_pct"] = relative("goals", "sog_all", HL_SHOOT, K_SHOOT)
    e = ew.pregame(pg, "player", ["points", "t_ev", "t_pp", "t_sh"], HL_RATE, GAP)
    hist = pg[~pg.pending]
    tt = e.t_ev + e.t_pp + e.t_sh
    out["q_pts"] = (e.points + K_MATE * hist.points.sum() / (hist.toi.sum() / 60)) / (tt + K_MATE)
    # last completed season's points per minute (regular season), for the 'recent season' term
    reg = hist[hist.game_type == 2]
    ls = (reg.groupby(["player", "season"]).agg(p=("points", "sum"), t=("toi", "sum")).reset_index())
    ls["season"] += 1
    m = pg[["player", "season"]].merge(ls, on=["player", "season"], how="left")
    out["ls_pts_pm"] = ((m.p + K_MATE * hist.points.sum() / (hist.toi.sum() / 60)) / (m.t / 60 + K_MATE)).to_numpy()
    return pd.concat([pg, pd.DataFrame(out, index=pg.index)], axis=1)


ROLE_FIT_SEASONS = (2021, 2022)


def _role_matrix(pg: pd.DataFrame, out: dict) -> pd.DataFrame:
    """Pregame role of each player-game: EV share relative to an average F / D (log), PP and
    PK share, centre, and NHL experience. Without history the slot he is given stands in."""
    n_grp = np.where(pg.F, 12.0, 6.0)
    ev = pd.Series(out["ev_share_16"], index=pg.index).fillna(pg.slot_ev)
    pp = pd.Series(out["pp_share_16"], index=pg.index).fillna(pg.slot_pp)
    sh = pd.Series(out["sh_share_16"], index=pg.index).fillna(pg.slot_sh)
    career = pg.groupby("player").cumcount()
    return pd.DataFrame({"const": 1.0, "ev": np.log((ev * n_grp).clip(0.2, 3)), "pp": pp.clip(0, 1),
                         "sh": sh.clip(0, 1), "is_c": (pg.pos == "C").astype(float),
                         "exp": np.log1p(np.minimum(career, 400)) / np.log(401)}, index=pg.index)


def _role_prior(pg: pd.DataFrame, role: pd.DataFrame, n: pd.Series, x: pd.Series) -> pd.Series:
    """Expected relative rate (1.0 = league average for F or D) given the role, per row."""
    import statsmodels.api as sm
    r0 = pd.Series(1.0, index=pg.index)
    fit_rows = ~pg.pending & pg.season.isin(ROLE_FIT_SEASONS) & n.notna() & (x > 0)
    for f in (True, False):
        cols = ["const", "ev", "pp", "sh", "exp"] + (["is_c"] if f else [])
        m = fit_rows & (pg.F == f)
        glm = sm.GLM(n[m], role.loc[m, cols], family=sm.families.Poisson(), offset=np.log(x[m])).fit()
        g = pg.F == f
        r0[g] = np.exp(role.loc[g, cols] @ glm.params)
    return r0.clip(0.2, 5)


def _mate_quality(pg: pd.DataFrame, fwd: tuple[str, str], partner: str) -> pd.Series:
    """Mean pregame points/60 of a player's linemates in that game (forwards: two linemates,
    defence: his partner)."""
    q = pg.set_index(["game_id", "player"]).q_pts
    look = lambda c: q.reindex(pd.MultiIndex.from_arrays([pg.game_id, pg[c].fillna(-1).astype(np.int64)])).to_numpy()
    fw = np.column_stack([look(fwd[0]), look(fwd[1])])
    n = (~np.isnan(fw)).sum(1)
    fw_mean = np.where(n > 0, np.nansum(fw, 1) / np.maximum(n, 1), np.nan)
    return pd.Series(np.where(pg.F, fw_mean, look(partner)), index=pg.index)


def _usual_mates(pg: pd.DataFrame) -> pd.Series:
    """The quality of the linemates a player has USUALLY had (EW over his past games, from the
    shift chart), which his own on-ice history already reflects."""
    mq = _mate_quality(pg, ("lm_f1", "lm_f2"), "lm_d1").where(~pg.pending)
    tmp = pd.DataFrame({"player": pg.player, "season": pg.season, "mq": mq, "one": mq.notna().astype(float)})
    e = ew.pregame(tmp, "player", ["mq", "one"], HL_SHARE[1], SHARE_GAP)
    return e.mq / e.one.where(e.one > 0)


def _mates(pg: pd.DataFrame) -> pd.Series:
    """Linemates tonight vs his usual linemates (log ratio of their points per 60): positive
    when he is promoted to better linemates than he has been playing with, 0 when unchanged.
    Comparing linemates with the player himself would mark every star down, since a star is
    nearly always better than his linemates and his own record already includes them."""
    now = _mate_quality(pg, ("given_m1", "given_m2"), "given_d1")
    return pd.Series(np.log(np.clip(now / pg.usual_mq, 0.5, 2)), index=pg.index).fillna(0.0)


def _team_states(tg: pd.DataFrame, pg: pd.DataFrame, goalies: pd.DataFrame) -> pd.DataFrame:
    tg = tg.copy()
    tg["team"], tg["opp"] = tg.team.astype(str), tg.opp.astype(str)
    evs = pg.groupby(["game_id", "team"]).toi_ev.sum().div(5).rename("ev_secs")
    tg = tg.merge(evs, left_on=["game_id", "team"], right_index=True, how="left")
    for grp in ("F", "D"):
        p = pg[pg.grp == grp].groupby(["game_id", "team"]).toi_ev.sum().rename(f"pool_{grp}")
        tg = tg.merge(p, left_on=["game_id", "team"], right_index=True, how="left")
    tg.loc[tg.pending, ["ev_secs", "pool_F", "pool_D"]] = np.nan
    tg["mins"] = (tg.ev_secs + tg.pp_secs + tg.sh_secs) / 60
    tg["ga_all"] = tg.goals_against
    tg["one"] = 1.0
    tg["tot_sog"] = tg.sog + tg.sog_against
    cols = ["one", "pp_secs", "sh_secs", "pool_F", "pool_D"]
    e = ew.pregame(tg, "team", cols, HL_TEAM, 10)
    st = pd.DataFrame(index=tg.index)
    st["ew_pp"] = e.pp_secs / e.one          # PP seconds a team usually gets
    st["ew_sh"] = e.sh_secs / e.one          # ...and spends shorthanded
    st["ew_pool_F"] = e.pool_F / e.one
    st["ew_pool_D"] = e.pool_D / e.one
    tg["grp"] = "T"

    def suppression(num: str, den: str, hl: float, k: float) -> pd.Series:
        """What a team allows relative to the league at the time (1.0 = average)."""
        env = league_env(tg, num, den)
        tmp = pd.DataFrame({"team": tg.team, "season": tg.season, "n": tg[num], "x": tg[den] * env})
        e = ew.pregame(tmp, "team", ["n", "x"], hl, 10)
        return (e.n + k * env) / (e.x + k * env)

    st["supp_ev"] = suppression("sa_ev", "ev_secs", HL_TEAM, 9000)   # EV shots allowed
    st["supp_pk"] = suppression("sa_pp", "sh_secs", HL_TEAM, 2000)   # shots allowed on the PK
    st["supp_g"] = suppression("ga_all", "mins", 40, 180)            # goals allowed
    st["supp_pkg"] = suppression("ga_sh", "sh_secs", 40, 3000)       # PP goals allowed on the PK
    tg = pd.concat([tg, st], axis=1)
    tg["rest"] = tg.groupby("team").date.diff().dt.days.clip(upper=4).fillna(4)
    # rink: total shots in a team's home games vs its road games (scorer bias + home effects)
    for side, flag in (("h", True), ("a", False)):
        part = tg[tg.is_home == flag]
        e = ew.pregame(part, "team", ["tot_sog", "one"], 160, 0)
        tg.loc[part.index, f"rink_{side}"] = (e.tot_sog + 20 * 60) / (e.one + 20)
    tg["rink_a"] = tg.groupby("team").rink_a.ffill()
    tg["rink_h"] = tg.groupby("team").rink_h.ffill()
    home_team = tg[tg.is_home].set_index("game_id")
    tg["rink"] = tg.game_id.map(home_team.rink_h / home_team.rink_a).fillna(1.0)
    tg = tg.merge(goalies, on=["game_id", "team"], how="left")  # the goalie this team starts
    tg["gsax"] = tg.gsax.fillna(0.0)
    return tg


def goalie_ratings(shots: pd.DataFrame, pending_goalies: pd.DataFrame | None = None) -> pd.DataFrame:
    """Pregame GSAx per shot faced (%, shrunk) for each team's starting goalie in each game.
    The starter is the goalie who faced the most shots; pending rows name their own."""
    s = shots[shots.event.isin(["goal", "sog"]) & ~shots.target_net_empty & shots.goalie.notna()]
    gg = (s.assign(ga=(s.event == "goal").astype(int))
          .groupby(["game_id", "opp", "goalie"], observed=True)
          .agg(date=("date", "first"), season=("season", "first"), sa=("ga", "size"), ga=("ga", "sum"),
               xga=("xg_on", "sum")).reset_index().rename(columns={"opp": "team"}))
    gg["team"] = gg.team.astype(str)
    gg["date"] = pd.to_datetime(gg.date.astype(str))
    gg["pending"] = False
    if pending_goalies is not None and len(pending_goalies):
        gg = pd.concat([gg, pending_goalies.assign(pending=True, sa=0, ga=0, xga=0.0)], ignore_index=True)
    gg = gg.sort_values(["date", "game_id"]).reset_index(drop=True)
    e = ew.pregame(gg, "goalie", ["sa", "ga", "xga"], HL_GOALIE, 10)
    gg["gsax"] = 100 * (e.xga - e.ga) / (e.sa + K_GOALIE)
    starters = gg.sort_values("sa").groupby(["game_id", "team"]).tail(1)
    return starters[["game_id", "team", "goalie", "gsax"]]


def goalie_latest(shots: pd.DataFrame) -> pd.Series:
    """Each goalie's GSAx per shot (%, shrunk) going into his next game, for overrides."""
    s = shots[shots.event.isin(["goal", "sog"]) & ~shots.target_net_empty & shots.goalie.notna()]
    gg = (s.assign(ga=(s.event == "goal").astype(int))
          .groupby(["game_id", "goalie"], observed=True)
          .agg(date=("date", "first"), season=("season", "first"), sa=("ga", "size"), ga=("ga", "sum"),
               xga=("xg_on", "sum")).reset_index())
    gg["date"] = pd.to_datetime(gg.date.astype(str))
    gg = gg.sort_values(["date", "game_id"]).reset_index(drop=True)
    e = ew.latest(gg, "goalie", ["sa", "ga", "xga"], HL_GOALIE, 10)
    return 100 * (e.xga - e.ga) / (e.sa + K_GOALIE)


def build(pg: pd.DataFrame, tg: pd.DataFrame, shots: pd.DataFrame,
          pending_goalies: pd.DataFrame | None = None) -> pd.DataFrame:
    """Every row of pg (history + pending) with its pregame features and opponent context."""
    pg = pg.copy()
    tg = tg.copy()
    if "pending" not in pg:
        pg["pending"] = False
    if "pending" not in tg:
        tg["pending"] = False
    pg["pending"] = pg.pending.fillna(False).astype(bool)
    tg["pending"] = tg.pending.fillna(False).astype(bool)
    pg = pg.sort_values(["date", "game_id", "team", "player"]).reset_index(drop=True)
    tg = tg.sort_values(["date", "game_id", "team"]).reset_index(drop=True)
    pg = _prep_players(pg, tg)
    pg = _add_age(pg)
    pg = _given_deployment(pg)
    pg = _slot_shares(pg)
    pg = _slot_shares(pg, ACTUAL, "aslot")  # tonight's actual slots: training only
    pg = _player_states(pg)
    pg["usual_mq"] = _usual_mates(pg)
    pg["mate"] = _mates(pg)
    goalies = goalie_ratings(shots, pending_goalies)
    tg = _team_states(tg, pg, goalies)
    own = tg[["game_id", "team", "ew_pp", "ew_sh", "ew_pool_F", "ew_pool_D", "rest", "rink"]]
    opp = tg[["game_id", "team", "ew_pp", "ew_sh", "supp_ev", "supp_pk", "supp_g", "supp_pkg", "rest", "gsax", "goalie"]]
    opp = opp.rename(columns={c: f"opp_{c}" for c in opp.columns if c not in ("game_id",)})
    pg = pg.merge(own, on=["game_id", "team"], how="left")
    pg = pg.merge(opp.rename(columns={"opp_team": "opp"}), on=["game_id", "opp"], how="left")
    # a team's first game (an expansion team, say) has no history: league-typical values
    for c in ("ew_pp", "ew_sh", "ew_pool_F", "ew_pool_D", "opp_ew_pp", "opp_ew_sh"):
        pg[c] = pg[c].fillna(pg[c].median())
    for c in ("opp_supp_ev", "opp_supp_pk", "opp_supp_g", "opp_supp_pkg", "rink"):
        pg[c] = pg[c].fillna(1.0)
    pg["opp_gsax"] = pg.opp_gsax.fillna(0.0)
    pg[["rest", "opp_rest"]] = pg[["rest", "opp_rest"]].fillna(4)
    return pg

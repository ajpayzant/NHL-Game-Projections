"""NHL skater game projections: Slate · Game · Lines & goalies · Player · Trends · Edges · Results · How it works.

    run_app.bat            # http://localhost:8670
"""
from __future__ import annotations

import json
import os

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import cloud
import config as C
import edges
import linecheck
import live
import markets
import model
import odds
import store
import tracking
import trends
from saves import live as saves_live

st.set_page_config(page_title="NHL Skater Projections", layout="wide")
PROJ, ACT, MUTED = "#2a78d6", "#eb6834", "#8a8984"   # palette slots 1-2 + muted ink


@st.cache_data(ttl=600, show_spinner="Loading the latest projections...")
def _sync() -> str | None:
    """On Streamlit Cloud: pull the refresh workflow's newest files (no-op locally)."""
    return cloud.sync()


_sync()
HIST_COLS = ["game_id", "date", "season", "game_type", "team", "opp", "is_home", "player", "name", "pos",
             "toi", "toi_pp", "toi_sh", "sog", "goals", "assists", "points", "g_pp", "a_pp", "line", "pp_unit", "pk_unit"]


# ---------------------------------------------------------------- data

@st.cache_resource
def _state(mtime: float) -> dict | None:
    return live.load_state()


def state() -> dict | None:
    p = live.LIVE / "state.pkl"
    return _state(p.stat().st_mtime) if p.exists() else None


@st.cache_data
def history(mtime: float) -> pd.DataFrame:
    h = pd.read_parquet(C.LAKE / "player_games.parquet", columns=HIST_COLS)
    for c in ("team", "opp", "name", "pos"):
        h[c] = h[c].astype(str)
    h["ppp"] = h.g_pp + h.a_pp
    return h


def hist() -> pd.DataFrame:
    return history((C.LAKE / "player_games.parquet").stat().st_mtime)


@st.cache_data(ttl=900, show_spinner="Reading Daily Faceoff lines and starters...")
def _dfo_deployments(day: str, built_at: str) -> dict:
    return live.deployments(state(), day, use_dfo=True, log=True, overrides=False)


def deployments(day: str) -> dict:
    s = state()
    deps = json.loads(json.dumps(_dfo_deployments(day, s["built_at"]), default=str))  # a copy
    deps = {t: {**d, "skaters": {int(k): v for k, v in d["skaters"].items()}} for t, d in deps.items()}
    for k, v in live.load_overrides().items():  # overrides are read fresh on every run
        d, t = k.split(":")
        if d == day and t in deps:
            deps[t] = {**live._from_json(v), "dfo": deps[t].get("dfo")}
    for d in deps.values():
        for k in ("goalie", "backup"):
            d[k] = int(d[k]) if d.get(k) not in (None, "None") else None
    return deps


def projections(day: str) -> pd.DataFrame:
    s = state()
    slate = s["slate"][s["slate"].date == pd.Timestamp(day)]
    p = live.project(slate, deployments(day), s)
    if p.empty:
        return p
    p = p.assign(pos=p.pos.astype(str))
    return p.join(season_avgs(tuple(p.player.tolist()), day), on="player")


@st.cache_data
def season_avgs(players: tuple, day: str) -> pd.DataFrame:
    """This season's (or, before 5 games, last season's) per-game averages, and last-10 TOI."""
    h = hist()
    h = h[h.player.isin(players) & (h.date < pd.Timestamp(day))].sort_values("date")
    out = pd.DataFrame(index=pd.Index(players, name="player"))
    cur = h[h.season == C.CURRENT_SEASON].groupby("player")
    prev = h[(h.season == C.CURRENT_SEASON - 1) & (h.game_type == 2)].groupby("player")
    n = cur.size().reindex(out.index).fillna(0)
    for c in ("points", "sog", "goals", "assists"):
        out[f"avg_{c}"] = np.where(n >= 5, cur[c].mean().reindex(out.index), prev[c].mean().reindex(out.index))
    out["l10_toi"] = h.groupby("player").toi.apply(lambda s: s.tail(10).mean() / 60).reindex(out.index)
    return out


def info(team: str) -> pd.DataFrame:
    """Today's roster: player, name, pos, shoots, number, headshot, birth."""
    r = state()["rosters"].get(team)
    return r.assign(player=r.player.astype(int)) if r is not None and len(r) else \
        pd.DataFrame(columns=["player", "name", "pos", "shoots", "number", "headshot", "birth"])


def names() -> pd.Series:
    """Names for every player id we know: today's rosters (goalies included) and history."""
    s = state()
    ros = [r[["player", "name"]] for r in s["rosters"].values() if r is not None and len(r)]
    allp = pd.concat(ros + [s["known"][["player", "name"]]], ignore_index=True)
    allp["player"] = allp.player.astype(int)
    return allp.drop_duplicates("player").set_index("player").name


def when(ts) -> str:
    """Daily Faceoff's ISO timestamp as local Eastern time."""
    try:
        return pd.Timestamp(ts).tz_convert("America/New_York").strftime("%b %d, %I:%M %p ET")
    except Exception:
        return str(ts or "")


def no_state():
    st.warning("No projections cached yet. Run `python live.py` (or refresh.bat) first.")
    st.stop()


def american(p) -> str:
    """Fair (no-vig) American odds for probability p: -150 means risk 150 to win 100."""
    if p is None or not np.isfinite(p) or p <= 0 or p >= 1:
        return "—"
    p = min(max(p, 0.005), 0.995)
    o = -100 * p / (1 - p) if p >= 0.5 else 100 * (1 - p) / p
    o = 5 * round(o / 5)
    return "EVEN" if abs(o) <= 100 else f"{o:+d}"


def odds_mode() -> bool:
    return st.session_state.get("prob_view", "%") == "Fair odds"


def odds_toggle():
    st.sidebar.segmented_control("Probabilities", ["%", "Fair odds"], default="%", key="prob_view",
                                 help="Fair odds = the no-vig American price implied by the model's probability.")


def pick_day(s: dict) -> str:
    days = sorted(s["games"].date.unique())
    if not days:
        st.info("No upcoming games in the next few days.")
        st.stop()
    return st.sidebar.selectbox("Date", days, key="day")


def pick_game(s: dict, day: str):
    games = s["games"][s["games"].date == day]
    label = {r.game_id: f"{r.away} @ {r.home}" for r in games.itertuples()}
    gid = st.sidebar.selectbox("Game", list(label), format_func=label.get, key=f"game_{day}")
    return games[games.game_id == gid].iloc[0]


# ---------------------------------------------------------------- shared table

# prop columns, in display order: (header, probability column). Shown as a % bar, or as the
# fair (no-vig) American price for the over when the sidebar is set to Fair odds.
PROP_COLS = [("o0.5 PTS", "p_pts_1"), ("o1.5 PTS", "p_pts_2"), ("o0.5 G", "p_g_1"), ("o1.5 G", "p_g_2"),
             ("o0.5 A", "p_a_1"), ("o1.5 SOG", "p_sog_2"), ("o2.5 SOG", "p_sog_3"), ("o3.5 SOG", "p_sog_4"),
             ("o0.5 PPP", "p_ppp_1")]
PROP_NAMES = [h for h, _ in PROP_COLS]

SPOT_ORDER = {"LW": 0, "C": 1, "RW": 2, "LD": 0, "RD": 1}


def player_table(p: pd.DataFrame, deps: dict, with_team: bool = True) -> pd.DataFrame:
    """Projection rows in depth-chart order, with readable column names."""
    meta = {}
    for team, g in p.groupby("team"):
        roster = info(team)
        if roster.empty:
            roster = g[["player", "pos"]]
        chart = live.assign_spots(deps[team], roster)
        for (grp, line, spot), pl in chart.items():
            meta[(team, pl)] = (spot, f"{grp}{line}", (0 if grp == "F" else 1, line, SPOT_ORDER[spot]))
    m = [meta.get((t, pl), ("", f"{gr}{int(l)}", (0 if gr == "F" else 1, int(l), 3)))
         for t, pl, gr, l in zip(p.team, p.player, p.grp, p.given_line)]
    t = p.assign(Spot=[x[0] for x in m], LineLab=[x[1] for x in m], order=[x[2] for x in m]).sort_values(["team", "order"])
    slot = np.where(t.Spot != "", t.LineLab.str[1:] + t.Spot, t.LineLab)  # 1LW, 2C, 3RD
    out = pd.DataFrame({
        "Team": t.team, "Slot": slot, "Line": t.LineLab, "Player": t["name"],
        "PP": t.given_pp.astype(int).map({0: "", 1: "PP1", 2: "PP2"}),
        "TOI": t.toi_hat, "PP TOI": t.toi_pp_hat, "SOG": t.sog_hat,
        "G": t.g_hat, "A": t.a_hat, "PTS": t.pts_hat, "PPP": t.ppp_hat,
        **{h: t[c] for h, c in PROP_COLS},
        "L10 TOI": t.l10_toi})
    return out if with_team else out.drop(columns=["Team", "L10 TOI"])


def show(t: pd.DataFrame, height="auto"):
    t = t.copy()
    pcols = [c for c in t.columns if c in PROP_NAMES]
    if odds_mode():  # same columns, shown as the fair price for the over
        for c in pcols:
            t[c] = t[c].map(american)
        pcols = []
    else:
        t[pcols] = (t[pcols] * 100).round()
    cfg = {c: st.column_config.NumberColumn(format="%.2f", width="small") for c in ("SOG", "G", "A", "PTS")}
    cfg["PPP"] = st.column_config.NumberColumn("PPP", format="%.2f", width="small",
                                               help="Power-play points (goals + assists on the power play)")
    cfg |= {c: st.column_config.NumberColumn(format="%.1f", width="small") for c in ("TOI", "PP TOI")}
    cfg |= {c: st.column_config.TextColumn(width="small", pinned=c != "PP") for c in ("Slot", "PP", "Team")}
    cfg["Player"] = st.column_config.TextColumn("Player", pinned=True)  # names stay put while the props scroll
    cfg["Line"] = None  # kept for filtering, not shown
    cfg |= {c: st.column_config.ProgressColumn(format="%d%%", min_value=0, max_value=100, width="small") for c in pcols}
    cfg["L10 TOI"] = st.column_config.NumberColumn("Last-10 TOI", format="%.1f")
    st.dataframe(t, hide_index=True, column_config=cfg, width="stretch", height=height)


# ---------------------------------------------------------------- goalies

@st.cache_resource
def _saves_state(mtime: float) -> dict | None:
    return saves_live.load_state()


def saves_state() -> dict | None:
    p = saves_live.STATE
    return _saves_state(p.stat().st_mtime) if p.exists() else None


@st.cache_data(show_spinner="Simulating goalie saves...")
def _team_goalies(built_at: str, game_id: int, team: str, key: str) -> list[dict]:
    """Saves projections for one team-game. `key` is the deployment that decides who starts
    (goalie, Daily Faceoff status, source), so a new report or override re-simulates."""
    S = saves_state()
    goalie, status, source = json.loads(key)
    cand = S["cand"][(S["cand"].game_id == game_id) & (S["cand"].team == team)]
    probs = saves_live.starter_probs(cand, {team: {"goalie": goalie, "goalie_status": status, "source": source}})
    return saves_live.project_team(S, game_id, team, probs, seed=game_id % 100000)


# saves props, in display order: shown like the skater props (% bar, or the fair price for the over)
SAVE_LINES = [21.5, 22.5, 23.5, 24.5, 25.5, 26.5, 27.5, 28.5, 29.5]
GOALIE_PROP_COLS = [(f"o{x:g} SV", f"p_over_{x}") for x in SAVE_LINES]
GOALIE_PROP_NAMES = [h for h, _ in GOALIE_PROP_COLS]
GOALIE_NOTE = ("Saves props are priced as if he starts (books void a goalie prop when he doesn't). "
               "Who starts follows Lines & goalies: an override there pins him.")


def goalie_table(day: str, deps: dict, game_id: int | None = None, backups: bool = True) -> pd.DataFrame:
    """One row per plausible starter (starter, then backup, for every team), most likely first."""
    S = saves_state()
    if S is None:
        return pd.DataFrame()
    games = state()["games"]
    games = games[games.date == day] if game_id is None else games[games.game_id == game_id]
    nm = names()
    rows = []
    for g in games.itertuples():
        for team, opp in ((g.away, g.home), (g.home, g.away)):
            d = deps.get(team, {})
            key = json.dumps([d.get("goalie"), d.get("goalie_status", ""), d.get("source", "")])
            for i, p in enumerate(_team_goalies(S["built_at"], int(g.game_id), team, key)):
                if i and not backups:
                    break
                rows.append({"Team": team, "Opp": opp, "Goalie": nm.get(p["goalie"], p["name"]),
                             "Role": "Starter" if i == 0 else "Backup", "Start %": p["p_start"],
                             "Status": p["status"], "SV": p["mean"], "Median": p["median"],
                             "Range": f"{p['p10']:.0f}–{p['p90']:.0f}", "SA": p["shots_faced"],
                             "GA": p["goals_against"], "SV%": p["exp_sv_pct"], "Pull %": p["pull_rate"],
                             **{h: p[c] for h, c in GOALIE_PROP_COLS}})
    return pd.DataFrame(rows)


def show_goalies(t: pd.DataFrame, height="auto"):
    t = t.copy()
    pcols = list(GOALIE_PROP_NAMES)
    if odds_mode():  # same columns, shown as the fair price for the over
        for c in pcols:
            t[c] = t[c].map(american)
        pcols = []
    else:
        t[pcols] = (t[pcols] * 100).round()
    t["Start %"] = (t["Start %"] * 100).round()
    t["Pull %"] = t["Pull %"] * 100
    cfg = {c: st.column_config.TextColumn(width="small", pinned=True) for c in ("Team", "Opp")}
    cfg["Goalie"] = st.column_config.TextColumn("Goalie", pinned=True)
    cfg["Role"] = st.column_config.TextColumn(width="small")
    cfg["Start %"] = st.column_config.ProgressColumn(
        format="%d%%", min_value=0, max_value=100, width="small",
        help="Chance he starts: the starter model, moved by Daily Faceoff news, or pinned by an override")
    cfg["Status"] = st.column_config.TextColumn(width="small", help="Daily Faceoff's report on him, or Override")
    cfg["SV"] = st.column_config.NumberColumn("SV", format="%.1f", width="small",
                                              help="Projected saves if he starts (includes the chance he is pulled)")
    cfg["Median"] = st.column_config.NumberColumn(format="%.0f", width="small")
    cfg["Range"] = st.column_config.TextColumn(width="small", help="10th to 90th percentile of his saves")
    cfg["SA"] = st.column_config.NumberColumn("SA", format="%.1f", width="small", help="Shots on goal he faces")
    cfg["GA"] = st.column_config.NumberColumn(format="%.2f", width="small")
    cfg["SV%"] = st.column_config.NumberColumn(format="%.3f", width="small", help="Expected save percentage")
    cfg["Pull %"] = st.column_config.NumberColumn(format="%.1f%%", width="small", help="Chance he doesn't finish")
    cfg |= {c: st.column_config.ProgressColumn(format="%d%%", min_value=0, max_value=100, width="small") for c in pcols}
    st.dataframe(t, hide_index=True, column_config=cfg, width="stretch", height=height)


# ---------------------------------------------------------------- game lines

def game_lines(p: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """Fair moneyline, puck line and total per game, from the summed player goal projections."""
    lam = p.groupby(["game_id", "team"]).g_hat.sum()
    g = games[games.game_id.isin(p.game_id.unique())].copy()
    g["lh"] = [lam.get((gid, t), np.nan) for gid, t in zip(g.game_id, g.home)]
    g["la"] = [lam.get((gid, t), np.nan) for gid, t in zip(g.game_id, g.away)]
    g = g.dropna(subset=["lh", "la"]).reset_index(drop=True)
    if g.empty:
        return g
    P = markets.game_probs(g.lh.to_numpy(), g.la.to_numpy())
    g["p_home"], g["p_home_m15"], g["p_away_m15"] = P["home_win"], P["home_m15"], P["away_m15"]
    lines = [markets.fair_total({t: P["over"][t][i] for t in markets.TOTALS}) for i in range(len(g))]
    g["total"] = lines
    g["p_over"] = [P["over"][t][i] for i, t in enumerate(lines)]
    g["over_by_line"] = [{t: float(P["over"][t][i]) for t in markets.TOTALS} for i in range(len(g))]  # alt totals
    # alt puck lines: P(the favourite, by expected goals, wins by more than k)
    g["fav_by"] = [{k: float((P["home_by"] if lh >= la else P["away_by"])[k][i]) for k in markets.SPREADS}
                   for i, (lh, la) in enumerate(zip(g.lh, g.la))]
    return g


def _fmt(p: float) -> str:
    return f"{american(p)} ({p:.0%})"


BOARD_CSS = """<style>
.gb{font-family:"Source Sans Pro","Segoe UI",sans-serif;max-width:860px;border:1px solid #e5e4df;border-radius:10px;padding:10px 12px 8px;
    background:#fff;color:#0b0b0b}
.gb .bar{display:flex;height:6px;border-radius:3px;overflow:hidden;margin:2px 0 4px}
.gb .bar span{display:block}
.gb .barlab{display:flex;justify-content:space-between;font-size:11px;color:#52514e;margin-bottom:6px}
.gb table{width:100%;border-collapse:separate;border-spacing:4px 4px}
.gb th{font-size:11px;font-weight:600;color:#8a8984;text-align:center;padding:0 4px;text-transform:uppercase;letter-spacing:.03em}
.gb th.t{text-align:left}
.gb td{text-align:center;border-radius:7px;padding:4px 6px;background:#f6f6f4;line-height:1.15}
.gb td.t{text-align:left;background:none;padding-left:0;white-space:nowrap}
.gb td.fav{background:rgba(42,120,214,.10)}
.gb .o{font-weight:700;font-size:15px}.gb .p{font-size:11px;color:#52514e}
.gb .tm{font-weight:700;font-size:15px}.gb .xg{font-size:11px;color:#52514e}
.gb.sm{padding:8px 9px 6px}.gb.sm .o,.gb.sm .tm{font-size:13px}.gb.sm td{padding:3px 4px}
.gb .ttl{font-size:12px;font-weight:600;color:#52514e;margin-bottom:2px}
.gb .ar{display:inline-flex;flex-direction:column;margin-left:5px;vertical-align:middle;gap:2px}
.gb .ar b{display:block;font-size:8px;line-height:8px;height:8px;font-weight:400;color:#8a8984;cursor:pointer;
    user-select:none;padding:0 2px}
.gb .ar b:hover{color:#2a78d6}
</style>"""


def _cell(line: str, p: float, fav: bool) -> str:
    return (f"<td class='{'fav' if fav else ''}'><div class='o'>{line}{american(p)}</div>"
            f"<div class='p'>{p:.0%}</div></td>")


def _step_cell(bid: str, key: str, label: str, p: float, fav: bool, title: str) -> str:
    """A price cell with small up / down arrows that step its market's line (see _step_script).
    key = market letter + side, e.g. 'to' (total over) or 'pf' (puck line favourite)."""
    return (f"<td id='{bid}-{key}' class='{'fav' if fav else ''}'><div class='o'><span class='tl'>{label}</span> "
            f"<span class='to'>{american(p)}</span><span class='ar' data-g='{key[0]}' title='{title}'>"
            f"<b data-d='1'>&#9650;</b><b data-d='-1'>&#9660;</b></span></div><div class='p'>{p:.0%}</div></td>")


def _step_rows(pairs: list) -> list:
    """[label, price, pct, tint] for both cells of each line, formatted here so the stepped
    prices match the rest of the app exactly. pairs = [(label_a, p_a, tint_a, label_b, p_b, tint_b)]."""
    return [[la, american(pa), f"{pa:.0%}", bool(fa), lb, american(pb), f"{pb:.0%}", bool(fb)]
            for la, pa, fa, lb, pb, fb in pairs]


def _step_script(bid: str, grp: str, keys: tuple[str, str], rows: list, start: int) -> str:
    """Steps one market of a board through its alternate lines in the browser (no rerun)."""
    return f"""<script>(function(){{
const L={json.dumps(rows)}, id="{bid}"; let i={start};
const b=document.getElementById(id); if(!b) return;
const a=document.getElementById(id+"-{keys[0]}"), c=document.getElementById(id+"-{keys[1]}");
const put=(td,l,od,pc,fav)=>{{td.querySelector(".tl").textContent=l;td.querySelector(".to").textContent=od;
  td.querySelector(".p").textContent=pc;td.classList.toggle("fav",fav);}};
b.querySelectorAll(".ar[data-g='{grp}'] b").forEach(x=>x.addEventListener("click",()=>{{
  i=Math.max(0,Math.min(L.length-1,i+Number(x.dataset.d)));const r=L[i];
  put(a,r[0],r[1],r[2],r[3]);put(c,r[4],r[5],r[6],r[7]);}}));
}})();</script>"""


def board_html(g, small: bool = False, title: str = "") -> str:
    """A compact sportsbook-style card: win-probability bar, then away / home rows with fair
    moneyline, puck line and total (odds bold, probability under it, favourite tinted). The
    arrows on the puck line and the total step through their alternate lines."""
    bid = f"gb{g.game_id}{'s' if small else ''}"
    pa, ph = 1 - g.p_home, g.p_home
    home_fav = g.lh >= g.la
    p15 = g.fav_by[1.5]
    rows = ""
    for side, team, lam, p_win in (("away", g.away, g.la, pa), ("home", g.home, g.lh, ph)):
        fav = (side == "home") == home_fav
        # favourite -1.5 / underdog +1.5 (= not losing by 2+); tint whichever side is above 50%
        pl = _step_cell(bid, "pf", "-1.5", p15, p15 >= 0.5, "Alternate puck lines") if fav else \
            _step_cell(bid, "pd", "+1.5", 1 - p15, 1 - p15 > 0.5, "Alternate puck lines")
        tot = _step_cell(bid, "to", f"o{g.total:g}", g.p_over, g.p_over >= 0.5, "Alternate totals") if side == "away" else \
            _step_cell(bid, "tu", f"u{g.total:g}", 1 - g.p_over, g.p_over < 0.5, "Alternate totals")
        rows += (f"<tr><td class='t'><span class='tm'>{team}</span> <span class='xg'>{lam:.2f} xG</span></td>"
                 f"{_cell('', p_win, p_win >= 0.5)}{pl}{tot}</tr>")
    bar = (f"<div class='bar'><span style='width:{pa:.1%};background:#eb6834'></span>"
           f"<span style='width:{ph:.1%};background:#2a78d6'></span></div>"
           f"<div class='barlab'><span>{g.away} {pa:.0%}</span><span>win probability</span><span>{ph:.0%} {g.home}</span></div>")
    ttl = f"<div class='ttl'>{title}</div>" if title else ""
    return (f"<div class='gb{' sm' if small else ''}' id='{bid}'>{ttl}{bar}<table><tr><th class='t'></th><th>Moneyline</th>"
            f"<th>Puck line</th><th>Total</th></tr>{rows}</table></div>"
            + _step_script(bid, "t", ("to", "tu"), _step_rows(
                [(f"o{t:g}", o, o >= 0.5, f"u{t:g}", 1 - o, o < 0.5) for t, o in sorted(g.over_by_line.items())]),
                sorted(g.over_by_line).index(g.total))
            + _step_script(bid, "p", ("pf", "pd"), _step_rows(
                [(f"-{k:g}", q, q >= 0.5, f"+{k:g}", 1 - q, 1 - q > 0.5) for k, q in sorted(g.fav_by.items())]), 0))


# ---------------------------------------------------------------- Slate

def page_slate():
    s = state() or no_state()
    day = pick_day(s)
    odds_toggle()
    st.title(f"Slate · {day}")
    p = projections(day)
    if p.empty:
        st.info("Nothing to project for this date.")
        return
    deps = deployments(day)
    with st.expander("Lineup sources and starting goalies"):
        nm = names()
        st.dataframe(pd.DataFrame([{
            "Team": t, "Lines from": d["source"] + (f" ({d['dfo']['source']})" if d.get("dfo") else ""),
            "Updated": when((d.get("dfo") or {}).get("updated")), "Goalie": nm.get(d.get("goalie"), d.get("goalie")),
            "Goalie status": d.get("goalie_status", "")} for t, d in sorted(deps.items())]), hide_index=True, width="stretch")
    with st.expander("Game lines (fair moneyline, puck line, total)"):
        gl = game_lines(p, s["games"])
        cards = [board_html(r, small=True, title=f"{r.away} @ {r.home} · "
                            + pd.Timestamp(r.start_utc).tz_convert("America/New_York").strftime("%I:%M %p ET"))
                 for r in gl.itertuples()]
        for i in range(0, len(cards), 3):
            cols = st.columns(3)
            for c, html in zip(cols, cards[i:i + 3]):
                with c:
                    st.html(BOARD_CSS + html, unsafe_allow_javascript=True)
        st.caption("Fair = no-vig prices from the projected score; overtime and shootouts count, as books grade them. "
                   "Blue = the more likely side of each bet. The arrows on the puck line and total step through alternate lines.")
    kind = st.segmented_control("Players", ["Skaters", "Goalies"], default="Skaters", key="slate_kind") or "Skaters"
    if kind == "Goalies":
        slate_goalies(s, day, p, deps)
        return
    c1, c2, c3 = st.columns([2, 2, 1])
    teams = c1.multiselect("Teams", sorted(p.team.unique()))
    pos = c2.multiselect("Position", ["C", "L", "R", "D"])
    sort = c3.selectbox("Sort by", ["PTS", "SOG", "G", "A", "PPP", "TOI"] + PROP_NAMES)
    q = p[p.team.isin(teams)] if teams else p
    q = q[q.pos.isin(pos)] if pos else q
    t = player_table(q, deps).sort_values(sort, ascending=False)
    st.caption(f"{p.game_id.nunique()} games · {len(t)} skaters · built {s['built_at']}")
    show(t, height=720)
    st.download_button("Download CSV", t.to_csv(index=False), f"projections_{day}.csv", "text/csv")


def slate_goalies(s: dict, day: str, p: pd.DataFrame, deps: dict):
    if saves_state() is None:
        st.info("No goalie projections cached yet. Run `python live.py` (or refresh.bat) first.")
        return
    c1, c2, c3 = st.columns([2, 2, 1])
    teams = c1.multiselect("Teams", sorted(p.team.unique()), key="g_teams")
    backups = c2.toggle("Include backups", value=False, key="g_backups")
    sort = c3.selectbox("Sort by", ["SV", "SA", "Start %", "Pull %"] + GOALIE_PROP_NAMES, key="g_sort")
    t = goalie_table(day, deps, backups=backups)
    if t.empty:
        st.info("No goalie projections for this date.")
        return
    t = t[t.Team.isin(teams)] if teams else t
    t = t.sort_values(sort, ascending=False)
    st.caption(f"{p.game_id.nunique()} games · {len(t)} goalies · built {saves_state()['built_at']}")
    show_goalies(t, height=min(720, 35 * (len(t) + 1) + 3))
    st.caption(GOALIE_NOTE)
    st.download_button("Download CSV", t.to_csv(index=False), f"goalies_{day}.csv", "text/csv")


# ---------------------------------------------------------------- Game

def team_panel(team: str, name: str, p: pd.DataFrame, d: dict):
    nm = names()
    tp = p[p.team == team]
    st.subheader(name)
    status = d.get("goalie_status", "")
    st.caption(f"Goalie **{nm.get(d.get('goalie'), '?')}**{(' · ' + status) if status else ''} · lines from {d['source']}"
               + (f" ({d['dfo']['source']})" if d.get("dfo") else ""))
    if tp.empty:
        return
    a, b, c = st.columns(3)
    a.metric("Goals", f"{tp.g_hat.sum():.2f}", border=True)
    b.metric("Assists", f"{tp.a_hat.sum():.2f}", border=True)
    c.metric("Shots on goal", f"{tp.sog_hat.sum():.1f}", border=True)
    a, b, c = st.columns(3)
    a.metric("Points", f"{tp.pts_hat.sum():.2f}", border=True)
    b.metric("PP minutes", f"{tp.team_pp.iloc[0] / 60:.1f}", border=True)
    c.metric("PK minutes", f"{tp.team_sh.iloc[0] / 60:.1f}", border=True)
    top = tp.nlargest(3, "pts_hat")
    st.caption("Top projected: " + " · ".join(f"**{n}** {v:.2f} pts" for n, v in zip(top["name"], top.pts_hat)))


def page_game():
    s = state() or no_state()
    day = pick_day(s)
    g = pick_game(s, day)
    odds_toggle()
    deps = deployments(day)
    p = projections(day)
    p = p[p.game_id == g.game_id]
    st.title(f"{g.away_name} @ {g.home_name}")
    st.caption(pd.Timestamp(g.start_utc).tz_convert("America/New_York").strftime("%a %b %d · %I:%M %p ET"))
    left, right = st.columns(2, gap="large")
    with left:
        team_panel(g.away, g.away_name, p, deps[g.away])
    with right:
        team_panel(g.home, g.home_name, p, deps[g.home])
    gl = game_lines(p, s["games"][s["games"].game_id == g.game_id])
    if len(gl):
        st.html(BOARD_CSS + board_html(gl.iloc[0]), unsafe_allow_javascript=True)
        st.caption("Fair (no-vig) prices from the projected score; overtime and shootouts count, as books grade them. "
                   "The arrows on the puck line and total step through alternate lines.")
    st.divider()
    c1, c2 = st.columns([2, 3])
    view = c1.segmented_control("Team", [g.away, g.home], default=g.away, key=f"gv_{g.game_id}") or g.away
    stat = c2.segmented_control("Chart", ["PTS", "SOG", "G", "A", "PPP", "TOI", "o0.5 PTS"], default="PTS", key="gstat") or "PTS"
    t = player_table(p[p.team == view], deps, with_team=False)
    fit = lambda df: 35 * (len(df) + 1) + 3  # every row visible, no inner scrolling
    st.markdown("**Forwards**")
    fw = t[t.Line.str.startswith("F")]
    show(fw, height=fit(fw))
    st.markdown("**Defence**")
    dm = t[t.Line.str.startswith("D")]
    show(dm, height=fit(dm))
    gt = goalie_table(day, deps, game_id=g.game_id)
    if len(gt):
        gt = gt[gt.Team == view].drop(columns=["Team", "Opp"])
        st.markdown("**Goalies**")
        show_goalies(gt, height=fit(gt))
        st.caption(GOALIE_NOTE)
    st.markdown(f"**Projected {stat}, {view}**")
    # chart a copy without dotted names (Altair reads "o0.5" as a nested field)
    ch = t[["Player", "Slot", "TOI", "SOG", "PTS"]].assign(val=t[stat].to_numpy(), p_pt=t["o0.5 PTS"].to_numpy())
    fmt = ".0%" if stat in PROP_NAMES else ".2f"
    bar = alt.Chart(ch).mark_bar(color=PROJ, cornerRadiusEnd=4, height=16).encode(
        y=alt.Y("Player:N", sort="-x", title=None, axis=alt.Axis(labelOverlap=False, labelLimit=200)),
        x=alt.X("val:Q", title=f"Projected {stat}", axis=alt.Axis(format=fmt)),
        tooltip=["Player", "Slot", alt.Tooltip("TOI:Q", format=".1f"), alt.Tooltip("SOG:Q", format=".2f"),
                 alt.Tooltip("PTS:Q", format=".2f"), alt.Tooltip("p_pt:Q", format=".0%", title="P(o0.5 PTS)")])
    text = bar.mark_text(align="left", dx=4, fontSize=11, color="#52514e").encode(text=alt.Text("val:Q", format=fmt))
    st.altair_chart((bar + text).properties(height=22 * len(ch) + 40), width="stretch")


# ---------------------------------------------------------------- Lines & goalies

def page_lines():
    s = state() or no_state()
    day = pick_day(s)
    g = pick_game(s, day)
    team = st.sidebar.radio("Team", [g.away, g.home], horizontal=True, key=f"lt_{g.game_id}")
    deps = deployments(day)
    d = deps[team]
    inf = info(team)
    nm = names()
    st.title(f"Lines & goalies · {team}")
    st.caption(f"{g.away} @ {g.home}, {day} · currently from **{d['source']}**"
               + (f" ({d['dfo']['source']}, updated {when(d['dfo']['updated'])})" if d.get("dfo") else "")
               + ". Saving stores a manual override for this team on this date, shared with everyone using "
               "the app; projections update at once.")
    if cloud.enabled() and store.backend() == "file":
        st.warning("Overrides are not connected to their gist (add gist_id and github_token to the app's secrets), "
                   "so saves only last until the app restarts.")
    sk_info = inf[inf.pos != "G"][["player", "name", "pos", "shoots"]]
    extra = [pl for pl in d["skaters"] if pl not in set(sk_info.player)]
    if extra:  # dressed per the line chart but not on today's NHL roster (e.g. a fresh call-up)
        sk_info = pd.concat([sk_info, pd.DataFrame({"player": extra, "name": [nm.get(pl, str(pl)) for pl in extra],
                                                    "pos": "?", "shoots": None})], ignore_index=True)
    pos_of = dict(zip(sk_info.player, sk_info.pos))
    label = {int(r.player): f"{r.name} ({r.pos})" for r in sk_info.itertuples()}
    fwd = sorted([pl for pl in label if pos_of[pl] != "D"], key=label.get)
    dmen = sorted([pl for pl in label if pos_of[pl] == "D"], key=label.get)
    chart = live.assign_spots(d, sk_info)
    goalies = inf[inf.pos == "G"]
    gopts = [int(x) for x in goalies.player]
    for x in (d.get("goalie"), d.get("backup")):
        if x and x not in gopts:
            gopts.append(x)
    gname = {x: nm.get(x, str(x)) for x in gopts} | dict(zip(goalies.player, goalies.name))
    fmt = lambda pl: "—" if pl is None else label.get(pl, str(pl))
    key = f"{day}{team}"
    view = st.segmented_control("View", ["Depth chart", "Table"], default="Depth chart", key="lines_view") or "Depth chart"
    proj = projections(day)
    proj_t = proj[proj.team == team].set_index("player")

    problems: list[str] = []
    if view == "Depth chart":
        new_chart = {}
        for grp, title, spots, n_lines, pool in (("F", "Forwards", live.F_SPOTS, 4, fwd), ("D", "Defence", live.D_SPOTS, 4, dmen)):
            st.markdown(f"##### {title}")
            widths = [0.5] + [3] * len(spots) + ([3] if grp == "D" else [])
            hdr = st.columns(widths)
            for c, h in zip(hdr[1:], spots):
                c.markdown(f"**{h}**")
            for line in range(1, n_lines + 1):
                cols = st.columns(widths, vertical_alignment="center")
                cols[0].markdown(f"**{line}**" if grp == "F" or line < 4 else "**7th**",
                                 help=None if grp == "F" or line < 4 else "Pair 4 is for a 7th defenceman "
                                 "(an 11 F + 7 D lineup); leave it empty for 12 + 6.")
                for c, spot in zip(cols[1:], spots):
                    cur = chart.get((grp, line, spot))
                    opts = [None] + pool + ([cur] if cur is not None and cur not in pool else [])
                    new_chart[(grp, line, spot)] = c.selectbox(
                        f"{grp}{line}{spot}", opts, index=opts.index(cur) if cur in opts else 0, format_func=fmt,
                        label_visibility="collapsed", key=f"{key}{grp}{line}{spot}")
        st.markdown("##### Goalies")
        c1, c2, _ = st.columns([3, 3, 3.5])
        starter = c1.selectbox("Starter", gopts, index=gopts.index(d["goalie"]) if d.get("goalie") in gopts else 0,
                               format_func=lambda x: gname.get(x, str(x)), key=f"{key}G1")
        back_opts = [None] + [x for x in gopts if x != starter]
        backup = c2.selectbox("Backup", back_opts, index=back_opts.index(d["backup"]) if d.get("backup") in back_opts else 0,
                              format_func=lambda x: "—" if x is None else gname.get(x, str(x)), key=f"{key}G2")
        st.markdown("##### Special teams")
        dressed = [pl for pl in new_chart.values() if pl is not None]
        cur_unit = lambda k, u: [pl for pl, v in d["skaters"].items() if v.get(k) == u and pl in dressed]
        short = lambda pl: label.get(pl, str(pl)).split(" (")[0]
        cs = st.columns(2)
        pp = {u: cs[u - 1].multiselect(f"PP{u} (up to 5)", dressed, cur_unit("pp", u), max_selections=5, format_func=short,
                                        key=f"{key}pp{u}") for u in (1, 2)}
        cs = st.columns(2)
        pk = {u: cs[u - 1].multiselect(f"PK{u} (up to 4)", dressed, cur_unit("pk", u), max_selections=4, format_func=short,
                                        key=f"{key}pk{u}") for u in (1, 2)}
        sk = {}
        for (grp, line, spot), pl in new_chart.items():
            if pl is not None:
                sk[int(pl)] = {"line": line, "spot": spot, "pp": 0, "pk": 0,
                               **({"toi": d["skaters"][pl]["toi"]} if d["skaters"].get(pl, {}).get("toi") else {})}
        for u in (1, 2):
            for pl in pp[u]:
                sk[pl]["pp"] = u
            for pl in pk[u]:
                if not sk[pl]["pk"]:
                    sk[pl]["pk"] = u
        new_dep = {"skaters": sk, "goalie": starter, "backup": backup}
        dupes = pd.Series(dressed, dtype="Int64").value_counts()
        problems = [f"{label.get(int(x), x)} is in two spots" for x in dupes[dupes > 1].index]
    else:
        spot_of = {pl: (line, spot) for (grp, line, spot), pl in chart.items()}
        sk0 = d["skaters"]
        tab = sk_info[["player", "name", "pos"]].copy()
        tab.insert(0, "dressed", tab.player.isin(list(sk0)))
        tab["line"] = tab.player.map(lambda x: spot_of.get(x, (None, None))[0])
        tab["spot"] = tab.player.map(lambda x: spot_of.get(x, (None, None))[1])
        tab["pp"] = tab.player.map(lambda x: (sk0.get(x) or {}).get("pp", 0))
        tab["pk"] = tab.player.map(lambda x: (sk0.get(x) or {}).get("pk", 0))
        tab["proj_toi"] = tab.player.map(proj_t.toi_hat)
        tab["toi"] = tab.player.map(lambda x: (sk0.get(x) or {}).get("toi"))
        tab = tab.sort_values(["dressed", "pos", "line", "spot"], ascending=[False, True, True, True])
        st.caption("**TOI override** sets a player's total minutes: his EV / PP / PK time is scaled to hit it, and the EV "
                   "minutes he gains or loses come out of (or go to) his teammates in proportion to their own EV time. "
                   "Leave it blank to let the model decide.")
        ed = st.data_editor(tab, hide_index=True, width="stretch", key=f"ed_{key}", column_config={
            "player": None,
            "dressed": st.column_config.CheckboxColumn("Dressed"),
            "name": st.column_config.TextColumn("Player", disabled=True),
            "pos": st.column_config.TextColumn("Pos", disabled=True),
            "line": st.column_config.NumberColumn("Line / pair", min_value=1, max_value=4, step=1),
            "spot": st.column_config.SelectboxColumn("Spot", options=["LW", "C", "RW", "LD", "RD"]),
            "pp": st.column_config.NumberColumn("PP unit", min_value=0, max_value=2, step=1),
            "pk": st.column_config.NumberColumn("PK unit", min_value=0, max_value=2, step=1),
            "proj_toi": st.column_config.NumberColumn("Projected TOI", disabled=True, format="%.1f"),
            "toi": st.column_config.NumberColumn("TOI override (min)", min_value=3.0, max_value=32.0, step=0.5, format="%.1f")})
        c1, _ = st.columns([3, 6.5])
        starter = c1.selectbox("Starting goalie", gopts, index=gopts.index(d["goalie"]) if d.get("goalie") in gopts else 0,
                               format_func=lambda x: gname.get(x, str(x)), key=f"{key}tG1")
        dr = ed[ed.dressed]
        new_dep = {"skaters": {int(r.player): {"line": int(r.line) if pd.notna(r.line) else None,
                                               "spot": r.spot if isinstance(r.spot, str) else None,
                                               "pp": int(r.pp or 0), "pk": int(r.pk or 0),
                                               **({"toi": float(r.toi)} if pd.notna(r.toi) else {})}
                               for r in dr.itertuples()},
                   "goalie": starter, "backup": next((x for x in gopts if x != starter), None)}
    nf = sum(pos_of.get(x) != "D" for x in new_dep["skaters"])
    nd = len(new_dep["skaters"]) - nf
    st.caption(f"{nf} forwards and {nd} defence dressed" + ("" if (nf, nd) in ((12, 6), (11, 7)) else " · a usual lineup is 12 + 6 or 11 + 7"))
    for msg in problems:
        st.error(msg)
    c1, c2, _ = st.columns([1.2, 2.3, 5])
    if c1.button("Save lines", type="primary", disabled=bool(problems)):
        live.save_override(day, team, new_dep)
        st.toast("Saved. Projections updated.")
        st.rerun()
    if c2.button("Reset to Daily Faceoff / last game"):
        live.save_override(day, team, None)
        for k in [k for k in st.session_state if str(k).startswith(key) or k == f"ed_{key}"]:
            del st.session_state[k]
        st.rerun()
    st.divider()
    st.markdown("##### Projections with the saved lines")
    st.html(depth_chart_html(d, proj_t, sk_info, nm))


def depth_chart_html(d: dict, proj_t: pd.DataFrame, sk_info: pd.DataFrame, nm: pd.Series) -> str:
    chart = live.assign_spots(d, sk_info)

    def cell(pl):
        if pl is None or pl not in proj_t.index:
            return "<td class='e'>—</td>"
        r = proj_t.loc[pl]
        pp = f" · PP{int(r.given_pp)}" if r.given_pp else ""
        return (f"<td><div class='n'>{r['name']}</div><div class='s'>{r.toi_hat:.1f} min{pp}</div>"
                f"<div class='s'>{r.pts_hat:.2f} pts · {r.sog_hat:.1f} SOG · {r.p_pts_1:.0%} 1+ pt</div></td>")

    rows_f = "".join(f"<tr><th>{l}</th>{''.join(cell(chart.get(('F', l, s))) for s in live.F_SPOTS)}</tr>" for l in range(1, 5))
    rows_d = "".join(f"<tr><th>{l if l < 4 else '7th'}</th>{''.join(cell(chart.get(('D', l, s))) for s in live.D_SPOTS)}"
                     "<td class='b'></td></tr>"
                     for l in range(1, 5) if l < 4 or any(('D', 4, s) in chart for s in live.D_SPOTS))
    css = """<style>
    .dc{border-collapse:separate;border-spacing:6px;width:100%;font-family:"Source Sans Pro",sans-serif}
    .dc th{color:#52514e;font-weight:600;font-size:13px;text-align:left;padding:2px 6px;width:28px}
    .dc td{background:rgba(42,120,214,.08);border-radius:8px;padding:8px 10px;vertical-align:top;width:32%}
    .dc td.e{color:#8a8984}.dc td.b{background:none}
    .dc .n{font-weight:600;font-size:14px}.dc .s{font-size:12px;color:#52514e}
    .h{font-weight:700;margin:8px 0 0 6px;font-family:"Source Sans Pro",sans-serif}
    </style>"""
    return (css + "<div class='h'>Forwards</div><table class='dc'><tr><th></th><th>LW</th><th>C</th><th>RW</th></tr>"
            + rows_f + "</table><div class='h'>Defence</div><table class='dc'><tr><th></th><th>LD</th><th>RD</th><th></th></tr>"
            + rows_d + "</table><div class='h'>Goalies</div><table class='dc'>"
            f"<tr><th>Starter</th><td><div class='n'>{nm.get(d.get('goalie'), '—')}</div></td><th>Backup</th><td><div class='n'>{nm.get(d.get('backup'), '—')}</div></td></tr></table>")


# ---------------------------------------------------------------- Player

MARKETS = [("SOG", "sog", 2), ("SOG", "sog", 3), ("SOG", "sog", 4), ("SOG", "sog", 5), ("Points", "points", 1),
           ("Points", "points", 2), ("Points", "points", 3), ("Goals", "goals", 1), ("Assists", "assists", 1),
           ("Assists", "assists", 2), ("PP points", "ppp", 1), ("PP points", "ppp", 2)]
PROB = {"sog": "p_sog_{}", "points": "p_pts_{}", "goals": "p_g_{}", "assists": "p_a_{}", "ppp": "p_ppp_{}"}


GOALIE_GAMES = C.DATA / "saves" / "goalie_games_app.parquet"   # the cloud copy (published by the refresh)


@st.cache_data
def _goalie_games(path: str, mtime: float) -> pd.DataFrame:
    g = pd.read_parquet(path)
    g["date"] = pd.to_datetime(g.date)
    g = g[g.toi > 0]
    return g.assign(sv_pct=g.saves / g.shots_against.where(g.shots_against > 0))


def goalie_games() -> pd.DataFrame:
    """Every goalie appearance since 2021-22 (the saves model's lake locally, a slim copy on the cloud)."""
    for p in (C.DATA / "saves" / "lake" / "goalies.parquet", GOALIE_GAMES):
        if p.exists():
            return _goalie_games(str(p), p.stat().st_mtime)
    return pd.DataFrame()


def day_goalies(day: str, deps: dict) -> list[dict]:
    """Every plausible starter on the date with his saves projection and his game."""
    S = saves_state()
    if S is None:
        return []
    out = []
    for g in state()["games"][state()["games"].date == day].itertuples():
        for team, opp, home in ((g.away, g.home, False), (g.home, g.away, True)):
            d = deps.get(team, {})
            k = json.dumps([d.get("goalie"), d.get("goalie_status", ""), d.get("source", "")])
            for i, q in enumerate(_team_goalies(S["built_at"], int(g.game_id), team, k)):
                out.append({**q, "team": team, "opp": opp, "is_home": home, "game_id": int(g.game_id),
                            "role": "Starter" if i == 0 else "Backup"})
    return out


def page_player():
    s = state() or no_state()
    day = pick_day(s)
    p = projections(day)
    if p.empty:
        st.stop()
    p = p.sort_values(["team", "pts_hat"], ascending=[True, False])
    deps = deployments(day)
    gl = day_goalies(day, deps)
    nm = names()
    # skaters and goalies in one list, grouped by team (goalies after their team's skaters)
    opts, lab = [], {}
    for team in sorted(set(p.team) | {g["team"] for g in gl}):
        for i, r in zip(p.index[p.team == team], p[p.team == team].itertuples()):
            opts.append(f"s{i}"); lab[f"s{i}"] = f"{r.name} · {r.team}"
        for j, g in enumerate(gl):
            if g["team"] == team:
                opts.append(f"g{j}"); lab[f"g{j}"] = f"{nm.get(g['goalie'], g['name'])} · {team} · G"
    sel = st.sidebar.selectbox("Player", opts, index=opts.index(f"s{p.pts_hat.idxmax()}"), format_func=lab.get,
                               key=f"pp_{day}", help="Skaters and goalies; type to search")
    if sel[0] == "g":
        page_goalie(day, gl[int(sel[1:])], deps)
        return
    i = int(sel[1:])
    r = p.loc[i]
    ph = hist()
    ph = ph[(ph.player == r.player) & (ph.date < pd.Timestamp(day))].sort_values("date")
    inf = info(r.team)
    me = inf[inf.player == r.player]
    me = me.iloc[0] if len(me) else None
    nm = names()
    deps = deployments(day)

    # ---- header
    c0, c1, c2 = st.columns([1, 4, 4])
    if me is not None and isinstance(me.get("headshot"), str):
        c0.image(me["headshot"], width=110)
    bits = [f"#{int(me['number'])}" if me is not None and pd.notna(me.get("number")) else None, r.team, r.pos,
            f"age {r.age:.0f}" if pd.notna(r.age) else None,
            f"shoots {me['shoots']}" if me is not None and isinstance(me.get("shoots"), str) else None]
    c1.title(r["name"])
    c1.markdown(" · ".join(b for b in bits if b))
    mates = [nm.get(int(x), "") for x in (r.given_m1, r.given_m2, r.given_d1) if pd.notna(x)]
    c2.markdown(f"**Tonight** {'vs' if r.is_home else '@'} {r.opp} · {day}  \n"
                f"**Deployment** {'line' if r.grp == 'F' else 'pair'} {int(r.given_line)}"
                f"{' · PP' + str(int(r.given_pp)) if r.given_pp else ''}{' · PK' + str(int(r.given_pk)) if r.given_pk else ''}  \n"
                f"**With** {', '.join(m for m in mates if m) or '—'}  \n"
                f"**Opposing goalie** {nm.get(deps[r.opp].get('goalie'), '?')}")

    # ---- projection vs averages
    st.markdown("#### Tonight's projection")
    cur = ph[ph.season == C.CURRENT_SEASON]
    last = ph[(ph.season == C.CURRENT_SEASON - 1) & (ph.game_type == 2)]
    cols = st.columns(7)
    for c, (title, v) in zip(cols, [("TOI", f"{r.toi_hat:.1f}"), ("Shots on goal", f"{r.sog_hat:.2f}"),
                                    ("Goals", f"{r.g_hat:.2f}"), ("Assists", f"{r.a_hat:.2f}"),
                                    ("Points", f"{r.pts_hat:.2f}"), ("PP points", f"{r.ppp_hat:.2f}"),
                                    ("P(1+ point)", f"{r.p_pts_1:.0%}")]):
        c.metric(title, v, border=True)
    stats = [("TOI", "toi", 60, r.toi_hat), ("PP TOI", "toi_pp", 60, r.toi_pp_hat), ("SOG", "sog", 1, r.sog_hat),
             ("Goals", "goals", 1, r.g_hat), ("Assists", "assists", 1, r.a_hat), ("Points", "points", 1, r.pts_hat),
             ("PP points", "ppp", 1, r.ppp_hat)]
    comp = pd.DataFrame({
        "Tonight": [v for *_, v in stats],
        **{f"{wl} ({len(w)} gp)": [w[c].mean() / f if len(w) else np.nan for _, c, f, _ in stats]
           for wl, w in (("Last 10", ph.tail(10)), ("This season", cur), ("Last season", last), ("Career", ph[ph.game_type == 2]))}},
        index=[s[0] for s in stats])
    st.dataframe(comp.style.format("{:.2f}", na_rep="—"), width="stretch")

    # ---- prop lines + hit rates
    st.markdown("#### Prop lines and hit rates")
    opp_games = ph[ph.opp == r.opp]
    windows = [("Last 5", ph.tail(5)), ("Last 10", ph.tail(10)), ("Last 20", ph.tail(20)),
               ("This season", cur), ("Last season", ph[(ph.season == C.CURRENT_SEASON - 1) & (ph.game_type == 2)]),
               (f"vs {r.opp}", opp_games)]
    rows = []
    for title, col, k in MARKETS:
        pr = r[PROB[col].format(k)]
        row = {"Line": f"{title} o{k - 0.5:g}", "Model": pr, "Over": american(pr), "Under": american(1 - pr)}
        for wl, w in windows:
            row[wl] = (w[col] >= k).mean() if len(w) else np.nan
        rows.append(row)
    hr = pd.DataFrame(rows)
    pc = [c for c in hr.columns if c not in ("Line", "Over", "Under")]
    hr[pc] = (hr[pc] * 100).round()
    cfg = {c: st.column_config.ProgressColumn(c, format="%d%%", min_value=0, max_value=100) for c in pc}
    cfg |= {"Over": st.column_config.TextColumn("Fair over", width="small"),
            "Under": st.column_config.TextColumn("Fair under", width="small")}
    st.dataframe(hr, hide_index=True, width="stretch", column_config=cfg, height=35 * (len(hr) + 1) + 3)
    fair = []
    for title, col in (("SOG", "sog"), ("Points", "points")):
        ks = [k for t2, c2, k in MARKETS if c2 == col]
        k = min(ks, key=lambda k: abs(r[PROB[col].format(k)] - 0.5))
        p_ = r[PROB[col].format(k)]
        fair.append(f"**{title} {k - 0.5:g}** (over {american(p_)} / under {american(1 - p_)})")
    st.caption("Fair line, the one closest to a coin flip: " + " · ".join(fair)
               + ". Fair odds carry no vig; a book's price has to beat them to be worth it. "
               + "Games in each window: " + " · ".join(f"{wl} {len(w)}" for wl, w in windows) + ".")

    # ---- game log
    st.markdown("#### Game log")
    c1, c2, c3 = st.columns([3, 2, 1.3])
    stat = c1.segmented_control("Stat", ["SOG", "Points", "Goals", "Assists", "PPP", "TOI", "PP TOI"], default="SOG", key="pl_stat") or "SOG"
    n = c2.segmented_control("Games", [10, 20, 40, 82], default=20, key="pl_n") or 20
    col = {"SOG": "sog", "Points": "points", "Goals": "goals", "Assists": "assists", "PPP": "ppp", "TOI": "toi", "PP TOI": "toi_pp"}[stat]
    scale = 60 if col in ("toi", "toi_pp") else 1
    proj_v = {"sog": r.sog_hat, "points": r.pts_hat, "goals": r.g_hat, "assists": r.a_hat, "ppp": r.ppp_hat, "toi": r.toi_hat, "toi_pp": r.toi_pp_hat}[col]
    default_line = {"sog": 2.5, "points": 0.5, "goals": 0.5, "assists": 0.5, "ppp": 0.5, "toi": round(proj_v * 2) / 2, "toi_pp": 1.5}[col]
    line = c3.number_input("Line", value=float(default_line), step=0.5, key=f"pl_line_{col}")
    log = ph.tail(n).copy()
    if len(log):
        log["value"] = log[col] / scale
        log["game"] = log.date.dt.strftime("%m-%d") + np.where(log.is_home, " v ", " @ ") + log.opp
        log["result"] = np.where(log.value > line, f"over {line:g}", f"under {line:g}")
        log["slot"] = (np.where(log.pos == "D", "D", "F") + log.line.fillna(0).astype(int).astype(str)
                       + np.where(log.pp_unit > 0, " · PP" + log.pp_unit.astype(int).astype(str), ""))
        log["toi_min"] = log.toi / 60
        bars = alt.Chart(log).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
            x=alt.X("game:N", sort=None, title=None, axis=alt.Axis(labelAngle=-45)),
            y=alt.Y("value:Q", title=stat),
            color=alt.Color("result:N", scale=alt.Scale(domain=[f"over {line:g}", f"under {line:g}"], range=[PROJ, "#c3c2b7"]),
                            legend=alt.Legend(title=None, orient="top")),
            tooltip=[alt.Tooltip("game:N", title="Game"), alt.Tooltip("value:Q", title=stat, format=".1f"),
                     alt.Tooltip("slot:N", title="Deployment"), alt.Tooltip("toi_min:Q", title="TOI", format=".1f")])
        rule = alt.Chart(pd.DataFrame({"y": [line]})).mark_rule(color=MUTED, strokeDash=[4, 3]).encode(y="y:Q")
        proj_rule = alt.Chart(pd.DataFrame({"y": [proj_v], "t": [f"{proj_v:.2f}"]})).mark_rule(color=ACT, strokeWidth=2).encode(
            y="y:Q", tooltip=alt.Tooltip("t:N", title="Tonight's projection"))
        st.altair_chart((bars + rule + proj_rule).properties(height=280), width="stretch")
        st.caption(f"Over {line:g} in {(log.value > line).mean():.0%} of the last {len(log)} games. "
                   f"Orange line: tonight's projection ({proj_v:.2f}). Dashed: the line.")
    else:
        st.info("No NHL games yet.")

    t1, t2, t3 = st.tabs(["Recent games", "Season by season", "Splits"])
    with t1:
        rec = ph.tail(15).iloc[::-1]
        st.dataframe(pd.DataFrame({
            "Date": rec.date.dt.strftime("%Y-%m-%d"), "Opp": np.where(rec.is_home, "vs ", "@ ") + rec.opp,
            "Line": np.where(rec.pos == "D", "D", "F") + rec.line.fillna(0).astype(int).astype(str),
            "PP": rec.pp_unit.map({0: "", 1: "PP1", 2: "PP2"}), "TOI": rec.toi / 60, "PP TOI": rec.toi_pp / 60,
            "SOG": rec.sog, "G": rec.goals, "A": rec.assists, "PTS": rec.points, "PPP": rec.ppp}), hide_index=True, width="stretch",
            column_config={"TOI": st.column_config.NumberColumn(format="%.1f"), "PP TOI": st.column_config.NumberColumn(format="%.1f")})
    with t2:
        reg = ph[ph.game_type == 2].groupby("season")
        ss = pd.DataFrame({"GP": reg.size(), "TOI/gm": reg.toi.mean() / 60, "PP TOI/gm": reg.toi_pp.mean() / 60,
                           "G": reg.goals.sum(), "A": reg.assists.sum(), "PTS": reg.points.sum(), "SOG": reg.sog.sum(),
                           "PTS/gm": reg.points.mean(), "SOG/gm": reg.sog.mean(),
                           "SH%": reg.goals.sum() / reg.sog.sum().replace(0, np.nan)})
        ss.index = [f"{y}-{str(y + 1)[2:]}" for y in ss.index]
        st.dataframe(ss.iloc[::-1], width="stretch", column_config={
            "TOI/gm": st.column_config.NumberColumn(format="%.1f"), "PP TOI/gm": st.column_config.NumberColumn(format="%.1f"),
            "PTS/gm": st.column_config.NumberColumn(format="%.2f"), "SOG/gm": st.column_config.NumberColumn(format="%.2f"),
            "SH%": st.column_config.NumberColumn(format="percent")})
        if len(ss) > 1:
            trend = ss.reset_index(names="season")[["season", "PTS/gm", "SOG/gm"]].melt("season", var_name="stat")
            st.altair_chart(alt.Chart(trend).mark_line(point=True, strokeWidth=2).encode(
                x=alt.X("season:N", title=None), y=alt.Y("value:Q", title="per game"),
                color=alt.Color("stat:N", scale=alt.Scale(range=[PROJ, ACT]), legend=alt.Legend(orient="top", title=None)),
                tooltip=["season", "stat", alt.Tooltip("value:Q", format=".2f")]).properties(height=220), width="stretch")
    with t3:
        recent = ph[ph.season >= C.CURRENT_SEASON - 1]

        def split(df, by, label):
            g = df.groupby(by)
            out = pd.DataFrame({"GP": g.size(), "TOI": g.toi.mean() / 60, "PTS/gm": g.points.mean(), "SOG/gm": g.sog.mean(),
                                "1+ pt": g.points.apply(lambda x: (x >= 1).mean()), "3+ SOG": g.sog.apply(lambda x: (x >= 3).mean())})
            out.index = [f"{label}: {k}" for k in out.index]
            return out
        parts = [split(recent.assign(k=np.where(recent.is_home, "home", "away")), "k", "Venue"),
                 split(recent.assign(k=np.where(recent.pos == "D", "D", "F") + recent.line.fillna(0).astype(int).astype(str)), "k", "Line"),
                 split(recent.assign(k=recent.pp_unit.map({0: "none", 1: "PP1", 2: "PP2"})), "k", "PP unit")]
        if len(opp_games):
            parts.append(split(opp_games.assign(k=r.opp), "k", "Career vs"))
        st.dataframe(pd.concat(parts), width="stretch", column_config={
            "TOI": st.column_config.NumberColumn(format="%.1f"), "PTS/gm": st.column_config.NumberColumn(format="%.2f"),
            "SOG/gm": st.column_config.NumberColumn(format="%.2f"),
            "1+ pt": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1),
            "3+ SOG": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1)})
        st.caption("Venue, line and PP-unit splits cover last season and this season; 'Career vs' covers every game in the data.")

    with st.expander("How tonight's projection is calculated"):
        breakdown(r, p, i, s)


def page_goalie(day: str, g: dict, deps: dict):
    """The Player page for a goalie: tonight's saves projection, saves prop lines with hit
    rates, his game log and seasons. Props are priced as if he starts, as books grade them."""
    nm = names()
    gid = int(g["goalie"])
    gh = goalie_games()
    mine = gh[(gh.goalie == gid) & (gh.date < pd.Timestamp(day))].sort_values("date") if len(gh) else gh
    starts = mine[mine.starter] if len(mine) else mine
    inf = info(g["team"])
    me = inf[inf.player == gid]
    me = me.iloc[0] if len(me) else None
    name = nm.get(gid, g["name"])

    # ---- header
    c0, c1, c2 = st.columns([1, 4, 4])
    if me is not None and isinstance(me.get("headshot"), str):
        c0.image(me["headshot"], width=110)
    age = None
    if me is not None and isinstance(me.get("birth"), str):
        age = (pd.Timestamp(day) - pd.Timestamp(me["birth"])).days / 365.25
    bits = [f"#{int(me['number'])}" if me is not None and pd.notna(me.get("number")) else None, g["team"], "G",
            f"age {age:.0f}" if age else None,
            f"catches {me['shoots']}" if me is not None and isinstance(me.get("shoots"), str) else None]
    c1.title(name)
    c1.markdown(" · ".join(b for b in bits if b))
    status = g.get("status") or ""
    c2.markdown(f"**Tonight** {'vs' if g['is_home'] else '@'} {g['opp']} · {day}  \n"
                f"**Start chance** {g['p_start']:.0%} ({g['role'].lower()}{', ' + status if status else ''})  \n"
                f"**Team shots against** {g['mu_team_sa']:.1f} expected  \n"
                f"**Lines & goalies** pins the starter: an override there sets this to 100%")

    # ---- projection vs averages
    st.markdown("#### Tonight's projection")
    st.caption("If he starts. Books void a goalie's props when he doesn't start, so they're priced the same way.")
    cols = st.columns(6)
    for c, (t, v) in zip(cols, [("Start %", f"{g['p_start']:.0%}"), ("Saves", f"{g['mean']:.1f}"),
                                ("Shots faced", f"{g['shots_faced']:.1f}"), ("Goals against", f"{g['goals_against']:.2f}"),
                                ("Save %", f"{g['exp_sv_pct']:.3f}"), ("Pulled %", f"{g['pull_rate']:.1%}")]):
        c.metric(t, v, border=True)
    cur = starts[starts.season == C.CURRENT_SEASON] if len(starts) else starts
    last = starts[(starts.season == C.CURRENT_SEASON - 1) & (starts.game_type == 2)] if len(starts) else starts
    car = starts[starts.game_type == 2] if len(starts) else starts
    wins = (("Last 10 starts", starts.tail(10)), ("This season", cur), ("Last season", last), ("Career", car))

    def avg(w, c):
        if not len(w):
            return np.nan
        return w.saves.sum() / w.shots_against.sum() if c == "sv_pct" else w[c].mean()
    comp = pd.DataFrame({"Tonight": [g["mean"], g["shots_faced"], g["goals_against"], g["exp_sv_pct"]],
                         **{f"{wl} ({len(w)} gp)": [avg(w, c) for c in ("saves", "shots_against", "goals_against", "sv_pct")]
                            for wl, w in wins}},
                        index=["Saves", "Shots against", "Goals against", "Save %"])
    st.dataframe(comp.style.format("{:.2f}", na_rep="—").format("{:.3f}", subset=pd.IndexSlice["Save %", :], na_rep="—"),
                 width="stretch")

    # ---- saves lines + hit rates
    st.markdown("#### Saves lines and hit rates")
    pmf = np.asarray(g["pmf"])
    x = np.arange(len(pmf))
    mid = int(round(g["median"]))
    windows = [("Last 5", starts.tail(5)), ("Last 10", starts.tail(10)), ("Last 20", starts.tail(20)), ("This season", cur),
               ("Last season", last), (f"vs {g['opp']}", starts[starts.opp == g["opp"]] if len(starts) else starts)]
    rows = []
    for ln in [k + 0.5 for k in range(mid - 6, mid + 6)]:
        pr = float(pmf[x > ln].sum())
        row = {"Line": f"Saves o{ln:g}", "Model": pr, "Over": american(pr), "Under": american(1 - pr)}
        for wl, w in windows:
            row[wl] = (w.saves > ln).mean() if len(w) else np.nan
        rows.append(row)
    hr = pd.DataFrame(rows)
    pc = [c for c in hr.columns if c not in ("Line", "Over", "Under")]
    hr[pc] = (hr[pc] * 100).round()
    cfg = {c: st.column_config.ProgressColumn(c, format="%d%%", min_value=0, max_value=100) for c in pc}
    cfg |= {"Over": st.column_config.TextColumn("Fair over", width="small"),
            "Under": st.column_config.TextColumn("Fair under", width="small")}
    st.dataframe(hr, hide_index=True, width="stretch", column_config=cfg, height=35 * (len(hr) + 1) + 3)
    k = min(range(len(rows)), key=lambda j: abs(rows[j]["Model"] - 0.5))
    st.caption(f"Fair line, the one closest to a coin flip: **{rows[k]['Line'][7:]} saves** (over {rows[k]['Over']} / "
               f"under {rows[k]['Under']}). Hit rates count his starts only. Games in each window: "
               + " · ".join(f"{wl} {len(w)}" for wl, w in windows) + ".")

    # ---- game log
    st.markdown("#### Game log")
    if not len(gh):
        st.warning("The goalie game log hasn't loaded yet; it arrives with the next data sync (within 10 minutes).")
        return
    if not len(starts):
        st.info("No NHL starts in the data (2021-22 on).")
        return
    c1, c2, c3 = st.columns([3, 2, 1.3])
    stat = c1.segmented_control("Stat", ["Saves", "Shots against", "Goals against", "Save %"], default="Saves",
                                key="gl_stat") or "Saves"
    n = c2.segmented_control("Starts", [10, 20, 40, 82], default=20, key="gl_n") or 20
    col = {"Saves": "saves", "Shots against": "shots_against", "Goals against": "goals_against", "Save %": "sv_pct"}[stat]
    proj_v = {"saves": g["mean"], "shots_against": g["shots_faced"], "goals_against": g["goals_against"],
              "sv_pct": g["exp_sv_pct"]}[col]
    default = {"saves": rows[k]["Line"][7:], "shots_against": round(proj_v) + 0.5, "goals_against": 2.5, "sv_pct": 0.9}[col]
    line = c3.number_input("Line", value=float(default), step=0.01 if col == "sv_pct" else 0.5, key=f"gl_line_{col}")
    log = starts.tail(n).copy()
    log["value"] = log[col]
    log["game"] = log.date.dt.strftime("%m-%d") + np.where(log.is_home, " v ", " @ ") + log.opp.astype(str)
    log["result"] = np.where(log.value > line, f"over {line:g}", f"under {line:g}")
    bars = alt.Chart(log).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
        x=alt.X("game:N", sort=None, title=None, axis=alt.Axis(labelAngle=-45)),
        y=alt.Y("value:Q", title=stat, scale=alt.Scale(zero=col != "sv_pct")),
        color=alt.Color("result:N", scale=alt.Scale(domain=[f"over {line:g}", f"under {line:g}"], range=[PROJ, "#c3c2b7"]),
                        legend=alt.Legend(title=None, orient="top")),
        tooltip=[alt.Tooltip("game:N", title="Game"), alt.Tooltip("value:Q", title=stat, format=".3f" if col == "sv_pct" else ".0f"),
                 alt.Tooltip("shots_against:Q", title="Shots"), alt.Tooltip("decision:N", title="Decision")])
    rule = alt.Chart(pd.DataFrame({"y": [line]})).mark_rule(color=MUTED, strokeDash=[4, 3]).encode(y="y:Q")
    proj_rule = alt.Chart(pd.DataFrame({"y": [proj_v], "t": [f"{proj_v:.2f}"]})).mark_rule(color=ACT, strokeWidth=2).encode(
        y="y:Q", tooltip=alt.Tooltip("t:N", title="Tonight's projection"))
    st.altair_chart((bars + rule + proj_rule).properties(height=280), width="stretch")
    st.caption(f"Over {line:g} in {(log.value > line).mean():.0%} of his last {len(log)} starts. "
               f"Orange line: tonight's projection ({proj_v:.2f}). Dashed: the line.")

    t1, t2 = st.tabs(["Recent starts", "Season by season"])
    with t1:
        rec = mine.tail(15).iloc[::-1]
        st.dataframe(pd.DataFrame({
            "Date": rec.date.dt.strftime("%Y-%m-%d"), "Opp": np.where(rec.is_home, "vs ", "@ ") + rec.opp.astype(str),
            "Started": np.where(rec.starter, "yes", "relief"), "TOI": rec.toi / 60, "Shots": rec.shots_against,
            "Saves": rec.saves, "GA": rec.goals_against, "Save %": rec.sv_pct, "Decision": rec.decision}),
            hide_index=True, width="stretch", column_config={"TOI": st.column_config.NumberColumn(format="%.1f"),
                                                             "Save %": st.column_config.NumberColumn(format="%.3f")})
    with t2:
        reg = mine[mine.game_type == 2].groupby("season")
        ss = pd.DataFrame({"GP": reg.size(), "Starts": reg.starter.sum(), "Shots/gm": reg.shots_against.mean(),
                           "Saves/gm": reg.saves.mean(), "GA/gm": reg.goals_against.mean(),
                           "Save %": reg.saves.sum() / reg.shots_against.sum().replace(0, np.nan)})
        ss.index = [f"{y}-{str(y + 1)[2:]}" for y in ss.index]
        st.dataframe(ss.iloc[::-1], width="stretch", column_config={
            "Shots/gm": st.column_config.NumberColumn(format="%.1f"), "Saves/gm": st.column_config.NumberColumn(format="%.1f"),
            "GA/gm": st.column_config.NumberColumn(format="%.2f"), "Save %": st.column_config.NumberColumn(format="%.3f")})


def breakdown(r, p, i, s):
    mult = model.multipliers(p.loc[[i]], p.loc[[i]], s["coefs"])
    left, right = st.columns(2)
    with left:
        st.markdown("**Ice time** = team minutes × his share")
        st.table(pd.DataFrame({"team minutes": [r.team_pool / 60, r.team_pp / 60, r.team_sh / 60],
                               "his share": [r.share_ev, r.share_pp, r.share_sh],
                               "his minutes": [r.toi_ev_hat, r.toi_pp_hat, r.toi_sh_hat]},
                              index=["EV (all " + ("forward" if r.grp == "F" else "defence") + " EV minutes)", "PP", "PK"]).round(3))
        st.markdown("**Shots** = Σ minutes × shots per minute × context")
        st.table(pd.DataFrame({"minutes": [r.toi_ev_hat, r.toi_pp_hat, r.toi_sh_hat],
                               "shots / min": [r.sog_pm_ev, r.sog_pm_pp, r.sog_pm_sh]},
                              index=["EV", "PP", "PK"]).assign(shots=lambda d: d["minutes"] * d["shots / min"]).round(3))
    with right:
        st.markdown(f"**Goals** = shots {r.sog_hat:.2f} × shooting % {r.sh_pct:.1%} (regressed) × context = **{r.g_hat:.2f}**")
        st.markdown(f"**Assists** = Σ minutes × team goals per minute on ice × his assist share × context = **{r.a_hat:.2f}**")
        st.markdown(f"**PP points** = {r.toi_pp_hat:.2f} PP minutes × {r.gf_pm_pp * 60:.2f} team PP goals per 60 with "
                    f"him on (÷ 60) × his PP point share {r.ipp_pp:.0%} × context = **{r.ppp_hat:.2f}**")
        st.table(pd.DataFrame({"minutes": [r.toi_ev_hat, r.toi_pp_hat], "on-ice GF / 60": [r.gf_pm_ev * 60, r.gf_pm_pp * 60],
                               "assist share": [r.ashare_ev, r.ashare_pp]}, index=["EV", "PP"]).round(3))
        m = pd.concat({k: v.iloc[0] for k, v in mult.items()}, axis=1).rename(columns={"sog": "SOG", "g": "Goals", "a": "Assists"})
        st.markdown("**Context multipliers** (1.00 = no effect)")
        st.dataframe(m.round(3), width="stretch")


# ---------------------------------------------------------------- How it works

def page_model():
    doc = C.ROOT / "HOW_IT_WORKS.md"
    st.markdown(doc.read_text(encoding="utf-8") if doc.exists() else "")
    bt = C.DATA / "backtest" / "backtest_last.parquet"
    if not bt.exists():
        return
    st.subheader("Calibration check (walk-forward, regular season)")
    b = pd.read_parquet(bt)
    b = b[(b.game_type == 2) & (b.gp_16 > 0)]
    opts = {"1+ point": ("p_pts_1", b.points >= 1), "1+ goal": ("p_g_1", b.goals >= 1),
            "1+ assist": ("p_a_1", b.assists >= 1), "2+ SOG": ("p_sog_2", b.sog >= 2),
            "3+ SOG": ("p_sog_3", b.sog >= 3), "4+ SOG": ("p_sog_4", b.sog >= 4)}
    which = st.segmented_control("Event", list(opts), default="1+ point") or "1+ point"
    col, y = opts[which]
    bins = pd.cut(b[col], np.linspace(0, 1, 21))
    t = pd.DataFrame({"pred": b[col], "act": y.astype(float)}).groupby(bins, observed=True).agg(
        predicted=("pred", "mean"), actual=("act", "mean"), n=("act", "size")).reset_index(drop=True)
    t = t[t.n >= 100]
    diag = alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]})).mark_line(color=MUTED, strokeDash=[4, 4], strokeWidth=1).encode(
        x=alt.X("x:Q", title="Projected probability", scale=alt.Scale(domain=[0, 1])),
        y=alt.Y("y:Q", title="Actual frequency", scale=alt.Scale(domain=[0, 1])))
    line = alt.Chart(t).mark_line(color=PROJ, strokeWidth=2, point=alt.OverlayMarkDef(size=80, filled=True, color=PROJ)).encode(
        x="predicted:Q", y="actual:Q",
        tooltip=[alt.Tooltip("predicted:Q", format=".1%", title="Projected"), alt.Tooltip("actual:Q", format=".1%", title="Actual"),
                 alt.Tooltip("n:Q", format=",", title="Player-games")])
    st.altair_chart((diag + line).properties(height=380), width="stretch")
    st.caption("On the dashed diagonal = perfectly calibrated.")
    with st.expander("Current coefficients"):
        st.json(state()["coefs"] if state() else {})


# ---------------------------------------------------------------- Edges

EDGE_TYPES = {"Moneyline": "Moneyline", "Puck line": "Puck line", "Total": "Total", "PTS": "Points", "SOG": "Shots",
              "G": "Goals", "A": "Assists", "PPP": "PP points", "Saves": "Saves"}
ODDS_COOLDOWN_MIN = 2   # the Refresh odds button: stops double clicks and public-app spam


@st.cache_data(ttl=600, show_spinner=False)
def _odds(mtime: float) -> pd.DataFrame | None:
    return odds.load()


@st.cache_data(show_spinner="Comparing prices with the projections...", max_entries=8)
def _edges(odds_mtime: float, day: str, built_at: str, deps_key: str) -> tuple[pd.DataFrame, dict]:
    """The full comparison, cached until the odds, the projections or a lineup change, so
    filtering and sorting the table is instant."""
    s, deps = state(), deployments(day)
    return edges.compare(_odds(odds_mtime), s["games"], projections(day), s["coefs"]["alpha"], goalie_lines(day, deps))


def goalie_lines(day: str, deps: dict) -> pd.DataFrame:
    """Every plausible starter's P(over) at each saves line, for pricing saves props."""
    S = saves_state()
    if S is None:
        return pd.DataFrame()
    nm = names()
    rows = []
    for g in state()["games"][state()["games"].date == day].itertuples():
        for team in (g.away, g.home):
            d = deps.get(team, {})
            k = json.dumps([d.get("goalie"), d.get("goalie_status", ""), d.get("source", "")])
            for p in _team_goalies(S["built_at"], int(g.game_id), team, k):
                rows.append({"game_id": int(g.game_id), "team": team, "goalie": nm.get(p["goalie"], p["name"]),
                             "mean": p["mean"], "p_over": {float(c[7:]): v for c, v in p.items() if c.startswith("p_over_")}})
    return pd.DataFrame(rows)


CHECK_MARKETS = {"Points": ("PTS", "pts_hat", "pts"), "Shots": ("SOG", "sog_hat", "sog"), "Goals": ("G", "g_hat", "g"),
                 "Assists": ("A", "a_hat", "a"), "PP points": ("PPP", "ppp_hat", "ppp"), "Saves": ("Saves", None, None),
                 "Moneyline": ("Moneyline", None, None), "Puck line": ("Puck line", None, None), "Total": ("Total", None, None)}
PLAY_EDGE = 0.08   # the backtest: below about 8 points the books' margin eats the edge


def _odds_in(label: str, default: str, key: str, help: str | None = None) -> float | None:
    raw = st.text_input(label, value=default, key=key, help=help).strip().replace("+", "")
    try:
        v = float(raw)
    except ValueError:
        return None
    return v if abs(v) >= 100 else None


def check_line(day: str, s: dict, expanded: bool = False):
    """Price one line typed in from a book against tonight's projection."""
    with st.expander("Check a line from a book", expanded=expanded):
        p = projections(day)
        games = s["games"][s["games"].date == day]
        if p.empty or games.empty:
            st.info("No projections for this date.")
            return
        c1, c2, c3 = st.columns([2, 2, 3])
        market = c1.selectbox("Market", list(CHECK_MARKETS), key="chk_market")
        glab = {r.game_id: f"{r.away} @ {r.home}" for r in games.itertuples()}
        gid = c2.selectbox("Game", list(glab), format_func=glab.get, key="chk_game")
        g = games[games.game_id == gid].iloc[0]
        kind, col, akey = CHECK_MARKETS[market]
        d1, d2, d3, d4 = st.columns(4)
        lam = p[p.game_id == gid].groupby("team").g_hat.sum()
        proj_text, win_push = "", None
        if market in ("Moneyline", "Puck line", "Total"):
            if market == "Total":
                side = d1.segmented_control("Side", ["Over", "Under"], default="Over", key="chk_ou") or "Over"
                line = d2.number_input("Line", value=6.5, step=0.5, key="chk_tot")
                win_push = linecheck.game(lam[g.home], lam[g.away], "Total", True, line, side == "Over")
                proj_text = f"{lam[g.home] + lam[g.away]:.2f} total goals projected"
                bet = f"{side} {line:g} goals, {glab[gid]}"
            else:
                team = c3.segmented_control("Team", [g.away, g.home], default=g.away, key="chk_team") or g.away
                line = 0.0 if market == "Moneyline" else d2.number_input(
                    "Handicap", value=-1.5, step=1.0, key="chk_pl", help="This team's puck line, e.g. -1.5 or +1.5")
                win_push = linecheck.game(lam[g.home], lam[g.away], market, team == g.home, line)
                opp = g.home if team == g.away else g.away
                proj_text = f"projected score {team} {lam[team]:.2f} – {lam[opp]:.2f} {opp}"
                bet = f"{team} {'moneyline' if market == 'Moneyline' else f'{line:+g}'}"
                side = None
        elif market == "Saves":
            gl = [x for x in day_goalies(day, deployments(day)) if x["game_id"] == gid]
            if not gl:
                st.info("No goalie projections for this game.")
                return
            nm = names()
            j = c3.selectbox("Goalie", range(len(gl)), format_func=lambda j: f"{nm.get(gl[j]['goalie'], gl[j]['name'])} "
                             f"· {gl[j]['team']} ({gl[j]['p_start']:.0%} to start)", key="chk_goalie")
            q = gl[j]
            side = d1.segmented_control("Side", ["Over", "Under"], default="Over", key="chk_sv_ou") or "Over"
            line = d2.number_input("Line", value=float(round(q["median"])) + 0.5, step=1.0, key="chk_sv")
            win_push = linecheck.from_pmf(np.asarray(q["pmf"]), line, side == "Over")
            proj_text = f"{q['mean']:.1f} saves projected if he starts ({q['shots_faced']:.1f} shots)"
            bet = f"{nm.get(q['goalie'], q['name'])} {side.lower()} {line:g} saves"
        else:
            pg = p[p.game_id == gid].sort_values(col, ascending=False)
            who = c3.selectbox("Player", pg.index, format_func=lambda i: f"{pg.at[i, 'name']} · {pg.at[i, 'team']}",
                               key="chk_player")
            r = pg.loc[who]
            side = d1.segmented_control("Side", ["Over", "Under"], default="Over", key="chk_p_ou") or "Over"
            default = {"Points": 0.5, "Shots": 2.5, "Goals": 0.5, "Assists": 0.5, "PP points": 0.5}[market]
            line = d2.number_input("Line", value=default, step=0.5, key=f"chk_line_{market}")
            win_push = linecheck.skater(float(r[col]), s["coefs"]["alpha"][akey], line, side == "Over")
            proj_text = f"{r[col]:.2f} {market.lower()} projected ({r.toi_hat:.1f} min)"
            bet = f"{r['name']} {side.lower()} {line:g} {market.lower()}"
        with d3:
            price = _odds_in("Book odds", "-110", "chk_odds", "American odds, e.g. -115 or +130")
        with d4:
            other = _odds_in("Other side's odds (optional)", "", "chk_other",
                             "Adding the other side's price removes the book's margin from its probability")
        if price is None:
            st.warning("Enter the book's odds as an American price, e.g. -115 or +130.")
            return
        ev = linecheck.evaluate(*win_push, price, other)
        if kind == "Moneyline":
            side = "Favourite" if ev["book_p"] >= 0.5 else "Underdog"
        elif kind == "Puck line":
            side = "Favourite -" if line < 0 else "Underdog +"
        a, b, c, d, e = st.columns(5)
        a.metric("Our probability", f"{ev['p']:.1%}", border=True,
                 help="Given no push" if ev["push"] > 0 else None)
        b.metric("Our fair odds", american(ev["p"]), border=True, help="The price with no margin, by the model")
        c.metric("Book probability", f"{ev['book_p']:.1%}", border=True,
                 help="Margin removed" if ev["novig"] else "Includes the book's margin; add the other side to remove it")
        d.metric("Edge", f"{100 * ev['edge']:+.1f} pts", border=True, help="Our probability minus the book's")
        e.metric("EV per $100", f"{ev['ev']:+.2f}", border=True, help="Expected profit at the book's odds, by the model")
        # how this market and side did in the 2024-26 backtest at an 8+ point edge (closing lines)
        bt = results("Backtest 2024-26")
        hist_txt, roi8 = "", None
        if len(bt):
            h8 = bt[(bt.snap == "close") & (bt.kind == kind) & (bt.side == side) & (bt.bucket.map(BUCKET_LO) >= PLAY_EDGE)]
            if h8.bets.sum() >= 20:
                roi8 = h8.pnl.sum() / h8.bets.sum()
                hist_txt = (f"Backtest, {EDGE_TYPES[kind].lower()} {side.lower()} bets with an 8+ point edge: "
                            f"{roi8:+.1%} ROI on {int(h8.bets.sum()):,} bets.")
        msg = f"**{bet}** at {price:+.0f}: {proj_text}."
        if ev["ev"] <= 0 or ev["edge"] < 0.02:
            st.error(f"No edge. {msg} The book's price is at or above the model's fair value.")
        elif ev["edge"] < PLAY_EDGE:
            st.warning(f"Small edge. {msg} Historically, edges under about 8 points have not beaten the books' margin.")
        elif roi8 is not None and roi8 < 0:
            st.warning(f"Edge, but this market has not paid historically. {msg} {hist_txt}")
        else:
            st.success(f"Edge worth a look. {msg} Check the lineup and goalie news before playing.")
        if hist_txt:
            st.caption(hist_txt + (f" Push chance {ev['push']:.0%}, refunded." if ev["push"] > 0 else ""))


def page_edges():
    s = state() or no_state()
    day = pick_day(s)
    st.title(f"Edges · {day}")
    o = _odds(odds.ODDS_FILE.stat().st_mtime) if odds.ODDS_FILE.exists() else None
    head, btn = st.columns([5, 1], vertical_alignment="bottom")
    if odds.api_key() is not None:   # odds update only when someone clicks (no automatic fetches)
        last = pd.Timestamp(o.fetched_at.iloc[0]) if o is not None and len(o) else None
        wait = last is not None and pd.Timestamp.now(tz="UTC") - last < pd.Timedelta(minutes=ODDS_COOLDOWN_MIN)
        if btn.button("Refresh odds", type="primary", disabled=wait, width="stretch",
                      help=f"Fetch the latest sportsbook prices from OpticOdds now (about one API call per 5 games). "
                           f"At most once every {ODDS_COOLDOWN_MIN} minutes."):
            with st.spinner("Fetching sportsbook odds..."):
                try:
                    odds.fetch(verbose=False)
                except Exception as e:
                    st.error(f"Could not fetch odds: {e}")
                    st.stop()
            _odds.clear()
            st.rerun()
    check_line(day, s, expanded=o is None or o.empty)
    if o is None or o.empty:
        head.info("No sportsbook odds yet. Click **Refresh odds** to fetch them (needs `opticodds_api_key` in the "
                  "app's secrets, or in `.streamlit/secrets.toml` locally).")
        return
    p = projections(day)
    if p.empty:
        st.info("Nothing to project for this date.")
        return
    deps = deployments(day)
    df, stats = _edges(odds.ODDS_FILE.stat().st_mtime, day, s["built_at"], json.dumps(deps, sort_keys=True, default=str))
    head.caption("Sportsbook prices compared with the model's own probability for the same bet. "
                 "Odds update only when you click Refresh odds.")
    if df.empty:
        st.info("No pregame odds match this date's games yet.")
        return
    df = df.assign(type=df.kind.map(EDGE_TYPES))
    shown = df[df.novig.notna()].sort_values("dec", ascending=False).drop_duplicates(["game_id", "market", "bet"])
    t_up = pd.Timestamp(o.fetched_at.iloc[0]).tz_convert("America/New_York")
    a, b, c, d = st.columns(4)
    a.metric("Odds updated", t_up.strftime("%I:%M %p ET").lstrip("0"), border=True, help=t_up.strftime("%a %b %d, %Y"))
    b.metric("Sportsbooks", f"{o.book.nunique()}", border=True)
    c.metric("Bets priced", f"{len(shown):,}", border=True, help="Distinct bets with both sides posted")
    d.metric("Positive EV", f"{(shown.ev > 0).sum():,}", border=True,
             help="Bets where the model's probability beats the best available price")

    c1, c2, c3, c4 = st.columns(4)
    game_order = df.sort_values("commence_time").game.drop_duplicates().tolist()
    games_sel = c1.multiselect("Game", game_order, key="edge_games")
    pool = df[df.game.isin(games_sel)] if games_sel else df
    plab = pool[pool.team.notna()].drop_duplicates("subject")    # players with a prop (skaters and goalies)
    plab = dict(zip(plab.subject, plab.subject + " (" + plab.team.astype(str) + ")"))
    players = c2.multiselect("Player", sorted(plab), format_func=plab.get, key="edge_players",
                             placeholder="Search a player",
                             help="Every line posted for him, whatever its EV")
    kind = c3.multiselect("Bet type", [k for k in EDGE_TYPES.values() if k in set(df.type)], key="edge_type")
    books = c4.multiselect("Sportsbooks", sorted(df.book.unique()), key="edge_books")
    c5, c6, c7, c8, _ = st.columns([1, 1, 1, 1, 2], vertical_alignment="bottom")
    min_ev = c5.number_input("Min EV %", value=0.0, step=1.0, format="%.0f", key="edge_min",
                             disabled=bool(players), help="Ignored while a player is picked")
    sort = c6.selectbox("Sort by", ["EV", "Edge"], key="edge_sort",
                        help="EV ranks by expected profit (it favours long odds); Edge by the probability gap")
    best = c7.toggle("Best price", value=True, key="edge_best", help="One row per bet, at the book with the best odds")
    both = c8.toggle("Both sides", value=True, key="edge_both",
                     help="Only bets the book posts both sides of, so its probability can be read without its margin. "
                          "Off adds one-sided longshots, where the model is least reliable.")
    q = df[df.game.isin(games_sel)] if games_sel else df
    q = q[q.subject.isin(players)] if players else q
    q = q[q.type.isin(kind)] if kind else q
    q = q[q.book.isin(books)] if books else q
    q = q[q.novig.notna()] if both else q
    if best:
        q = q.sort_values("dec", ascending=False).drop_duplicates(["game_id", "market", "bet"])
    q = q if players else q[q.ev * 100 >= min_ev]
    q = q.sort_values("ev" if sort == "EV" else "edge", ascending=False)
    t = pd.DataFrame({"Game": q.game, "Bet": q.subject, "Line": q.line, "Projection": q.proj_text,
                      "Book": q.book, "Book odds": [f"{x:+.0f}" for x in q.price], "Our odds": q.p_model.map(american),
                      "Book %": (q.novig * 100).round(), "Our %": (q.p_model * 100).round(),
                      "Edge": q.edge * 100, "EV": q.ev * 100})
    bar = lambda h: st.column_config.ProgressColumn(format="%d%%", min_value=0, max_value=100, width="small", help=h)
    st.dataframe(t, hide_index=True, width="stretch", height=720, column_config={
        "Game": st.column_config.TextColumn(width="small", pinned=True),
        "Bet": st.column_config.TextColumn(width="medium", pinned=True, help="The player, the team, or the game total"),
        "Line": st.column_config.TextColumn(help="The bet as the book lists it"),
        "Projection": st.column_config.TextColumn(
            help="The model's number for this bet: the player's projected stat, the projected total, or the "
                 "projected score (this team first)"),
        "Book": st.column_config.TextColumn(width="small"),
        "Book odds": st.column_config.TextColumn(width="small", help="The book's price"),
        "Our odds": st.column_config.TextColumn(width="small", help="The model's fair price (no margin)"),
        "Book %": bar("The book's probability with its margin removed"),
        "Our %": bar("The model's probability (saves: as if he starts)"),
        "Edge": st.column_config.NumberColumn(format="%+.1f", width="small",
                                              help="Our % minus Book %, in percentage points"),
        "EV": st.column_config.NumberColumn(format="%+.1f%%", width="small",
                                            help="Expected profit per $100 bet at the book's odds, by the model")})
    st.caption("**Book %** the book's probability without its margin · **Our %** the model's · **Edge** Our % − Book % · "
               "**EV** expected profit per $100 at the book's odds. Saves are priced as if the goalie starts. Books often "
               "know about late scratches and line changes first, so check the lineup before trusting a big edge. "
               f"Matched {stats.get('skater props matched', 0) + stats.get('saves props matched', 0):,} of "
               f"{stats.get('skater props', 0) + stats.get('saves props', 0):,} player prices to tonight's projections.")
    st.download_button("Download CSV", t.to_csv(index=False), f"edges_{day}.csv", "text/csv")


# ---------------------------------------------------------------- Results

RESULT_FILES = {"Live (this season)": tracking.RESULTS_LIVE, "Backtest 2024-26": tracking.RESULTS_BACKTEST}
EDGE_MIN = {"All bets": None, "Edge 0%+": 0.0, "Edge 2%+": 0.02, "Edge 5%+": 0.05, "Edge 8%+": 0.08, "Edge 12%+": 0.12}
BUCKET_LO = {lab: lo for lo, _, lab in tracking.BUCKETS}


@st.cache_data(ttl=600, show_spinner=False)
def _results(path: str, mtime: float) -> pd.DataFrame:
    return pd.read_parquet(path)


def results(name: str) -> pd.DataFrame:
    p = RESULT_FILES[name]
    if not p.exists():   # locally, the published copy (the cloud app syncs it like the projections)
        try:
            cloud._download(p.name, p)
        except Exception:
            return pd.DataFrame()
    return _results(str(p), p.stat().st_mtime)


def page_results():
    st.title("Results")
    st.caption("The model's projections graded against the sportsbooks' opening (10 AM ET) and closing lines: "
               "1-unit bets at the best price of DraftKings, FanDuel, BetMGM, Caesars and Fanatics.")
    c1, c2, c3, c4 = st.columns([2, 1, 2, 1], vertical_alignment="bottom")
    first = "Live (this season)" if not results("Live (this season)").empty else "Backtest 2024-26"
    src = c1.segmented_control("Results", list(RESULT_FILES), default=first, key="res_src") or first
    snap = c2.segmented_control("Price", ["Close", "Open"], default="Close", key="res_snap") or "Close"
    r = results(src)
    if r.empty:
        st.info("No graded games yet. Each morning's refresh grades the previous day's games against their "
                "opening and closing lines (it needs the `ODDS_API_KEY` secret in the GitHub repo).")
        return
    r = r[r.snap == snap.lower()]
    kinds = [k for k in EDGE_TYPES if k in set(r.kind)]
    kind = c3.multiselect("Bet type", kinds, format_func=EDGE_TYPES.get, key="res_kind")
    th = EDGE_MIN[c4.selectbox("Bets", list(EDGE_MIN), index=4, key="res_edge",
                               help="Only bets where the model's probability beat the books' no-vig "
                                    "probability by at least this much")]
    q = r[r.kind.isin(kind)] if kind else r
    sel = q if th is None else q[q.bucket.map(BUCKET_LO) >= th]
    bets, wins, pnl = sel.bets.sum(), sel.wins.sum(), sel.pnl.sum()
    acc = q[q.primary]
    a, b, c, d = st.columns(4)
    a.metric("Bets", f"{int(bets):,}", border=True, help=f"{q.date.min()} to {q.date.max()}" if len(q) else None)
    b.metric("Win rate", f"{wins / bets:.1%}" if bets else "—", border=True)
    c.metric("ROI", f"{pnl / bets:+.1%}" if bets else "—", border=True, help="Profit per unit staked")
    d.metric("Profit", f"{pnl:+,.1f} u" if bets else "—", border=True, help="Units won at 1 unit per bet")

    def table(by: list[str]) -> pd.DataFrame:
        g = sel.groupby(by, observed=True)[["bets", "wins", "pnl"]].sum().reset_index()
        g = g[g.bets > 0]
        g["Win %"] = 100 * g.wins / g.bets
        g["ROI"] = 100 * g.pnl / g.bets
        return g

    roi = st.column_config.NumberColumn(format="%+.1f%%", width="small")
    left, right = st.columns([3, 2], gap="large")
    with left:
        st.markdown("**By market and side**")
        t = table(["kind", "side"])
        t = t.assign(kind=t.kind.map(EDGE_TYPES)).rename(columns={"kind": "Market", "side": "Side", "bets": "Bets",
                                                                  "pnl": "Profit (u)"})
        t["order"] = t.Market.map({v: i for i, v in enumerate(EDGE_TYPES.values())})
        st.dataframe(t.sort_values(["order", "Side"])[["Market", "Side", "Bets", "Win %", "ROI", "Profit (u)"]],
                     hide_index=True, width="stretch", height=35 * (len(t) + 1) + 3, column_config={
                         "Win %": st.column_config.NumberColumn(format="%.1f%%", width="small"), "ROI": roi,
                         "Profit (u)": st.column_config.NumberColumn(format="%+.1f", width="small")})
    with right:
        st.markdown("**By edge**")
        e = q.groupby("bucket", observed=True)[["bets", "wins", "pnl"]].sum().reset_index()
        e["lo"] = e.bucket.map(BUCKET_LO)
        e = e.sort_values("lo")
        e = pd.DataFrame({"Model edge": e.bucket, "Bets": e.bets, "Win %": 100 * e.wins / e.bets, "ROI": 100 * e.pnl / e.bets})
        st.dataframe(e, hide_index=True, width="stretch", column_config={
            "Win %": st.column_config.NumberColumn(format="%.1f%%", width="small"), "ROI": roi})
        if acc.bets.sum():
            st.caption(f"Accuracy (log loss, lower is better): model **{acc.ll_model.sum() / acc.bets.sum():.4f}** · "
                       f"books' no-vig **{acc.ll_mkt.sum() / acc.bets.sum():.4f}**, on {int(acc.bets.sum()):,} bets "
                       "(all edges, one side per bet)")
    daily = sel.groupby("date").pnl.sum().sort_index().cumsum().reset_index()
    if len(daily) > 1:
        daily["date"] = pd.to_datetime(daily.date)
        ch = alt.Chart(daily).mark_line(color=PROJ, strokeWidth=2).encode(
            x=alt.X("date:T", title=None), y=alt.Y("pnl:Q", title="Cumulative profit (units)"),
            tooltip=[alt.Tooltip("date:T"), alt.Tooltip("pnl:Q", format="+.1f", title="Profit")])
        zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color=MUTED, strokeDash=[4, 4]).encode(y="y:Q")
        st.markdown("**Running profit**")
        st.altair_chart((zero + ch).properties(height=260), width="stretch")
    st.caption("Edge = the model's probability minus the books' consensus no-vig probability. Saves are graded only "
               "when the goalie started (books void the prop otherwise); pushes and void bets are left out. "
               "Backtest = 2024-26 game lines and a 2025-26 prop sample, priced with last game's lineups.")


# ---------------------------------------------------------------- trends

TEAM_GAMES = C.LAKE / "team_games.parquet"


@st.cache_data
def _team_games(path: str, mtime: float) -> pd.DataFrame:
    t = pd.read_parquet(path)
    return t.assign(date=pd.to_datetime(t.date), team=t.team.astype(str), opp=t.opp.astype(str),
                    last_period_type=t.last_period_type.astype(str))


def team_games() -> pd.DataFrame:
    return _team_games(str(TEAM_GAMES), TEAM_GAMES.stat().st_mtime) if TEAM_GAMES.exists() else pd.DataFrame()


def careers() -> pd.DataFrame | None:
    return pd.read_parquet(trends.CAREERS) if trends.CAREERS.exists() else None


@st.cache_data(show_spinner="Finding streaks, slumps and milestones...", max_entries=8)
def _trends(day: str, built_at: str, deps_key: str, hist_mtime: float) -> dict:
    s, deps, h, p = state(), deployments(day), hist(), projections(day)
    starters = pd.DataFrame([{"goalie": g["goalie"], "team": g["team"]} for g in day_goalies(day, deps)
                             if g["role"] == "Starter"], columns=["goalie", "team"])
    gg = goalie_games()
    return {"form": trends.form(h, p, day), "streaks": trends.streaks(h, p, day), "matchups": trends.matchups(h, p, day),
            "milestones": trends.milestones(careers(), h, p, starters, day, s["coefs"]["alpha"]),
            "goalies": trends.goalie_form(gg, starters.goalie, day) if len(gg) and len(starters) else pd.DataFrame(),
            "starters": starters}


def _tonight(opp, is_home) -> list[str]:
    return [f"{'vs' if h else '@'} {o}" for o, h in zip(opp, is_home)]


PCT = dict(format="%d%%", min_value=0, max_value=100, width="small")
STREAK_P = {"Point streak": "p_pts_1", "Goal streak": "p_g_1", "Assist streak": "p_a_1", "Multi-point streak": "p_pts_2",
            "3+ shots streak": "p_sog_3", "Goal drought": "p_g_1", "Point drought": "p_pts_1", "Shot slump (under 2)": "p_sog_2"}
STREAK_BET = {"p_pts_1": "1+ point", "p_g_1": "a goal", "p_a_1": "1+ assist", "p_pts_2": "2+ points", "p_sog_3": "3+ shots",
              "p_sog_2": "2+ shots"}
REGULAR_PTS = 0.35   # usual points per game for a skater to count as a regular (Hot & cold)


def page_trends():
    s = state() or no_state()
    day = pick_day(s)
    st.title(f"Trends · {day}")
    p = projections(day)
    if p.empty:
        st.info("Nothing to project for this date.")
        return
    deps = deployments(day)
    T = _trends(day, s["built_at"], json.dumps(deps, sort_keys=True, default=str),
                (C.LAKE / "player_games.parquet").stat().st_mtime)
    nm = names()
    games = s["games"][s["games"].date == day]
    opp = {t: (o, h) for g in games.itertuples() for t, o, h in ((g.home, g.away, True), (g.away, g.home, False))}
    glab = {f"{g.away} @ {g.home}": {g.away, g.home} for g in games.itertuples()}
    st.caption("Streaks, slumps, matchups and milestones for tonight's players, from games before this date. "
               "Context only: none of it changes a projection (recent form added nothing in testing once ice time "
               "and longer-run rates are known), so each table shows what the model expects tonight next to it.")
    c1, c2, _ = st.columns([2, 2, 3])
    sel = c1.multiselect("Game", list(glab), key="tr_games", placeholder="All games")
    teams = set().union(*(glab[g] for g in sel)) if sel else set(opp)
    tsel = c2.multiselect("Team", sorted(teams), key="tr_teams", placeholder="All teams")
    teams = set(tsel) if tsel else teams

    def keep(df: pd.DataFrame) -> pd.DataFrame:
        return df[df.team.isin(teams)] if len(df) else df

    fm, sk, mu, ms = keep(T["form"]), keep(T["streaks"]), keep(T["matchups"]), T["milestones"]
    if len(ms):
        team_of = dict(zip(p.player, p.team)) | dict(zip(T["starters"].goalie, T["starters"].team))
        ms = ms.assign(team=ms.player.map(team_of))
        ms = ms[ms.team.isin(teams)]

    # ---- storylines: who, in the label; the number, as the value
    a, b, c, d = st.columns(4)
    tile = dict(delta_color="off", delta_arrow="off", border=True)
    hs = sk[(sk.kind == "Hot") & sk.streak.str.contains("Point streak")] if len(sk) else sk
    if len(hs):
        r = hs.sort_values(["games", "pts_hat"]).iloc[-1]
        a.metric(f"Longest point streak · {r['name']}", f"{r.games} games", r.during.rsplit(",", 1)[0], **tile)
    else:
        a.metric("Longest point streak", "—", border=True)
    reg = fm[fm.base_pts >= REGULAR_PTS] if len(fm) else fm
    for col, title, asc in ((b, "Hottest regular", False), (c, "Coldest regular", True)):
        if len(reg):
            r = reg.sort_values("heat", ascending=asc).iloc[0]
            col.metric(f"{title} · {r['name']}", f"{int(r.pts)} PTS in {int(r.gp)}",
                       f"usually {r.base_pts * r.gp:.1f} · {int(r.sog)} SOG (usually {r.base_sog * r.gp:.0f})", **tile)
        else:
            col.metric(title, "—", border=True)
    car = ms[(ms.scope == "Career") & (ms.stat != "gp") & (ms.target > 1)] if len(ms) else ms
    if len(car):
        r = car.sort_values(["needs", "target"], ascending=[True, False]).iloc[0]
        chance = f" · {r.p_tonight:.0%} tonight" if np.isfinite(r.p_tonight) else ""
        d.metric(f"Milestone watch · {nm.get(r.player, str(r.player))}", r.milestone.replace(" career", ""),
                 f"needs {r.needs}{chance}", **tile)
    else:
        d.metric("Milestone watch", "—", border=True)

    tabs = st.tabs(["Hot & cold", "Streaks", "Matchups", "Milestones", "Hit rates", "Teams"])
    with tabs[0]:
        trends_form(fm, T["goalies"], T["starters"], teams, nm, opp)
    with tabs[1]:
        trends_streaks(sk)
    with tabs[2]:
        trends_matchups(mu, games, teams, day)
    with tabs[3]:
        trends_milestones(ms, nm, opp)
    with tabs[4]:
        trends_hits(keep(p), s, day)
    with tabs[5]:
        trends_teams(games, teams, opp, day)


def trends_form(fm: pd.DataFrame, gf: pd.DataFrame, starters: pd.DataFrame, teams: set, nm: pd.Series, opp: dict):
    if fm.empty:
        st.info("Not enough games yet this season.")
        return
    depth = st.toggle("Include depth players", value=False, key="tr_depth",
                      help=f"Off: only skaters who usually score {REGULAR_PTS} points a game or more. A small usual rate "
                           "makes a short hot run look extreme.")
    f = fm if depth else fm[fm.base_pts >= REGULAR_PTS]

    def table(x: pd.DataFrame):
        st.dataframe(pd.DataFrame({
            "Player": x["name"], "Team": x.team, "Tonight": _tonight(x.opp, x.is_home),
            "Last 10": [f"{int(g)} G, {int(a)} A, {int(sg)} SOG" for g, a, sg in zip(x.g, x.a, x.sog)],
            "PTS/gm": x.pts_gm, "Usual PTS": x.base_pts, "SOG/gm": x.sog_gm, "Usual SOG": x.base_sog,
            "TOI": x.toi, "Usual TOI": x.base_toi, "Heat": x.heat, "Proj PTS": x.pts_hat, "What's behind it": x.driver}),
            hide_index=True, width="stretch", column_config={
                "Player": st.column_config.TextColumn(pinned=True), "Team": st.column_config.TextColumn(width="small"),
                **{c: st.column_config.NumberColumn(format="%.2f", width="small")
                   for c in ("PTS/gm", "Usual PTS", "SOG/gm", "Usual SOG", "Proj PTS")},
                **{c: st.column_config.NumberColumn(format="%.1f", width="small") for c in ("TOI", "Usual TOI")},
                "What's behind it": st.column_config.TextColumn(width="large"),
                "Heat": st.column_config.NumberColumn(format="%+.1f", width="small",
                                                      help="Points and shots in the last 10 against his usual rates, in "
                                                           "standard deviations (0 = as usual)")})
    st.markdown("#### Running hot")
    table(f.sort_values("heat", ascending=False).head(15))
    st.markdown("#### Running cold")
    table(f.sort_values("heat").head(15))
    st.caption("Usual = this season and last season's regular season before the last 10 games (at least 20 games). "
               "What's behind it: ice time, shot volume or shooting % where they differ clearly from usual. A run built "
               "on shooting % alone tends to fade; one built on more ice time or more shots tends to last.")

    if len(gf):
        st.markdown("#### Goalies, last 5 starts")
        g = gf.merge(starters, on="goalie")
        g = g[g.team.isin(teams)].sort_values("sv_diff", ascending=False)
        st.dataframe(pd.DataFrame({
            "Goalie": g.goalie.map(nm), "Team": g.team,
            "Tonight": [("vs " if opp[t][1] else "@ ") + opp[t][0] for t in g.team],
            "Record": g.record, "Streak": g.streak, "SV% last 5": g.sv_last, "SV% season": g.sv_season,
            "SV% last season": g.sv_prev, "vs usual": g.sv_diff * 1000, "GAA last 5": g.gaa_last, "Saves/gm": g.saves_last}),
            hide_index=True, width="stretch", column_config={
                **{c: st.column_config.NumberColumn(format="%.3f", width="small")
                   for c in ("SV% last 5", "SV% season", "SV% last season")},
                "vs usual": st.column_config.NumberColumn(format="%+.0f", width="small",
                                                          help="Save % in the last 5 starts minus this season's (last "
                                                               "season's before 5 starts), in points: .010 = 10"),
                "GAA last 5": st.column_config.NumberColumn(format="%.2f", width="small"),
                "Saves/gm": st.column_config.NumberColumn(format="%.1f", width="small")})
        st.caption("Tonight's expected starters (see Lines & goalies). Record: W-L-OT in those starts.")


def trends_streaks(sk: pd.DataFrame):
    if sk.empty:
        st.info("No active streaks of note.")
        return

    def table(x: pd.DataFrame):
        pcol = [STREAK_P[v.split(" · ")[0].rsplit(" ", 1)[0]] for v in x.streak]
        pr = [getattr(r, c) for r, c in zip(x.itertuples(), pcol)]
        st.dataframe(pd.DataFrame({
            "Player": x["name"], "Team": x.team, "Tonight": _tonight(x.opp, x.is_home), "Streak": x.streak, "Games": x.games,
            "Since": pd.to_datetime(x.since).dt.strftime("%b %d"), "During it": x.during,
            "Tonight's bet": [STREAK_BET[c] for c in pcol], "Model %": [round(100 * v) for v in pr],
            "Fair odds": [american(v) for v in pr]}),
            hide_index=True, width="stretch", column_config={
                "Player": st.column_config.TextColumn(pinned=True), "Team": st.column_config.TextColumn(width="small"),
                "Games": st.column_config.NumberColumn(width="small"), "Model %": st.column_config.ProgressColumn(**PCT),
                "Fair odds": st.column_config.TextColumn(width="small")})
    st.markdown("#### Active streaks")
    table(sk[sk.kind == "Hot"].sort_values(["games", "pts_hat"], ascending=False))
    st.markdown("#### Slumps")
    cold = sk[sk.kind == "Cold"]
    if len(cold):
        table(cold.sort_values(["games", "pts_hat"], ascending=False))
    else:
        st.caption("No slumps of note.")
    st.caption("Counted in consecutive games played this season. A slump is listed only for a player who usually "
               "produces (a goal drought from 0.30 goals a game, a point drought from 0.65 points a game, a shot slump "
               "from 3 shots a game), at least 3 games long and 2 goals or points below his usual pace. Model % is "
               "the projection's chance of the streak's bet tonight.")


def trends_matchups(mu: pd.DataFrame, games: pd.DataFrame, teams: set, day: str):
    if len(mu):
        def table(x: pd.DataFrame):
            st.dataframe(pd.DataFrame({
                "Player": x["name"], "Team": x.team, "Tonight": _tonight(x.opp, x.is_home), "GP vs": x.gp_vs.astype(int),
                "PTS vs": x.pts_vs.astype(int), "G vs": x.g_vs.astype(int), "PTS/gm vs": x.pts_gm_vs,
                "PTS/gm others": x.pts_gm, "SOG/gm vs": x.sog_gm_vs, "SOG/gm others": x.sog_gm,
                "Last meeting": x["last"], "Proj PTS": x.pts_hat}),
                hide_index=True, width="stretch", column_config={
                    "Player": st.column_config.TextColumn(pinned=True), "Team": st.column_config.TextColumn(width="small"),
                    **{c: st.column_config.NumberColumn(format="%.2f")
                       for c in ("PTS/gm vs", "PTS/gm others", "SOG/gm vs", "SOG/gm others", "Proj PTS")}})
        st.markdown("#### Plays well against tonight's opponent")
        table(mu[mu.score > 0].sort_values("score", ascending=False).head(15))
        st.markdown("#### Struggles against tonight's opponent")
        table(mu[mu.score < 0].sort_values("score").head(15))
        st.caption("Every game since 2018-19 against tonight's opponent (at least 5), next to his rate against everyone "
                   "else over the same years, ranked by the gap shrunk toward zero for small samples. Matchup history "
                   "mostly reflects who was on both teams back then, so treat it as colour rather than signal.")
    tg = team_games()
    opts = [g for g in games.itertuples() if {g.home, g.away} & teams]
    if tg.empty or not opts:
        return
    st.markdown("#### Head to head")
    g = st.selectbox("Game", opts, format_func=lambda g: f"{g.away} @ {g.home}", key="tr_h2h")
    m = trends.head_to_head(tg, g.home, g.away, day, n=10)
    if m.empty:
        st.info("No meetings since 2018-19.")
        return
    hw = int((m.goals > m.goals_against).sum())
    tot = m.goals + m.goals_against
    a, b, c = st.columns(3)
    a.metric(f"Last {len(m)} meetings", f"{g.home} {hw}–{len(m) - hw} {g.away}", border=True)
    b.metric("Goals per game", f"{tot.mean():.1f}", f"over 5.5 in {(tot > 5.5).mean():.0%}", delta_color="off", border=True)
    c.metric("Went to OT / SO", f"{m.last_period_type.isin(['OT', 'SO']).mean():.0%}", border=True)
    st.dataframe(pd.DataFrame({
        "Date": m.date.dt.strftime("%Y-%m-%d"), "Venue": np.where(m.is_home, f"at {g.home}", f"at {g.away}"),
        "Score": [f"{g.home} {x}–{y} {g.away}" for x, y in zip(m.goals, m.goals_against)],
        "Winner": np.where(m.goals > m.goals_against, g.home, g.away),
        "Final": m.last_period_type.replace({"REG": ""}), "Total": tot.astype(int),
        f"{g.home} SOG": m.sog, f"{g.away} SOG": m.sog_against}), hide_index=True, width="stretch")


def trends_milestones(ms: pd.DataFrame, nm: pd.Series, opp: dict):
    if ms.empty:
        st.info("No milestones within reach tonight.")
        return
    scope = st.segmented_control("Show", ["All", "Career", "Season"], default="All", key="tr_ms") or "All"
    x = ms if scope == "All" else ms[ms.scope == scope]
    x = x.assign(order=x.stat.eq("gp")).sort_values(["needs", "order", "p_tonight"], ascending=[True, True, False])
    st.dataframe(pd.DataFrame({
        "Player": x.player.map(nm), "Team": x.team,
        "Tonight": [("vs " if opp[t][1] else "@ ") + opp[t][0] for t in x.team],
        "Milestone": x.milestone, "Now": x.now, "Needs": x.needs, "Model % tonight": (x.p_tonight * 100).round()}),
        hide_index=True, width="stretch", column_config={
            "Player": st.column_config.TextColumn(pinned=True), "Team": st.column_config.TextColumn(width="small"),
            "Now": st.column_config.NumberColumn(format="%d", width="small"),
            "Needs": st.column_config.NumberColumn(width="small"), "Model % tonight": st.column_config.ProgressColumn(**PCT)})
    c = careers()
    asof = (f" Career totals are the NHL's regular-season numbers (refreshed {c.fetched_at.iloc[0][:10]})."
            if c is not None else " Career totals arrive with the next refresh.")
    st.caption("Career: every 50 goals; every 100 assists, points and games (games only when tonight is the game); "
               "goalie wins every 50 and shutouts every 10 (expected starters). Season: 10, 20, 30... goals and 25, "
               "50... assists and points. First NHL goal or point for anyone without one." + asof +
               " Model % tonight: the projection's chance of getting there in this game.")


HIT_STATS = {"Points": ("points", "pts_hat", "pts", 0.5), "Goals": ("goals", "g_hat", "g", 0.5),
             "Assists": ("assists", "a_hat", "a", 0.5), "Shots on goal": ("sog", "sog_hat", "sog", 2.5),
             "PP points": ("ppp", "ppp_hat", "ppp", 0.5)}


def trends_hits(p: pd.DataFrame, s: dict, day: str):
    c1, c2, c3, c4 = st.columns([3.2, 1, 2.4, 1.6])
    stat = c1.segmented_control("Stat", list(HIT_STATS), default="Points", key="tr_hstat") or "Points"
    col, hat, a, default = HIT_STATS[stat]
    line = c2.number_input("Over", value=default, min_value=0.5, step=1.0, format="%.1f", key=f"tr_hline_{col}")
    win = c3.segmented_control("Games", ["Last 5", "Last 10", "Last 20", "Season"], default="Last 10", key="tr_hwin") or "Last 10"
    least = c4.slider("Hit at least", 0, 100, 70, 5, format="%d%%", key="tr_hmin")
    k = int(np.floor(line)) + 1
    n = None if win == "Season" else int(win.split()[1])
    hr = trends.hit_rates(hist(), p, day, col, k, n)
    hr = hr[(hr.gp >= (3 if n is None else min(n, 5))) & (hr.rate * 100 >= least)]
    if hr.empty:
        st.info("Nobody has cleared that line that often.")
        return
    proj = p.set_index("player")[hat].reindex(hr.index)
    hr = hr.assign(proj=proj, model=model.prob_at_least(proj.to_numpy(), s["coefs"]["alpha"][a], k))
    hr = hr.sort_values(["rate", "gp", "model"], ascending=False)
    st.dataframe(pd.DataFrame({
        "Player": hr["name"], "Team": hr.team, "Tonight": _tonight(hr.opp, hr.is_home),
        "Hit": [f"{int(h)}/{int(g)}" for h, g in zip(hr.hits, hr.gp)], "Hit rate": (hr.rate * 100).round(),
        "This season": (hr.season_rate * 100).round(), "In a row": hr.run.astype(int),
        "Projection": hr.proj, "Model %": (hr.model * 100).round(), "Fair odds": [american(v) for v in hr.model]}),
        hide_index=True, width="stretch", column_config={
            "Player": st.column_config.TextColumn(pinned=True), "Team": st.column_config.TextColumn(width="small"),
            "Hit": st.column_config.TextColumn(width="small"), "Hit rate": st.column_config.ProgressColumn(**PCT),
            "This season": st.column_config.ProgressColumn(**PCT), "In a row": st.column_config.NumberColumn(width="small"),
            "Projection": st.column_config.NumberColumn(format="%.2f", width="small"),
            "Model %": st.column_config.ProgressColumn(**PCT), "Fair odds": st.column_config.TextColumn(width="small")})
    st.caption(f"Over {line:g} {stat.lower()} in the games he played (last season's games fill a window early in the "
               "season). Model % is tonight's projection for the same line; a hit rate far above it is usually a "
               "small sample or a role he no longer has.")


def trends_teams(games: pd.DataFrame, teams: set, opp: dict, day: str):
    tg = team_games()
    if tg.empty:
        st.info("Team results arrive with the next refresh.")
        return
    win = st.segmented_control("Games", ["Last 5", "Last 10", "Season"], default="Last 10", key="tr_twin") or "Last 10"
    n = None if win == "Season" else int(win.split()[1])
    tf = trends.team_form(tg, sorted(teams), day, n)
    if tf.empty:
        st.info("No games yet this season.")
        return
    tf = tf.loc[[t for g in games.itertuples() for t in (g.away, g.home) if t in tf.index]]
    pct = lambda v: (v * 100).round()
    st.dataframe(pd.DataFrame({
        "Team": tf.index, "Tonight": [("vs " if opp[t][1] else "@ ") + opp[t][0] for t in tf.index],
        "Record": tf.record, "Streak": tf.streak, "Last 5": tf.last5, "GF/gm": tf.gf, "GA/gm": tf.ga,
        "SF/gm": tf.sf, "SA/gm": tf.sa, "PP G/gm": tf.ppg, "Total o5.5": pct(tf.o55), "Total o6.5": pct(tf.o65),
        "Scored 3+": pct(tf.gf3), "Allowed ≤2": pct(tf.ga2), "Won by 2+": pct(tf.cover), "To OT": pct(tf.ot_rate)}),
        hide_index=True, width="stretch", column_config={
            "Team": st.column_config.TextColumn(width="small", pinned=True),
            "Last 5": st.column_config.TextColumn(width="small", help="Oldest to newest; O = overtime or shootout loss"),
            **{c: st.column_config.NumberColumn(format="%.2f", width="small") for c in ("GF/gm", "GA/gm", "PP G/gm")},
            **{c: st.column_config.NumberColumn(format="%.1f", width="small") for c in ("SF/gm", "SA/gm")},
            **{c: st.column_config.ProgressColumn(**PCT) for c in ("Total o5.5", "Total o6.5", "Scored 3+",
                                                                   "Allowed ≤2", "Won by 2+", "To OT")}},
        height=35 * (len(tf) + 1) + 3)
    st.caption(f"{win + ' games of this season' if n else 'This season'}, in tonight's game order. Record W-L-OT. "
               "Won by 2+ = covered -1.5 (empty-netters count); a shootout win counts one goal for the winner, as the "
               "final score does.")


_start = os.environ.get("APP_PAGE", "slate")  # headless tests pick the page to open
PAGES = [("slate", page_slate, "Slate"), ("game", page_game, "Game"), ("lines", page_lines, "Lines & goalies"),
         ("player", page_player, "Player"), ("trends", page_trends, "Trends"), ("edges", page_edges, "Edges"), ("results", page_results, "Results"),
         ("model", page_model, "How it works")]
st.navigation([st.Page(f, title=t, url_path=k, default=_start == k) for k, f, t in PAGES]).run()

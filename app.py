"""NHL skater game projections: Slate · Game · Lines & goalies · Player · How it works.

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
import live
import markets
import model
import store

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
</style>"""


def _cell(line: str, p: float, fav: bool) -> str:
    return (f"<td class='{'fav' if fav else ''}'><div class='o'>{line}{american(p)}</div>"
            f"<div class='p'>{p:.0%}</div></td>")


def board_html(g, small: bool = False, title: str = "") -> str:
    """A compact sportsbook-style card: win-probability bar, then away / home rows with fair
    moneyline, puck line and total (odds bold, probability under it, favourite tinted)."""
    pa, ph = 1 - g.p_home, g.p_home
    home_fav = g.lh >= g.la
    rows = ""
    for side, team, lam, p_win in (("away", g.away, g.la, pa), ("home", g.home, g.lh, ph)):
        fav = (side == "home") == home_fav
        own_m15 = g.p_home_m15 if side == "home" else g.p_away_m15
        opp_m15 = g.p_away_m15 if side == "home" else g.p_home_m15
        # favourite -1.5 / underdog +1.5 (= not losing by 2+); tint whichever side is above 50%
        pl = _cell("-1.5 ", own_m15, own_m15 >= 0.5) if fav else _cell("+1.5 ", 1 - opp_m15, 1 - opp_m15 > 0.5)
        tot = _cell(f"o{g.total:g} ", g.p_over, g.p_over >= 0.5) if side == "away" else \
            _cell(f"u{g.total:g} ", 1 - g.p_over, g.p_over < 0.5)
        rows += (f"<tr><td class='t'><span class='tm'>{team}</span> <span class='xg'>{lam:.2f} xG</span></td>"
                 f"{_cell('', p_win, p_win >= 0.5)}{pl}{tot}</tr>")
    bar = (f"<div class='bar'><span style='width:{pa:.1%};background:#eb6834'></span>"
           f"<span style='width:{ph:.1%};background:#2a78d6'></span></div>"
           f"<div class='barlab'><span>{g.away} {pa:.0%}</span><span>win probability</span><span>{ph:.0%} {g.home}</span></div>")
    ttl = f"<div class='ttl'>{title}</div>" if title else ""
    return (f"<div class='gb{' sm' if small else ''}'>{ttl}{bar}<table><tr><th class='t'></th><th>Moneyline</th>"
            f"<th>Puck line</th><th>Total</th></tr>{rows}</table></div>")


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
                    st.html(BOARD_CSS + html)
        st.caption("Fair = no-vig prices from the projected score; overtime and shootouts count, as books grade them. "
                   "Blue = the more likely side of each bet.")
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
        st.html(BOARD_CSS + board_html(gl.iloc[0]))
        st.caption("Fair (no-vig) prices from the projected score; overtime and shootouts count, as books grade them.")
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
        for grp, title, spots, n_lines, pool in (("F", "Forwards", live.F_SPOTS, 4, fwd), ("D", "Defence", live.D_SPOTS, 3, dmen)):
            st.markdown(f"##### {title}")
            widths = [0.5] + [3] * len(spots) + ([3] if grp == "D" else [])
            hdr = st.columns(widths)
            for c, h in zip(hdr[1:], spots):
                c.markdown(f"**{h}**")
            for line in range(1, n_lines + 1):
                cols = st.columns(widths, vertical_alignment="center")
                cols[0].markdown(f"**{line}**")
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
    st.caption(f"{nf} forwards and {nd} defence dressed" + ("" if (nf, nd) == (12, 6) else " · a usual lineup is 12 + 6"))
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
    rows_d = "".join(f"<tr><th>{l}</th>{''.join(cell(chart.get(('D', l, s))) for s in live.D_SPOTS)}<td class='b'></td></tr>"
                     for l in range(1, 4))
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


def page_player():
    s = state() or no_state()
    day = pick_day(s)
    p = projections(day)
    if p.empty:
        st.stop()
    p = p.sort_values(["team", "pts_hat"], ascending=[True, False])
    lab = {i: f"{r.name} · {r.team}" for i, r in zip(p.index, p.itertuples())}
    keys = list(lab)
    i = st.sidebar.selectbox("Player", keys, index=keys.index(p.pts_hat.idxmax()), format_func=lab.get, key=f"pp_{day}")
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


_start = os.environ.get("APP_PAGE", "slate")  # headless tests pick the page to open
PAGES = [("slate", page_slate, "Slate"), ("game", page_game, "Game"), ("lines", page_lines, "Lines & goalies"),
         ("player", page_player, "Player"), ("model", page_model, "How it works")]
st.navigation([st.Page(f, title=t, url_path=k, default=_start == k) for k, f, t in PAGES]).run()

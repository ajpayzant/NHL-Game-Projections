"""How good are Daily Faceoff's line charts, and when? python -m research.lineup_timing [bundle_dir]

Every Daily Faceoff read is logged (data/dfo, locally and in the release bundle). Each chart
carries DFO's own `updated` time, so the version that stood at puck drop (+10 min) can be
rebuilt even when it was only fetched later. For every 2026-27 team-game:
    last    each player's previous game (what the model falls back to)
    pre     the latest chart fetched before puck drop (what the app had at best)
    start   the newest chart version DFO had at puck drop + 10 min
    tonight the lines that actually played (the oracle)
1. lineup accuracy of each vs what actually dressed / played (dressed, F lines, D pairs, PP1)
2. projection accuracy (walk-forward, model fitted on 2021-26) with each lineup, and the line-slot
   trust weight slot_w (0 = treat as stale, 0.5 = current setting for updated charts, 1 = fully)
Output: research/lineup_timing_out.txt
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import backtest as B
import config as C
import features
import live
import model

OUT = Path(__file__).with_name("lineup_timing_out.txt")
log = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    log.append(s)


def snapshots(bundle: Path) -> pd.DataFrame:
    paths = [C.DFO_DIR / "lines_snapshots.parquet", bundle / "data/dfo/lines_snapshots.parquet",
             C.DFO_DIR / "capture_lines.parquet", bundle / "data/dfo/capture_lines.parquet"]   # + puck-drop captures
    parts = [pd.read_parquet(p) for p in paths if p.exists()]
    d = pd.concat(parts, ignore_index=True).drop_duplicates()
    d["fetched"] = pd.to_datetime(d.fetched_at, utc=True)
    d["upd"] = pd.to_datetime(d.updated, utc=True, errors="coerce")
    d["number"] = pd.to_numeric(d.number, errors="coerce")
    return d


def chart(rows: pd.DataFrame, pool: pd.DataFrame, known: pd.DataFrame) -> dict:
    """{player: {grp, line, pp, pk}} from one chart, matched the way the app matches."""
    rows = rows.assign(player=live._match(rows, pool, known)).dropna(subset=["player"])
    out = {}
    for r in rows.itertuples():
        g = str(r.group)
        if re.fullmatch(r"[fd]\d", g):
            out.setdefault(int(r.player), {"grp": g[0].upper(), "line": int(g[1]), "pp": 0, "pk": 0})
    for r in rows.itertuples():
        g = str(r.group)
        if int(r.player) in out and re.fullmatch(r"(pp|pk)\d", g):
            out[int(r.player)][g[:2]] = int(g[2])
    return out


def main():
    bundle = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / "nhlgp_bundle"
    snap = snapshots(bundle)
    pg = pd.read_parquet(C.LAKE / "player_games.parquet")
    for c in ("team", "name", "pos"):
        pg[c] = pg[c].astype(str)
    tg = pd.read_parquet(C.LAKE / "team_games.parquet", columns=["game_id", "date", "team", "start_utc", "season", "game_type"])
    tg = tg[(tg.season == C.CURRENT_SEASON) & (tg.game_type == 2)].copy()
    tg["team"] = tg.team.astype(str)
    tg["day"] = pd.to_datetime(tg.date).dt.strftime("%Y-%m-%d")
    tg["start"] = pd.to_datetime(tg.start_utc, utc=True)
    st = live.load_state()
    numbers = pd.concat([r[["player", "number"]] for r in st["rosters"].values() if r is not None and len(r)])
    numbers = numbers.assign(player=numbers.player.astype(int)).drop_duplicates("player").set_index("player").number
    cur = pg[pg.season == C.CURRENT_SEASON]
    known = pg[pg.season >= C.CURRENT_SEASON - 1].drop_duplicates("player", keep="last")[["player", "name"]]
    known = known.assign(number=known.player.map(numbers))

    rows_pre, rows_start, timing = {}, {}, []
    for g in tg.itertuples():
        s = snap[(snap.team == g.team) & (snap.date == g.day)]
        if s.empty:
            continue
        # a chart that is just tonight's finished game is not a pregame chart
        s = s[~s.source.astype(str).str.contains(f"Last Game ({g.day})", regex=False)]
        pool = cur[cur.team == g.team].drop_duplicates("player")[["player", "name"]]
        pool = pool.assign(number=pool.player.map(numbers))
        pre = s[s.fetched < g.start]
        if len(pre):
            f = pre.fetched.max()
            rows_pre[(g.game_id, g.team)] = chart(pre[pre.fetched == f], pool, known)
        ok = s[s.upd <= g.start + pd.Timedelta(minutes=10)]
        if len(ok):
            u = ok.upd.max()
            v = ok[ok.upd == u]
            v = v[v.fetched == v.fetched.min()]
            rows_start[(g.game_id, g.team)] = chart(v, pool, known)
            timing.append({"game_id": g.game_id, "team": g.team,
                           "pre_fetch_min": (g.start - pre.fetched.max()).total_seconds() / 60 if len(pre) else np.nan,
                           "start_upd_min": (g.start - u).total_seconds() / 60,
                           "late_update": len(pre) > 0 and u > pre.upd.max(),
                           "source": str(v.source.iloc[0])})
    tm = pd.DataFrame(timing)
    say(f"team-games this season: {len(tg)}; with a pre-start fetch {len(rows_pre)}, with a puck-drop chart {len(rows_start)}")
    say(f"chart changed after the last pre-start fetch (a late update we only saw later): {tm.late_update.mean():.1%}")
    say("puck-drop chart last updated, minutes before start: " + str(tm.start_upd_min.describe(percentiles=[.1, .25, .5, .75]).round(0).to_dict()))
    say("puck-drop chart based on: " + str(tm.source.str.startswith("Last Game").map({True: "a 'Last Game' chart", False: "a reporter / projected chart"}).value_counts().to_dict()))

    # ---- 1. lineup accuracy
    act = cur[(cur.game_type == 2) & (cur.toi > 0)]
    actual = {k: g for k, g in act.groupby(["game_id", "team"])}
    prev = pg.sort_values("date").groupby("player")[["line", "pp_unit", "game_id"]].shift(1)
    last_line = pg.assign(pline=prev.line, ppp=prev.pp_unit)
    lastd = {k: g for k, g in last_line[last_line.season == C.CURRENT_SEASON].groupby(["game_id", "team"])}
    # last game's dressed list = the team's previous game's skaters
    team_prev = {}
    for t, g in act.sort_values("date").groupby("team"):
        ids = list(g.game_id.drop_duplicates())
        for a, b in zip(ids[1:], ids[:-1]):
            team_prev[(a, t)] = set(g[g.game_id == b].player)

    def score(kind: str, lineup: dict) -> dict:
        res = []
        for k, dep in lineup.items():
            a = actual.get(k)
            if a is None or not dep:
                continue
            dressed = set(a.player)
            listed = set(dep)
            al = dict(zip(a.player, a.line))
            app = dict(zip(a.player, a.pp_unit))
            both = listed & dressed
            line_ok = np.mean([dep[p]["line"] == al.get(p) for p in both]) if both else np.nan
            pp1_act = {p for p, u in app.items() if u == 1}
            pp1_dep = {p for p in listed if dep[p]["pp"] == 1}
            res.append({"listed_dressed": len(both) / max(len(listed), 1), "dressed_listed": len(both) / max(len(dressed), 1),
                        "line_ok": line_ok, "pp1_ok": len(pp1_act & pp1_dep) / max(len(pp1_act), 1),
                        "perfect_18": float(listed == dressed)})
        r = pd.DataFrame(res)
        return {"lineup": kind, "team-games": len(r), **r.mean().round(3).to_dict()}

    last_lineups = {}
    for k, prev_set in team_prev.items():
        if k in actual:
            g = lastd.get(k)
            if g is None:
                continue
            pl = dict(zip(g.player, zip(g.pline, g.ppp)))
            # the previous game's skaters with their slots then
            pg_prev = act[(act.team == k[1]) & act.player.isin(prev_set)]
            dep = {}
            for p in prev_set:
                ln, u = pl.get(p, (np.nan, 0))
                dep[p] = {"line": ln, "pp": u}
            last_lineups[k] = dep
    common = set(rows_pre) & set(rows_start) & set(last_lineups)
    say(f"\n1. LINEUP ACCURACY vs what actually played ({len(common)} team-games with all three)")
    acc = pd.DataFrame([score("last game", {k: last_lineups[k] for k in common}),
                        score("latest chart before puck drop", {k: rows_pre[k] for k in common}),
                        score("chart at puck drop + 10 min", {k: rows_start[k] for k in common})])
    say(acc.to_string(index=False))
    say("  listed_dressed = share of the listed skaters who played; dressed_listed = share of those who played that were listed;"
        " line_ok = same F line / D pair; pp1_ok = share of the actual PP1 that was listed on PP1; perfect_18 = exact 18")
    # by how long before the start the pre chart was fetched
    pre_t = tm.set_index(["game_id", "team"]).pre_fetch_min
    rr = []
    for k in common:
        s1 = score("x", {k: rows_pre[k]})
        rr.append({"mins": pre_t.get(k), "line_ok": s1.get("line_ok"), "dressed_listed": s1.get("dressed_listed")})
    rr = pd.DataFrame(rr)
    rr["bucket"] = pd.cut(rr.mins, [0, 30, 60, 120, 240, 2000])
    say("\n  latest pre-start chart, by minutes before puck drop it was fetched:")
    say(rr.groupby("bucket", observed=True).agg(n=("line_ok", "size"), line_ok=("line_ok", "mean"),
                                                dressed_listed=("dressed_listed", "mean")).round(3).to_string())

    # ---- 2. projection accuracy with each lineup
    say("\n2. PROJECTION ACCURACY (walk-forward 2026-27, same rows)")
    base_pg, tgh = features.load_history()
    shots = pd.read_parquet(C.LAKE / "shots_xg.parquet",
                            columns=["game_id", "date", "season", "opp", "goalie", "event", "xg_on", "target_net_empty"])
    feat_last = features.build(base_pg.assign(slot_w=0.0), tgh, shots)
    co = model.fit(feat_last[feat_last.season.between(B.FIRST_TRAIN, C.CURRENT_SEASON - 1)])
    keys = set(rows_start) & set(rows_pre)

    def variant(lineups: dict | None, w: float, tonight: bool = False) -> pd.DataFrame:
        p = base_pg.copy()
        p["slot_w"] = 0.0
        if tonight:
            for gcol, a in (("given_line", "line"), ("given_pp", "pp_unit"), ("given_pk", "pk_unit"),
                            ("given_m1", "lm_f1"), ("given_m2", "lm_f2"), ("given_d1", "lm_d1")):
                p[gcol] = np.where(p.season == C.CURRENT_SEASON, p[a].astype(float), np.nan)
            p.loc[p.season == C.CURRENT_SEASON, "slot_w"] = w
        elif lineups:
            for c in ("given_line", "given_pp", "given_pk", "given_m1", "given_m2", "given_d1"):
                p[c] = np.nan
            idx = {(gid, t, pl): i for i, gid, t, pl in zip(p.index, p.game_id, p.team.astype(str), p.player)}
            for (gid, t), dep in lineups.items():
                if (gid, t) not in keys:
                    continue
                by_line = {}
                for pl, d in dep.items():
                    by_line.setdefault((d["grp"], d["line"]), []).append(pl)
                for pl, d in dep.items():
                    i = idx.get((gid, t, pl))
                    if i is None:
                        continue
                    mates = [q for q in by_line[(d["grp"], d["line"])] if q != pl]
                    p.at[i, "given_line"], p.at[i, "given_pp"], p.at[i, "given_pk"] = d["line"], d["pp"], d["pk"]
                    if d["grp"] == "F":
                        p.at[i, "given_m1"] = mates[0] if mates else np.nan
                        p.at[i, "given_m2"] = mates[1] if len(mates) > 1 else np.nan
                    else:
                        p.at[i, "given_d1"] = mates[0] if mates else np.nan
                    p.at[i, "slot_w"] = w
        f = features.build(p, tgh, shots)
        t = f[(f.season == C.CURRENT_SEASON) & (f.toi > 0) & (f.game_type == 2)]
        t = t[[(g, tm_) in keys for g, tm_ in zip(t.game_id, t.team.astype(str))]]
        pr = model.predict(t, co)
        return pd.concat([t[["game_id", "team", "player", "toi", "sog", "goals", "assists", "points", "ppp"]], pr], axis=1)

    runs = [("last game's lines", None, 0.0, False), ("pre-start chart, w 0.5 (app now)", rows_pre, 0.5, False),
            ("pre-start chart, w 1.0", rows_pre, 1.0, False), ("puck-drop chart, w 0.5", rows_start, 0.5, False),
            ("puck-drop chart, w 1.0", rows_start, 1.0, False), ("actual lines (oracle)", None, 1.0, True)]
    for lab, lu, w, ton in runs:
        d = variant(lu, w, ton)
        d = d.loc[:, ~d.columns.duplicated()]
        say(f"  {lab:34s} n {len(d):5,} | TOI MAE {np.abs(d.toi_hat - d.toi / 60).mean():.3f} | " +
            " | ".join(f"{y} {B.pois_dev(d[y], d[h]):.4f}" for y, h in (("sog", "sog_hat"), ("goals", "g_hat"),
                                                                        ("assists", "a_hat"), ("points", "pts_hat"), ("ppp", "ppp_hat"))))
    OUT.write_text("\n".join(log), encoding="utf-8")


if __name__ == "__main__":
    main()

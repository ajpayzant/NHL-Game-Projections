"""How the projections are doing this season (2026-27 to date): python -m research.season_accuracy [bundle_dir]

A. Walk-forward on this season (fitted on 2021-22..2025-26 only) with last game's lines and with
   tonight's actual lines, against the same early-season window (each team's first N games) of
   2023-24..2025-26, so early-season noise is compared with early-season noise. Bias by group.
B. The projections the app actually published (tracking snapshots in the release bundle, Daily
   Faceoff lines) against what happened: accuracy, scratches, skaters it missed, goalie saves.
Output: research/season_accuracy_out.txt
"""
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import backtest as B
import config as C
import model

OUT = Path(__file__).with_name("season_accuracy_out.txt")
STATS = [("sog", "sog_hat", "s2d_sog", "l10_sog"), ("goals", "g_hat", "s2d_goals", "l10_goals"),
         ("assists", "a_hat", "s2d_assists", "l10_assists"), ("points", "pts_hat", "s2d_points", "l10_points"),
         ("ppp", "ppp_hat", "s2d_ppp", "l10_ppp")]
lines_out = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    lines_out.append(s)


def walk_forward(lines: str) -> pd.DataFrame:
    f = B.feature_table(lines)
    base = B.baselines(f)
    out = []
    for yr in (2023, 2024, 2025, 2026):
        co = model.fit(f[f.season.between(B.FIRST_TRAIN, yr - 1)])
        test = f[(f.season == yr) & (f.toi > 0) & (f.game_type == 2)]
        p = model.predict(test, co)
        p = pd.concat([p, model.probabilities(p, co)], axis=1)
        keep = ["game_id", "date", "season", "team", "opp", "player", "name", "pos", "grp", "toi", "sog", "goals",
                "assists", "points", "ppp", "line", "given_line", "pp_unit", "given_pp", "age"]
        out.append(pd.concat([test[[c for c in keep if c in test]], p, base.loc[test.index]], axis=1))
    d = pd.concat(out)
    d = d.loc[:, ~d.columns.duplicated()]
    d["date"] = pd.to_datetime(d.date)
    d["team_gn"] = d.groupby(["season", "team"]).date.rank(method="dense")   # the team's nth game
    return d


def summary(d: pd.DataFrame, label: str) -> None:
    toi = d.toi / 60 if d.toi.max() > 200 else d.toi
    cells = [f"n {len(d):6,}", f"TOI MAE {np.abs(d.toi_hat - toi).mean():.2f} (bias {(d.toi_hat - toi).mean():+.2f})"]
    for y, hat, s2d, l10 in STATS:
        cells.append(f"{y} dev {B.pois_dev(d[y], d[hat]):.4f} vs s2d {B.pois_dev(d[y], d[s2d]):.4f} "
                     f"ratio {d[hat].sum() / max(d[y].sum(), 1):.3f}")
    say(f"  {label:26s} " + " | ".join(cells))


def bias_table(d: pd.DataFrame, by: str, label: str) -> None:
    g = d.groupby(by, observed=True)
    t = pd.DataFrame({"n": g.size(), "TOI bias": g.apply(lambda x: (x.toi_hat - x.toi / 60).mean()),
                      **{f"{y} pred/act": g.apply(lambda x, y=y, h=h: x[h].sum() / max(x[y].sum(), 1))
                         for y, h, *_ in STATS if y in ("sog", "goals", "assists", "points")}})
    say(f"\n  bias by {label}:")
    say(t.round(3).to_string())


def main():
    bundle = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / "nhlgp_bundle"
    say("A. WALK-FORWARD")
    d_last, d_ton = walk_forward("last"), walk_forward("tonight")
    gn = int(d_last[d_last.season == 2026].team_gn.max())
    say(f"this season: through {d_last[d_last.season == 2026].date.max():%Y-%m-%d}, up to {gn} games per team\n")
    for lab, d in (("last game's lines", d_last), ("tonight's actual lines", d_ton)):
        say(f"[{lab}]")
        for yr in (2023, 2024, 2025, 2026):
            x = d[(d.season == yr) & (d.team_gn <= gn)]
            summary(x, f"{yr}-{str(yr + 1)[2:]} first {gn} games")
        summary(d[(d.season < 2026)], "2023-26 full seasons")
        say("")

    cur = d_last[d_last.season == 2026].copy()
    prior = d_last[(d_last.season < 2026) & (d_last.team_gn <= gn)].copy()
    # league level: did scoring move this season?
    say("league per skater-game (actual): " + " | ".join(
        f"{y}: {prior[y].mean():.3f} early 2023-26 -> {cur[y].mean():.3f} now" for y, *_ in STATS))
    say("league TOI/skater-game: " + f"{(prior.toi / 60).mean():.2f} -> {(cur.toi / 60).mean():.2f}")

    # groups
    hist = pd.read_parquet(C.LAKE / "player_games.parquet", columns=["player", "season", "team", "date", "game_type"])
    prev = hist[(hist.season < 2026) & (hist.game_type == 2)]
    nhl_gp = prev.groupby("player").size()
    last_team = prev.sort_values("date").groupby("player").team.last().astype(str)
    for x in (cur, prior):
        x["experience"] = pd.cut(x.player.map(nhl_gp).fillna(0), [-1, 0, 81, 10000], labels=["rookie", "<82 GP", "veteran"])
        x["tier"] = pd.cut(x.pts_hat, [0, .3, .5, .7, .9, 5], labels=["<.3", ".3-.5", ".5-.7", ".7-.9", ".9+"])
        x["slot"] = x.grp.astype(str) + x.given_line.fillna(0).astype(int).astype(str)
    cur["moved"] = np.where(cur.player.map(last_team).isna(), "new", np.where(cur.player.map(last_team) == cur.team.astype(str), "same team", "changed team"))
    for by, lab in (("experience", "NHL experience"), ("moved", "team change (this season)"), ("tier", "projected points tier"),
                    ("slot", "line slot"), ("grp", "position")):
        bias_table(cur, by, lab + " - 2026-27")
        if by != "moved":
            bias_table(prior, by, lab + f" - 2023-26 first {gn} games (reference)")

    # teams: projected vs actual team totals (sum of skaters), this season
    t = cur.groupby("team").agg(gp=("game_id", "nunique"), g_hat=("g_hat", "sum"), g=("goals", "sum"),
                                sog_hat=("sog_hat", "sum"), sog=("sog", "sum"))
    t = t.assign(goals_gm_pred=t.g_hat / t.gp, goals_gm_act=t.g / t.gp, sog_gm_pred=t.sog_hat / t.gp, sog_gm_act=t.sog / t.gp)
    t["sog_ratio"] = t.sog_hat / t.sog
    t["goal_ratio"] = t.g_hat / t.g
    say("\n  teams this season, skater totals per game (pred vs actual), sorted by shots ratio:")
    say(t.sort_values("sog_ratio")[["gp", "sog_gm_pred", "sog_gm_act", "sog_ratio", "goals_gm_pred", "goals_gm_act", "goal_ratio"]].round(2).to_string())
    # what the noise floor is for a team's SOG ratio over ~10 games: ~ sqrt(28*10)/280
    say(f"  (noise: a team's {gn}-game shots total has sd about {np.sqrt(28 * gn) / (28 * gn):.1%}, goals about {np.sqrt(3 * gn) / (3 * gn):.0%})")

    # calibration this season
    say("\n  calibration this season (pred -> actual):")
    for col, y, k in (("p_pts_1", "points", 1), ("p_g_1", "goals", 1), ("p_a_1", "assists", 1), ("p_sog_3", "sog", 3), ("p_sog_2", "sog", 2)):
        z = cur.assign(b=pd.cut(cur[col], [0, .15, .3, .45, .6, .75, 1]), hit=cur[y] >= k).groupby("b", observed=True)
        z = pd.DataFrame({"n": z.size(), "pred": z[col].mean(), "act": z.hit.mean()})
        say(f"  {col:8s} logloss {B.log_loss(cur[y] >= k, cur[col]):.4f} | " + "  ".join(f"{r.pred:.2f}->{r.act:.2f} ({r.n})" for r in z.itertuples()))

    # biggest player misses this season (TOI and shots), min 6 games
    pm = cur.groupby(["player", "name", "team"]).agg(gp=("game_id", "size"), toi_hat=("toi_hat", "mean"),
                                                     toi=("toi", lambda s: s.mean() / 60), sog_hat=("sog_hat", "mean"),
                                                     sog=("sog", "mean"), pts_hat=("pts_hat", "mean"), pts=("points", "mean"))
    pm = pm[pm.gp >= 6].assign(toi_err=lambda x: x.toi_hat - x.toi, sog_err=lambda x: x.sog_hat - x.sog)
    say("\n  largest TOI misses this season (6+ games), projected - actual:")
    say(pm.reindex(pm.toi_err.abs().sort_values(ascending=False).index).head(15).round(2).to_string())

    # ---------------------------------------------------------------- B. published snapshots
    say("\n\nB. WHAT THE APP PUBLISHED (tracking snapshots)")
    files = sorted(glob.glob(str(bundle / "data" / "tracking" / "snap_*_skaters.parquet")))
    pg = pd.read_parquet(C.LAKE / "player_games.parquet", columns=["game_id", "player", "team", "toi", "sog", "goals",
                                                                  "assists", "points", "g_pp", "a_pp", "date"])
    pg["ppp"] = pg.g_pp + pg.a_pp
    done = set(pg.game_id)
    snaps = []
    for fn in files:
        s = pd.read_parquet(fn)
        s["day"] = Path(fn).name[5:15]
        snaps.append(s[s.game_id.isin(done)])
    sn = pd.concat(snaps, ignore_index=True)
    say(f"days {sn.day.min()} .. {sn.day.max()}, {sn.game_id.nunique()} finished games, {len(sn):,} projected skater rows")
    m = sn.merge(pg[["game_id", "player", "toi", "sog", "goals", "assists", "points", "ppp"]], on=["game_id", "player"], how="left")
    scratched = m.toi.isna() | (m.toi == 0)
    say(f"projected but did not play: {scratched.sum()} ({scratched.mean():.1%}); their projected points {m[scratched].pts_hat.sum():.1f}")
    played = pg[pg.game_id.isin(sn.game_id.unique()) & (pg.toi > 0)]
    missed = played.merge(sn[["game_id", "player"]], on=["game_id", "player"], how="left", indicator=True)
    missed = missed[missed._merge == "left_only"]
    say(f"played but not projected: {len(missed)} of {len(played)} ({len(missed) / len(played):.1%}); their actual points {missed.points.sum():.0f}")
    mm = m[~scratched]
    for y, hat, *_ in STATS:
        say(f"  {y:8s} published dev {B.pois_dev(mm[y], mm[hat]):.4f}  ratio {mm[hat].sum() / mm[y].sum():.3f}")
    # same rows from the walk-forward (last lines / tonight lines) for comparison
    for lab, d in (("walk-forward last lines", d_last), ("walk-forward tonight lines", d_ton)):
        w = d[d.season == 2026].merge(mm[["game_id", "player"]], on=["game_id", "player"])
        say(f"  same rows, {lab}: " + " | ".join(f"{y} {B.pois_dev(w[y], w[hat]):.4f}" for y, hat, *_ in STATS))
    # by day: is anything drifting?
    byday = mm.groupby("day").apply(lambda x: pd.Series({"n": len(x), "pts_ratio": x.pts_hat.sum() / x.points.sum(),
                                                         "sog_ratio": x.sog_hat.sum() / x.sog.sum()}))
    say("\n  by day:\n" + byday.round(3).to_string())

    # goalies
    gfiles = [f.replace("skaters", "goalies") for f in files]
    gs = pd.concat([pd.read_parquet(f).assign(day=Path(f).name[5:15]) for f in gfiles if Path(f).exists()], ignore_index=True)
    gl = pd.read_parquet(C.DATA / "saves" / "lake" / "goalies.parquet", columns=["game_id", "goalie", "team", "starter", "saves", "shots_against"])
    st_ = gl[gl.starter]
    gs = gs[gs.game_id.isin(st_.game_id)]
    first = gs.drop(columns="goalie").groupby(["game_id", "team"]).head(1)   # the app's first goalie per team = its expected starter
    j = first.merge(st_[["game_id", "team", "goalie", "saves"]], on=["game_id", "team"], how="left")
    right = j.goalie_id == j.goalie
    say(f"\n  goalies: {len(j)} team-games, expected starter right {right.mean():.1%}")
    jr = j[right]
    say(f"  saves when the starter was right: MAE {np.abs(jr['mean'] - jr.saves).mean():.2f}, bias {(jr['mean'] - jr.saves).mean():+.2f}, "
        f"corr {np.corrcoef(jr['mean'], jr.saves)[0, 1]:.2f} (n {len(jr)}); constant-mean MAE {np.abs(jr.saves.mean() - jr.saves).mean():.2f}")
    OUT.write_text("\n".join(lines_out), encoding="utf-8")


if __name__ == "__main__":
    main()

"""Walk-forward backtest: each season is projected by a model fitted on earlier seasons only.

The lineup is who actually dressed (Daily Faceoff gets that right on the day); the lines,
units and linemates assumed are the player's previous game ("last game"), which is what a
line chart shows the morning after, or optionally that night's actual ones ("tonight", a
slightly optimistic stand-in for good pregame line info). Baselines a bettor might use:
the player's season-to-date per-game average and his last-10 average.

    python backtest.py                 # last-game lines
    python backtest.py --lines tonight
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

import config as C
import features
import model

TEST_SEASONS = (2023, 2024, 2025)
FIRST_TRAIN = 2021  # 2018-19 to 2020-21 only warm up the running sums


def feature_table(lines: str = "last") -> pd.DataFrame:
    pg, tg = features.load_history()
    pg["slot_w"] = 1.0 if lines == "tonight" else 0.0
    if lines == "tonight":
        for g, a in (("given_line", "line"), ("given_pp", "pp_unit"), ("given_pk", "pk_unit"),
                     ("given_m1", "lm_f1"), ("given_m2", "lm_f2"), ("given_d1", "lm_d1")):
            pg[g] = pg[a].astype(float)
    shots = pd.read_parquet(C.LAKE / "shots_xg.parquet",
                            columns=["game_id", "date", "season", "opp", "goalie", "event", "xg_on", "target_net_empty"])
    return features.build(pg, tg, shots)


def baselines(d: pd.DataFrame) -> pd.DataFrame:
    """Season-to-date and last-10 per-game averages (falling back to last season, then the
    position average) for each target."""
    d = d.sort_values(["date", "game_id"])
    out = pd.DataFrame(index=d.index)
    d = d.assign(toi_min=d.toi / 60)
    for tgt in ("toi_min", "sog", "goals", "assists", "points", "ppp"):
        g = d.groupby(["player", "season"])[tgt]
        s2d = g.transform(lambda s: s.shift(1).expanding().mean())
        last_season = d.groupby(["player", "season"])[tgt].mean().rename("ls").reset_index()
        last_season["season"] += 1
        ls = d[["player", "season"]].merge(last_season, on=["player", "season"], how="left").ls.to_numpy()
        pos_avg = d.groupby(["season", "grp"])[tgt].transform("mean")
        n = g.cumcount()
        # blend toward last season while this season is young (10 games of last-season weight)
        s2d = np.where(np.isnan(ls), s2d, (s2d.fillna(0) * n + ls * 10) / (n + 10))
        out[f"s2d_{tgt}"] = pd.Series(s2d, index=d.index).fillna(pos_avg)
        l10 = d.groupby("player")[tgt].transform(lambda s: s.shift(1).rolling(10, min_periods=1).mean())
        out[f"l10_{tgt}"] = l10.fillna(out[f"s2d_{tgt}"])
    return out.sort_index()


def pois_dev(y, mu) -> float:
    y, mu = np.asarray(y, float), np.clip(np.asarray(mu, float), 1e-6, None)
    t = np.where(y > 0, y * np.log(np.where(y > 0, y, 1) / mu), 0.0)
    return float(2 * np.mean(t - (y - mu)))


def log_loss(y, p) -> float:
    y, p = np.asarray(y, float), np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def run(lines: str = "last", verbose: bool = True) -> pd.DataFrame:
    f = feature_table(lines)
    base = baselines(f)
    preds = []
    for yr in TEST_SEASONS:
        train = f[f.season.between(FIRST_TRAIN, yr - 1)]
        co = model.fit(train)
        test = f[(f.season == yr) & (f.toi > 0)]
        p = model.predict(test, co)
        p = pd.concat([p, model.probabilities(p, co)], axis=1)
        preds.append(pd.concat([test[["game_id", "date", "season", "game_type", "team", "opp", "player", "name",
                                      "pos", "grp", "toi", "sog", "goals", "assists", "points", "ppp", "gp_16",
                                      "line", "given_line", "pp_unit", "given_pp"]], p,
                                base.loc[test.index]], axis=1))
        if verbose:
            print(f"{yr}: fitted on {FIRST_TRAIN}-{yr - 1} ({len(train):,} rows), scored {len(test):,}", flush=True)
    out = pd.concat(preds)
    C.DATA.joinpath("backtest").mkdir(exist_ok=True)
    out.to_parquet(C.DATA / "backtest" / f"backtest_{lines}.parquet", index=False)
    if verbose:
        report(out)
    return out


def report(out: pd.DataFrame) -> None:
    out = out[out.gp_16 > 0]  # a player's NHL debut has no history; the baselines are blind too
    for gt, lab in ((2, "regular season"), (3, "playoffs")):
        d = out[out.game_type == gt]
        if d.empty:
            continue
        print(f"\n=== {lab}: {len(d):,} player-games ===")
        rows = []
        y_toi = d.toi / 60
        for name, pred_toi, sog, g, a, pts, ppp in (
                ("model", d.toi_hat, d.sog_hat, d.g_hat, d.a_hat, d.pts_hat, d.ppp_hat),
                ("season-to-date avg", d.s2d_toi_min, d.s2d_sog, d.s2d_goals, d.s2d_assists, d.s2d_points, d.s2d_ppp),
                ("last-10 avg", d.l10_toi_min, d.l10_sog, d.l10_goals, d.l10_assists, d.l10_points, d.l10_ppp)):
            rows.append({"": name,
                         "TOI MAE": np.mean(np.abs(y_toi - pred_toi)),
                         "SOG MAE": np.mean(np.abs(d.sog - sog)),
                         "SOG dev": pois_dev(d.sog, sog), "G dev": pois_dev(d.goals, g),
                         "A dev": pois_dev(d.assists, a), "PTS dev": pois_dev(d.points, pts),
                         "PTS MAE": np.mean(np.abs(d.points - pts)), "PPP dev": pois_dev(d.ppp, ppp),
                         "r SOG": np.corrcoef(d.sog, sog)[0, 1], "r PTS": np.corrcoef(d.points, pts)[0, 1]})
        print(pd.DataFrame(rows).set_index("").round(4).to_string())
        print("\nby season (model):")
        print(d.groupby("season").apply(lambda s: pd.Series({
            "TOI MAE": np.mean(np.abs(s.toi / 60 - s.toi_hat)), "SOG MAE": np.mean(np.abs(s.sog - s.sog_hat)),
            "PTS dev": pois_dev(s.points, s.pts_hat), "mean pts": s.points.mean(), "mean pts_hat": s.pts_hat.mean(),
            "mean sog": s.sog.mean(), "mean sog_hat": s.sog_hat.mean()}), include_groups=False).round(4).to_string())
        print("\ncalibration (predicted vs actual frequency):")
        for col, y, lab2 in (("p_ppp_1", d.ppp >= 1, "1+ PPP"), ("p_pts_1", d.points >= 1, "1+ point"), ("p_pts_2", d.points >= 2, "2+ points"),
                             ("p_g_1", d.goals >= 1, "1+ goal"), ("p_a_1", d.assists >= 1, "1+ assist"),
                             ("p_sog_2", d.sog >= 2, "2+ SOG"), ("p_sog_3", d.sog >= 3, "3+ SOG"),
                             ("p_sog_4", d.sog >= 4, "4+ SOG"), ("p_sog_5", d.sog >= 5, "5+ SOG")):
            bins = pd.cut(d[col], [0, .05, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1])
            t = pd.DataFrame({"pred": d[col], "act": y}).groupby(bins, observed=True).agg(
                n=("act", "size"), pred=("pred", "mean"), act=("act", "mean"))
            t = t[t.n >= 200]
            cells = "  ".join(f"{r.pred:.2f}->{r.act:.2f}" for r in t.itertuples())
            print(f"  {lab2:10s} logloss {log_loss(y, d[col]):.4f} | {cells}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--lines", choices=["last", "tonight"], default="last")
    run(ap.parse_args().lines)

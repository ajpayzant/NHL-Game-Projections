"""The research matrix: every pregame variable we can build, grouped by data type, next to
the structured model's walk-forward prediction. Built once, used by research/models.py.

    python -m research.matrix
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import backtest
import config as C
import ew

GROUPS: dict[str, list[str]] = {}


def add(group: str, df: pd.DataFrame, cols: dict[str, pd.Series]) -> None:
    for k, v in cols.items():
        df[k] = np.asarray(v, dtype=float)
    GROUPS.setdefault(group, []).extend(cols)


def rolling_means(f: pd.DataFrame) -> dict[str, pd.Series]:
    out = {}
    f = f.assign(toi_min=f.toi / 60)
    g = f.groupby("player")
    for tgt in ("sog", "goals", "assists", "points", "toi_min"):
        sh = g[tgt].shift(1)
        for n in (5, 10, 20):
            out[f"raw_{tgt}_l{n}"] = sh.groupby(f.player).transform(lambda s: s.rolling(n, min_periods=1).mean())
        out[f"raw_{tgt}_s2d"] = f.groupby(["player", "season"])[tgt].transform(lambda s: s.shift(1).expanding().mean())
        ls = f.groupby(["player", "season"])[tgt].mean().rename("v").reset_index()
        ls["season"] += 1
        out[f"raw_{tgt}_lastseason"] = f[["player", "season"]].merge(ls, on=["player", "season"], how="left").v.to_numpy()
        out[f"raw_{tgt}_career"] = g[tgt].transform(lambda s: s.shift(1).expanding().mean())
    return out


def build() -> pd.DataFrame:
    f = backtest.feature_table("last")
    bt = pd.read_parquet(C.DATA / "backtest" / "backtest_last.parquet",
                         columns=["game_id", "player", "toi_hat", "sog_hat", "g_hat", "a_hat", "pts_hat"])
    f = f.merge(bt, on=["game_id", "player"], how="left")
    f = f.sort_values(["date", "game_id", "team", "player"]).reset_index(drop=True)
    pg = f
    t = lambda c: pg[c]
    # 1 usage (structured inputs)
    add("usage", f, {c: f[c] for c in ["ev_share_4", "ev_share_16", "pp_share_4", "pp_share_16", "sh_share_4",
                                         "sh_share_16", "slot_ev", "slot_pp", "slot_sh", "given_line", "given_pp",
                                         "given_pk", "gp_16"]})
    # 2 structured rates
    add("rates", f, {c: f[c] for c in ["sog_pm_ev", "sog_pm_pp", "sog_pm_sh", "sh_pct", "gf_pm_ev", "gf_pm_pp",
                                         "ashare_ev", "ashare_pp"]})
    # 3 raw per-game averages
    add("raw_avgs", f, rolling_means(f))
    # 4 shot attempts and shot quality (EW, 80 games)
    f["t_all"] = (f.toi_ev + f.toi_pp + f.toi_sh) / 60
    f["icf_all"] = f.icf_ev + f.icf_pp + f.icf_sh
    f["ixg_all"] = f.ixg_ev + f.ixg_pp + f.ixg_sh
    e = ew.pregame(f, "player", ["icf_ev", "icf_pp", "t_ev", "t_pp", "ixg_all", "icf_all", "sog_all", "t_all"], 80, 20)
    add("attempts_xg", f, {"icf_pm_ev": (e.icf_ev + 30) / (e.t_ev + 120), "icf_pm_pp": (e.icf_pp + 10) / (e.t_pp + 30),
                           "ixg_pm": (e.ixg_all + 5) / (e.t_all + 120), "ixg_per_shot": (e.ixg_all + 5) / (e.sog_all + 50),
                           "sog_per_att": (e.sog_all + 25) / (e.icf_all + 50)})
    # 5 on-ice for / against (EV, per minute)
    f["ton_ev2"] = f.t_ev.where(f.toi_5v5.notna())
    e = ew.pregame(f, "player", ["oxgf_ev", "osf_ev", "oga_ev", "oxga_ev", "osa_ev", "ton_ev2"], 80, 20)
    add("on_ice", f, {f"{c}_pm": (e[c] + 1) / (e.ton_ev2 + 60) for c in ["oxgf_ev", "osf_ev", "oga_ev", "oxga_ev", "osa_ev"]}
        | {"oxg_share": (e.oxgf_ev + 5) / (e.oxgf_ev + e.oxga_ev + 10)})
    # 6 deployment extras: offensive-zone start share, penalties drawn per 60
    for c in ("fo_oz", "fo_dz", "fo_nz", "pen_drawn", "pen_taken", "hits", "blocks", "takeaways", "giveaways"):
        f[c] = f[c].astype(float).where(f.toi_5v5.notna() | ~c.startswith("fo"))
    e = ew.pregame(f, "player", ["fo_oz", "fo_dz", "fo_nz", "pen_drawn", "pen_taken", "t_all"], 40, 10)
    add("zone_pen", f, {"oz_share": (e.fo_oz + 5) / (e.fo_oz + e.fo_dz + 10),
                        "fo_per_min": (e.fo_oz + e.fo_dz + e.fo_nz + 5) / (e.t_all + 15),
                        "pen_drawn_pm": (e.pen_drawn + 1) / (e.t_all + 60), "pen_taken_pm": (e.pen_taken + 1) / (e.t_all + 60)})
    # 7 physical / style
    e = ew.pregame(f, "player", ["hits", "blocks", "takeaways", "giveaways", "t_all"], 80, 20)
    add("style", f, {f"{c}_pm": (e[c] + 1) / (e.t_all + 60) for c in ["hits", "blocks", "takeaways", "giveaways"]})
    # 8 bio
    bio = pd.read_parquet(C.LAKE / "bio.parquet")
    f = f.merge(bio, on="player", how="left")
    age = (f.date - pd.to_datetime(f.birth)).dt.days / 365.25
    career = f.groupby("player").cumcount()
    add("bio", f, {"age": age, "height": f.height, "weight": f.weight, "draft_overall": f.draft_overall.fillna(250),
                   "career_gp": career, "season_gp": f.groupby(["player", "season"]).cumcount(),
                   "is_C": (f.pos == "C"), "is_D": (f.pos == "D")})
    # 9 team / opponent / schedule
    tg = pd.read_parquet(C.LAKE / "team_games.parquet")
    tg["team"] = tg.team.astype(str)
    tg["one"] = 1.0
    tg["gf_all"] = tg.goals
    tg["xgf_all"] = tg.xgf_ev + tg.xgf_pp + tg.xgf_sh
    tg["xga_all"] = tg.xga_ev + tg.xga_pp + tg.xga_sh
    e = ew.pregame(tg, "team", ["gf_all", "goals_against", "xgf_all", "xga_all", "one", "sf_ev", "sa_ev"], 30, 10)
    tg["team_gf_pg"] = e.gf_all / e.one
    tg["team_xg_share"] = e.xgf_all / (e.xgf_all + e.xga_all)
    tg["team_sf_pg"] = e.sf_ev / e.one
    tm = tg[["game_id", "team", "team_gf_pg", "team_xg_share", "team_sf_pg"]]
    f = f.merge(tm, on=["game_id", "team"], how="left")
    f = f.merge(tm.rename(columns={"team": "opp", "team_gf_pg": "opp_gf_pg", "team_xg_share": "opp_xg_share",
                                   "team_sf_pg": "opp_sf_pg"}), on=["game_id", "opp"], how="left")
    add("team_ctx", f, {c: f[c] for c in ["team_gf_pg", "team_xg_share", "team_sf_pg", "opp_gf_pg", "opp_xg_share",
                                            "opp_sf_pg", "opp_supp_ev", "opp_supp_pk", "opp_supp_g", "opp_gsax",
                                            "ew_pp", "ew_sh", "opp_ew_pp", "opp_ew_sh", "rink", "rest", "opp_rest"]}
        | {"home": f.is_home, "playoffs": f.game_type == 3})
    # 10 linemates: their structured rates
    q = f.set_index(["game_id", "player"])[["sog_pm_ev", "gf_pm_ev", "q_pts"]]
    vals = {}
    for c in ("given_m1", "given_m2", "given_d1"):
        k = pd.MultiIndex.from_arrays([f.game_id, f[c].fillna(-1).astype(np.int64)])
        vals[c] = q.reindex(k).to_numpy()
    fw = np.nanmean(np.stack([vals["given_m1"], vals["given_m2"]]), axis=0)
    add("linemates", f, {"mate": f.mate, "mate_sog_pm": fw[:, 0], "mate_gf_pm": fw[:, 1], "mate_q_pts": fw[:, 2],
                         "partner_q_pts": vals["given_d1"][:, 2]})
    keep = (["game_id", "date", "season", "game_type", "team", "opp", "player", "name", "pos", "grp",
             "toi", "sog", "goals", "assists", "points", "toi_hat", "sog_hat", "g_hat", "a_hat", "pts_hat"]
            + [c for cs in GROUPS.values() for c in cs])
    out = f[keep].copy()
    for c in [c for cs in GROUPS.values() for c in cs]:
        out[c] = out[c].astype("float32")
    out.to_parquet(C.DATA / "research_matrix.parquet", index=False)
    pd.Series({g: ",".join(c) for g, c in GROUPS.items()}).to_json(C.DATA / "research_groups.json")
    print(out.shape, {g: len(c) for g, c in GROUPS.items()})
    return out


if __name__ == "__main__":
    build()

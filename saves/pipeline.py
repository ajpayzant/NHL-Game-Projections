"""Build the modelling tables from the lake and cache them.

    data/lake/team_games_xg.parquet  team-games with xG sums (the input to features)
    data/lake/matrix.parquet   pregame features (team states, lineups) + targets, one row per team-game
    data/lake/goalies.parquet  goalie-games with xG faced and pregame talent state
    data/lake/pulls.parquet    one row per goal a starter allowed, pulled after it or not

    python pipeline.py
"""
from __future__ import annotations

import pandas as pd

from . import config as C
from . import features as F
from . import goalie_model as GM
from . import parse
from . import xg


def build(verbose: bool = True) -> dict[str, pd.DataFrame]:
    shots = xg.score(parse.load("shots"), verbose=verbose)
    tg, gg = F.add_xg_to_games(parse.load("team_games"), parse.load("goalie_games"), shots)
    M = F.build_matrix(tg, parse.load("skater_games"))
    # the matchup-quality target for the save model
    M = M.merge(tg[["game_id", "team", "xga_on"]], on=["game_id", "team"], how="left")
    hist_path = C.LAKE / "goalie_history.parquet"
    hist = pd.read_parquet(hist_path) if hist_path.exists() else None
    gg = pd.concat([gg, GM.goalie_states(gg, hist)], axis=1)
    pulls = GM.pull_events(shots, gg)
    exit_rate = pulls.attrs["exit_rate"]
    pulls = pulls.assign(exit_rate=exit_rate, intermission_share=pulls.attrs["intermission_share"])
    out = {"team_games_xg": tg, "matrix": M, "goalies": gg,
           "pulls": pulls[["game_id", "goalie", "k", "period", "pull", "exit_rate", "intermission_share"]]}
    for k, v in out.items():
        v.to_parquet(C.LAKE / f"{k}.parquet", index=False)
    if verbose:
        print(f"matrix {M.shape}, goalies {gg.shape}, pulls {len(pulls):,} (exit rate {exit_rate:.4f})")
    return out


def load() -> dict[str, pd.DataFrame]:
    out = {k: pd.read_parquet(C.LAKE / f"{k}.parquet") for k in ("team_games_xg", "matrix", "goalies", "pulls")}
    for a in ("exit_rate", "intermission_share"):
        out["pulls"].attrs[a] = float(out["pulls"][a].iloc[0])
    return out


if __name__ == "__main__":
    build()

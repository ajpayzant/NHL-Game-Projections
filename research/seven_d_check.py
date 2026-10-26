"""Before / after the 7th-D change (a 7th defenceman is pair 4 on his own): backtest errors
overall, in 11 F + 7 D games, and for the 7th D himself.

    python -m research.seven_d_check
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config as C

R = C.ROOT / "research"


def pois_dev(y, mu) -> float:
    mu = np.clip(mu, 1e-9, None)
    return float(np.mean(2 * (np.where(y > 0, y * np.log(np.where(y > 0, y, 1) / mu), 0) - (y - mu))))


def summary(d: pd.DataFrame) -> dict:
    err = d.toi_hat - d.toi / 60  # toi is in seconds, toi_hat in minutes
    return {"n": len(d), "TOI MAE": err.abs().mean(), "TOI bias": err.mean(),
            "SOG dev": pois_dev(d.sog.to_numpy(), d.sog_hat.to_numpy()),
            "PTS dev": pois_dev(d.points.to_numpy(), d.pts_hat.to_numpy())}


def main() -> None:
    for lines in ("last", "tonight"):
        before = pd.read_parquet(R / f"bt_before_{lines}.parquet")
        after = pd.read_parquet(C.DATA / "backtest" / f"backtest_{lines}.parquet")
        rows = []
        for lab, d in (("before", before), ("after", after)):
            d = d[d.game_type == 2]
            nd = d.groupby(["game_id", "team"]).grp.transform(lambda g: (g == "D").sum())
            seven = d[nd == 7]
            # the 7th D: lowest-TOI defenceman in a 7-D game
            sd = seven[seven.grp == "D"]
            last_d = sd[sd.toi == sd.groupby(["game_id", "team"]).toi.transform("min")]
            for part, x in (("all", d), ("11F/7D games", seven), ("  their F", seven[seven.grp == "F"]),
                            ("  their D1-6", sd.drop(last_d.index)), ("  7th D", last_d)):
                rows.append({"lines": lines, "set": part, "version": lab, **summary(x)})
        t = pd.DataFrame(rows).set_index(["lines", "set", "version"])
        print(t.round(4).to_string(), "\n")


if __name__ == "__main__":
    main()

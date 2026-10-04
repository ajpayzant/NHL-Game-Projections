"""One-at-a-time re-check of the main memory / shrinkage settings now that history starts in
2018. Validation: 2022-23, fitted on 2021-22 (test seasons untouched)."""
import numpy as np, pandas as pd
import backtest, features, model
from backtest import pois_dev
BASE = dict(HL_RATE=80, K_SOG={"ev": 120, "pp": 40, "sh": 40}, HL_SHOOT=320, K_SHOOT=200, HL_GF=80, HL_ASSIST_SHARE=160)
TRIALS = [("baseline", {}), ("HL_RATE=160", {"HL_RATE": 160}), ("HL_RATE=40", {"HL_RATE": 40}),
          ("K_SOG ev=60", {"K_SOG": {"ev": 60, "pp": 20, "sh": 20}}), ("K_SOG ev=240", {"K_SOG": {"ev": 240, "pp": 80, "sh": 80}}),
          ("HL_SHOOT=640", {"HL_SHOOT": 640}), ("HL_SHOOT=160", {"HL_SHOOT": 160}), ("K_SHOOT=400", {"K_SHOOT": 400}),
          ("K_SHOOT=100", {"K_SHOOT": 100}), ("HL_GF=160", {"HL_GF": 160}), ("HL_ASSIST_SHARE=320", {"HL_ASSIST_SHARE": 320})]
for name, ch in TRIALS:
    for k, v in (BASE | ch).items():
        setattr(features, k, v)
    f = backtest.feature_table("last")
    co = model.fit(f[f.season == 2021])
    te = f[(f.season == 2022) & (f.toi > 0) & (f.gp_16 > 0) & (f.game_type == 2)]
    p = model.predict(te, co)
    print(f"{name:20s} SOG {pois_dev(te.sog, p.sog_hat):.5f}  G {pois_dev(te.goals, p.g_hat):.5f}  "
          f"A {pois_dev(te.assists, p.a_hat):.5f}  PTS {pois_dev(te.points, p.pts_hat):.5f}", flush=True)

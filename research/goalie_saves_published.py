"""Published goalie projections (tracking snapshots, Oct 18-26) vs the walk-forward on the same starts:
python -m research.goalie_saves_published [bundle_dir]"""
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from saves import config as SC

bundle = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / "nhlgp_bundle"
gs = pd.concat([pd.read_parquet(f).assign(day=Path(f).name[5:15])
                for f in sorted(glob.glob(str(bundle / "data/tracking/snap_*_goalies.parquet")))], ignore_index=True)
w = pd.read_parquet(SC.SAVES / "backtest" / "preds_2026.parquet")
m = gs.merge(w[["game_id", "goalie", "saves", "m_mean", "m_mu_used", "sa_faced", "shots_against"]],
             left_on=["game_id", "goalie_id"], right_on=["game_id", "goalie"])
m = m.drop_duplicates(["game_id", "goalie_id"])
print(f"published starts matched to the actual starter: {len(m)}")
print(f"  published mean {m['mean'].mean():.2f}  walk-forward {m.m_mean.mean():.2f}  actual {m.saves.mean():.2f}")
print(f"  bias published {(m['mean'] - m.saves).mean():+.2f}  walk-forward {(m.m_mean - m.saves).mean():+.2f}  "
      f"(sd of a {len(m)}-start mean ~{np.std(m.saves - m.m_mean) / np.sqrt(len(m)):.2f})")
print(f"  MAE published {np.abs(m['mean'] - m.saves).mean():.2f}  walk-forward {np.abs(m.m_mean - m.saves).mean():.2f}")
print(f"  published - walk-forward per start: mean {(m['mean'] - m.m_mean).mean():+.2f}, sd {(m['mean'] - m.m_mean).std():.2f}")
print(f"  team SA on these games: model mu {m.m_mu_used.mean():.2f}, actual {m.sa_faced.mean():.2f}")

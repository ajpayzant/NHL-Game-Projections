"""Team markets from the player projections: are moneyline, puck line and totals calibrated?

Team expected goals = sum of its skaters' projected goals. Scores: regulation goals Poisson,
ties go to OT / shootout (winner +1 goal, as books grade). Checked walk-forward 2023-26."""
import numpy as np
import pandas as pd

import config as C
import markets

b = pd.read_parquet(C.DATA / "backtest" / "backtest_last.parquet")
tg = pd.read_parquet(C.LAKE / "team_games.parquet")
tg = tg[tg.season.between(2023, 2025)]
lam = b.groupby(["game_id", "team"]).g_hat.sum().rename("lam").reset_index()
h = tg[tg.is_home][["game_id", "season", "game_type", "team", "opp", "goals", "goals_against", "last_period_type"]]
h = h.merge(lam, on=["game_id", "team"]).merge(lam.rename(columns={"team": "opp", "lam": "lam_a"}), on=["game_id", "opp"])
h = h.rename(columns={"lam": "lam_h", "goals": "gh", "goals_against": "ga"})
print(f"{len(h):,} games. Mean projected goals home {h.lam_h.mean():.3f} away {h.lam_a.mean():.3f}; "
      f"actual (final, incl. SO winner) home {h.gh.mean():.3f} away {h.ga.mean():.3f}")
reg = h.last_period_type == "REG"
print(f"share of games to OT/SO: {1 - reg.mean():.3f}")

h = h[h.season >= 2024].reset_index(drop=True)  # out of sample: parameters were fitted on 2023-24
reg = h.last_period_type == "REG"
print(f"checking on {len(h):,} games of 2024-25 and 2025-26")
for name, kw in (("plain Poisson", {"en": {1: 0, 2: 0, 3: 0}, "share": 0.025, "tie": 0.0, "stretch": 1.0}),
                 ("final model", {})):
    P = markets.game_probs(h.lam_h.to_numpy(), h.lam_a.to_numpy(), **kw)
    hw = (h.gh > h.ga).to_numpy()
    fav_h = h.lam_h.to_numpy() >= h.lam_a.to_numpy()
    fav_cover = np.where(fav_h, h.gh - h.ga >= 2, h.ga - h.gh >= 2)
    p_cover = np.where(fav_h, P["home_m15"], P["away_m15"])
    tot = (h.gh + h.ga).to_numpy()
    print(f"\n{name}")
    for lab, p, y in (("home win", P["home_win"], hw), ("favourite -1.5", p_cover, fav_cover),
                      ("over 5.5", P["over"][5.5], tot > 5.5), ("over 6.5", P["over"][6.5], tot > 6.5),
                      ("tie after regulation", P["reg_tie"], ~reg.to_numpy())):
        bins = pd.cut(p, [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1])
        t = pd.DataFrame({"p": p, "y": y}).groupby(bins, observed=True).agg(n=("y", "size"), p=("p", "mean"), y=("y", "mean"))
        t = t[t.n >= 100]
        ll = -np.mean(y * np.log(np.clip(p, 1e-6, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-6, 1)))
        print(f"  {lab:22s} mean pred {np.mean(p):.3f} actual {np.mean(y):.3f}  logloss {ll:.4f} | "
              + "  ".join(f"{r.p:.2f}->{r.y:.2f}" for r in t.itertuples()))

"""Fixes for the long-odds calibration found by research/full_model_check.py, walk-forward:
python -m research.tail_fix  -> research/tail_fix_out.txt
  A. SOG: the new shot terms with / without the defence term, one NB dispersion vs one per F / D
  B. tail recalibration for 2+/3+/4+ goals, points, assists, PPP and 4+/5+/6+ SOG: logit(P) mapped
     by a + b*logit(P) fitted per (stat, line) on the training seasons' own fitted rows, applied to
     the test season
"""
from pathlib import Path

import numpy as np
import pandas as pd

import backtest as B
import model

OUT = Path(__file__).with_name("tail_fix_out.txt")
CACHE = Path(__file__).parent / "_cache" / "ft_last.parquet"
log = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    log.append(s)


if CACHE.exists():
    f = pd.read_parquet(CACHE)
else:
    f = B.feature_table("last")
    CACHE.parent.mkdir(exist_ok=True)
    f.to_parquet(CACHE)
COLS = ["game_id", "season", "team", "player", "grp", "toi", "sog", "goals", "assists", "points", "ppp", "given_pp", "gp_16"]
HATS = ["sog_hat", "g_hat", "a_hat", "pts_hat", "ppp_hat"]


def alpha_by(y, mu, grp):
    return {g: model._nb_alpha(y[grp == g], mu[grp == g]) for g in ("F", "D")}


def run(defence: bool):
    model.USE_PHASES = model.USE_SHOT_TERMS = True
    if not defence:
        model.SHOT_TERMS = ["attempts", "own_team"]
    else:
        model.SHOT_TERMS = ["attempts", "own_team", "defence"]
    tr_parts, te_parts = [], []
    for yr in (2023, 2024, 2025, 2026):
        tr = f[f.season.between(B.FIRST_TRAIN, yr - 1) & (f.toi > 0) & f.gp_16.gt(0)]
        co = model.fit(tr)
        te = f[(f.season == yr) & (f.toi > 0) & (f.game_type == 2) & f.gp_16.gt(0)]
        for part, d, lst in (("train", tr[tr.season >= yr - 2], tr_parts), ("test", te, te_parts)):
            p = model.predict(d, co)
            lst.append(d[COLS].assign(**{h: p[h].to_numpy() for h in HATS}, fold=yr,
                                      a_sog=co["alpha"]["sog"]).reset_index(drop=True))
        print("defence" if defence else "no defence", yr, flush=True)
    return pd.concat(tr_parts, ignore_index=True), pd.concat(te_parts, ignore_index=True)


TAILS = [("goals", "g_hat", (2, 3)), ("points", "pts_hat", (2, 3, 4)), ("assists", "a_hat", (2,)), ("ppp", "ppp_hat", (2,)),
         ("sog", "sog_hat", (2, 3, 4, 5, 6))]


def probs(d, hat, k, alpha):
    return model.prob_at_least(d[hat].to_numpy(), alpha, k)


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def recal(p_tr, y_tr, p_te):
    import statsmodels.api as sm
    X = sm.add_constant(logit(p_tr))
    m = sm.GLM(y_tr, X, family=sm.families.Binomial()).fit()
    a, b = m.params
    return 1 / (1 + np.exp(-(a + b * logit(p_te)))), (a, b)


def tiers(d):
    r = d.pts_hat.rank(pct=True)
    return pd.cut(r, [0, .4, .8, .95, .99, 1], labels=["bottom 40%", "40-80%", "80-95%", "top 5%", "top 1%"])


res = {}
for defence in (False, True):
    tr, te = run(defence)
    lab = "with defence term" if defence else "without defence term"
    say(f"\n===== {lab} =====")
    say("SOG projected/actual by position and tier:")
    t = te.assign(q=pd.qcut(te.sog_hat, 5, labels=["q1", "q2", "q3", "q4", "q5"])).groupby(["grp", "q"], observed=True)
    say(t.apply(lambda x: x.sog_hat.sum() / x.sog.sum()).unstack().round(3).to_string())
    say(f"SOG deviance {B.pois_dev(te.sog, te.sog_hat):.4f}")
    rows = []
    for col, hat, ks in TAILS:
        for k in ks:
            for yr in sorted(te.fold.unique()):
                pass
            # one alpha (the model's) vs per-position alpha (SOG only), and recalibrated
            out = {"line": f"{k}+ {col}"}
            p_one = np.empty(len(te)); p_grp = np.empty(len(te)); p_rec = np.empty(len(te))
            for yr in sorted(te.fold.unique()):
                mt, mr = (te.fold == yr).to_numpy(), (tr.fold == yr).to_numpy()
                a1 = te.a_sog[mt].iloc[0] if col == "sog" else 0.0
                p_one[mt] = probs(te[mt], hat, k, a1)
                if col == "sog":
                    ag = alpha_by(tr[mr].sog.to_numpy(float), tr[mr].sog_hat.to_numpy(), tr[mr].grp.to_numpy())
                    for g in ("F", "D"):
                        m2 = mt & (te.grp == g).to_numpy()
                        p_grp[m2] = probs(te[m2], hat, k, ag[g])
                    base_tr = np.empty(mr.sum())
                    for g in ("F", "D"):
                        mm = (tr[mr].grp == g).to_numpy()
                        base_tr[mm] = probs(tr[mr][mm], hat, k, ag[g])
                    base_te = p_grp[mt]
                else:
                    p_grp[mt] = p_one[mt]
                    base_tr = probs(tr[mr], hat, k, 0.0)
                    base_te = p_one[mt]
                p_rec[mt], _ = recal(base_tr, (tr[mr][col] >= k).to_numpy(float), base_te)
            y = (te[col] >= k).to_numpy(float)
            for nm, p in (("model", p_one), ("per-pos alpha", p_grp), ("recalibrated", p_rec)):
                if nm == "per-pos alpha" and col != "sog":
                    continue
                out[f"{nm} ratio"] = y.mean() / p.mean()
                out[f"{nm} ll"] = B.log_loss(y, p)
                tt = tiers(te)
                cells = []
                for lev in tt.cat.categories:
                    m = (tt == lev).to_numpy()
                    cells.append(f"{y[m].mean() / max(p[m].mean(), 1e-9):.2f}")
                out[f"{nm} by tier (actual/model)"] = " ".join(cells)
            rows.append(out)
    tab = pd.DataFrame(rows)
    pd.set_option("display.width", 300); pd.set_option("display.max_colwidth", 60)
    say("\nlong-odds lines: actual/model ratio, log loss, ratio by points tier (bottom 40% .. top 1%)")
    say(tab.round(5).to_string(index=False))
OUT.write_text("\n".join(log), encoding="utf-8")

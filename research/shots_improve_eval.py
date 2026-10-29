"""Shots-on-goal improvements, walk-forward: python -m research.shots_improve_eval  (after shots_improve_build)

Every variant refits the skater SOG Poisson GLM exactly as model.fit does (same base, offset and
terms), with one change, on FIRST_TRAIN..season-1, and is scored on that season's regular
season. Output: research/shots_improve_out.txt
"""
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

import backtest as B
import model
from research.shots_improve_build import CACHE, SEASONS

OUT = Path(__file__).with_name("shots_improve_out.txt")
log = []
XCOLS = ["x_" + c for c in model.TERMS["sog"]]


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    log.append(s)


def lg(x):
    return np.log(np.clip(np.asarray(x, float), 0.5, 2.0))


def mix(d, ev, pp):
    """A rating weighted by tonight's EV / PP share of ice time (PK and the rest at 1.0)."""
    ev = np.nan_to_num(np.asarray(ev, float), nan=1.0)
    pp = np.nan_to_num(np.asarray(pp, float), nan=1.0)
    return d.w_ev * ev + d.w_pp * pp + (1 - d.w_ev - d.w_pp)


def attempt_base(d, hl):
    ot = np.nan_to_num(d.ot_rel, nan=1.0)
    return (d.toi_ev_hat * d.env_sog_ev * np.nan_to_num(d[f"att_rel_ev_{hl}"], nan=1.0) * ot
            + d.toi_pp_hat * d.env_sog_pp * np.nan_to_num(d[f"att_rel_pp_{hl}"], nan=1.0) * ot
            + d.toi_sh_hat * d.sog_pm_sh)


# variant -> function(d) returning (design DataFrame, base Series)
def v_current(d):
    return d[XCOLS].copy(), d.base_sog


def v_add(name, fn):
    def f(d):
        X = d[XCOLS].copy()
        X[name] = fn(d)
        return X, d.base_sog
    return f


def v_opp(col):
    def f(d):
        X = d[XCOLS].copy()
        X["x_opp_shots"] = lg(mix(d, d[col], d.opp_supp_pk))
        return X, d.base_sog
    return f


def v_att_replace(hl, blend=None):
    def f(d):
        ab = attempt_base(d, hl)
        base = ab if blend is None else np.exp((1 - blend) * np.log(d.base_sog.clip(1e-4)) + blend * np.log(ab.clip(1e-4)))
        X = d[XCOLS].copy()
        X["x_spread"] = np.log(base.clip(1e-4) / model.TYPICAL["sog"])
        return X, base
    return f


VARIANTS = {
    "current": v_current,
    "own team gen hl5": v_add("own_gen", lambda d: lg(mix(d, d.gen_ev_5, d.gen_pp_5))),
    "own team gen hl10": v_add("own_gen", lambda d: lg(mix(d, d.gen_ev_10, d.gen_pp_10))),
    "own team gen hl20": v_add("own_gen", lambda d: lg(mix(d, d.gen_ev_20, d.gen_pp_20))),
    "own team gen adj hl20": v_add("own_gen", lambda d: lg(mix(d, d.gen_adj_ev_20, d.gen_pp_10))),
    "opp supp raw (rebuilt)": v_opp("supp_raw_ev"),
    "opp supp adjusted hl20": v_opp("supp_adj_ev_20"),
    "opp supp adjusted hl40": v_opp("supp_adj_ev_40"),
    "attempts term hl40": v_add("att", lambda d: np.log((attempt_base(d, 40) / d.base_sog).clip(0.33, 3))),
    "attempts term hl80": v_add("att", lambda d: np.log((attempt_base(d, 80) / d.base_sog).clip(0.33, 3))),
    "attempts base hl80": v_att_replace(80),
    "attempts base 50% hl80": v_att_replace(80, 0.5),
}


def fit_predict(tr, te, fn):
    Xtr, btr = fn(tr)
    Xte, bte = fn(te)
    off_tr = np.log(btr.clip(1e-4))
    m = sm.GLM(tr.sog.to_numpy(float), Xtr.to_numpy(float), family=sm.families.Poisson(), offset=off_tr.to_numpy()).fit()
    coef = dict(zip(Xtr.columns, m.params))
    mu_tr = btr * np.exp(Xtr.to_numpy(float) @ m.params)
    alpha = model._nb_alpha(tr.sog, mu_tr)
    mu = bte * np.exp(Xte.to_numpy(float) @ m.params)
    return pd.Series(np.asarray(mu), index=te.index), alpha, coef


def ll_line(y, mu, alpha, k):
    p = np.clip(model.prob_at_least(mu, alpha, k), 1e-6, 1 - 1e-6)
    o = (np.asarray(y) >= k).astype(float)
    return float(-(o * np.log(p) + (1 - o) * np.log(1 - p)).mean())


def team_spread(te, mu, early_only=False):
    d = te.assign(mu=mu)
    d["gn"] = d.groupby("team").date.rank(method="dense")
    if early_only:
        d = d[d.gn <= 14]
    t = d.groupby("team").agg(h=("mu", "sum"), a=("sog", "sum"))
    r = t.h / t.a
    return float(r.std()), float((np.sqrt(t.a) / t.a).mean())


def main():
    data = {yr: (pd.read_parquet(CACHE / f"shots_{yr}_train.parquet"), pd.read_parquet(CACHE / f"shots_{yr}_test.parquet"))
            for yr in SEASONS}
    res, preds = [], {}
    for name, fn in VARIANTS.items():
        for yr, (tr, te) in data.items():
            mu, alpha, coef = fit_predict(tr, te, fn)
            preds[(name, yr)] = (mu, alpha)
            y = te.sog.to_numpy(float)
            gm = te.assign(mu=mu).groupby(["game_id", "team"]).agg(h=("mu", "sum"), a=("sog", "sum"))
            sd_all, nz_all = team_spread(te, mu)
            sd_e, nz_e = team_spread(te, mu, True)
            res.append({"variant": name, "season": yr, "n": len(te), "dev": B.pois_dev(y, mu), "MAE": np.abs(mu - y).mean(),
                        "ll1.5": ll_line(y, mu, alpha, 2), "ll2.5": ll_line(y, mu, alpha, 3), "ll3.5": ll_line(y, mu, alpha, 4),
                        "ratio": mu.sum() / y.sum(), "team MAE": np.abs(gm.h - gm.a).mean(), "team corr": np.corrcoef(gm.h, gm.a)[0, 1],
                        "team sd": sd_all, "team noise": nz_all, "team sd 1-14": sd_e, "noise 1-14": nz_e,
                        "coef": {k: round(v, 3) for k, v in coef.items() if k not in XCOLS or k in ("x_opp_shots", "x_spread")}})
        print(name, "done", flush=True)
    R = pd.DataFrame(res)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 80)
    say("1. SKATER SOG, walk-forward by season (lower dev / MAE / log loss is better)")
    for yr in SEASONS:
        say(f"\n  {yr}-{str(yr + 1)[2:]}" + (" (to date)" if yr == 2026 else ""))
        say(R[R.season == yr].drop(columns=["season"]).round(4).to_string(index=False))
    say("\n  POOLED 2023-26 full seasons (row-weighted), change vs current:")
    P = R[R.season < 2026].copy()
    agg = P.groupby("variant").apply(lambda g: pd.Series({k: np.average(g[k], weights=g.n) for k in
                                                          ("dev", "MAE", "ll1.5", "ll2.5", "ll3.5", "team MAE", "team corr", "team sd", "team sd 1-14")}))
    cur = agg.loc["current"]
    rel = agg.copy()
    for k in ("dev", "MAE", "ll1.5", "ll2.5", "ll3.5", "team MAE"):
        rel[k + " %"] = 100 * (agg[k] / cur[k] - 1)
    say(rel.round(4).to_string())
    say("\n  per-season dev change vs current (consistency):")
    piv = R.pivot(index="variant", columns="season", values="dev")
    say((100 * (piv / piv.loc["current"] - 1)).round(3).to_string())
    OUT.write_text("\n".join(log), encoding="utf-8")
    pd.to_pickle({"R": R, "preds": preds}, CACHE / "shots_eval.pkl")


if __name__ == "__main__":
    main()

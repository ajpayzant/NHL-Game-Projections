"""Model bake-off on the research matrix, walk-forward (test 2024-25 and 2025-26).

    S          structured model (production)
    LGB-all    gradient boosting, Poisson loss, every variable
    LGB-simple gradient boosting on simple stats only (raw averages, usage, bio, team context)
    S+boost    gradient boosting on the structured model's errors (structured log-prediction as offset)
    GLM-wide   ridge Poisson regression on every variable (standardised)
    blend      w * S + (1 - w) * LGB-all, w fitted on the previous season

    python -m research.models
"""
from __future__ import annotations

import json
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

import config as C

TARGETS = {"toi": ("toi_min", "toi_hat"), "sog": ("sog", "sog_hat"), "goals": ("goals", "g_hat"),
           "assists": ("assists", "a_hat"), "points": ("points", "pts_hat")}
PARAMS = dict(learning_rate=0.04, num_leaves=31, min_data_in_leaf=400, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, verbose=-1, num_threads=4)


def pois_dev(y, mu):
    y, mu = np.asarray(y, float), np.clip(np.asarray(mu, float), 1e-6, None)
    t = np.where(y > 0, y * np.log(np.where(y > 0, y, 1) / mu), 0.0)
    return float(2 * np.mean(t - (y - mu)))


def score(name, y, p, is_toi):
    return np.mean(np.abs(y - p)) if is_toi else pois_dev(y, p)


def fit_lgb(Xtr, ytr, Xte, is_toi, init_tr=None, init_te=None, rounds=None):
    obj = "regression" if is_toi else "poisson"
    params = PARAMS | {"objective": obj}
    if rounds is None:  # pick rounds on the last 20% of training rows (they are in date order)
        cut = int(len(Xtr) * 0.8)
        dtr = lgb.Dataset(Xtr[:cut], ytr[:cut], init_score=None if init_tr is None else init_tr[:cut])
        dva = lgb.Dataset(Xtr[cut:], ytr[cut:], init_score=None if init_tr is None else init_tr[cut:])
        m = lgb.train(params, dtr, 2000, valid_sets=[dva], callbacks=[lgb.early_stopping(50, verbose=False)])
        rounds = max(m.best_iteration, 10)
    m = lgb.train(params, lgb.Dataset(Xtr, ytr, init_score=init_tr), rounds)
    raw = m.predict(Xte, raw_score=True)
    if init_te is not None:
        raw = raw + init_te
    p = raw if is_toi else np.exp(raw)
    return p, m, rounds


def ridge_poisson(Xtr, ytr, Xte, lam=1.0, iters=25):
    mu_x, sd_x = np.nanmean(Xtr, 0), np.nanstd(Xtr, 0) + 1e-9
    Z = lambda X: np.column_stack([np.ones(len(X)), np.nan_to_num((X - mu_x) / sd_x)])
    A, B = Z(Xtr), Z(Xte)
    b = np.zeros(A.shape[1]); b[0] = np.log(max(ytr.mean(), 1e-3))
    pen = lam * np.eye(A.shape[1]); pen[0, 0] = 0
    for _ in range(iters):
        eta = np.clip(A @ b, -10, 5); mu = np.exp(eta)
        g = A.T @ (ytr - mu) - pen @ b
        H = (A * mu[:, None]).T @ A + pen
        b = b + np.linalg.solve(H, g)
    return np.exp(np.clip(B @ b, -10, 5))


def main(targets=None):
    m = pd.read_parquet(C.DATA / "research_matrix.parquet")
    groups = {g: c.split(",") for g, c in json.loads((C.DATA / "research_groups.json").read_text()).items()}
    allf = [c for cs in groups.values() for c in cs]
    simple = groups["raw_avgs"] + groups["usage"] + groups["bio"] + groups["team_ctx"]
    m["toi_min"] = m.toi / 60
    m = m[(m.toi > 0) & (m.gp_16 > 0)].reset_index(drop=True)
    results, preds = [], []
    for tname in targets or TARGETS:
        y_col, s_col = TARGETS[tname]
        is_toi = tname == "toi"
        prev_blend = {}
        for test in (2023, 2024, 2025):
            tr = m[(m.season >= 2022) & (m.season < test)]
            te = m[m.season == test]
            ytr, yte = tr[y_col].to_numpy(), te[y_col].to_numpy()
            out = {"S": te[s_col].to_numpy()}
            out["LGB-all"], _, _ = fit_lgb(tr[allf].to_numpy(), ytr, te[allf].to_numpy(), is_toi)
            if test >= 2024:
                out["LGB-simple"], _, _ = fit_lgb(tr[simple].to_numpy(), ytr, te[simple].to_numpy(), is_toi)
                trr = tr[tr.season >= 2023]  # structured predictions exist out of sample from 2023
                link = (lambda v: v) if is_toi else (lambda v: np.log(np.clip(v, 1e-4, None)))
                out["S+boost"], _, _ = fit_lgb(trr[allf].to_numpy(), trr[y_col].to_numpy(), te[allf].to_numpy(), is_toi,
                                               init_tr=link(trr[s_col].to_numpy()), init_te=link(te[s_col].to_numpy()))
                if not is_toi:
                    out["GLM-wide"] = ridge_poisson(tr[allf].to_numpy(float), ytr, te[allf].to_numpy(float), lam=50.0)
                # blend weight from the previous season's out-of-sample predictions
                ws = np.linspace(0, 1, 21)
                ps, pl, yp = prev_blend[test - 1]
                w = ws[np.argmin([score(None, yp, wi * ps + (1 - wi) * pl, is_toi) for wi in ws])]
                out[f"blend(w_S={w:.2f})"] = w * out["S"] + (1 - w) * out["LGB-all"]
            prev_blend[test] = (out["S"], out["LGB-all"], yte)
            if test >= 2024:
                for k, p in out.items():
                    results.append({"target": tname, "season": test, "model": k.split("(")[0] if k.startswith("blend") else k,
                                    "label": k, "score": score(k, yte, p, is_toi)})
                preds.append(pd.DataFrame({"target": tname, "game_id": te.game_id.values, "player": te.player.values,
                                           **{k.split("(")[0]: v for k, v in out.items()}}))
            print(tname, test, {k: round(score(k, yte, p, is_toi), 4) for k, p in out.items()}, flush=True)
    r = pd.DataFrame(results)
    print("\n(TOI: MAE in minutes; others: Poisson deviance; lower is better; mean of 2024-25 and 2025-26)")
    print(r.groupby(["target", "model"]).score.mean().unstack("model").round(4).to_string())
    pd.concat(preds).to_parquet(C.DATA / "research_model_preds.parquet", index=False)


if __name__ == "__main__":
    main(sys.argv[1:] or None)

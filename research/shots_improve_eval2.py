"""Second round: shorter attempt half-lives, the combination, and breakdowns for the winner.
python -m research.shots_improve_eval2  (after shots_improve_build and shots_improve_eval)"""
import numpy as np
import pandas as pd

import backtest as B
import config as C
import ew
import features as F
import model
from research.shots_improve_build import CACHE, SEASONS
from research.shots_improve_eval import (OUT, VARIANTS, XCOLS, attempt_base, fit_predict, lg, ll_line, mix,
                                         team_spread, v_add)

log = [OUT.read_text(encoding="utf-8")] if OUT.exists() else []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    log.append(s)


# attempts at shorter half-lives, from player_games (same construction as the build, pregame only)
pg = pd.read_parquet(C.LAKE / "player_games.parquet", columns=["game_id", "date", "season", "player", "pos", "toi_ev", "toi_pp",
                                                               "icf_ev", "icf_pp"])
pg = pg.sort_values(["date", "game_id"]).reset_index(drop=True)
pg["grp"] = np.where(pg.pos.astype(str) == "D", "D", "F")
pg["pending"] = False
for s in ("ev", "pp"):
    pg[f"t_{s}"] = pg[f"toi_{s}"] / 60
extra = {}
for s, k in (("ev", 120), ("pp", 40)):
    env = F.league_env(pg, f"icf_{s}", f"t_{s}")
    for hl in (10, 20):
        tmp = pd.DataFrame({"player": pg.player, "season": pg.season, "n": pg[f"icf_{s}"], "x": pg[f"t_{s}"] * env})
        e = ew.pregame(tmp, "player", ["n", "x"], hl, F.GAP)
        extra[f"att_rel_{s}_{hl}"] = ((e.n + k * env) / (e.x + k * env)).to_numpy()
ex = pd.DataFrame(extra).assign(game_id=pg.game_id.to_numpy(), player=pg.player.to_numpy())

data = {}
for yr in SEASONS:
    tr = pd.read_parquet(CACHE / f"shots_{yr}_train.parquet").merge(ex, on=["game_id", "player"], how="left")
    te = pd.read_parquet(CACHE / f"shots_{yr}_test.parquet").merge(ex, on=["game_id", "player"], how="left")
    data[yr] = (tr, te)

att = lambda hl: (lambda d: np.log((attempt_base(d, hl) / d.base_sog).clip(0.33, 3)))
gen5 = lambda d: lg(mix(d, d.gen_ev_5, d.gen_pp_5))


def both(hl):
    def f(d):
        X = d[XCOLS].copy()
        X["att"] = att(hl)(d)
        X["own_gen"] = gen5(d)
        return X, d.base_sog
    return f


V = {"current": VARIANTS["current"], "attempts term hl10": v_add("att", att(10)), "attempts term hl20": v_add("att", att(20)),
     "attempts term hl40": VARIANTS["attempts term hl40"], "attempts hl40 + own gen hl5": both(40),
     "attempts hl20 + own gen hl5": both(20)}
rows, keep = [], {}
for name, fn in V.items():
    for yr, (tr, te) in data.items():
        mu, alpha, coef = fit_predict(tr, te, fn)
        keep[(name, yr)] = (mu, alpha, te)
        y = te.sog.to_numpy(float)
        sd, nz = team_spread(te, mu)
        gm = te.assign(mu=mu).groupby(["game_id", "team"]).agg(h=("mu", "sum"), a=("sog", "sum"))
        rows.append({"variant": name, "season": yr, "n": len(te), "dev": B.pois_dev(y, mu), "MAE": np.abs(mu - y).mean(),
                     "ll1.5": ll_line(y, mu, alpha, 2), "ll2.5": ll_line(y, mu, alpha, 3), "ll3.5": ll_line(y, mu, alpha, 4),
                     "team MAE": np.abs(gm.h - gm.a).mean(), "team corr": np.corrcoef(gm.h, gm.a)[0, 1], "team sd": sd, "noise": nz,
                     "coef": {k: round(v, 3) for k, v in coef.items() if k in ("att", "own_gen", "x_spread")}})
R = pd.DataFrame(rows)
pd.set_option("display.width", 250)
say("\n\n2. ROUND TWO: attempt half-lives and the combination (dev % vs current by season)")
piv = R.pivot(index="variant", columns="season", values="dev")
say((100 * (piv / piv.loc["current"] - 1)).round(3).to_string())
P = R[R.season < 2026]
agg = P.groupby("variant").apply(lambda g: pd.Series({k: np.average(g[k], weights=g.n) for k in
                                                      ("dev", "MAE", "ll1.5", "ll2.5", "ll3.5", "team MAE", "team corr", "team sd", "noise")}))
cur = agg.loc["current"]
for k in ("dev", "MAE", "ll1.5", "ll2.5", "ll3.5", "team MAE"):
    agg[k + " %"] = 100 * (agg[k] / cur[k] - 1)
say("\n  pooled 2023-26:\n" + agg.round(4).to_string())
say("\n  coefficients (2025-26 fit): " + str(R[R.season == 2025].set_index("variant").coef.to_dict()))

# breakdowns for the best combination vs current, pooled 2023-26 and 2026-27
best = "attempts hl20 + own gen hl5" if agg.loc["attempts hl20 + own gen hl5", "dev"] <= agg.loc["attempts hl40 + own gen hl5", "dev"] else "attempts hl40 + own gen hl5"
say(f"\n3. BREAKDOWN: {best} vs current")
for lab, yrs in (("2023-26", (2023, 2024, 2025)), ("2026-27", (2026,))):
    parts = []
    for name in ("current", best):
        for yr in yrs:
            mu, alpha, te = keep[(name, yr)]
            parts.append(te[["grp", "sog", "player", "gp_16"]].assign(mu=mu.to_numpy(), alpha=alpha, variant=name, season=yr))
    D = pd.concat(parts, ignore_index=True)
    cur_mu = D[D.variant == "current"].mu.to_numpy()
    D["tier"] = pd.cut(np.tile(cur_mu, 2), [0, 1, 2, 3, 99], labels=["<1", "1-2", "2-3", "3+"])
    D["history"] = pd.cut(D.gp_16, [-1, 4, 10, 99], labels=["thin (<5 wtd gp)", "5-10", "10+"])
    for by in ("tier", "grp", "history"):
        t = D.groupby([by, "variant"], observed=True).apply(lambda g: pd.Series({
            "n": len(g), "dev": B.pois_dev(g.sog, g.mu), "ratio": g.mu.sum() / g.sog.sum(),
            "ll2.5": ll_line(g.sog, g.mu, g.alpha.iloc[0], 3)})).unstack("variant")
        say(f"\n  {lab} by {by}:\n" + t.round(4).to_string())
    # calibration at 2.5 for the best
    b = D[D.variant == best]
    p3 = model.prob_at_least(b.mu.to_numpy(), b.alpha.iloc[0], 3)
    z = pd.DataFrame({"p": p3, "o": (b.sog >= 3).to_numpy()}).assign(bin=lambda x: pd.cut(x.p, [0, .1, .2, .3, .4, .5, .6, .7, 1]))
    z = z.groupby("bin", observed=True).agg(n=("o", "size"), pred=("p", "mean"), act=("o", "mean"))
    say(f"\n  {lab} calibration P(3+ SOG), {best}: " + "  ".join(f"{r.pred:.2f}->{r.act:.2f}({r.n})" for r in z.itertuples()))
pd.to_pickle({"R": R, "keep": {k: (v[0], v[1]) for k, v in keep.items()}, "best": best}, CACHE / "shots_eval2.pkl")
OUT.write_text("\n".join(log), encoding="utf-8")

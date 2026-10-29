"""Goalie saves: what limits accuracy and what could improve it (walk-forward evidence).

    python -m research.goalie_saves_backtest     # first: preds_<season> for 2023-26
    python -m research.goalie_saves_research     # -> research/goalie_saves_research_out.txt

Regular season, the goalie who actually started. Saves = shots he faced x his save %.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from saves import config as SC

BT = SC.SAVES / "backtest"
OUT = Path(__file__).with_name("goalie_saves_research_out.txt")
LINES = (22.5, 24.5, 26.5, 28.5, 30.5)
SEASONS = (2023, 2024, 2025)
log = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    log.append(s)


def load(season):
    p = pd.read_parquet(BT / f"preds_{season}.parquet")
    c = np.load(BT / f"cdf_{season}.npy")
    keep = (p.game_type == 2).to_numpy()
    p = p[keep].reset_index(drop=True)
    return p.assign(cdf=list(c[keep])), c[keep]


def ll_lines(y, pover):
    """Mean over/under log loss across LINES; pover: (n, len(LINES))."""
    out = []
    for j, ln in enumerate(LINES):
        p = np.clip(pover[:, j], 1e-4, 1 - 1e-4)
        o = (y > ln).astype(float)
        out.append(-(o * np.log(p) + (1 - o) * np.log(1 - p)).mean())
    return float(np.mean(out))


def nb_pover(mean, var, lines=LINES):
    """P(saves > line) for a negative binomial with this mean and variance (Poisson if var <= mean)."""
    mean, var = np.asarray(mean, float), np.asarray(var, float)
    var = np.maximum(var, mean * 1.0001)
    n = mean ** 2 / (var - mean)
    q = n / (n + mean)
    return np.column_stack([stats.nbinom.sf(np.floor(ln), n, q) for ln in lines])


def card(y, mean, pover, label):
    e = mean - y
    say(f"  {label:44s} MAE {np.abs(e).mean():.3f}  RMSE {np.sqrt((e ** 2).mean()):.3f}  corr {np.corrcoef(mean, y)[0, 1]:.3f}  "
        f"bias {e.mean():+.2f}  O/U logloss {ll_lines(y, pover):.4f}" if pover is not None else
        f"  {label:44s} MAE {np.abs(e).mean():.3f}  RMSE {np.sqrt((e ** 2).mean()):.3f}  corr {np.corrcoef(mean, y)[0, 1]:.3f}  bias {e.mean():+.2f}")


def main():
    P = {s: load(s) for s in (*SEASONS, 2026)}
    allp = pd.concat([P[s][0] for s in SEASONS], ignore_index=True)
    allc = np.concatenate([P[s][1] for s in SEASONS])
    y = allp.saves.to_numpy().astype(float)
    pover_model = np.column_stack([1 - allc[:, int(ln)] for ln in LINES])

    say("1. CURRENT MODEL, walk-forward 2023-24..2025-26 regular season (actual starter)")
    for s in SEASONS:
        d, c = P[s]
        card(d.saves.to_numpy(float), d.m_mean.to_numpy(), np.column_stack([1 - c[:, int(ln)] for ln in LINES]), f"{s}-{str(s + 1)[2:]}")
    card(y, allp.m_mean.to_numpy(), pover_model, "pooled")
    lg = y.mean()
    card(y, np.full_like(y, lg), nb_pover(np.full_like(y, lg), np.full_like(y, y.var())), "constant (pooled mean, hindsight)")
    # irreducible error if the model's distribution is right: expected |X - mean| under its own pmf
    k = np.arange(71)
    pmf = np.diff(np.concatenate([np.zeros((len(allc), 1)), allc], axis=1), axis=1)
    exp_mae = (pmf * np.abs(k[None, :] - allp.m_mean.to_numpy()[:, None])).sum(axis=1).mean()
    say(f"  expected MAE if the model's distributions were exactly right: {exp_mae:.3f}  (actual {np.abs(allp.m_mean - y).mean():.3f})")
    say(f"  model sd {allp.m_sd.mean():.2f} vs actual residual sd {np.std(y - allp.m_mean):.2f}")

    say("\n2. WHERE THE ERROR COMES FROM")
    pulled = (allp.goalies_used > 1).to_numpy()
    say(f"  starts with a pull / exit: {pulled.mean():.1%}; MAE there {np.abs(allp.m_mean - y)[pulled].mean():.2f}, "
        f"finished starts {np.abs(allp.m_mean - y)[~pulled].mean():.2f}; share of total abs error from pulls "
        f"{np.abs(allp.m_mean - y)[pulled].sum() / np.abs(allp.m_mean - y).sum():.1%}")
    sa = allp.sa_faced.to_numpy(float)
    mu = allp.m_mu_used.to_numpy(float)
    say(f"  team shots faced: model mu corr {np.corrcoef(mu, sa)[0, 1]:.3f}, MAE {np.abs(mu - sa).mean():.2f}, "
        f"bias {np.mean(mu - sa):+.2f}; actual sd {sa.std():.2f}, model mu sd {mu.std():.2f}")
    # Poisson/NB noise floor on SA: if mu were exactly right, MAE of NB(mu, alpha) draws
    rng = np.random.default_rng(0)
    a = float(allp.alpha.iloc[0])
    draws = rng.poisson(rng.gamma(1 / a, mu * a))
    say(f"  SA noise floor (draws from NB around the model's own mu): MAE {np.abs(draws - mu).mean():.2f}; "
        f"variance of actual SA explained by mu: {np.corrcoef(mu, sa)[0, 1] ** 2:.1%}")
    sv_act = (allp.saves / allp.shots_against.where(allp.shots_against > 0)).to_numpy()
    p_goal = allp.m_p_goal.to_numpy()
    # ceilings
    ceil_sa = allp.shots_against.to_numpy() * (1 - p_goal)
    card(y, ceil_sa, None, "ceiling: starter's shots known, sv% = model")
    ceil_team = allp.m_mean.to_numpy() * sa / mu
    card(y, ceil_team, None, "ceiling: team shots known (pulls unknown)")
    ceil_sv = allp.m_shots_faced.to_numpy() * np.nan_to_num(sv_act, nan=1 - p_goal)
    card(y, ceil_sv, None, "ceiling: game's sv% known, shots = model")
    # how predictable is save %? binomial deviance of goals given shots, model p vs league p
    ga = allp.goals_against.to_numpy(float)
    n = allp.shots_against.to_numpy(float)
    ok = n > 0

    def bdev(p):
        p = np.clip(p, 1e-4, 1 - 1e-4)
        return -(ga[ok] * np.log(p[ok]) + (n[ok] - ga[ok]) * np.log(1 - p[ok])).sum() / n[ok].sum()
    p_league = np.full_like(p_goal, ga[ok].sum() / n[ok].sum())
    say(f"  save %: per-shot log loss league-constant {bdev(p_league):.5f} vs model {bdev(p_goal):.5f} "
        f"(gain {100 * (1 - bdev(p_goal) / bdev(p_league)):.2f}%); corr(pred sv%, game sv%) "
        f"{np.corrcoef((1 - p_goal)[ok], sv_act[ok])[0, 1]:.3f}")

    say("\n3a. SHOTS FACED FROM THE SKATER MODEL (sum of the opposing skaters' projected SOG)")
    sk = {}
    for lab, f in (("last lines", "backtest_last"), ("tonight lines", "backtest_tonight")):
        b = pd.read_parquet(Path("data/backtest") / f"{f}.parquet", columns=["game_id", "team", "game_type", "sog_hat", "sog"])
        b = b[b.game_type == 2].groupby(["game_id", "team"]).agg(sk_sog=("sog_hat", "sum"), sk_act=("sog", "sum")).reset_index()
        sk[lab] = b.rename(columns={"team": "opp"})   # the opposing team's shots = this goalie's team's shots faced
    for lab, b in sk.items():
        m = allp.merge(b, on=["game_id", "opp"], how="left")
        ok2 = m.sk_sog.notna().to_numpy()
        mm = m[ok2]
        s_sa, s_mu, s_sk = mm.sa_faced.to_numpy(float), mm.m_mu_used.to_numpy(float), mm.sk_sog.to_numpy(float)
        # the skater sum counts empty-net shots and all skaters; rescale with the prior seasons' ratio (walk-forward)
        scaled = np.empty(len(mm))
        blend_w = {}
        for s in SEASONS:
            tr, te = (mm.season != s).to_numpy(), (mm.season == s).to_numpy()
            r = s_sa[tr].sum() / s_sk[tr].sum()
            scaled[te] = s_sk[te] * r
            # blend weight (log scale) chosen on the other seasons
            ws = np.linspace(0, 1, 21)
            err = [np.mean((s_sa[tr] - np.exp(w * np.log(s_sk[tr] * r) + (1 - w) * np.log(s_mu[tr]))) ** 2) for w in ws]
            blend_w[s] = ws[int(np.argmin(err))]
        blended = np.empty(len(mm))
        for s in SEASONS:
            te = (mm.season == s).to_numpy()
            w = blend_w[s]
            blended[te] = np.exp(w * np.log(scaled[te]) + (1 - w) * np.log(s_mu[te]))
        say(f"  [{lab}] n {len(mm):,}: team SA corr  saves-model {np.corrcoef(s_mu, s_sa)[0, 1]:.3f} | skater-sum "
            f"{np.corrcoef(scaled, s_sa)[0, 1]:.3f} | blend {np.corrcoef(blended, s_sa)[0, 1]:.3f} (weights on skater sum "
            f"{blend_w}); SA MSE {np.mean((s_mu - s_sa) ** 2):.2f} / {np.mean((scaled - s_sa) ** 2):.2f} / {np.mean((blended - s_sa) ** 2):.2f}")
        yy = mm.saves.to_numpy(float)
        base_mean = mm.m_mean.to_numpy()
        var = (mm.m_sd.to_numpy()) ** 2
        for name, new_mu in (("skater sum", scaled), ("blend", blended)):
            new_mean = base_mean * new_mu / s_mu
            card(yy, new_mean, nb_pover(new_mean, var * (new_mean / base_mean)), f"saves with SA = {name} ({lab})")
        card(yy, base_mean, nb_pover(base_mean, var), f"saves, current (same rows, NB approx)")
        idx = np.nonzero(ok2)[0]
        card(yy, base_mean, pover_model[idx], f"saves, current (same rows, its own pmf)")

    say("\n3b. LEFTOVER SIGNAL IN SHOTS FACED: residual log(SA/mu) on candidate pregame inputs (fit on 2 seasons, test on the 3rd)")
    b = sk["last lines"]
    gl = pd.read_parquet(Path("data/backtest") / "backtest_last.parquet", columns=["game_id", "team", "game_type", "g_hat"])
    gl = gl[gl.game_type == 2].groupby(["game_id", "team"]).g_hat.sum().rename("g_for").reset_index()
    m = allp.merge(gl, on=["game_id", "team"], how="left").merge(gl.rename(columns={"team": "opp", "g_for": "g_against"}),
                                                                   on=["game_id", "opp"], how="left")
    m = m.merge(b[["game_id", "opp", "sk_sog"]], on=["game_id", "opp"], how="left")
    m["proj_gd"] = m.g_for - m.g_against           # positive: this goalie's team expected to lead
    m["resid"] = np.log(m.sa_faced.clip(lower=1) / m.m_mu_used)
    m["log_sk_ratio"] = np.log(m.sk_sog / m.m_mu_used)
    m["month"] = pd.to_datetime(m.date).dt.month
    m["early"] = (m.ctx_gp < 10).astype(float) if "ctx_gp" in m else 0.0
    cands = ["proj_gd", "ctx_home", "ctx_own_b2b", "ctx_opp_b2b", "ctx_venue", "log_sk_ratio", "early"]
    m = m.dropna(subset=["resid", "proj_gd", "log_sk_ratio"])
    for c in cands:
        if c not in m:
            continue
        gains = []
        for s in SEASONS:
            tr, te = m[m.season != s], m[m.season == s]
            X = np.column_stack([np.ones(len(tr)), tr[c].astype(float)])
            beta = np.linalg.lstsq(X, tr.resid, rcond=None)[0]
            pred = beta[0] + beta[1] * te[c].astype(float)
            gains.append(1 - np.mean((te.resid - pred) ** 2) / np.mean((te.resid - tr.resid.mean()) ** 2))
        say(f"  {c:14s} out-of-season R2 of the residual: " + " ".join(f"{g:+.4f}" for g in gains) + f"  (slope {beta[1]:+.4f})")
    say(f"  residual by month (actual / model SA): " + " ".join(
        f"{mo}:{np.exp(g.resid.mean()):.3f}" for mo, g in m.groupby("month")))

    say("\n3c. SAVE %: how much to trust goalie talent (multiplier k on the shrunk talent; 1 = current)")
    t = allp.g_talent.to_numpy()
    gr = allp.m_goal_ratio.to_numpy()
    for kk in (0.0, 0.5, 1.0, 1.5, 2.0, 3.0):
        pk = np.clip(p_goal + (1 - kk) * t * gr, 0.02, 0.3)
        say(f"  k {kk:.1f}: per-shot log loss {bdev(pk):.5f}" + ("   <- current" if kk == 1.0 else ""))
    # matchup quality: does the predicted per-shot xG track the game's actual xG per shot?
    xg_ps = (allp.xg_on_faced / allp.pbp_sa.where(allp.pbp_sa > 0)).to_numpy()
    mx = p_goal + t * gr
    okx = np.isfinite(xg_ps)
    say(f"  matchup shot quality: corr(predicted xG/shot, game's actual xG/shot) {np.corrcoef(mx[okx], xg_ps[okx])[0, 1]:.3f}; "
        f"sd of game xG/shot {np.nanstd(xg_ps):.4f} vs sd of prediction {mx.std():.4f}")
    say(f"  talent spread: sd of talent {t.std() * 1000:.1f} goals per 1000 shots; sd of game sv% {np.nanstd(sv_act):.4f}")

    say("\n3d. DISTRIBUTION: calibration of P(over) at common lines, and the spread")
    for j, ln in enumerate(LINES):
        say(f"  o{ln}: predicted {pover_model[:, j].mean():.3f}  actual {(y > ln).mean():.3f}")
    a = pd.DataFrame({"p": pover_model.ravel(), "o": (y[:, None] > np.array(LINES)[None, :]).ravel().astype(float)})
    a["bin"] = pd.cut(a.p, np.linspace(0, 1, 11))
    say("  pooled reliability: " + "  ".join(f"{r.p:.2f}->{r.o:.2f}" for r in a.groupby("bin", observed=True)[["p", "o"]].mean().itertuples()))
    pit = np.array([c[int(v) - 1] if v >= 1 else 0.0 for c, v in zip(allc, y)]) + 0.5 * pmf[np.arange(len(y)), np.minimum(y.astype(int), 70)]
    hist = np.histogram(pit, bins=10, range=(0, 1))[0] / len(pit)
    say("  PIT deciles (flat = calibrated spread; U = too narrow): " + " ".join(f"{h:.3f}" for h in hist))
    for scale in (0.9, 1.0, 1.1, 1.2):
        mean = allp.m_mean.to_numpy()
        var = (allp.m_sd.to_numpy() * scale) ** 2
        say(f"  NB approx with sd x{scale}: O/U logloss {ll_lines(y, nb_pover(mean, var)):.4f}")

    say("\n3e. MARKET: closing saves lines 2025-26 (The Odds API cache, best of 5 books; p_mkt = consensus no-vig)")
    bets = pd.read_parquet("data/odds_hist/bets.parquet")
    sv = bets[(bets.kind == "Saves") & (bets.snap == "close")].copy()
    sv["game_id"] = sv.game_id.astype("int64")
    sv["pid"] = sv.pid.astype("int64")
    d25 = P[2025][0]
    cdf25 = np.stack(d25.cdf.to_numpy())
    key = {(g, gl_): i for i, (g, gl_) in enumerate(zip(d25.game_id, d25.goalie))}
    sv["i"] = [key.get((g, p)) for g, p in zip(sv.game_id, sv.pid)]
    sv = sv[sv.i.notna() & sv.won.notna()].copy()
    sv["i"] = sv.i.astype(int)
    sv["p_ours"] = [1 - cdf25[i, int(np.floor(pt))] if s == "Over" else cdf25[i, int(np.floor(pt))]
                    for i, pt, s in zip(sv.i, sv.point, sv.side)]
    over = sv[sv.side == "Over"]

    def ll(p, o):
        p = np.clip(p, 1e-4, 1 - 1e-4)
        return float(-(o * np.log(p) + (1 - o) * np.log(1 - p)).mean())
    o = over.won.to_numpy(float)
    say(f"  {len(over)} goalie-lines (started goalie only): log loss ours {ll(over.p_ours, o):.4f} | market {ll(over.p_mkt, o):.4f} | "
        + " | ".join(f"blend {w:.2f} {ll(w * over.p_ours + (1 - w) * over.p_mkt, o):.4f}" for w in (0.25, 0.5, 0.75)))
    say(f"  mean line {over.point.mean():.1f}; our mean saves on those starts {d25.m_mean.to_numpy()[over.i].mean():.2f}; "
        f"actual {d25.saves.to_numpy()[over.i].mean():.2f}; market implied P(over) {over.p_mkt.mean():.3f}, ours {over.p_ours.mean():.3f}, actual {o.mean():.3f}")
    # where we and the market disagree: who was right?
    gap = over.p_ours - over.p_mkt
    for lo, hi in ((0.05, 1), (-1, -0.05)):
        q = over[(gap > lo) & (gap <= hi)] if lo > 0 else over[(gap > lo) & (gap < hi)]
        say(f"  we are {'higher' if lo > 0 else 'lower'} than market by 5+ pts: n {len(q)}, ours {q.p_ours.mean():.3f}, market {q.p_mkt.mean():.3f}, actual {q.won.mean():.3f}")
    # ROI at the best price, by edge vs the market's no-vig
    both = sv.assign(edge=sv.p_ours - sv.p_mkt)
    dec = np.where(both.best > 0, 1 + both.best / 100, 1 + 100 / both.best.abs())
    both = both.assign(dec=dec, pnl=np.where(both.won == 1, dec - 1, -1.0))
    for e in (0.03, 0.05, 0.08):
        q = both[both.edge >= e]
        say(f"  bets with edge >= {e:.0%}: n {len(q)}, ROI {q.pnl.mean():+.1%} (+/- {q.pnl.std() / np.sqrt(max(len(q), 1)):.1%})")

    say("\n4. THIS SEASON (2026-27 to date, walk-forward = what live.build_state does)")
    d, c = P[2026]
    yy = d.saves.to_numpy(float)
    card(yy, d.m_mean.to_numpy(), np.column_stack([1 - c[:, int(ln)] for ln in LINES]), f"2026-27 n {len(d)}")
    say(f"  team SA: model mu {d.m_mu_used.mean():.2f} vs actual {d.sa_faced.mean():.2f} (bias {(d.m_mu_used - d.sa_faced).mean():+.2f}); "
        f"starter shots model {d.m_shots_faced.mean():.2f} vs actual {d.shots_against.mean():.2f}; "
        f"sv% model {d.m_exp_sv_pct.mean():.4f} vs actual {d.saves.sum() / d.shots_against.sum():.4f}")
    say(f"  online shot_bias at the last day {d.m_shot_bias.iloc[-1]:+.3f}, goal_ratio {d.m_goal_ratio.iloc[-1]:.3f}")
    for s in SEASONS:
        x = P[s][0]
        x = x[pd.to_datetime(x.date) <= pd.Timestamp(f"{s}-10-27")]
        say(f"  same window {s}-{str(s + 1)[2:]}: n {len(x)}, bias {(x.m_mean - x.saves).mean():+.2f}, SA bias {(x.m_mu_used - x.sa_faced).mean():+.2f}, "
            f"MAE {np.abs(x.m_mean - x.saves).mean():.2f}, corr {np.corrcoef(x.m_mean, x.saves)[0, 1]:.3f}")
    # league shot level vs the training seasons
    say("  league SA per team-game by season (actual): " + " ".join(
        f"{s}:{P[s][0].sa_faced.mean():.2f}" for s in (*SEASONS, 2026)))
    OUT.write_text("\n".join(log), encoding="utf-8")


if __name__ == "__main__":
    main()

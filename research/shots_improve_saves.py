"""Carry the SOG candidates into goalie saves, and the market blends (SOG and saves props).
python -m research.shots_improve_saves  (after shots_improve_eval2)"""
import numpy as np
import pandas as pd

import model
from research.goalie_saves_research import BT, LINES, ll_lines, load, nb_pover
from research.shots_improve_build import CACHE, SEASONS
from research.shots_improve_eval import OUT
from research.oddsutil import logit

log = [OUT.read_text(encoding="utf-8")] if OUT.exists() else []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    log.append(s)


E = pd.read_pickle(CACHE / "shots_eval2.pkl")
keep, best = E["keep"], E["best"]
tests = {yr: pd.read_parquet(CACHE / f"shots_{yr}_test.parquet", columns=["game_id", "team", "player", "sog"]) for yr in SEASONS}


def team_sums(name):
    out = []
    for yr in SEASONS:
        mu, _ = keep[(name, yr)]
        t = tests[yr].assign(mu=mu.to_numpy())
        out.append(t.groupby(["game_id", "team"]).mu.sum().rename("sk").reset_index().assign(season=yr))
    return pd.concat(out).rename(columns={"team": "opp"})   # the shooting team = the goalie's opponent


say("\n\n4. CARRIED INTO GOALIE SAVES (actual starter, regular season; saves mean rescaled by the new shots-faced mean)")
P = pd.concat([load(s)[0] for s in SEASONS], ignore_index=True)
P = P[["game_id", "season", "opp", "saves", "sa_faced", "m_mean", "m_sd", "m_mu_used"]]


def card(lab, d, mean, sd):
    y = d.saves.to_numpy(float)
    e = mean - y
    pov = nb_pover(mean, sd ** 2)
    return {"variant": lab, "n": len(d), "MAE": np.abs(e).mean(), "RMSE": np.sqrt((e ** 2).mean()), "corr": np.corrcoef(mean, y)[0, 1],
            "bias": e.mean(), "O/U ll (NB approx)": ll_lines(y, pov)}


rows = []
for name in ("current", best):
    s = team_sums(name)
    d = P.merge(s, on=["game_id", "season", "opp"], how="inner")
    for yr_set, lab in (((2023, 2024, 2025), "2023-26"), ((2026,), "2026-27")):
        m = d[d.season.isin(yr_set)].copy()
        if name == "current":
            rows.append({"set": lab, **card("saves model as is", m, m.m_mean.to_numpy(), m.m_sd.to_numpy())})
        # walk-forward blend of the skater sum into shots faced: scale and weight chosen on the OTHER seasons
        new_mu = np.empty(len(m))
        for s_ in sorted(m.season.unique()):
            tr, te = d[d.season != s_], (m.season == s_).to_numpy()
            r = tr.sa_faced.sum() / tr.sk.sum()
            ws = np.linspace(0, 1, 21)
            best_w, best_dev = 0.0, np.inf
            for w in ws:
                mu = np.exp(w * np.log(tr.sk * r) + (1 - w) * np.log(tr.m_mu_used))
                dev = np.mean(2 * (tr.sa_faced * np.log(np.maximum(tr.sa_faced, 1e-9) / mu) - (tr.sa_faced - mu)))
                if dev < best_dev:
                    best_w, best_dev = w, dev
            x = m[te]
            new_mu[te] = np.exp(best_w * np.log(x.sk * r) + (1 - best_w) * np.log(x.m_mu_used))
            if name == best and lab == "2023-26":
                say(f"  {name}: season {s_} blend weight on the skater sum {best_w:.2f} (chosen on the other seasons)")
        k = new_mu / m.m_mu_used.to_numpy()
        rows.append({"set": lab, **card(f"shots faced blended with {name}", m, m.m_mean.to_numpy() * k, m.m_sd.to_numpy() * np.sqrt(k))})
        sa = m.sa_faced.to_numpy(float)
        say(f"  {lab} {name:30s} team shots faced: corr saves-model mu {np.corrcoef(m.m_mu_used, sa)[0, 1]:.4f} | "
            f"skater sum {np.corrcoef(m.sk, sa)[0, 1]:.4f} | blend {np.corrcoef(new_mu, sa)[0, 1]:.4f}")
pd.set_option("display.width", 220)
say(pd.DataFrame(rows).round(4).to_string(index=False))

say("\n5. MARKET BLENDS (The Odds API closing lines, 2025-26 sample; weights fitted on one half of the season, scored on the other)")
bets = pd.read_parquet("data/odds_hist/bets.parquet")


def ll(p, o):
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
    return float(-(o * np.log(p) + (1 - o) * np.log(1 - p)).mean())


sog = bets[(bets.kind == "SOG") & (bets.snap == "close") & (bets.side == "Over") & bets.won.notna()].copy()
sog["game_id"], sog["pid"] = sog.game_id.astype("int64"), sog.pid.astype("int64")
te25 = tests[2025]
for name in ("current", best):
    mu, alpha = keep[(name, 2025)]
    look = dict(zip(zip(te25.game_id, te25.player), mu.to_numpy()))
    sog[f"mu_{name}"] = [look.get((g, p)) for g, p in zip(sog.game_id, sog.pid)]
    sog[f"p_{name}"] = [float(model.prob_at_least([m_], alpha, int(np.floor(pt)) + 1)[0]) if m_ is not None else np.nan
                        for m_, pt in zip(sog[f"mu_{name}"], sog.point)]
q = sog.dropna(subset=[f"p_{n}" for n in ("current", best)]).sort_values("game_id")
o = q.won.to_numpy(float)
half = q.game_id.rank(pct=True).to_numpy() <= 0.5
say(f"  SOG overs: {len(q):,} player-lines ({q.game_id.nunique()} games) | log loss market {ll(q.p_mkt, o):.4f} | current {ll(q.p_current, o):.4f} | {best} {ll(q[f'p_{best}'], o):.4f}")
for name in ("current", best):
    p = q[f"p_{name}"].to_numpy()
    res = []
    for a, b in ((half, ~half), (~half, half)):
        ws = np.linspace(0, 1, 21)
        w = ws[np.argmin([ll(1 / (1 + np.exp(-(w_ * logit(p[a]) + (1 - w_) * logit(q.p_mkt.to_numpy()[a])))), o[a]) for w_ in ws])]
        bl = 1 / (1 + np.exp(-(w * logit(p[b]) + (1 - w) * logit(q.p_mkt.to_numpy()[b]))))
        res.append((w, ll(bl, o[b]), ll(p[b], o[b]), ll(q.p_mkt.to_numpy()[b], o[b])))
    say(f"  {name}: out-of-half blends " + " | ".join(f"w_model {w:.2f}: blend {x:.4f} vs model {m_:.4f} vs market {k:.4f}" for w, x, m_, k in res))

sv = bets[(bets.kind == "Saves") & (bets.snap == "close") & (bets.side == "Over") & bets.won.notna()].copy()
sv["game_id"], sv["pid"] = sv.game_id.astype("int64"), sv.pid.astype("int64")
d25, c25 = load(2025)
idx = {(g, gl): i for i, (g, gl) in enumerate(zip(d25.game_id, d25.goalie))}
sv["i"] = [idx.get((g, p)) for g, p in zip(sv.game_id, sv.pid)]
sv = sv[sv.i.notna()].copy()
sv["i"] = sv.i.astype(int)
sv["p_model"] = [1 - c25[i, int(np.floor(pt))] for i, pt in zip(sv.i, sv.point)]
sv = sv.sort_values("game_id")
o = sv.won.to_numpy(float)
half = sv.game_id.rank(pct=True).to_numpy() <= 0.5
res = []
for a, b in ((half, ~half), (~half, half)):
    ws = np.linspace(0, 1, 21)
    w = ws[np.argmin([ll(1 / (1 + np.exp(-(w_ * logit(sv.p_model.to_numpy()[a]) + (1 - w_) * logit(sv.p_mkt.to_numpy()[a])))), o[a]) for w_ in ws])]
    bl = 1 / (1 + np.exp(-(w * logit(sv.p_model.to_numpy()[b]) + (1 - w) * logit(sv.p_mkt.to_numpy()[b]))))
    res.append((w, ll(bl, o[b]), ll(sv.p_model.to_numpy()[b], o[b]), ll(sv.p_mkt.to_numpy()[b], o[b])))
say(f"  saves overs: {len(sv)} goalie-lines | log loss market {ll(sv.p_mkt, o):.4f} | saves model {ll(sv.p_model, o):.4f}")
say("  out-of-half blends " + " | ".join(f"w_model {w:.2f}: blend {x:.4f} vs model {m_:.4f} vs market {k:.4f}" for w, x, m_, k in res))
OUT.write_text("\n".join(log), encoding="utf-8")

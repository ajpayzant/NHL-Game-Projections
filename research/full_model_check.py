"""The review build against the live model, walk-forward 2023-24 .. 2026-27 (regular season, last
game's lines): python -m research.full_model_check -> research/full_model_check_out.txt

    live   no season stages, no shot terms (what the cloud app runs)
    new    model.USE_PHASES + model.USE_SHOT_TERMS (the review build)

1. accuracy by season (TOI, SOG, G, A, PTS, PPP) and team SOG totals
2. calibration of SOG by position and tier (the study flagged defence slightly lower)
3. long-odds props: 2+/3+ goals, 2+/3+/4+ points, 2+ assists, 4+/5+/6+ SOG, 2+ PPP, by player type
   (projected-points tier incl. top 1% / 5%, ice time, PP1, shot volume, shooting talent), for both
"""
from pathlib import Path

import numpy as np
import pandas as pd

import backtest as B
import model

OUT = Path(__file__).with_name("full_model_check_out.txt")
log = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    log.append(s)


PROPS = [("2+ goals", "goals", 2, "g_hat", "g"), ("3+ goals", "goals", 3, "g_hat", "g"),
         ("2+ points", "points", 2, "pts_hat", "pts"), ("3+ points", "points", 3, "pts_hat", "pts"),
         ("4+ points", "points", 4, "pts_hat", "pts"), ("2+ assists", "assists", 2, "a_hat", "a"),
         ("4+ SOG", "sog", 4, "sog_hat", "sog"), ("5+ SOG", "sog", 5, "sog_hat", "sog"), ("6+ SOG", "sog", 6, "sog_hat", "sog"),
         ("2+ PPP", "ppp", 2, "ppp_hat", "ppp")]

CACHE = Path(__file__).parent / "_cache" / "ft_last.parquet"   # research/tail_fix.py writes it
f = pd.read_parquet(CACHE) if CACHE.exists() else B.feature_table("last")
runs = {}
for name, ph, sh in (("live", False, False), ("new", True, True)):
    model.USE_PHASES, model.USE_SHOT_TERMS, model.USE_TAIL_FIX = ph, sh, sh
    parts = []
    for yr in (2023, 2024, 2025, 2026):
        co = model.fit(f[f.season.between(B.FIRST_TRAIN, yr - 1)])
        t = f[(f.season == yr) & (f.toi > 0) & (f.game_type == 2)]
        p = model.predict(t, co)
        pr = model.probabilities(p, co)
        keep = t[["game_id", "season", "team", "player", "grp", "toi", "toi_pp", "sog", "goals", "assists", "points", "ppp",
                  "given_pp", "sh_pct", "gp_16"]]
        tails = {f"q_{col}_{k}": model.prop_prob(kind, p[hat], k, co) for _, col, k, hat, kind in PROPS}
        parts.append(pd.concat([keep, p[["toi_hat", "sog_hat", "g_hat", "a_hat", "pts_hat", "ppp_hat"]], pr,
                                pd.DataFrame(tails, index=p.index)], axis=1))
        print(name, yr, flush=True)
    runs[name] = pd.concat(parts, ignore_index=True)
model.USE_PHASES = model.USE_SHOT_TERMS = model.USE_TAIL_FIX = True

STATS = [("sog", "sog_hat"), ("goals", "g_hat"), ("assists", "a_hat"), ("points", "pts_hat"), ("ppp", "ppp_hat")]
say("1. ACCURACY (Poisson deviance; lower is better)")
rows = []
for name, d in runs.items():
    for yr, g in list(d.groupby("season")) + [("2023-26", d[d.season < 2026])]:
        r = {"model": name, "season": yr, "TOI MAE": np.abs(g.toi_hat - g.toi / 60).mean()}
        r |= {y: B.pois_dev(g[y], g[h]) for y, h in STATS}
        tm = g.groupby(["game_id", "team"]).agg(h=("sog_hat", "sum"), a=("sog", "sum"))
        r["team SOG MAE"] = np.abs(tm.h - tm.a).mean()
        rows.append(r)
say(pd.DataFrame(rows).round(4).to_string(index=False))
a, b = runs["live"], runs["new"]
say("\n  change new vs live, 2023-26: " + " | ".join(
    f"{y} {100 * (B.pois_dev(b[b.season < 2026][y], b[b.season < 2026][h]) / B.pois_dev(a[a.season < 2026][y], a[a.season < 2026][h]) - 1):+.2f}%"
    for y, h in STATS))

say("\n2. SOG CALIBRATION, projected / actual (2023-27)")
for name, d in runs.items():
    d = d.assign(tier=pd.qcut(d.sog_hat, 5, labels=["q1 low", "q2", "q3", "q4", "q5 high"]))
    t = d.groupby(["grp", "tier"], observed=True).apply(lambda x: x.sog_hat.sum() / x.sog.sum()).unstack()
    say(f"  [{name}]\n" + t.round(3).to_string())

# ---------------------------------------------------------------- 3. long odds



def p_at(d, hat, kind, k):
    col = {"g_hat": "goals", "pts_hat": "points", "a_hat": "assists", "sog_hat": "sog", "ppp_hat": "ppp"}[hat]
    return d[f"q_{col}_{k}"].to_numpy()


def segments(d: pd.DataFrame) -> dict[str, pd.Series]:
    pts_rank = d.pts_hat.rank(pct=True)
    return {
        "points tier": pd.cut(pts_rank, [0, .4, .8, .95, .99, 1], labels=["below avg (bottom 40%)", "average (40-80%)",
                                                                           "good (80-95%)", "top 5%", "top 1%"]),
        "ice time": pd.cut(d.toi_hat, [0, 14, 17, 20, 40], labels=["<14", "14-17", "17-20", "20+"]),
        "PP unit": d.given_pp.map({0: "no PP", 1: "PP1", 2: "PP2"}),
        "shot volume": pd.cut(d.sog_hat, [0, 1.5, 2.5, 3.5, 20], labels=["<1.5", "1.5-2.5", "2.5-3.5", "3.5+"]),
        "shooting talent": pd.cut(d.sh_pct, [0, .07, .10, .13, 1], labels=["<7%", "7-10%", "10-13%", "13%+"]),
    }


say("\n3. LONG-ODDS PROPS: model % -> actual % (n), ratio, and z = (actual - model) / standard error")
for name, d in runs.items():
    d = d[d.gp_16 > 0].copy()
    say(f"\n  [{name}] overall")
    for lab, col, k, hat, kind in PROPS:
        p = p_at(d, hat, kind, k)
        y = (d[col] >= k).to_numpy(float)
        se = np.sqrt(np.sum(p * (1 - p))) / len(p)
        say(f"    {lab:10s} model {100 * p.mean():5.2f}% actual {100 * y.mean():5.2f}% ratio {y.mean() / p.mean():.3f} "
            f"z {(y.mean() - p.mean()) / se:+.1f}  logloss {B.log_loss(y, p):.4f}")
    for seg, s in segments(d).items():
        say(f"\n  [{name}] by {seg}")
        hdr = None
        tab = []
        for lab, col, k, hat, kind in PROPS:
            p = p_at(d, hat, kind, k)
            y = (d[col] >= k).to_numpy(float)
            row = {"prop": lab}
            for lev in s.cat.categories if hasattr(s, "cat") else sorted(s.dropna().unique()):
                m = (s == lev).to_numpy()
                if m.sum() < 300:
                    row[str(lev)] = ""
                    continue
                pm, ym = p[m].mean(), y[m].mean()
                z = (ym - pm) / (np.sqrt(np.sum(p[m] * (1 - p[m]))) / m.sum())
                row[str(lev)] = f"{100 * pm:.1f}->{100 * ym:.1f} ({z:+.1f})"
            tab.append(row)
        say(pd.DataFrame(tab).to_string(index=False))
OUT.write_text("\n".join(log), encoding="utf-8")

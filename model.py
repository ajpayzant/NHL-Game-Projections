"""The projection: ice time by strength first, then shots, goals and assists on top of it.

    Team minutes   PP / SH seconds  = f(team's usual PP time, opponent's usual SH time)
                   EV pool (F / D)  = f(team's usual EV pool, expected special-teams time)
    Ice time       share of each pool = own recent share (4 / 16 games) + the team's usual
                   share for the slot he is given (line, PP unit, PK unit); shares are then
                   rescaled so a team's minutes add up
    Shots          sum over EV / PP / SH of minutes x his shot rate, times multipliers for
                   the opponent's shot suppression, the rink, home ice and rest
    Goals          shots x his shooting % (heavily regressed), times the opposing goalie
    Assists        minutes x on-ice goals for per minute x his share of those goals as
                   assists, times the opponent's goal suppression, home ice and linemates

Each multiplier is one coefficient of a Poisson GLM fitted with the base as the offset, so
the base is what the player's own history says and the coefficients are how much context
moves it. Everything is stored in a small dict so a projection can be explained line by line.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

import config as C

ST = ("ev", "pp", "sh")
PP_SKATERS, PK_SKATERS = 5.0, 4.0
SOG_LINES = (1, 2, 3, 4, 5, 6)


def _ols(X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    m = y.notna() & X.notna().all(axis=1)
    b = np.linalg.lstsq(X[m].to_numpy(float), y[m].to_numpy(float), rcond=None)[0]
    return dict(zip(X.columns, map(float, b)))


def _apply(X: pd.DataFrame, b: dict[str, float]) -> pd.Series:
    return sum(X[k] * v for k, v in b.items())


# ---------------------------------------------------------------- team minutes and ice time

# Stage of the regular season, by the team's game number. Referees call more penalties in
# October, and the running PP / PK averages lag that: low early, then high once October's games
# are in them. Each stage gets its own offset (games 61+ are the baseline).
PHASES = ((1, 7), (8, 14), (15, 25), (26, 40), (41, 60))
USE_PHASES = True


def _phases(d: pd.DataFrame) -> pd.DataFrame:
    gn = d["team_gn"] if "team_gn" in d else pd.Series(np.nan, index=d.index)
    reg = d.game_type == 2
    return pd.DataFrame({f"ph_{a}": (reg & gn.between(a, b)).astype(float) for a, b in PHASES}, index=d.index)


def _team_X(d: pd.DataFrame) -> dict[str, pd.DataFrame]:
    one = pd.Series(1.0, index=d.index)
    po = (d.game_type == 3).astype(float)  # playoff games run longer (20-minute overtimes)
    ph = _phases(d) if USE_PHASES else pd.DataFrame(index=d.index)
    return {"pp": pd.concat([pd.DataFrame({"const": one, "ew_pp": d.ew_pp, "opp_ew_sh": d.opp_ew_sh, "playoffs": po}), ph], axis=1),
            "sh": pd.concat([pd.DataFrame({"const": one, "ew_sh": d.ew_sh, "opp_ew_pp": d.opp_ew_pp, "playoffs": po}), ph], axis=1)}


def _share_X(d: pd.DataFrame, st: str, slot_col: str) -> pd.DataFrame:
    own = {"ev": "ev_share", "pp": "pp_share", "sh": "sh_share"}[st]
    slot = d[slot_col]
    own16 = d[f"{own}_16"].fillna(slot)
    po = (d.game_type == 3).astype(float)
    return pd.DataFrame({"const": 1.0,
                         "own_4": d[f"{own}_4"].fillna(slot),   # no history: the slot is all we know
                         "own_16": own16,
                         "slot": slot,
                         # playoff benches shorten: big-minute players get more, depth less
                         "po": po, "po_own": po * own16}, index=d.index)


def fit_toi(d: pd.DataFrame) -> dict:
    """d: training rows (finished games) from features.build."""
    tm = d.drop_duplicates(["game_id", "team"])
    co: dict = {"team": {}, "share": {}}
    X = _team_X(tm)
    co["team"]["pp"] = _ols(X["pp"], tm.pp_secs)
    co["team"]["sh"] = _ols(X["sh"], tm.sh_secs)
    exp_st = _apply(X["pp"], co["team"]["pp"]) + _apply(X["sh"], co["team"]["sh"])
    for grp in ("F", "D"):
        pool = tm[f"ew_pool_{grp}"]
        y = d.drop_duplicates(["game_id", "team", "grp"]).query("grp == @grp").set_index(["game_id", "team"]).pool_ev
        y = y.reindex(pd.MultiIndex.from_frame(tm[["game_id", "team"]])).to_numpy()
        Xp = pd.DataFrame({"const": 1.0, "ew_pool": pool, "exp_special": exp_st,
                           "playoffs": (tm.game_type == 3).astype(float)}, index=tm.index)
        co["team"][f"pool_{grp}"] = _ols(Xp, pd.Series(y, index=tm.index))
    targets = {"ev": d.share_ev,
               "pp": d.toi_pp / d.pp_secs.where(d.pp_secs > 0),
               "sh": d.toi_sh / d.sh_secs.where(d.sh_secs > 0)}
    for st in ST:
        for grp in ("F", "D"):
            m = d.grp == grp
            # Two fits, used according to how fresh the line information is (slot_w):
            #   prev - slots from the player's last game, i.e. a stale line chart (slot_w = 0)
            #   act  - tonight's actual slots (slot_w = 1; leaks a little, since tonight's line
            #          ranks come from tonight's ice time, so live use stops at 0.5)
            b_prev = _ols(_share_X(d[m], st, f"slot_{st}"), targets[st][m])
            act = m & d.line.notna()
            b_act = _ols(_share_X(d[act], st, f"aslot_{st}"), targets[st][act])
            co["share"][f"{st}_{grp}"] = {"prev": b_prev, "act": b_act}
    return co


def predict_toi(d: pd.DataFrame, co: dict) -> pd.DataFrame:
    X = _team_X(d)
    out = pd.DataFrame(index=d.index)
    out["team_pp"] = _apply(X["pp"], co["team"]["pp"]).clip(lower=0)
    out["team_sh"] = _apply(X["sh"], co["team"]["sh"]).clip(lower=0)
    pool = pd.Series(np.nan, index=d.index)
    for grp in ("F", "D"):
        m = d.grp == grp
        Xp = pd.DataFrame({"const": 1.0, "ew_pool": d.loc[m, f"ew_pool_{grp}"],
                           "exp_special": out.team_pp[m] + out.team_sh[m],
                           "playoffs": (d.game_type[m] == 3).astype(float)})
        pool[m] = _apply(Xp, co["team"][f"pool_{grp}"])
    out["team_pool"] = pool
    for st in ST:
        sh = pd.Series(np.nan, index=d.index)
        for grp in ("F", "D"):
            m = d.grp == grp
            X = _share_X(d[m], st, f"slot_{st}")
            b = co["share"][f"{st}_{grp}"]
            w = d.slot_w[m] if "slot_w" in d else 0.0
            sh[m] = (1 - w) * _apply(X, b["prev"]) + w * _apply(X, b["act"])
        out[f"share_{st}"] = sh.clip(lower=0)
    # minutes add up within a team: F and D each fill their EV pool; 5 skaters fill a PP, 4 a PK
    key = [d.game_id, d.team]
    out["share_ev"] = out.share_ev / out.share_ev.groupby([d.game_id, d.team, d.grp]).transform("sum")
    out["share_pp"] = (out.share_pp * PP_SKATERS / out.share_pp.groupby(key).transform("sum")).clip(upper=1)
    out["share_sh"] = (out.share_sh * PK_SKATERS / out.share_sh.groupby(key).transform("sum")).clip(upper=1)
    out["toi_ev_hat"] = out.share_ev * out.team_pool / 60
    out["toi_pp_hat"] = out.share_pp * out.team_pp / 60
    out["toi_sh_hat"] = out.share_sh * out.team_sh / 60
    out["toi_hat"] = out.toi_ev_hat + out.toi_pp_hat + out.toi_sh_hat
    if "toi_target" in d and d.toi_target.notna().any():
        out = _override_toi(d, out)
    return out


def _override_toi(d: pd.DataFrame, out: pd.DataFrame) -> pd.DataFrame:
    """A user-set total TOI: the player's EV / PP / PK minutes are scaled to hit it, and the
    EV minutes he gains or loses come out of (go to) his F or D teammates in proportion to
    their own EV time, so the team's EV pool still adds up."""
    out = out.copy()
    tgt = d.toi_target
    fixed = tgt.notna() & (out.toi_hat > 0)
    scale = (tgt / out.toi_hat).where(fixed, 1.0)
    new_ev = out.toi_ev_hat * scale
    delta = (new_ev - out.toi_ev_hat).where(fixed, 0.0)
    grp = [d.game_id, d.team, d.grp]
    free_ev = out.toi_ev_hat.where(~fixed, 0.0)
    give = delta.groupby(grp).transform("sum")
    free_tot = free_ev.groupby(grp).transform("sum")
    adj = (1 - give / free_tot.where(free_tot > 0)).clip(lower=0.3).fillna(1.0)
    out["toi_ev_hat"] = np.where(fixed, new_ev, out.toi_ev_hat * adj)
    out["toi_pp_hat"] = out.toi_pp_hat * scale
    out["toi_sh_hat"] = out.toi_sh_hat * scale
    out["toi_hat"] = out.toi_ev_hat + out.toi_pp_hat + out.toi_sh_hat
    return out


# ---------------------------------------------------------------- shots, goals, assists

def _bases(d: pd.DataFrame, t: pd.DataFrame) -> pd.DataFrame:
    b = pd.DataFrame(index=d.index)
    b["base_sog"] = sum(t[f"toi_{s}_hat"] * d[f"sog_pm_{s}"] for s in ST)
    b["base_a"] = (t.toi_ev_hat * d.gf_pm_ev * d.ashare_ev + t.toi_pp_hat * d.gf_pm_pp * d.ashare_pp
                   + t.toi_sh_hat * d.a_pm_sh)
    # power-play points: PP minutes x team PP goals per minute with him on x his PP IPP
    b["base_ppp"] = t.toi_pp_hat * d.gf_pm_pp * d.ipp_pp
    return b


def _rel(d: pd.DataFrame, col: str) -> pd.Series:
    return d[col].fillna(1.0) if col in d else pd.Series(1.0, index=d.index)


def _ctx(d: pd.DataFrame, t: pd.DataFrame) -> pd.DataFrame:
    tot = t.toi_hat.where(t.toi_hat > 0, 1.0)
    w_ev, w_pp = t.toi_ev_hat / tot, t.toi_pp_hat / tot
    supp = w_ev * d.opp_supp_ev + w_pp * d.opp_supp_pk + (1 - w_ev - w_pp)
    # shots implied by his shot attempts x his on-target share, against the shots-on-goal base
    base_sog = sum(t[f"toi_{s}_hat"] * d[f"sog_pm_{s}"] for s in ST)
    ot = _rel(d, "ot_rel")
    att_base = (t.toi_ev_hat * _rel(d, "env_sog_ev") * _rel(d, "att_rel_ev") * ot
                + t.toi_pp_hat * _rel(d, "env_sog_pp") * _rel(d, "att_rel_pp") * ot + t.toi_sh_hat * d.sog_pm_sh)
    att = np.log((att_base / base_sog.clip(1e-4)).clip(0.33, 3)) if "att_rel_ev" in d else 0.0
    gen = w_ev * _rel(d, "gen_ev") + w_pp * _rel(d, "gen_pp") + (1 - w_ev - w_pp)
    return pd.DataFrame({
        "defence": (d.grp == "D").astype(float),
        "attempts": att,
        "own_team": np.log(gen.clip(0.5, 2)),
        "const": 1.0,
        "opp_shots": np.log(supp.clip(0.5, 2)),
        "rink": np.log(d.rink.clip(0.8, 1.25)),
        "home": d.is_home.astype(float),
        "b2b": (d.rest == 1).astype(float),
        "opp_b2b": (d.opp_rest == 1).astype(float),
        "playoffs": (d.game_type == 3).astype(float),
        "opp_goalie": d.opp_gsax.fillna(0.0),
        "opp_goals": np.log(d.opp_supp_g.clip(0.5, 2)),
        "opp_pk": np.log(d.opp_supp_pkg.clip(0.5, 2)),
        "linemates": d.mate,
        # aging: multi-season histories lag players who are still improving (under 28) and
        # flatter those in decline (over 28); per year away from 28, two separate slopes
        "younger": np.minimum(d.age - 28, 0).clip(-9, 0),
        "older": np.maximum(d.age - 28, 0).clip(0, 12),
    }, index=d.index)


# Shot attempts and the own team's shot generation (research/shots_improve_*): SOG deviance
# -0.19% pooled 2023-27, every season, -1.9% for short histories; team-total spread 0.033 -> 0.026.
USE_SHOT_TERMS = True
SHOT_TERMS = ["attempts", "own_team", "defence"]   # defence: the attempts term alone left D ~3-5% low
TERMS = {"sog": ["const", "spread", "younger", "older", "opp_shots", "rink", "home", "b2b", "opp_b2b", "playoffs"],
         "g": ["const", "spread", "younger", "older", "opp_goalie", "home", "playoffs"],
         "a": ["const", "spread", "younger", "older", "opp_goals", "opp_goalie", "home", "linemates", "playoffs"],
         "ppp": ["const", "spread", "younger", "older", "opp_pk", "opp_goalie", "home", "playoffs"]}
# "spread" is log(base / a typical base): a coefficient above 0 means the history-only base is
# too regressed and the projection is stretched away from the middle by that power
TYPICAL = {"sog": 1.5, "g": 0.15, "a": 0.25, "ppp": 0.12}


def _X(X: pd.DataFrame, base: pd.Series, kind: str) -> pd.DataFrame:
    terms = TERMS[kind] + (SHOT_TERMS if kind == "sog" and USE_SHOT_TERMS else [])
    return X.assign(spread=np.log(base.clip(1e-4) / TYPICAL[kind]))[terms]


def _glm(y, X, offset) -> dict[str, float]:
    m = sm.GLM(y, X, family=sm.families.Poisson(), offset=offset).fit()
    return dict(zip(X.columns, map(float, m.params)))


def _nb_alpha(y, mu) -> float:
    """Method-of-moments overdispersion: Var = mu + alpha * mu^2."""
    y, mu = np.asarray(y, float), np.asarray(mu, float)
    return float(max(0.0, np.sum((y - mu) ** 2 - mu) / np.sum(mu ** 2)))


def fit(d: pd.DataFrame) -> dict:
    """Fit every piece on finished games `d` (rows with some history)."""
    d = d[(d.toi > 0) & d.gp_16.gt(0)]
    co = {"toi": fit_toi(d)}
    t = predict_toi(d, co["toi"])
    b = _bases(d, t)
    X = _ctx(d, t)
    co["sog"] = _glm(d.sog, _X(X, b.base_sog, "sog"), np.log(b.base_sog.clip(1e-4)))
    mu_sog = _mu(_X(X, b.base_sog, "sog"), b.base_sog, co["sog"])
    base_g = mu_sog * d.sh_pct
    co["g"] = _glm(d.goals, _X(X, base_g, "g"), np.log(base_g.clip(1e-4)))
    co["a"] = _glm(d.assists, _X(X, b.base_a, "a"), np.log(b.base_a.clip(1e-4)))
    mu_g = _mu(_X(X, base_g, "g"), base_g, co["g"])
    mu_a = _mu(_X(X, b.base_a, "a"), b.base_a, co["a"])
    co["ppp"] = _glm(d.ppp, _X(X, b.base_ppp, "ppp"), np.log(b.base_ppp.clip(1e-4)))
    mu_ppp = _mu(_X(X, b.base_ppp, "ppp"), b.base_ppp, co["ppp"])
    co["alpha"] = {"sog": _nb_alpha(d.sog, mu_sog), "g": _nb_alpha(d.goals, mu_g),
                   "a": _nb_alpha(d.assists, mu_a), "pts": _nb_alpha(d.points, mu_g + mu_a),
                   "ppp": _nb_alpha(d.ppp, mu_ppp)}
    resid = d.toi / 60 - t.toi_hat
    co["toi_sd"] = {grp: float(resid[d.grp == grp].std()) for grp in ("F", "D")}
    recent = (d.season >= d.season.max() - 1).to_numpy()
    co["tail"] = _fit_tails({"sog": (d.sog, mu_sog), "g": (d.goals, mu_g), "a": (d.assists, mu_a),
                             "pts": (d.points, mu_g + mu_a), "ppp": (d.ppp, mu_ppp)}, co["alpha"], recent)
    return co


# ---------------------------------------------------------------- long-odds calibration
# The count distributions are right through the common lines, but at the long-odds ones the
# Poisson / NB tail was off by up to 15% (4+ points 0.38% projected vs 0.32% actual; 3+ goals
# 10% high; 4+/5+ SOG 3-4% low; research/full_model_check.py). For each line in TAIL_LINES the
# log-odds are remapped (a + b * logit p), fitted with the model on its last two training
# seasons; higher lines use the highest fitted line's map. Walk-forward 2023-27 the ratios
# come to 1.00-1.02 and every line's log loss improves slightly (research/tail_fix.py).
TAIL_LINES = {"sog": (2, 3, 4, 5, 6), "g": (2, 3), "a": (2,), "pts": (2, 3, 4), "ppp": (2,)}
USE_TAIL_FIX = True


def _logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _fit_tails(series: dict, alpha: dict, rows: np.ndarray) -> dict:
    out = {}
    for kind, ks in TAIL_LINES.items():
        y, mu = series[kind]
        y, mu = np.asarray(y, float)[rows], np.asarray(mu, float)[rows]
        out[kind] = {}
        for k in ks:
            X = sm.add_constant(_logit(prob_at_least(mu, alpha[kind], k)))
            hit = (y >= k).astype(float)
            if hit.sum() < 30:
                continue
            a, b = sm.GLM(hit, X, family=sm.families.Binomial()).fit().params
            out[kind][str(k)] = [float(a), float(b)]
    return out


def prop_prob(kind: str, mu, k: int, co: dict) -> np.ndarray:
    """P(stat >= k) for kind in sog / g / a / pts / ppp: the fitted count distribution with the
    long-odds remap. `co` is the fitted coefficients (or just their "alpha" dict: no remap)."""
    alpha = co["alpha"] if "alpha" in co else co
    tail = (co.get("tail") or {}).get(kind) if "alpha" in co and USE_TAIL_FIX else None
    mu = np.atleast_1d(np.asarray(mu, float))
    if k <= 1 or not tail:
        return prob_at_least(mu, alpha[kind], k)
    top = max(int(x) for x in tail)
    cols = []
    for j in range(1, k + 1):
        p = prob_at_least(mu, alpha[kind], j)
        ab = tail.get(str(min(j, top)))
        if j >= 2 and ab:
            p = 1 / (1 + np.exp(-(ab[0] + ab[1] * _logit(p))))
        cols.append(p)
    # P(X >= k) can never exceed P(X >= k - 1)
    return np.minimum.accumulate(np.column_stack(cols), axis=1)[:, -1]


def _mu(X: pd.DataFrame, base: pd.Series, b: dict[str, float]) -> pd.Series:
    return base * np.exp(sum(X[k] * v for k, v in b.items()))


def predict(d: pd.DataFrame, co: dict) -> pd.DataFrame:
    t = predict_toi(d, co["toi"])
    b = _bases(d, t)
    X = _ctx(d, t)
    out = pd.concat([t, b], axis=1)
    out["sog_hat"] = _mu(_X(X, b.base_sog, "sog"), b.base_sog, co["sog"])
    out["base_g"] = out.sog_hat * d.sh_pct
    out["g_hat"] = _mu(_X(X, out.base_g, "g"), out.base_g, co["g"])
    out["a_hat"] = _mu(_X(X, b.base_a, "a"), b.base_a, co["a"])
    out["pts_hat"] = out.g_hat + out.a_hat
    out["ppp_hat"] = _mu(_X(X, b.base_ppp, "ppp"), b.base_ppp, co["ppp"]) if "ppp" in co else np.nan
    return out


LABELS = {"const": "calibration", "spread": "spread (stars vs depth)", "opp_shots": "opponent shot suppression",
          "rink": "rink scorer", "home": "home ice", "b2b": "own back-to-back", "opp_b2b": "opponent back-to-back",
          "playoffs": "playoffs", "opp_goalie": "opposing goalie", "opp_goals": "opponent goal suppression",
          "linemates": "linemates", "opp_pk": "opponent penalty kill", "younger": "age (under 28)", "older": "age (over 28)",
          "attempts": "shot attempts", "own_team": "own team's shot volume", "defence": "defence"}


def multipliers(d: pd.DataFrame, pred: pd.DataFrame, co: dict) -> dict[str, pd.DataFrame]:
    """Per-term multipliers exp(coef * x) behind each projection, for explaining it."""
    X = _ctx(d, pred)
    out = {}
    for kind, base in (("sog", pred.base_sog), ("g", pred.base_g), ("a", pred.base_a), ("ppp", pred.base_ppp)):
        if kind not in co:
            continue
        Xk = _X(X, base, kind)
        out[kind] = pd.DataFrame({LABELS[k]: np.exp(Xk[k] * v) for k, v in co[kind].items()}, index=d.index)
    return out


def prob_at_least(mu, alpha: float, k: int) -> np.ndarray:
    """P(X >= k) under a negative binomial with mean mu (Poisson when alpha is ~0)."""
    mu = np.asarray(mu, float)
    if alpha < 1e-4:
        return stats.poisson.sf(k - 1, mu)
    n = 1.0 / alpha
    return stats.nbinom.sf(k - 1, n, n / (n + mu))


def probabilities(pred: pd.DataFrame, co: dict) -> pd.DataFrame:
    out = pd.DataFrame(index=pred.index)
    for k in SOG_LINES:
        out[f"p_sog_{k}"] = prop_prob("sog", pred.sog_hat, k, co)
    for k in (1, 2):
        out[f"p_g_{k}"] = prop_prob("g", pred.g_hat, k, co)
        out[f"p_a_{k}"] = prop_prob("a", pred.a_hat, k, co)
    for k in (1, 2, 3):
        out[f"p_pts_{k}"] = prop_prob("pts", pred.pts_hat, k, co)
    if "ppp" in co["alpha"]:
        for k in (1, 2):
            out[f"p_ppp_{k}"] = prop_prob("ppp", pred.ppp_hat, k, co)
    return out


def save(co: dict, name: str = "coefs.json") -> None:
    C.MODELS.mkdir(parents=True, exist_ok=True)
    (C.MODELS / name).write_text(json.dumps(co, indent=2))


def load(name: str = "coefs.json") -> dict:
    return json.loads((C.MODELS / name).read_text())

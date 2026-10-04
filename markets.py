"""Team markets from the player projections: moneyline, puck line (+/-1.5) and total.

A team's expected goals is the sum of its skaters' projected goals (which include overtime and
empty-net goals). The final score is built in three steps, because hockey scores are not plain
Poisson:

1. Regulation goals with both goalies in: independent Poisson for each team, using the
   expected goals less the share that are scored in overtime or into an empty net.
2. Empty nets: a team leading by 1-3 late adds an empty-net goal with probability EN[lead]
   (and possibly a second). This is why 2-goal wins are rarer than 3-goal wins in the NHL.
3. Ties go to overtime / shootout; the winner gets +1 goal, as books grade the final score.
   The better team is slightly more likely to win it.

EN, OT_TILT, TIE_BOOST, STRETCH and NON_REG_SHARE were fitted on 2023-24 (fit_params) and
checked on 2024-25 and 2025-26 (research/team_markets.py).
"""
from __future__ import annotations

import numpy as np
from scipy import stats

N = 13                      # regulation goals 0..12 per team
NON_REG_SHARE = 0.081       # expected goals held out of regulation play (OT, empty net); set so the
                            # expected final total equals the sum of the player projections
EN = {1: 0.49, 2: 0.65, 3: 0.18}   # P(leader adds an empty-net goal) by lead; fitted on 2023-24
EN_SECOND = 0.0                    # a second empty-net goal: fitted as ~0
OT_TILT = 0.53              # 0 = OT is a coin flip, 1 = OT win share follows expected goals
TIE_BOOST = 0.77            # tied games are tighter than independent scoring implies (more OT)
STRETCH = 1.40              # >1 widens the gap: summed player projections compress team differences
TOTALS = [x + 0.5 for x in range(3, 10)]


def _stretch(lh, la, s):
    g = np.sqrt(lh * la)
    return g * (lh / g) ** s, g * (la / g) ** s


def final_scores(lam_h, lam_a, en=None, en2=None, tilt=None, share=None, tie=None, stretch=None) -> np.ndarray:
    """P(final home goals = i, away = j) for each game: array (n, N+3, N+3)."""
    en, en2 = en or EN, EN_SECOND if en2 is None else en2
    tilt = OT_TILT if tilt is None else tilt
    share = NON_REG_SHARE if share is None else share
    tie = TIE_BOOST if tie is None else tie
    lh, la = _stretch(np.asarray(lam_h, float), np.asarray(lam_a, float), STRETCH if stretch is None else stretch)
    k = np.arange(N)
    ph = stats.poisson.pmf(k[None, :], (lh * (1 - share))[:, None])
    pa = stats.poisson.pmf(k[None, :], (la * (1 - share))[:, None])
    reg = ph[:, :, None] * pa[:, None, :]
    reg = reg * np.where(np.eye(N, dtype=bool), 1 + tie, 1.0)[None]
    reg = reg / reg.sum((1, 2), keepdims=True)
    final_scores.last_reg_tie = np.trace(reg, axis1=1, axis2=2)
    out = np.zeros((len(lh), N + 3, N + 3))
    p_ot = 0.5 + tilt * (lh / (lh + la) - 0.5)
    for i in range(N):
        for j in range(N):
            p = reg[:, i, j]
            d = i - j
            if d == 0:
                out[:, i + 1, j] += p * p_ot
                out[:, i, j + 1] += p * (1 - p_ot)
                continue
            e1 = en.get(abs(d), 0.0)
            # leader adds 0, 1 or 2 empty-net goals
            for extra, w in ((0, 1 - e1), (1, e1 * (1 - en2)), (2, e1 * en2)):
                if d > 0:
                    out[:, i + extra, j] += p * w
                else:
                    out[:, i, j + extra] += p * w
    return out


def game_probs(lam_h, lam_a, **kw) -> dict:
    F = final_scores(lam_h, lam_a, **kw)
    m = np.arange(F.shape[1])
    margin = m[:, None] - m[None, :]
    total = m[:, None] + m[None, :]
    reg_tie = final_scores.last_reg_tie
    return {
        "home_win": (F * (margin > 0)).sum((1, 2)),
        "home_m15": (F * (margin >= 2)).sum((1, 2)),    # home -1.5 covers
        "away_m15": (F * (margin <= -2)).sum((1, 2)),   # away -1.5 covers
        "over": {t: (F * (total > t)).sum((1, 2)) for t in TOTALS},
        "reg_tie": reg_tie,
        "exp_total": (F * total).sum((1, 2)),
    }


def fit_params(lam_h, lam_a, gh, ga) -> dict:
    """Maximum-likelihood EN / OT parameters from final scores (gh, ga)."""
    from scipy.optimize import minimize
    gh, ga = np.minimum(np.asarray(gh), N + 2), np.minimum(np.asarray(ga), N + 2)

    sig = lambda v: 1 / (1 + np.exp(-v))

    def unpack(x):
        e1, e2, e3, tilt = sig(x[:4])
        return dict(en={1: e1, 2: e2, 3: e3}, en2=0.0, tilt=tilt, tie=np.exp(x[4]) - 1, stretch=np.exp(x[5]),
                    share=0.15 * sig(x[6]))

    def nll(x):
        F = final_scores(lam_h, lam_a, **unpack(x))
        return -np.mean(np.log(np.clip(F[np.arange(len(gh)), gh, ga], 1e-12, 1)))
    lo = lambda p: np.log(p / (1 - p))
    x0 = np.array([lo(0.28), lo(0.49), lo(0.11), 0.0, np.log(1.3), 0.0, lo(0.081 / 0.15)])
    r = minimize(nll, x0, method="Nelder-Mead", options={"maxiter": 1500, "xatol": 1e-3, "fatol": 1e-7})
    return unpack(r.x) | {"nll": r.fun}


def fair_total(over: dict) -> float:
    """The x.5 total whose over is closest to 50%."""
    return min(over, key=lambda t: abs(float(np.asarray(over[t]).ravel()[0]) - 0.5))

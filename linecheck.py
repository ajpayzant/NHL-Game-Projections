"""Price one betting line by hand: the Edges tab's "Check a line".

Each market uses the same distribution the rest of the tool prices with:
    skater props   negative binomial around the projection (model.prob_at_least)
    goalie saves   the saves model's simulated distribution, as if he starts
    game lines     the projected final score (markets.final_scores)
Whole-number lines can push: the push is refunded, so the edge is measured on the
probability of winning given no push, and the EV counts the push as money back.
"""
from __future__ import annotations

import numpy as np

import markets
import model


def to_decimal(american: float) -> float:
    a = float(american)
    return 1 + a / 100 if a > 0 else 1 + 100 / abs(a)


def skater(mean: float, alpha: float, line: float, over: bool) -> tuple[float, float]:
    """(P(win), P(push)) for an over / under on a skater stat."""
    at_least = lambda k: float(model.prob_at_least([mean], alpha, int(k))[0]) if k > 0 else 1.0
    if float(line).is_integer():
        k = int(line)
        push = at_least(k) - at_least(k + 1)
        return (at_least(k + 1) if over else 1 - at_least(k)), push
    k = int(np.floor(line)) + 1
    return (at_least(k) if over else 1 - at_least(k)), 0.0


def from_pmf(pmf: np.ndarray, line: float, over: bool) -> tuple[float, float]:
    """(P(win), P(push)) from a probability mass over 0, 1, 2, ... (goalie saves)."""
    x = np.arange(len(pmf))
    win = pmf[x > line].sum() if over else pmf[x < line].sum()
    return float(win), float(pmf[x == line].sum())


def game(lam_home: float, lam_away: float, market: str, home_side: bool, line: float = 0.0,
         over: bool = True) -> tuple[float, float]:
    """(P(win), P(push)) for a moneyline, puck line (line = this team's handicap, e.g. -1.5) or total."""
    F = markets.final_scores(np.array([lam_home]), np.array([lam_away]))[0]
    m = np.arange(F.shape[0])
    margin = m[:, None] - m[None, :]
    if market == "Total":
        t = m[:, None] + m[None, :]
        win = F[t > line].sum() if over else F[t < line].sum()
        return float(win), float(F[t == line].sum())
    mg = margin if home_side else -margin
    if market == "Moneyline":
        return float(F[mg > 0].sum()), 0.0
    v = mg + line
    return float(F[v > 0].sum()), float(F[v == 0].sum())


def evaluate(p_win: float, p_push: float, odds: float, other_odds: float | None = None) -> dict:
    """Our probability against the book's for one side at its price.
    book_p is the book's no-vig probability when the other side's price is given, else the
    raw implied probability (which still includes the book's margin)."""
    dec = to_decimal(odds)
    p = p_win / max(1 - p_push, 1e-9)                    # given no push
    implied = 1 / dec
    if other_odds is not None:
        o = 1 / to_decimal(other_odds)
        book_p, novig = implied / (implied + o), True
    else:
        book_p, novig = implied, False
    ev = 100 * (p_win * (dec - 1) - (1 - p_win - p_push))  # push = stake back
    return {"p": p, "push": p_push, "book_p": book_p, "novig": novig, "edge": p - book_p, "ev": ev, "dec": dec}

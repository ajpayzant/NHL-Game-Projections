"""Compare sportsbook prices (odds.py) with the model's probabilities, for the Edges tab.

Nothing here changes a projection: it reads the same projections the other tabs show and
prices each book's line with the model's own distributions
    skater props   model.prob_at_least with the fitted overdispersion (any x.5 line)
    goalie saves   the saves model's P(over) for that goalie, as if he starts (books void
                   a goalie prop when he doesn't)
    game lines     the full projected final score (markets.final_scores), so any spread or
                   total a book posts is priced exactly

For each bet: the book's price, its no-vig probability (from that book's two sides), the
market consensus (average no-vig across books), the model's probability, the edge (model -
market) and the expected value of a $1 bet at that book's price.
"""
from __future__ import annotations

import difflib

import numpy as np
import pandas as pd

import markets
import model
from live import _surname, key

PROPS = {"player_points": ("PTS", "pts_hat", "pts"), "player_shots_on_goal": ("SOG", "sog_hat", "sog"),
         "player_goals": ("G", "g_hat", "g"), "player_assists": ("A", "a_hat", "a"),
         "player_power_play_points": ("PPP", "ppp_hat", "ppp")}
KIND = {"h2h": "Moneyline", "spreads": "Puck line", "totals": "Total", "player_total_saves": "Saves",
        **{k: v[0] for k, v in PROPS.items()}}


def decimal(price: pd.Series) -> pd.Series:
    price = price.astype(float)
    return np.where(price > 0, 1 + price / 100, 1 + 100 / price.abs())


def _devig(o: pd.DataFrame) -> pd.Series:
    """No-vig probability of each outcome from its book's two sides of the same bet."""
    imp = 1 / o.dec
    grp = [o.event_id, o.book, o.market, o.player.fillna(""), o.pair]
    tot = imp.groupby(grp).transform("sum")
    n = imp.groupby(grp).transform("size")
    return (imp / tot).where(n == 2)


def match_games(odds: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """Attach our game_id by team names (The Odds API uses full names, as the schedule does)."""
    g = games.assign(k=[key(h) + "|" + key(a) for h, a in zip(games.home_name, games.away_name)])
    gid = dict(zip(g.k, g.game_id))
    return odds.assign(game_id=[gid.get(key(h) + "|" + key(a)) for h, a in zip(odds.home, odds.away)])


def compare(odds: pd.DataFrame, games: pd.DataFrame, proj: pd.DataFrame, alpha: dict,
            goalies: pd.DataFrame | None = None, now: pd.Timestamp | None = None) -> tuple[pd.DataFrame, dict]:
    """One row per (bet, book) with the model's probability. Also returns match counts."""
    now = now or pd.Timestamp.now(tz="UTC")
    o = match_games(odds, games)
    o = o[o.game_id.notna() & (pd.to_datetime(o.commence_time, utc=True) > now)].copy()   # pregame only
    o["game_id"] = o.game_id.astype(int)
    o = o[o.game_id.isin(proj.game_id.unique())]
    stats = {"prices": len(o)}
    if o.empty:
        return pd.DataFrame(), stats
    o["dec"] = decimal(o.price)
    # the two sides of a bet: over/under at one point, or two teams (spreads by |point|)
    o["pair"] = np.where(o.market == "spreads", o.point.abs(), o.point.fillna(0.0))
    o["novig"] = _devig(o)
    o["p_model"] = np.nan
    o["proj"] = np.nan   # the model's projected number for the stat (total goals, SOG, saves...)
    o["proj_own"] = o["proj_opp"] = np.nan   # game lines: projected goals, this team / opponent
    gm = games.set_index("game_id")
    lam = proj.groupby(["game_id", "team"]).g_hat.sum()

    # ---- game lines from the projected final score
    for gid, idx in o[o.market.isin(["h2h", "spreads", "totals"])].groupby("game_id").groups.items():
        g = gm.loc[gid]
        lh, la = lam.get((gid, g.home), np.nan), lam.get((gid, g.away), np.nan)
        if not (np.isfinite(lh) and np.isfinite(la)):
            continue
        F = markets.final_scores(np.array([lh]), np.array([la]))[0]
        m = np.arange(F.shape[0])
        margin = m[:, None] - m[None, :]          # home - away
        total = m[:, None] + m[None, :]
        for i in idx:
            r = o.loc[i]
            if r.market != "h2h" and float(r.point).is_integer():   # whole-number lines can push: skipped
                continue
            side_home = key(r["name"]) == key(g.home_name)
            if r.market in ("h2h", "spreads"):   # projected score, this team first
                o.at[i, "proj_own"], o.at[i, "proj_opp"] = (lh, la) if side_home else (la, lh)
            if r.market == "h2h":
                p = F[margin > 0].sum() if side_home else F[margin < 0].sum()
            elif r.market == "spreads":
                mg = margin if side_home else -margin  # this team's margin
                p = F[mg + r.point > 0].sum() / max(1 - F[mg + r.point == 0].sum(), 1e-9)
            else:
                over = str(r["name"]).lower() == "over"
                win = (total > r.point) if over else (total < r.point)
                p = F[win].sum() / max(1 - F[total == r.point].sum(), 1e-9)
                o.at[i, "proj"] = float((F * total).sum())
            o.at[i, "p_model"] = p

    # ---- skater props
    pr = proj.assign(k=proj["name"].map(key), sur=proj["name"].map(_surname))
    look = {(gid, k): row for gid, k, row in zip(pr.game_id, pr.k, pr.itertuples(index=False))}
    # spelling variants ("Dmitriy" / "Dmitri", "Maxwell" / "Max"): a surname unique in that game
    once = pr.groupby(["game_id", "sur"]).sur.transform("size") == 1
    by_sur = {(gid, s): row for gid, s, row, u in zip(pr.game_id, pr.sur, pr.itertuples(index=False), once) if u}
    sk = o[o.market.isin(list(PROPS))]
    matched = 0
    for i, r in sk.iterrows():
        row = look.get((r.game_id, key(r.player)))
        if row is None:
            row = by_sur.get((r.game_id, _surname(r.player)))
            if row is not None and not _same_first(row.name, r.player):   # Florian vs Arber Xhekaj
                row = None
        if row is None or r.point is None or not np.isfinite(r.point) or float(r.point).is_integer():
            continue
        _, col, a = PROPS[r.market]
        p_over = float(model.prob_at_least([getattr(row, col)], alpha[a], int(np.floor(r.point)) + 1)[0])
        o.at[i, "p_model"] = p_over if str(r["name"]).lower() == "over" else 1 - p_over
        o.at[i, "proj"] = getattr(row, col)
        o.at[i, "team"] = row.team
        matched += 1
    stats["skater props"], stats["skater props matched"] = len(sk), matched

    # ---- goalie saves (as if he starts)
    sv = o[o.market == "player_total_saves"]
    gmatch = 0
    if goalies is not None and len(goalies):
        gl = {(gid, key(n)): (t, d, m) for gid, n, t, d, m in
              zip(goalies.game_id, goalies.goalie, goalies.team, goalies.p_over, goalies["mean"])}
        for i, r in sv.iterrows():
            hit = gl.get((r.game_id, key(r.player)))
            if hit is None or r.point is None or not np.isfinite(r.point) or float(r.point).is_integer():
                continue
            team, p_over, mean = hit
            p = p_over.get(float(r.point))
            if p is None:
                continue
            o.at[i, "p_model"] = p if str(r["name"]).lower() == "over" else 1 - p
            o.at[i, "proj"] = mean
            o.at[i, "team"] = team
            gmatch += 1
    stats["saves props"], stats["saves props matched"] = len(sv), gmatch

    o = o[o.p_model.notna()].copy()
    o["kind"] = o.market.map(KIND)
    o["bet"] = [_label(r) for r in o.itertuples(index=False)]
    o["game"] = [f"{gm.loc[g].away} @ {gm.loc[g].home}" for g in o.game_id]
    o["date"] = [gm.loc[g].date for g in o.game_id]
    o["market_p"] = o.groupby(["game_id", "market", "bet"]).novig.transform("mean")
    o["books"] = o.groupby(["game_id", "market", "bet"]).book.transform("nunique")
    o["edge"] = o.p_model - o.novig          # vs this book, so Edge = Our % - Book % as shown
    o["ev"] = o.p_model * o.dec - 1
    # plain-language columns for the tab: who, which line, and what the model projects
    abbr = {key(n): a for n, a in zip(games.home_name, games.home)} | \
        {key(n): a for n, a in zip(games.away_name, games.away)}
    o["subject"] = [r.player if r.market in PROPS or r.market == "player_total_saves"
                    else ("Game total" if r.market == "totals" else abbr.get(key(r.name), r.name))
                    for r in o.itertuples(index=False)]
    o["line"] = [_line(r) for r in o.itertuples(index=False)]
    o["proj_text"] = [_proj_text(r) for r in o.itertuples(index=False)]
    return o, stats


WORDS = {"PTS": "points", "SOG": "shots", "G": "goals", "A": "assists", "PPP": "PP points", "Saves": "saves"}


def _line(r) -> str:
    """'Under 1.5 shots', 'Moneyline', 'Puck line -1.5', 'Over 6.5 goals'."""
    pt = f"{r.point:g}" if r.point is not None and np.isfinite(r.point) else ""
    if r.market == "h2h":
        return "Moneyline"
    if r.market == "spreads":
        return f"Puck line {'+' if r.point > 0 else ''}{pt}"
    side = "Over" if str(r.name).lower() == "over" else "Under"
    return f"{side} {pt} {'goals' if r.market == 'totals' else WORDS[KIND[r.market]]}"


def _proj_text(r) -> str:
    """The model's own number for the bet: '1.24 shots', '23.0 saves', '6.12 goals', '3.19-2.83'."""
    if r.market in ("h2h", "spreads"):
        return f"{r.proj_own:.2f}–{r.proj_opp:.2f}" if np.isfinite(r.proj_own) else ""
    if not np.isfinite(r.proj):
        return ""
    if r.market == "totals":
        return f"{r.proj:.2f} goals"
    word = WORDS[KIND[r.market]]
    return f"{r.proj:.1f} {word}" if r.market == "player_total_saves" else f"{r.proj:.2f} {word}"


def _same_first(a: str, b: str) -> bool:
    """First names that are the same person spelled two ways: same initial (Max / Maxwell,
    Zack / Zachary) or a close transliteration (Yegor / Egor)."""
    fa, fb = key(str(a).split()[0]), key(str(b).split()[0])
    return fa[:1] == fb[:1] or difflib.SequenceMatcher(None, fa, fb).ratio() >= 0.75


def _label(r) -> str:
    pt = f"{r.point:g}" if r.point is not None and np.isfinite(r.point) else ""
    if r.market == "h2h":
        return f"{r.name} ML"
    if r.market == "spreads":
        return f"{r.name} {'+' if r.point > 0 else ''}{pt}"
    if r.market == "totals":
        return f"{r.name} {pt}"
    side = "o" if str(r.name).lower() == "over" else "u"
    return f"{r.player} {side}{pt} {KIND[r.market]}"

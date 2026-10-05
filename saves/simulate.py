"""Monte Carlo distribution of a starting goalie's saves in one game.

    shots on goal the team faces   NegBin(mu, alpha)         (shots_model)
    each shot                      goal with p_starter       (goalie_model.SavePct), lower in games
                                   with more shots than expected (score effects, script_slope)
    after each goal he allows      pulled with hazard(k, period); a small goal-free exit rate
    after a pull                   the rest of the shots go to the backup (p_backup)

Shot times are uniform over regulation, which is all the hazard needs (it keys on period).
Returns the full distribution of the starter's saves; the backup's saves come along so a
team's total can be shown too.
"""
from __future__ import annotations

import numpy as np

MAX_SHOTS = 90


def negbin(rng: np.random.Generator, mu: float, alpha: float, n: int) -> np.ndarray:
    """NB2: variance = mu + alpha * mu^2, as a gamma-Poisson mixture."""
    if alpha <= 1e-6:
        return rng.poisson(mu, n)
    lam = rng.gamma(1.0 / alpha, mu * alpha, n)
    return rng.poisson(lam)


def simulate(mu: float, alpha: float, p_starter: float, p_backup: float, pull_model,
             n: int = 20000, seed: int | None = 0, script_slope: float = 0.0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    N = np.minimum(negbin(rng, mu, alpha, n), MAX_SHOTS)
    # per-game shot quality falls as volume rises; normalised so expected goals are unchanged
    script = 1 + script_slope * (N - mu)
    script = np.clip(script / (1 + script_slope * (1 + alpha * mu)), 0.2, 3.0)
    p_starter = np.clip(p_starter * script, 0.005, 0.6)
    p_backup = np.clip(p_backup * script, 0.005, 0.6)
    width = int(N.max()) if len(N) else 0
    width = max(width, 1)
    idx = np.arange(width)[None, :]
    live = idx < N[:, None]
    # fraction of regulation; padding sorts to the end so a game's N shots span the whole game
    t = np.sort(np.where(live, rng.random((n, width)), 2.0), axis=1)
    period = np.minimum((t * 3).astype(int) + 1, 3)
    goal_s = (rng.random((n, width)) < p_starter[:, None]) & live
    k = np.cumsum(goal_s, axis=1)                                # goals allowed so far (starter)
    pull_now = np.zeros((n, width), dtype=bool)
    gi = np.nonzero(goal_s)
    pull_now[gi] = rng.random(len(gi[0])) < pull_model.hazard(k[gi], period[gi])
    # goal-free exits: a random moment in a small share of games
    exit_game = rng.random(n) < pull_model.exit_rate
    exit_at = np.where(exit_game, (rng.random(n) * N).astype(int), MAX_SHOTS + 1)
    has_pull = pull_now.any(axis=1)
    first_pull = np.where(has_pull, pull_now.argmax(axis=1), MAX_SHOTS + 1)
    # some pulls wait for the intermission: he then faces the rest of that period's shots
    wait = has_pull & (rng.random(n) < pull_model.intermission_share)
    if wait.any():
        rows = np.nonzero(wait)[0]
        pull_period = period[rows, first_pull[rows]]
        shots_to_end = (live[rows] & (period[rows] <= pull_period[:, None])).sum(axis=1)
        first_pull[rows] = shots_to_end - 1
    # he is in net for shots 0..first_pull (inclusive: the goal that got him pulled)
    last_in = np.minimum(first_pull, exit_at)
    in_net = live & (idx <= last_in[:, None])
    starter_sa = in_net.sum(axis=1)
    starter_ga = (goal_s & in_net).sum(axis=1)
    backup_sa = N - starter_sa
    backup_ga = rng.binomial(backup_sa, p_backup)
    return {
        "team_sa": N,
        "saves": starter_sa - starter_ga,
        "shots_faced": starter_sa,
        "goals_against": starter_ga,
        "pulled": last_in < N - 1,
        "backup_saves": backup_sa - backup_ga,
        "team_saves": N - starter_ga - backup_ga,
    }


def summarize(sims: dict[str, np.ndarray], lines: tuple[float, ...] = ()) -> dict:
    sv = sims["saves"]
    out = {
        "mean": float(sv.mean()), "median": float(np.median(sv)), "sd": float(sv.std()),
        "p10": float(np.percentile(sv, 10)), "p25": float(np.percentile(sv, 25)),
        "p75": float(np.percentile(sv, 75)), "p90": float(np.percentile(sv, 90)),
        "shots_faced": float(sims["shots_faced"].mean()),
        "goals_against": float(sims["goals_against"].mean()),
        "pull_rate": float(sims["pulled"].mean()),
        "team_saves": float(sims["team_saves"].mean()),
    }
    for ln in lines:
        out[f"p_over_{ln}"] = float((sv > ln).mean())
    return out


def pmf(sims: dict[str, np.ndarray], key: str = "saves", upto: int = 70) -> np.ndarray:
    v = np.minimum(sims[key], upto)
    return np.bincount(v, minlength=upto + 1) / len(v)

"""Dixon-Coles (1997): per-team attack/defense strength via Poisson goal
counts, fit separately per domestic league (the 8 football-data.co.uk
leagues) since attack/defense parameters are only comparable within a pool of
teams that actually play each other in a connected round-robin — which the
three UEFA competitions and the Azerbaijani league don't reliably offer (see
model/league_strength.py for how those are handled instead, via the Elo
bridge).

Two corrections on top of plain independent-Poisson:
- rho: fixes the well-documented underestimation of 0-0/1-0/0-1/1-1 scorelines.
- time decay: exp(-xi * days_ago), xi converted from Dixon & Coles' published
  half-week value (0.0065) to a daily rate, so recent form dominates without
  discarding older matches outright.

Identifiability: the likelihood is invariant to shifting every attack value by
+c and every defense value by -c (they cancel in both lambda and mu), so
attack/defense need a normalizing constraint. Rather than a constrained
optimizer, we fit freely and re-center attack to sum-to-zero afterward,
shifting defense by the same amount — mathematically equivalent, much simpler.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
from scipy.optimize import minimize
from scipy.stats import poisson

XI_PER_DAY = 0.0065 / 3.5  # Dixon & Coles' published per-half-week decay, converted to per-day


class LeagueFit:
    def __init__(self, attack: dict[int, float], defense: dict[int, float], home_adv: float, rho: float, match_counts: dict[int, int]):
        self.attack = attack
        self.defense = defense
        self.home_adv = home_adv
        self.rho = rho
        self.match_counts = match_counts

    def lambdas(self, home_team_id: int, away_team_id: int) -> tuple[float, float] | None:
        if home_team_id not in self.attack or away_team_id not in self.attack:
            return None
        log_lambda = self.attack[home_team_id] + self.defense[away_team_id] + self.home_adv
        log_mu = self.attack[away_team_id] + self.defense[home_team_id]
        return float(np.exp(log_lambda)), float(np.exp(log_mu))


def tau(home_goals: np.ndarray, away_goals: np.ndarray, lam: np.ndarray, mu: np.ndarray, rho: float) -> np.ndarray:
    tau = np.ones_like(lam)
    m00 = (home_goals == 0) & (away_goals == 0)
    m10 = (home_goals == 1) & (away_goals == 0)
    m01 = (home_goals == 0) & (away_goals == 1)
    m11 = (home_goals == 1) & (away_goals == 1)
    tau[m00] = 1 - lam[m00] * mu[m00] * rho
    tau[m10] = 1 + mu[m10] * rho
    tau[m01] = 1 + lam[m01] * rho
    tau[m11] = 1 - rho
    return np.clip(tau, 1e-6, None)


def fit_league(matches: list[dict], as_of: dt.datetime) -> LeagueFit:
    """matches: dicts with home_team_id, away_team_id, home_goals, away_goals,
    utc_kickoff (all required, all finished matches only)."""
    team_ids = sorted({m["home_team_id"] for m in matches} | {m["away_team_id"] for m in matches})
    idx = {tid: i for i, tid in enumerate(team_ids)}
    n = len(team_ids)
    if n < 4 or len(matches) < 10:
        raise ValueError(f"not enough data to fit a league: {n} teams, {len(matches)} matches")

    home_idx = np.array([idx[m["home_team_id"]] for m in matches])
    away_idx = np.array([idx[m["away_team_id"]] for m in matches])
    home_goals = np.array([m["home_goals"] for m in matches], dtype=float)
    away_goals = np.array([m["away_goals"] for m in matches], dtype=float)
    days_ago = np.array([max((as_of - m["utc_kickoff"]).total_seconds() / 86400, 0.0) for m in matches])
    weights = np.exp(-XI_PER_DAY * days_ago)

    x0 = np.concatenate([np.zeros(n), np.zeros(n), [0.25], [-0.05]])
    bounds = [(-3, 3)] * n + [(-3, 3)] * n + [(-2, 2), (-0.4, 0.4)]

    def neg_log_likelihood(params: np.ndarray) -> float:
        attack, defense = params[:n], params[n : 2 * n]
        home_adv, rho = params[2 * n], params[2 * n + 1]

        log_lambda = attack[home_idx] + defense[away_idx] + home_adv
        log_mu = attack[away_idx] + defense[home_idx]
        lam, mu = np.exp(log_lambda), np.exp(log_mu)

        ll = poisson.logpmf(home_goals, lam) + poisson.logpmf(away_goals, mu)
        ll = ll + np.log(tau(home_goals, away_goals, lam, mu, rho))
        return float(-np.sum(weights * ll))

    result = minimize(neg_log_likelihood, x0, method="L-BFGS-B", bounds=bounds)

    attack, defense = result.x[:n], result.x[n : 2 * n]
    home_adv, rho = float(result.x[2 * n]), float(result.x[2 * n + 1])

    shift = attack.mean()
    attack = attack - shift
    defense = defense + shift

    match_counts: dict[int, int] = {}
    for m in matches:
        match_counts[m["home_team_id"]] = match_counts.get(m["home_team_id"], 0) + 1
        match_counts[m["away_team_id"]] = match_counts.get(m["away_team_id"], 0) + 1

    return LeagueFit(
        attack=dict(zip(team_ids, attack.tolist())),
        defense=dict(zip(team_ids, defense.tolist())),
        home_adv=home_adv,
        rho=rho,
        match_counts=match_counts,
    )

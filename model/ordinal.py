"""Ordered logistic regression on (Elo difference, home advantage, rest-day
difference), predicting the 1X2 outcome directly. A second, much simpler
model than Dixon-Coles — averaging two differently-biased models reliably
beats either alone (see the plan's rationale for this ensemble). It's also
useful precisely where Dixon-Coles is weakest: it needs no per-league pool at
all, just Elo (which is already cross-league-anchored, see model/elo.py) and
rest days, so it degrades gracefully for thin-history and cross-league
fixtures instead of falling all the way back to a pure Elo-bridge guess.

Proportional-odds parametrization: a single latent "home-favouredness" score
    eta = b_elo * elo_diff + b_home * is_home_advantage + b_rest * rest_diff
cut by two ordered thresholds c1 < c2 (enforced via c2 = c1 + exp(gap), so the
optimizer can't accidentally invert them) into away / draw / home probabilities:
    P(away) = sigma(c1 - eta)
    P(draw) = sigma(c2 - eta) - sigma(c1 - eta)
    P(home) = 1 - sigma(c2 - eta)
Outcome codes match model.backtest._actual_outcome: 0=home, 1=draw, 2=away.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

OUTCOME_HOME, OUTCOME_DRAW, OUTCOME_AWAY = 0, 1, 2
MIN_ROWS_TO_FIT = 50


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


class OrdinalFit:
    def __init__(self, beta: np.ndarray, c1: float, c2: float, feature_scale: np.ndarray):
        self.beta = beta
        self.c1 = c1
        self.c2 = c2
        self.feature_scale = feature_scale

    def predict_proba(self, elo_diff: float, home_advantage: float, rest_diff: float) -> tuple[float, float, float]:
        x = np.array([elo_diff, home_advantage, rest_diff]) / self.feature_scale
        eta = float(np.dot(self.beta, x))
        p_away = float(_sigmoid(np.array(self.c1 - eta)))
        cdf_2 = float(_sigmoid(np.array(self.c2 - eta)))
        p_draw = cdf_2 - p_away
        p_home = 1.0 - cdf_2
        # Clip only for float noise right at a boundary; the model is
        # monotonic by construction so this should never fire by more than
        # rounding error.
        return max(p_home, 1e-9), max(p_draw, 1e-9), max(p_away, 1e-9)


def fit(rows: list[dict]) -> OrdinalFit | None:
    """rows: [{"elo_diff": float, "home_advantage": 0|1, "rest_diff": float,
    "outcome": 0|1|2}, ...]. Returns None below MIN_ROWS_TO_FIT — a caller
    should fall back to Dixon-Coles/Elo-bridge alone in that case, same as
    every other thin-data fallback in this codebase."""
    if len(rows) < MIN_ROWS_TO_FIT:
        return None

    elo_diff = np.array([r["elo_diff"] for r in rows], dtype=float)
    home_adv = np.array([r["home_advantage"] for r in rows], dtype=float)
    rest_diff = np.array([r["rest_diff"] for r in rows], dtype=float)
    outcome = np.array([r["outcome"] for r in rows])

    # Scale features to comparable magnitude — L-BFGS-B converges poorly when
    # one feature ranges in the hundreds (Elo) and another in single digits
    # (rest days).
    feature_scale = np.array([max(float(np.std(elo_diff)), 1.0), 1.0, max(float(np.std(rest_diff)), 1.0)])
    X = np.vstack([elo_diff, home_adv, rest_diff]).T / feature_scale

    def neg_log_likelihood(params: np.ndarray) -> float:
        beta = params[:3]
        c1, log_gap = params[3], params[4]
        c2 = c1 + np.exp(log_gap)
        eta = X @ beta

        p_away = _sigmoid(c1 - eta)
        cdf_2 = _sigmoid(c2 - eta)
        p_draw = np.clip(cdf_2 - p_away, 1e-10, None)
        p_home = np.clip(1 - cdf_2, 1e-10, None)
        p_away = np.clip(p_away, 1e-10, None)

        ll = np.where(
            outcome == OUTCOME_HOME,
            np.log(p_home),
            np.where(outcome == OUTCOME_DRAW, np.log(p_draw), np.log(p_away)),
        )
        return float(-np.sum(ll))

    x0 = np.array([0.5, 0.3, 0.0, -0.4, 0.0])
    result = minimize(neg_log_likelihood, x0, method="L-BFGS-B")

    beta = result.x[:3]
    c1 = float(result.x[3])
    c2 = c1 + float(np.exp(result.x[4]))
    return OrdinalFit(beta=beta, c1=c1, c2=c2, feature_scale=feature_scale)

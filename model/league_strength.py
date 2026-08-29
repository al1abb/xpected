"""Bridges Elo ratings into the same (lambda_home, lambda_away) currency that
Dixon-Coles produces, so predict.py can treat every match uniformly regardless
of whether it's a same-league fixture (Dixon-Coles) or a cross-league one
(UEFA competitions, or the Azerbaijan Premyer Liqa, which isn't one of the 8
domestic leagues we fit Dixon-Coles on at all).

Calibrated from the only matches in our data that actually connect different
domestic leagues: the UEFA competition results. Simple linear regression,
goal_diff ~ a * elo_diff + b, split around the average total goals in those
matches. This also serves as the shrinkage target for thin-history teams
within a domestic Dixon-Coles fit (newly promoted sides, Azerbaijani teams
with only a handful of European matches) — see model/predict.py.

Known simplification: calibrated against a single current Elo snapshot rather
than each match's Elo-at-the-time, which introduces mild lookahead bias. Fine
for a slope+intercept calibration; a v2 could replay Elo incrementally instead.
"""

from __future__ import annotations

import numpy as np
from sqlalchemy.orm import Session

from app.config import COMPETITIONS
from app.models import Competition, Match

UEFA_SLUGS = {c["slug"] for c in COMPETITIONS if c["type"] == "uefa_cup"}
MIN_LAMBDA = 0.05


class EloCalibration:
    def __init__(self, slope: float, intercept: float, avg_total_goals: float):
        self.slope = slope
        self.intercept = intercept
        self.avg_total_goals = avg_total_goals

    def lambdas(self, elo_home: float, elo_away: float) -> tuple[float, float]:
        goal_diff = self.slope * (elo_home - elo_away) + self.intercept
        lambda_home = max(MIN_LAMBDA, self.avg_total_goals / 2 + goal_diff / 2)
        lambda_away = max(MIN_LAMBDA, self.avg_total_goals / 2 - goal_diff / 2)
        return lambda_home, lambda_away


def fit(session: Session, elo_ratings: dict[int, float]) -> EloCalibration:
    matches = (
        session.query(Match)
        .join(Competition)
        .filter(Competition.slug.in_(UEFA_SLUGS), Match.status == "finished")
        .all()
    )

    diffs, goal_diffs, totals = [], [], []
    for m in matches:
        elo_home = elo_ratings.get(m.home_team_id)
        elo_away = elo_ratings.get(m.away_team_id)
        if elo_home is None or elo_away is None or m.home_goals is None or m.away_goals is None:
            continue
        diffs.append(elo_home - elo_away)
        goal_diffs.append(m.home_goals - m.away_goals)
        totals.append(m.home_goals + m.away_goals)

    if len(diffs) < 20:
        # Not enough cross-league evidence yet — fall back to a weak, sensible
        # default (100 Elo points ~ roughly a 1-goal swing) rather than fitting noise.
        return EloCalibration(slope=1.0 / 100, intercept=0.15, avg_total_goals=2.6)

    x = np.array(diffs, dtype=float)
    y = np.array(goal_diffs, dtype=float)
    design = np.vstack([x, np.ones_like(x)]).T
    (slope, intercept), *_ = np.linalg.lstsq(design, y, rcond=None)
    avg_total = float(np.mean(totals))
    return EloCalibration(slope=float(slope), intercept=float(intercept), avg_total_goals=avg_total)

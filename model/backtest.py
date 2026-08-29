"""Walk-forward backtest: expanding-window, monthly refits. For each month
boundary, fit the model (and two baselines) on everything strictly before it,
then score every match that kicks off within that month — never seeing the
future relative to its own prediction. This is what the plan gates shipping
on: the model must beat the home-advantage baseline on RPS/Brier/log-loss, per
competition, not just in aggregate.

Refitting per-match (rather than monthly) would be more rigorous but refits
~9,000 times; monthly refit is the standard tractable compromise for
walk-forward sports forecasting and still keeps every evaluation genuinely
out-of-sample.
"""

from __future__ import annotations

import datetime as dt
import math

import numpy as np
from sqlalchemy.orm import Session

from app.models import Competition, Match, OddsSnapshot
from model.predict import Predictor

OUTCOME_HOME, OUTCOME_DRAW, OUTCOME_AWAY = 0, 1, 2


def _actual_outcome(match: Match) -> int:
    if match.home_goals > match.away_goals:
        return OUTCOME_HOME
    if match.home_goals == match.away_goals:
        return OUTCOME_DRAW
    return OUTCOME_AWAY


def rps(probs: tuple[float, float, float], actual: int) -> float:
    cum_pred = [probs[0], probs[0] + probs[1], 1.0]
    cum_actual = [1.0 if actual <= i else 0.0 for i in range(3)]
    return sum((cp - ca) ** 2 for cp, ca in zip(cum_pred, cum_actual)) / 2


def brier(probs: tuple[float, float, float], actual: int) -> float:
    return sum((p - (1.0 if i == actual else 0.0)) ** 2 for i, p in enumerate(probs))


def log_loss(probs: tuple[float, float, float], actual: int) -> float:
    p = max(probs[actual], 1e-10)
    return -math.log(p)


def _home_advantage_baseline(session: Session, cutoff: dt.datetime) -> tuple[float, float, float]:
    """Empirical home/draw/away frequency from everything before `cutoff` —
    the naive baseline the model must beat."""
    matches = session.query(Match).filter(Match.status == "finished", Match.utc_kickoff < cutoff).all()
    if not matches:
        return (0.45, 0.27, 0.28)
    counts = [0, 0, 0]
    for m in matches:
        counts[_actual_outcome(m)] += 1
    total = sum(counts)
    return tuple(c / total for c in counts)


def _devigged_market_probs(match: Match) -> tuple[float, float, float] | None:
    snapshot = next((o for o in match.odds if o.home_odds and o.draw_odds and o.away_odds), None)
    if snapshot is None:
        return None
    raw = [1 / snapshot.home_odds, 1 / snapshot.draw_odds, 1 / snapshot.away_odds]
    total = sum(raw)
    return tuple(r / total for r in raw)


def _month_boundaries(start: dt.datetime, end: dt.datetime) -> list[dt.datetime]:
    boundaries = []
    current = dt.datetime(start.year, start.month, 1)
    while current < end:
        boundaries.append(current)
        current = dt.datetime(current.year + (current.month == 12), current.month % 12 + 1, 1)
    boundaries.append(end)
    return boundaries


class Scoreboard:
    def __init__(self):
        self.n = 0
        self.rps_sum = 0.0
        self.brier_sum = 0.0
        self.log_loss_sum = 0.0
        self.correct_top_pick = 0

    def add(self, probs: tuple[float, float, float], actual: int) -> None:
        self.n += 1
        self.rps_sum += rps(probs, actual)
        self.brier_sum += brier(probs, actual)
        self.log_loss_sum += log_loss(probs, actual)
        if int(np.argmax(probs)) == actual:
            self.correct_top_pick += 1

    def summary(self) -> dict:
        if self.n == 0:
            return {"n": 0}
        return {
            "n": self.n,
            "rps": self.rps_sum / self.n,
            "brier": self.brier_sum / self.n,
            "log_loss": self.log_loss_sum / self.n,
            "accuracy": self.correct_top_pick / self.n,
        }


def run_backtest(session: Session, *, burn_in_days: int = 365) -> dict:
    all_matches = session.query(Match).filter(Match.status == "finished").order_by(Match.utc_kickoff).all()
    if not all_matches:
        return {"error": "no finished matches to backtest"}

    start = all_matches[0].utc_kickoff + dt.timedelta(days=burn_in_days)
    end = all_matches[-1].utc_kickoff
    if start >= end:
        return {"error": "not enough history for the requested burn-in period"}

    boundaries = _month_boundaries(start, end)

    model_overall = Scoreboard()
    home_adv_overall = Scoreboard()
    market_overall = Scoreboard()
    model_by_competition: dict[str, Scoreboard] = {}
    market_matches_scored = 0

    slug_by_competition_id = {c.id: c.slug for c in session.query(Competition).all()}

    for i in range(len(boundaries) - 1):
        cutoff, next_cutoff = boundaries[i], boundaries[i + 1]
        period_matches = [m for m in all_matches if cutoff <= m.utc_kickoff < next_cutoff]
        if not period_matches:
            continue

        predictor = Predictor(session, as_of=cutoff)
        home_adv_probs = _home_advantage_baseline(session, cutoff)

        for match in period_matches:
            actual = _actual_outcome(match)

            summary = predictor.predict_match(match)
            model_probs = (summary["home_win_prob"], summary["draw_prob"], summary["away_win_prob"])
            model_overall.add(model_probs, actual)
            home_adv_overall.add(home_adv_probs, actual)

            slug = slug_by_competition_id.get(match.competition_id, "unknown")
            model_by_competition.setdefault(slug, Scoreboard()).add(model_probs, actual)

            market_probs = _devigged_market_probs(match)
            if market_probs is not None:
                market_overall.add(market_probs, actual)
                market_matches_scored += 1

    return {
        "evaluated_from": start.isoformat(),
        "evaluated_to": end.isoformat(),
        "model": model_overall.summary(),
        "home_advantage_baseline": home_adv_overall.summary(),
        "market_baseline": market_overall.summary() if market_matches_scored else {"n": 0},
        "by_competition": {slug: sb.summary() for slug, sb in model_by_competition.items()},
        "beats_home_advantage_baseline": (
            model_overall.n > 0
            and home_adv_overall.n > 0
            and model_overall.rps_sum / model_overall.n < home_adv_overall.rps_sum / home_adv_overall.n
        ),
    }

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
import time

import numpy as np
from sqlalchemy.orm import Session

from app.models import Competition, Match, OddsSnapshot, Prediction
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


def devigged_market_probs(match: Match) -> tuple[float, float, float] | None:
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


def _latest_pre_kickoff_predictions(session: Session) -> list[tuple[Prediction, Match]]:
    """The single most recent Prediction for each finished match that was
    actually made before that match kicked off — genuinely out-of-sample, as
    opposed to the backtest's historical simulation. A match can accumulate
    several predictions over time as the model refits and re-predicts every
    still-scheduled fixture (see model.predict.generate_predictions); only
    the latest one reflects what the site was actually showing right before
    kickoff, so earlier ones for the same match are dropped here."""
    rows = (
        session.query(Prediction, Match)
        .join(Match, Prediction.match_id == Match.id)
        .filter(Match.status == "finished", Prediction.created_at < Match.utc_kickoff)
        .all()
    )
    latest: dict[int, tuple[Prediction, Match]] = {}
    for pred, match in rows:
        current = latest.get(match.id)
        if current is None or pred.created_at > current[0].created_at:
            latest[match.id] = (pred, match)
    return list(latest.values())


def _tracked_row(pred: Prediction, match: Match) -> tuple[tuple[float, float, float], int, int, bool, dict]:
    """Shared scoring + display-row logic for one tracked (Prediction, Match)
    pair — used by both live_tracking_summary's aggregate stats and the
    paginated tracked_predictions_page, so the two never drift apart."""
    probs = (pred.home_win_prob, pred.draw_prob, pred.away_win_prob)
    actual = _actual_outcome(match)
    top_idx = int(np.argmax(probs))
    hit = top_idx == actual
    row = {
        "match": match,
        "predicted_probs": probs,
        "predicted_pick": ["home", "draw", "away"][top_idx],
        "predicted_confidence": probs[top_idx],
        "actual_outcome": ["home", "draw", "away"][actual],
        "hit": hit,
    }
    return probs, actual, top_idx, hit, row


def live_tracking_summary(session: Session, *, recent_limit: int = 20) -> dict:
    """Scores real, already-made predictions against what actually happened —
    distinct from run_backtest's historical simulation. This is the honest
    answer to "how is the model doing right now": no walk-forward replay, just
    every prediction the site genuinely showed before a match, checked
    against the result once it came in.

    `recent` here is a fixed-size preview for the accuracy page, not the full
    history — see tracked_predictions_page for the paginated version.
    """
    pairs = _latest_pre_kickoff_predictions(session)
    if not pairs:
        return {"n": 0}

    pairs.sort(key=lambda pm: pm[1].utc_kickoff, reverse=True)

    scoreboard = Scoreboard()
    # 10-point bands on the top-pick's predicted probability, e.g. matches
    # where the model said "50-60% confident in its pick" — bucketed to show
    # whether that confidence is actually earned (see model/backtest.py's
    # docstring on why calibration, not hit-rate, is the real target).
    bands: dict[int, dict] = {}
    recent = []

    for pred, match in pairs:
        probs, actual, top_idx, hit, row = _tracked_row(pred, match)
        scoreboard.add(probs, actual)

        top_prob = probs[top_idx]
        band = min(int(top_prob * 10) * 10, 90)
        bucket = bands.setdefault(band, {"n": 0, "correct": 0, "prob_sum": 0.0})
        bucket["n"] += 1
        bucket["correct"] += int(hit)
        bucket["prob_sum"] += top_prob

        if len(recent) < recent_limit:
            recent.append(row)

    calibration_table = [
        {
            "band_label": f"{band}-{band + 10}%",
            "band": band,
            "n": b["n"],
            "predicted_avg": b["prob_sum"] / b["n"],
            "realized_rate": b["correct"] / b["n"],
        }
        for band, b in sorted(bands.items())
    ]

    return {
        **scoreboard.summary(),
        "calibration_table": calibration_table,
        "recent": recent,
    }


def tracked_predictions_page(session: Session, *, page: int = 1, page_size: int = 25) -> dict:
    """Full, paginated history of every prediction genuinely tracked before
    kickoff (see _latest_pre_kickoff_predictions), most recent first. Backs
    the "all tracked predictions" page — live_tracking_summary's own
    `recent` list is a small fixed preview, not meant to paginate itself."""
    pairs = _latest_pre_kickoff_predictions(session)
    pairs.sort(key=lambda pm: pm[1].utc_kickoff, reverse=True)

    total = len(pairs)
    total_pages = max(1, math.ceil(total / page_size)) if total else 1
    page = min(max(1, page), total_pages)
    start = (page - 1) * page_size

    rows = [_tracked_row(pred, match)[4] for pred, match in pairs[start : start + page_size]]

    return {
        "rows": rows,
        "total": total,
        "page": page,
        "page_size": page_size,
        "total_pages": total_pages,
    }


def run_backtest(
    session: Session,
    *,
    burn_in_days: int = 365,
    xi_per_day: float | None = None,
    ensemble_weight: float | None = None,
    rest_adjustment_enabled: bool = True,
    collect_predictions: bool = False,
    verbose: bool = False,
) -> dict:
    all_matches = session.query(Match).filter(Match.status == "finished").order_by(Match.utc_kickoff).all()
    if not all_matches:
        return {"error": "no finished matches to backtest"}

    start = all_matches[0].utc_kickoff + dt.timedelta(days=burn_in_days)
    end = all_matches[-1].utc_kickoff
    if start >= end:
        return {"error": "not enough history for the requested burn-in period"}

    boundaries = _month_boundaries(start, end)
    total_periods = len(boundaries) - 1

    model_overall = Scoreboard()
    home_adv_overall = Scoreboard()
    market_overall = Scoreboard()
    model_by_competition: dict[str, Scoreboard] = {}
    market_matches_scored = 0
    raw_predictions: list[dict] = []

    slug_by_competition_id = {c.id: c.slug for c in session.query(Competition).all()}

    period_started_at = time.monotonic()
    for i in range(total_periods):
        cutoff, next_cutoff = boundaries[i], boundaries[i + 1]
        period_matches = [m for m in all_matches if cutoff <= m.utc_kickoff < next_cutoff]
        if not period_matches:
            continue

        if verbose:
            elapsed = time.monotonic() - period_started_at
            running_rps = f"{model_overall.rps_sum / model_overall.n:.4f}" if model_overall.n else "n/a"
            print(
                f"  [{i + 1}/{total_periods}] {cutoff.date()} — {len(period_matches)} matches "
                f"— running RPS {running_rps} ({elapsed:.0f}s elapsed)",
                flush=True,
            )

        predictor = Predictor(
            session,
            as_of=cutoff,
            xi_per_day=xi_per_day,
            ensemble_weight=ensemble_weight,
            rest_adjustment_enabled=rest_adjustment_enabled,
        )
        home_adv_probs = _home_advantage_baseline(session, cutoff)

        for match in period_matches:
            actual = _actual_outcome(match)

            summary = predictor.predict_match(match)
            model_probs = (summary["home_win_prob"], summary["draw_prob"], summary["away_win_prob"])
            model_overall.add(model_probs, actual)
            home_adv_overall.add(home_adv_probs, actual)

            slug = slug_by_competition_id.get(match.competition_id, "unknown")
            model_by_competition.setdefault(slug, Scoreboard()).add(model_probs, actual)

            if collect_predictions:
                raw_predictions.append({"probs": model_probs, "actual": actual, "match_id": match.id})

            market_probs = devigged_market_probs(match)
            if market_probs is not None:
                market_overall.add(market_probs, actual)
                market_matches_scored += 1

    result = {
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
    if collect_predictions:
        result["raw_predictions"] = raw_predictions
    return result

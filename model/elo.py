"""Unified Elo rating, replayed from our own match history across every
competition. This is the cross-league bridge: a Bundesliga team and an
Azerbaijani team never share a domestic table, but if they've ever met (or
each played someone who played someone...) in a UEFA competition, Elo
propagates a comparable rating through that chain. ClubElo would seed this
with a richer prior if reachable (see ingest/clubelo.py's documented
connectivity issue) — absent that, every team starts at BASE_RATING and the
rating is earned purely from results in our own database, which is enough:
Elo needs no external prior to be internally consistent.

Formula: standard logistic expected-score with a fixed home-advantage offset,
K scaled by margin of victory (the widely-used World Football Elo ratings
formula: https://www.eloratings.net/about) so a 4-0 moves ratings more than a
1-0.
"""

from __future__ import annotations

import datetime as dt

from sqlalchemy.orm import Session

from app.models import Match

BASE_RATING = 1500.0
K_BASE = 20.0
HOME_ADVANTAGE = 100.0


def _goal_diff_multiplier(goal_diff: int) -> float:
    if goal_diff <= 1:
        return 1.0
    if goal_diff == 2:
        return 1.5
    return (11 + goal_diff) / 8


def _expected_home_score(elo_home: float, elo_away: float) -> float:
    return 1.0 / (1.0 + 10 ** (((elo_away) - (elo_home + HOME_ADVANTAGE)) / 400))


def compute_ratings(
    session: Session,
    *,
    as_of: dt.datetime | None = None,
    exclude_match_id: int | None = None,
) -> dict[int, float]:
    """Replay every finished match up to `as_of` (inclusive) in chronological
    order and return {team_id: rating}. Teams never seen start implicitly at
    BASE_RATING when looked up (not present in the returned dict).

    `exclude_match_id` lets the backtest evaluate a match using ratings built
    from everything *except* that match, without a second query round-trip.
    """
    query = session.query(Match).filter(Match.status == "finished")
    if as_of is not None:
        query = query.filter(Match.utc_kickoff <= as_of)
    query = query.order_by(Match.utc_kickoff.asc(), Match.id.asc())

    ratings: dict[int, float] = {}
    for match in query.all():
        if match.id == exclude_match_id:
            continue
        if match.home_goals is None or match.away_goals is None:
            continue

        home_elo = ratings.get(match.home_team_id, BASE_RATING)
        away_elo = ratings.get(match.away_team_id, BASE_RATING)

        goal_diff = match.home_goals - match.away_goals
        if goal_diff > 0:
            actual_home = 1.0
        elif goal_diff == 0:
            actual_home = 0.5
        else:
            actual_home = 0.0

        expected_home = _expected_home_score(home_elo, away_elo)
        k = K_BASE * _goal_diff_multiplier(abs(goal_diff))
        delta = k * (actual_home - expected_home)

        ratings[match.home_team_id] = home_elo + delta
        ratings[match.away_team_id] = away_elo - delta

    return ratings


def match_count_by_team(session: Session, *, as_of: dt.datetime | None = None) -> dict[int, int]:
    """How many finished matches each team has played up to `as_of` — used
    for thin-history shrinkage (see model/league_strength.py)."""
    query = session.query(Match).filter(Match.status == "finished")
    if as_of is not None:
        query = query.filter(Match.utc_kickoff <= as_of)

    counts: dict[int, int] = {}
    for match in query.all():
        counts[match.home_team_id] = counts.get(match.home_team_id, 0) + 1
        counts[match.away_team_id] = counts.get(match.away_team_id, 0) + 1
    return counts

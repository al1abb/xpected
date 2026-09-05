"""PES-style A-E form grade for a team, computed entirely from matches already
in the database — no new ingest.

Recent-form lists and head-to-head history already exist (app/main.py's
`_team_form` / `_head_to_head`, feeding team.html, match.html and
compare.html) — this module does not duplicate them. What's new here is the
single-letter/arrow grade: recent points-per-game measured against a team's
OWN season points-per-game, so "good form" means better than they normally
are, not just "a good team".

Deliberately team-level only. Per-player form was investigated (a partial
version is buildable from football-data.org's scorers list) and dropped: it
would only ever cover the ~49 players per league who have scored or assisted,
and the call was everyone-or-nobody. See future-plans.md.

Important: none of this feeds the prediction model. `model/dixon_coles.py`
already applies its own exponential time decay so recent matches dominate its
fit, and `model/predict.py`'s rest-days adjustment already tracks each team's
most recent match date. This module exists purely to *display* what the model
already accounts for — it must never be advertised as improving accuracy.
"""

from __future__ import annotations

import datetime as dt

from sqlalchemy import or_
from sqlalchemy.orm import Session

from app.models import Match
from ingest.seasons import current_season_start_year

FORM_LOOKBACK = 6
# A team needs at least this many matches played in the CURRENT season before
# its season points-per-game is a trustworthy baseline to grade form against —
# early in a season a handful of results swing PPG wildly. Below this, the
# grade is withheld rather than shown wrong (see form_grade's docstring).
MIN_SEASON_MATCHES_FOR_GRADE = 5

# Recent points-per-game as a ratio of the team's own season points-per-game.
# Thresholds are deliberately centred on 1.0 (exactly matching their own
# season rate = "normal"), not on an absolute PPG scale, since a top-of-table
# side and a relegation-zone side should both be able to show any grade.
_GRADE_THRESHOLDS = [
    (1.20, "A", "↑"),  # comfortably above their own season rate
    (1.05, "B", "↗"),
    (0.95, "C", "→"),  # roughly their own normal
    (0.80, "D", "↘"),
    (0.00, "E", "↓"),
]


def _team_matches(
    session: Session,
    team_id: int,
    *,
    since: dt.datetime | None = None,
    before: dt.datetime | None = None,
    limit: int | None = None,
) -> list[Match]:
    query = session.query(Match).filter(
        Match.status == "finished",
        Match.home_goals.isnot(None),
        Match.away_goals.isnot(None),
        or_(Match.home_team_id == team_id, Match.away_team_id == team_id),
    )
    if since is not None:
        query = query.filter(Match.utc_kickoff >= since)
    if before is not None:
        query = query.filter(Match.utc_kickoff < before)
    query = query.order_by(Match.utc_kickoff.desc())
    if limit is not None:
        query = query.limit(limit)
    return query.all()


def _points(match: Match, team_id: int) -> int:
    is_home = match.home_team_id == team_id
    gf = match.home_goals if is_home else match.away_goals
    ga = match.away_goals if is_home else match.home_goals
    if gf > ga:
        return 3
    if gf == ga:
        return 1
    return 0


def form_grade(session: Session, team_id: int, *, as_of: dt.datetime | None = None) -> dict | None:
    """PES-style A-E letter + arrow, from recent points-per-game measured
    against this team's OWN season points-per-game — "good form" means
    better than they normally are, not just "a good team". A relegation
    candidate on a genuine unbeaten run grades the same as a title
    contender doing the equivalent relative to their own level.

    Returns None when there isn't enough of *this season* played yet for the
    baseline to mean anything (see MIN_SEASON_MATCHES_FOR_GRADE) — grading
    off 2-3 matches at the start of a season would just be noise dressed up
    as a signal, and the existing plain W/D/L form list already covers that
    gap without needing a grade.
    """
    as_of = as_of or dt.datetime.utcnow()
    season_start = dt.datetime(current_season_start_year(as_of.date()), 7, 1)
    season_matches = _team_matches(session, team_id, since=season_start, before=as_of)
    if len(season_matches) < MIN_SEASON_MATCHES_FOR_GRADE:
        return None

    season_ppg = sum(_points(m, team_id) for m in season_matches) / len(season_matches)
    if season_ppg == 0:
        return None  # cannot form a meaningful ratio against a zero baseline

    # Recent form uses the SAME cross-season window app/main.py's _team_form
    # shows (last N matches, any season) — not season_matches[:N] — so a team
    # just a few games into a new season can still be graded against last
    # season's tail rather than being compared to an identical subset of
    # itself (which would force every grade to read as exactly "normal").
    recent = _team_matches(session, team_id, before=as_of, limit=FORM_LOOKBACK)
    recent_ppg = sum(_points(m, team_id) for m in recent) / len(recent)

    ratio = recent_ppg / season_ppg
    for threshold, letter, arrow in _GRADE_THRESHOLDS:
        if ratio >= threshold:
            return {"letter": letter, "arrow": arrow, "ratio": ratio, "recent_ppg": recent_ppg, "season_ppg": season_ppg}
    return {"letter": "E", "arrow": "↓", "ratio": ratio, "recent_ppg": recent_ppg, "season_ppg": season_ppg}

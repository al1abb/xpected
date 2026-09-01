"""Unified Elo rating, replayed from our own match history across every
competition, then anchored onto ClubElo's globally-comparable scale.

Why the anchor is necessary, not cosmetic: Elo is zero-sum *within a rating
pool*. Our leagues are almost entirely closed pools — teams mostly only play
inside their own domestic league, so wins and losses cancel out and every
league's internal ratings drift toward the same ~1500 average regardless of
how strong the league actually is. Only ~1,000 UEFA matches connect the pools
at all, which is nowhere near enough to correct a ~300 Elo-point gap between,
say, the Premier League and the Azerbaijan Premyer Liqa. The practical result
without this fix: teams from weaker leagues (e.g. Porto, at 1780 internally)
outrank teams from stronger ones (e.g. Manchester City, at 1755) purely
because they dominate a smaller pond — confirmed against a real UEFA fixture
during development, where a purely-internal Elo model favoured the weaker
side by double digits.

ClubElo (see ingest/clubelo.py) rates ~600 clubs across 55 countries on one
shared scale and is the fix: teams it covers use its rating directly; teams
it doesn't (thin lower-division and lesser Azerbaijani sides) keep their
*internally earned* rank ordering but get rescaled onto the ClubElo scale via
an affine transform fit against teams in the same domestic league that ARE
covered. This preserves "which of our own teams is better" (real results)
while fixing "how does that compare across leagues" (ClubElo's job).

Fails soft: if ClubElo is unreachable, `fetch_snapshot` returns {} and every
team keeps its plain internal rating — today's pre-anchor behaviour — rather
than the whole prediction pipeline breaking.

Internal replay formula: standard logistic expected-score with a fixed
home-advantage offset, K scaled by margin of victory (the widely-used World
Football Elo ratings formula: https://www.eloratings.net/about) so a 4-0
moves ratings more than a 1-0.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
from sqlalchemy.orm import Session

from app.models import Competition, EloRating, Match
from ingest import clubelo
from ingest.seasons import current_season_start_year

BASE_RATING = 1500.0
K_BASE = 20.0
HOME_ADVANTAGE = 100.0

# `elo_ratings.source` values. 'clubelo' rows are written by
# ingest/clubelo.py and hold ClubElo's raw published numbers; 'internal' rows
# are the *blended* output of compute_ratings below — our replayed ratings
# after anchoring — which is what the site actually displays.
SOURCE_INTERNAL = "internal"
SOURCE_CLUBELO = clubelo.SOURCE

# Below this many teams paired in both our data and ClubElo for a given
# domestic league, an affine fit is too noisy to trust — fall back to the
# global (all-teams, all-leagues) transform instead.
MIN_PAIRED_FOR_LEAGUE_TRANSFORM = 3


def _goal_diff_multiplier(goal_diff: int) -> float:
    if goal_diff <= 1:
        return 1.0
    if goal_diff == 2:
        return 1.5
    return (11 + goal_diff) / 8


def _expected_home_score(elo_home: float, elo_away: float) -> float:
    return 1.0 / (1.0 + 10 ** (((elo_away) - (elo_home + HOME_ADVANTAGE)) / 400))


def _replay_internal(
    session: Session,
    *,
    as_of: dt.datetime | None = None,
    exclude_match_id: int | None = None,
) -> dict[int, float]:
    """Replay every finished match up to `as_of` (inclusive) in chronological
    order and return {team_id: rating}, on our own internal scale (mean drifts
    toward BASE_RATING per closed pool — see module docstring). Teams never
    seen start implicitly at BASE_RATING when looked up (not present in the
    returned dict).

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


def _team_primary_league(session: Session, as_of: dt.datetime | None) -> dict[int, int]:
    """team_id -> id of the domestic league (Competition.type == 'league') the
    team has played the most finished matches in, up to `as_of`. Used to pick
    which league's teams anchor a thin-history team's rescale."""
    query = session.query(Match).join(Competition, Match.competition_id == Competition.id).filter(
        Competition.type == "league", Match.status == "finished"
    )
    if as_of is not None:
        query = query.filter(Match.utc_kickoff <= as_of)

    counts: dict[int, dict[int, int]] = {}
    for m in query.all():
        for team_id in (m.home_team_id, m.away_team_id):
            bucket = counts.setdefault(team_id, {})
            bucket[m.competition_id] = bucket.get(m.competition_id, 0) + 1

    return {team_id: max(comp_counts, key=comp_counts.get) for team_id, comp_counts in counts.items()}


def _fit_affine(xs: list[float], ys: list[float]):
    """Least-squares y = a*x + b. Degenerates to a pure offset if `xs` has no
    spread (e.g. only one paired team) rather than raising on a singular fit."""
    xs_arr = np.asarray(xs, dtype=float)
    ys_arr = np.asarray(ys, dtype=float)
    if len(xs_arr) < 2 or float(np.std(xs_arr)) < 1e-6:
        offset = float(np.mean(ys_arr) - np.mean(xs_arr)) if len(xs_arr) else 0.0
        return lambda x: x + offset
    design = np.vstack([xs_arr, np.ones_like(xs_arr)]).T
    (a, b), *_ = np.linalg.lstsq(design, ys_arr, rcond=None)
    a, b = float(a), float(b)
    return lambda x: a * x + b


def _anchor_to_clubelo(
    internal: dict[int, float],
    clubelo_ratings: dict[int, float],
    team_league: dict[int, int],
) -> dict[int, float]:
    if not clubelo_ratings:
        return dict(internal)

    paired = [(tid, internal[tid], clubelo_ratings[tid]) for tid in internal if tid in clubelo_ratings]
    if len(paired) < 2:
        return dict(internal)

    global_transform = _fit_affine([p[1] for p in paired], [p[2] for p in paired])

    by_league: dict[int, list[tuple[float, float]]] = {}
    for tid, i_val, c_val in paired:
        league = team_league.get(tid)
        if league is not None:
            by_league.setdefault(league, []).append((i_val, c_val))

    league_transforms = {
        league: _fit_affine([p[0] for p in pts], [p[1] for p in pts])
        for league, pts in by_league.items()
        if len(pts) >= MIN_PAIRED_FOR_LEAGUE_TRANSFORM
    }

    result: dict[int, float] = {}
    for tid, i_val in internal.items():
        if tid in clubelo_ratings:
            result[tid] = clubelo_ratings[tid]
            continue
        transform = league_transforms.get(team_league.get(tid), global_transform)
        result[tid] = transform(i_val)
    return result


def _clubelo_snapshot_date(as_of: dt.datetime | None) -> dt.date:
    """Live use (`as_of=None`) fetches today's actual ClubElo ranking — one
    request a day. Historical use (walk-forward backtest, ~monthly cutoffs
    over many seasons) snaps to that cutoff's season-start date instead: a
    cross-league scale anchor doesn't need month-level precision, domestic
    leagues realign every close season anyway (see ingest/seasons.py), and it
    turns what would be ~1 request per month (~80+ over an 8-season backtest)
    into ~1 per season (~8). ClubElo's free endpoint has no documented rate
    limit, but hammering it with dozens of rapid historical-date requests was
    observed, in practice, to make it silently stop responding (timeouts, not
    429s) rather than reject cleanly — this keeps total request volume low
    enough to stay a good citizen of a free, unauthenticated API.

    Still zero lookahead: a season-start snapshot is always dated on or
    before every cutoff that season resolves to, never after.
    """
    if as_of is None:
        return dt.date.today()
    season_start_year = current_season_start_year(as_of.date())
    return dt.date(season_start_year, 7, 1)


def compute_ratings(
    session: Session,
    *,
    as_of: dt.datetime | None = None,
    exclude_match_id: int | None = None,
) -> dict[int, float]:
    """{team_id: rating}, anchored onto ClubElo's cross-league-comparable scale
    where possible (see module docstring). Teams never seen are absent from
    the dict (callers default to BASE_RATING on lookup)."""
    internal = _replay_internal(session, as_of=as_of, exclude_match_id=exclude_match_id)
    if not internal:
        return internal

    snapshot_date = _clubelo_snapshot_date(as_of)
    clubelo_ratings = clubelo.fetch_snapshot(session, snapshot_date)
    if not clubelo_ratings:
        return internal
    team_league = _team_primary_league(session, as_of)
    return _anchor_to_clubelo(internal, clubelo_ratings, team_league)


def persist_ratings(session: Session, ratings: dict[int, float], *, as_of: dt.date | None = None) -> int:
    """Store `ratings` (compute_ratings' blended output) as `elo_ratings` rows
    with source='internal', so readers can look them up instead of recomputing.

    Called from the offline refresh, never from a web request. compute_ratings
    costs 8-12s and makes a live ClubElo HTTP call, which is fine in a GitHub
    Actions run but not inside a serverless request (where /tmp/raw starts
    empty, so the fetch can never be cache-warm, and a slow ClubElo can exceed
    the function's whole time budget). Persisting here is what lets
    app/main.py answer team/compare pages with a plain SELECT.

    Idempotent per day: `elo_ratings` is unique on
    (team_id, as_of_date, source), so re-running a refresh updates in place.
    """
    as_of = as_of or dt.date.today()
    existing = {
        row.team_id: row
        for row in session.query(EloRating).filter_by(as_of_date=as_of, source=SOURCE_INTERNAL).all()
    }
    for team_id, elo_value in ratings.items():
        row = existing.get(team_id)
        if row is None:
            session.add(EloRating(team_id=team_id, as_of_date=as_of, elo=elo_value, source=SOURCE_INTERNAL))
        else:
            row.elo = elo_value
    session.commit()
    return len(ratings)


def load_persisted_ratings(session: Session) -> dict[int, float]:
    """{team_id: rating} from the most recent persisted rating per team.

    Prefers source='internal' (compute_ratings' blended, cross-league-anchored
    output — what persist_ratings writes). Falls back per-team to the raw
    'clubelo' rows, which ingest/clubelo.py has always written: for any club
    ClubElo covers, the anchored rating IS its ClubElo rating (see
    _anchor_to_clubelo), so that fallback is the same number rather than an
    approximation. Teams in neither are simply absent, and callers default
    them to BASE_RATING exactly as they do for compute_ratings' output.

    Returns {} on a database that has never been refreshed, which callers must
    tolerate — it degrades to every team at BASE_RATING, not an error.
    """
    ratings: dict[int, float] = {}
    for source in (SOURCE_CLUBELO, SOURCE_INTERNAL):
        # Ascending date so a later snapshot overwrites an earlier one, and
        # 'internal' is applied second so it wins over the clubelo fallback.
        rows = (
            session.query(EloRating.team_id, EloRating.elo)
            .filter(EloRating.source == source)
            .order_by(EloRating.as_of_date.asc())
            .all()
        )
        for team_id, elo_value in rows:
            ratings[team_id] = elo_value
    return ratings


def match_count_by_team(session: Session, *, as_of: dt.datetime | None = None) -> dict[int, int]:
    """How many finished matches each team has played up to `as_of` — used
    for thin-history shrinkage (see model/predict.py)."""
    query = session.query(Match).filter(Match.status == "finished")
    if as_of is not None:
        query = query.filter(Match.utc_kickoff <= as_of)

    counts: dict[int, int] = {}
    for match in query.all():
        counts[match.home_team_id] = counts.get(match.home_team_id, 0) + 1
        counts[match.away_team_id] = counts.get(match.away_team_id, 0) + 1
    return counts

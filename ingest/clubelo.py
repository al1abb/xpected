"""ClubElo ingest: free, global club Elo ratings — the cross-league bridge that
makes a Champions League fixture between a Bundesliga team and an Azerbaijani
team predictable at all (see model/league_strength.py and model/elo.py).

Reachability note (superseded): earlier development found `https://api.clubelo.com`
unreachable from this project's network. That turned out to be a scheme
problem, not a host problem — plain `http://api.clubelo.com` works fine and is
what CLUBELO_API_BASE points at. Verified reachable for both today's ratings
and arbitrary historical dates (`/YYYY-MM-DD`), which is what lets the
backtest use point-in-time ratings without lookahead bias.

Resolve-only by design: ClubElo covers ~600 clubs across 55 countries, far
more than the ~12 competitions this app tracks. Using get_or_create_team here
would create a `Team` row for every foreign club ClubElo happens to rank —
confirmed in practice (an earlier version of this ingest created 332 junk
teams from a single run). `resolve_existing_team` instead matches against
teams we already track and skips — logging to `unresolved_aliases` — anything
it can't confidently place, the same manual-review discipline used elsewhere.
"""

from __future__ import annotations

import csv
import datetime as dt
import io

from sqlalchemy.orm import Session

from app.config import CLUBELO_API_BASE
from app.models import EloRating, IngestLog, Team
from ingest.cache import fetch_text
from ingest.clubelo_aliases import CLUBELO_TO_CANONICAL
from ingest.resolve import build_alias_pool, link_alias, resolve_existing_team

SOURCE = "clubelo"

# ClubElo re-ranks continuously, so today's file is refetched daily; a past
# date's ratings are final the moment that date has fully elapsed, so those
# are cached effectively forever (10 years — long enough to never re-fetch a
# closed date, short enough not to be "literally forever" if the API's date
# semantics ever change).
_PAST_DATE_MAX_AGE_HOURS = 24 * 365 * 10
_TODAY_MAX_AGE_HOURS = 24


def parse_ranking_csv(text: str) -> list[dict]:
    reader = csv.DictReader(io.StringIO(text))
    rows = []
    for row in reader:
        club = (row.get("Club") or "").strip()
        elo = row.get("Elo")
        if not club or elo is None:
            continue
        try:
            elo_value = float(elo)
        except ValueError:
            continue
        rows.append(
            {
                "club": club,
                "country": (row.get("Country") or "").strip() or None,
                "elo": elo_value,
            }
        )
    return rows


def _resolve_club(session: Session, club_name: str, *, context: str, pool: dict) -> Team | None:
    canonical = CLUBELO_TO_CANONICAL.get(club_name)
    if canonical is not None:
        team = session.query(Team).filter_by(canonical_name=canonical).one_or_none()
        if team is not None:
            link_alias(session, team, club_name, SOURCE)
            return team
    return resolve_existing_team(session, club_name, SOURCE, context=context, pool=pool)


def fetch_snapshot(session: Session, on_date: dt.date) -> dict[int, float]:
    """{team_id: elo} as ClubElo rated it on `on_date` (or the most recent date
    on/before it that ClubElo published — its `/YYYY-MM-DD` endpoint already
    resolves to the latest snapshot at or before the requested date).

    Disk-cached: past dates forever, today for 24h. Fails soft — an empty dict
    on any network error, so callers (model/elo.py) can fall back to
    internal-only ratings rather than break the whole prediction pipeline.
    """
    today = dt.date.today()
    max_age = _TODAY_MAX_AGE_HOURS if on_date >= today else _PAST_DATE_MAX_AGE_HOURS
    url = f"{CLUBELO_API_BASE}/{on_date.isoformat()}"

    try:
        text = fetch_text(url, subdir="clubelo", max_age_hours=max_age)
    except RuntimeError:
        return {}

    # Built once and reused for every row below — rebuilding it per row (as a
    # naive per-name resolve would) means re-querying and re-scoring against
    # every alias in the database ~600 times per snapshot fetch, which is the
    # difference between this taking a second and taking minutes.
    pool = build_alias_pool(session, exclude_source=SOURCE)

    ratings: dict[int, float] = {}
    for row in parse_ranking_csv(text):
        team = _resolve_club(session, row["club"], context=f"clubelo snapshot {on_date.isoformat()}", pool=pool)
        if team is not None:
            ratings[team.id] = row["elo"]
    session.flush()
    return ratings


def ingest_current_ratings(session: Session, as_of: dt.date | None = None) -> int:
    """Persist today's ClubElo ratings to `elo_ratings` (source='clubelo') —
    mainly for freshness/debugging visibility; model/elo.py fetches its own
    snapshot on demand rather than reading this table."""
    as_of = as_of or dt.date.today()
    started = dt.datetime.utcnow()

    ratings = fetch_snapshot(session, as_of)
    if not ratings:
        session.add(
            IngestLog(
                source=SOURCE,
                started_at=started,
                finished_at=dt.datetime.utcnow(),
                status="error",
                message=f"no ClubElo data resolved for {as_of.isoformat()}",
            )
        )
        session.commit()
        return 0

    count = 0
    for team_id, elo_value in ratings.items():
        existing = (
            session.query(EloRating).filter_by(team_id=team_id, as_of_date=as_of, source=SOURCE).one_or_none()
        )
        if existing is None:
            session.add(EloRating(team_id=team_id, as_of_date=as_of, elo=elo_value, source=SOURCE))
            count += 1
        else:
            existing.elo = elo_value

    session.add(
        IngestLog(
            source=SOURCE,
            started_at=started,
            finished_at=dt.datetime.utcnow(),
            status="ok",
            rows_ingested=count,
        )
    )
    session.commit()
    return count

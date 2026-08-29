"""ClubElo ingest: free, global club Elo ratings — the cross-league bridge that
makes a Champions League fixture between a Bundesliga team and an Azerbaijani
team predictable at all (see model/league_strength.py).

Note: during development, api.clubelo.com was unreachable (connection timeout)
from this project's network — DNS resolved fine, but the host refused/ignored
the TCP connection, while clubelo.com (the website, different IP) was fine.
This looks like the server blocking the sandbox's outbound IP range rather than
a code problem. This module is written to work once network access is
available (e.g. running refresh.py from a normal residential/VPS IP), and it
fails soft: an unreachable ClubElo is logged to ingest_log as an error and the
rest of the pipeline (Dixon-Coles on in-league results) continues without it.
"""

from __future__ import annotations

import csv
import datetime as dt
import io

from sqlalchemy.orm import Session

from app.config import CLUBELO_API_BASE
from app.models import EloRating, IngestLog
from ingest.cache import fetch_text
from ingest.resolve import get_or_create_team

SOURCE = "clubelo"


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


def ingest_current_ratings(session: Session, as_of: dt.date | None = None) -> int:
    as_of = as_of or dt.date.today()
    started = dt.datetime.utcnow()
    url = f"{CLUBELO_API_BASE}/{as_of.isoformat()}"

    try:
        text = fetch_text(url, subdir="clubelo", max_age_hours=24)
    except RuntimeError as exc:
        session.add(
            IngestLog(
                source=SOURCE,
                started_at=started,
                finished_at=dt.datetime.utcnow(),
                status="error",
                message=str(exc),
            )
        )
        session.commit()
        return 0

    rows = parse_ranking_csv(text)
    count = 0
    for row in rows:
        team = get_or_create_team(session, row["club"], SOURCE, context="clubelo ranking")
        if team.country is None and row["country"]:
            team.country = row["country"]

        existing = (
            session.query(EloRating).filter_by(team_id=team.id, as_of_date=as_of, source=SOURCE).one_or_none()
        )
        if existing is None:
            session.add(EloRating(team_id=team.id, as_of_date=as_of, elo=row["elo"], source=SOURCE))
            count += 1
        else:
            existing.elo = row["elo"]

    session.add(
        IngestLog(
            source=SOURCE,
            started_at=started,
            finished_at=dt.datetime.utcnow(),
            status="ok" if rows else "partial",
            rows_ingested=count,
        )
    )
    session.commit()
    return count

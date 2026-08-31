"""One-time backfill for the football-data.co.uk UK-local-vs-UTC bug fixed in
ingest/footballdata_csv.py::_parse_date. Every `Match` row already ingested
with source="football_data" has its kickoff time stored as if UK local time
were UTC — off by exactly 1h during BST (late March-late October), correct
during GMT (winter). This recomputes the true UTC kickoff for every such row
by reinterpreting the stored (wrong) value as Europe/London local time and
converting properly — the same fix now applied at ingest time.

Two outcomes per row that actually changes:
  - No colliding row from another source: update utc_kickoff in place.
  - A colliding row exists (same competition/teams, another source, kickoff
    within a few hours of the corrected time) — this is the same real
    fixture duplicated because the other source already had the correct
    time (confirmed: only the current 2026/27 season can collide, since
    fixturedownload.com — the only other near-term source — doesn't cover
    past seasons). Merge them: keep whichever row holds more prediction
    history (matching scripts/merge_teams.py's precedent), correct its
    kickoff, and drop the other along with its predictions/odds.

Historical rows (pre-current-season) are never expected to collide — verified
live before writing this script (see conversation) — so the vast majority of
this run is a plain in-place time correction with zero merge decisions.

Usage: python scripts/fix_footballdata_timezone.py [--dry-run]
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy.orm import Session

from app.db import SessionLocal, init_db
from app.models import Match, OddsSnapshot, Prediction

SOURCE = "football_data"
_UK_TZ = ZoneInfo("Europe/London")
_COLLISION_WINDOW = dt.timedelta(hours=3)


def _corrected_kickoff(wrong_utc: dt.datetime) -> dt.datetime:
    local = wrong_utc.replace(tzinfo=_UK_TZ)
    return local.astimezone(dt.timezone.utc).replace(tzinfo=None)


def _find_collision(session: Session, match: Match, corrected: dt.datetime) -> Match | None:
    return (
        session.query(Match)
        .filter(
            Match.id != match.id,
            Match.competition_id == match.competition_id,
            Match.home_team_id == match.home_team_id,
            Match.away_team_id == match.away_team_id,
            Match.utc_kickoff >= corrected - _COLLISION_WINDOW,
            Match.utc_kickoff <= corrected + _COLLISION_WINDOW,
        )
        .one_or_none()
    )


def _prediction_count(session: Session, match_id: int) -> int:
    return session.query(Prediction).filter_by(match_id=match_id).count()


def _drop_match(session: Session, match: Match) -> None:
    session.query(Prediction).filter_by(match_id=match.id).delete(synchronize_session=False)
    session.query(OddsSnapshot).filter_by(match_id=match.id).delete(synchronize_session=False)
    session.delete(match)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="report what would change without writing")
    args = parser.parse_args()

    init_db()
    session = SessionLocal()
    try:
        rows = session.query(Match).filter_by(source=SOURCE).all()
        updated = 0
        merged = 0
        unchanged = 0
        date_shifted = 0

        for match in rows:
            corrected = _corrected_kickoff(match.utc_kickoff)
            if corrected == match.utc_kickoff:
                unchanged += 1
                continue

            if corrected.date() != match.utc_kickoff.date():
                date_shifted += 1
                print(f"  NOTE date shift: match {match.id} {match.utc_kickoff} -> {corrected}")

            collision = _find_collision(session, match, corrected)
            if collision is not None:
                merged += 1
                keep, drop = (
                    (match, collision)
                    if _prediction_count(session, match.id) >= _prediction_count(session, collision.id)
                    else (collision, match)
                )
                print(f"  merge: match {match.id} ({match.source}) + {collision.id} ({collision.source}) -> keep {keep.id}")
                if not args.dry_run:
                    _drop_match(session, drop)
                    session.flush()
                    keep.utc_kickoff = corrected
            else:
                updated += 1
                if not args.dry_run:
                    match.utc_kickoff = corrected

        if args.dry_run:
            session.rollback()
        else:
            session.commit()

        print(
            f"\n{'[dry run] ' if args.dry_run else ''}"
            f"{len(rows)} football_data rows: {unchanged} unchanged, {updated} corrected in place, "
            f"{merged} merged with a duplicate, {date_shifted} crossed a date boundary."
        )
    finally:
        session.close()


if __name__ == "__main__":
    main()

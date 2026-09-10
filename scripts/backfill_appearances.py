"""Manual, multi-day historical backfill of per-player match stats from
bigballsdata.com — see ingest/bigballs_history.py's docstring for why this
reaches back to 2019/20 for free and why it writes to its own gitignored
SQLite store (data/appearances.sqlite) instead of app.db.

Run this by hand, repeatedly, over several days against the free 2000/day
API budget — NOT part of any scheduled job. Each run spends up to
`--budget` requests (default 1800, leaving headroom under the 2000/day cap
for the daily refresh's own bigballsdata calls) and stops cleanly; state is
resumable via appearances.sqlite's own backfilled_dates table, so simply
re-running this script the next day continues where the last one left off.

Usage:
    python scripts/backfill_appearances.py                  # default budget, all 5 leagues, 5 seasons back
    python scripts/backfill_appearances.py --budget 500      # a smaller slice, e.g. to test
    python scripts/backfill_appearances.py --seasons-back 8  # reach back to 2019/20
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.db import SessionLocal, init_db
from app.models import Competition, Match
from ingest.bigballs import LEAGUE_CODES
from ingest.bigballs_history import _connect, backfill_competition
from ingest.seasons import current_season_start_year


def _dates_for_competition(session, slug: str, seasons_back: int) -> list[str]:
    """Every distinct date this competition has a FINISHED match on record,
    oldest first — the real schedule, so no request is ever spent on a date
    with nothing to find. Bounded to `seasons_back` seasons so an early,
    smaller-scope run doesn't try to enumerate matches from before this
    app's own data begins."""
    competition = session.query(Competition).filter_by(slug=slug).one()
    earliest_season_start_year = current_season_start_year() - seasons_back + 1
    rows = (
        session.query(Match.utc_kickoff)
        .filter(
            Match.competition_id == competition.id,
            Match.status == "finished",
            Match.utc_kickoff >= f"{earliest_season_start_year}-07-01",
        )
        .distinct()
        .all()
    )
    return sorted({kickoff.date().isoformat() for (kickoff,) in rows})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget", type=int, default=1800, help="max API requests to spend this run")
    parser.add_argument("--seasons-back", type=int, default=5, help="how many seasons of dates to consider")
    parser.add_argument(
        "--competitions",
        nargs="*",
        default=list(LEAGUE_CODES),
        help=f"which competitions to backfill (default: all of {list(LEAGUE_CODES)})",
    )
    args = parser.parse_args()

    init_db()
    session = SessionLocal()
    conn = _connect()
    try:
        remaining = args.budget
        for slug in args.competitions:
            if remaining <= 0:
                print(f"{slug}: skipped, budget exhausted")
                continue
            dates = _dates_for_competition(session, slug, args.seasons_back)
            result = backfill_competition(session, conn, slug, dates, max_requests=remaining)
            print(f"{slug}: {result}")
            remaining -= result.get("requests_spent", 0)
        print(f"budget remaining this run: {remaining}")
    finally:
        conn.close()
        session.close()


if __name__ == "__main__":
    main()

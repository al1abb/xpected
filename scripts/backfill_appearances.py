"""Manual, multi-day historical backfill of per-player match stats from
bigballsdata.com — see ingest/bigballs_history.py's docstring for why this
reaches back to 2019/20 for free and why it writes to its own gitignored
SQLite store (data/appearances.sqlite) instead of app.db.

Run this by hand, repeatedly, over several days against the free 2000/day
API budget — NOT part of any scheduled job. Each run spends up to
`--budget` requests (default 1000, leaving headroom under the 2000/day cap
for the daily refresh's and the 15-minute lineup syncs' own bigballsdata
calls) and stops cleanly; state is
resumable via appearances.sqlite's own backfilled_dates table, so simply
re-running this script the next day continues where the last one left off.

An API refusal is not a failure of this script: a 429 (daily quota spent)
stops the run early, and a 403/404 on one date is recorded and skipped (see
ingest/bigballs_history.py). Both are printed as GitHub Actions warnings
with the API's own error message, and the run still exits 0 — its progress
is real and already saved, so a red run would only mean a failure email for
something the next run handles by itself.

Usage:
    python scripts/backfill_appearances.py                  # default budget (1000), all 5 leagues, 5 seasons back
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


def _warn(message: str) -> None:
    # `::warning::` shows up as an annotation on the Actions run page (a
    # plain print would be buried in the log) without failing the run.
    print(f"::warning::{message}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget", type=int, default=1000, help="max API requests to spend this run")
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
        rate_limited = False
        for slug in args.competitions:
            if rate_limited:
                print(f"{slug}: skipped, API rate limit hit")
                continue
            if remaining <= 0:
                print(f"{slug}: skipped, budget exhausted")
                continue
            dates = _dates_for_competition(session, slug, args.seasons_back)
            result = backfill_competition(session, conn, slug, dates, max_requests=remaining)
            errors = result.pop("errors", [])
            print(f"{slug}: {result}", flush=True)
            for error in errors:
                _warn(f"{slug}: {error}")
            remaining -= result.get("requests_spent", 0)
            rate_limited = rate_limited or bool(result.get("rate_limited"))
        print(f"budget remaining this run: {remaining}")
    finally:
        conn.close()
        session.close()


if __name__ == "__main__":
    main()

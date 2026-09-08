"""Lightweight, frequent lineup-only sync for matches close to or past
kickoff — the companion sync_lineups.py can't be, since it only ever asks
for finished matches (see ingest/bigballs.py's docstring for why that's a
separate function, not a flag on the same one).

Deliberately narrow, same principle as close_out_finished_matches.py: no
stats, no model refit, no full competition backfill — just "is a lineup
published yet for something kicking off soon or already underway." Meant to
run every ~15 minutes via .github/workflows/close-out-finished.yml, piggy-
backing on that job's existing schedule and commit step rather than adding a
second frequent workflow (this repo's data/app.db is git-linked to a Vercel
deploy — every commit that changes it redeploys, and that job's own history
documents the deploy-budget reason its frequency is already capped).

UNTESTED end-to-end as of writing — see sync_upcoming_lineups's docstring in
ingest/bigballs.py for exactly what's confirmed (the "nothing published yet"
path) vs. not (a real pre-kickoff or live lineup actually landing).

Usage: python scripts/sync_upcoming_lineups.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.db import SessionLocal, init_db
from ingest.bigballs import sync_all_upcoming


def main() -> None:
    init_db()
    session = SessionLocal()
    try:
        for slug, result in sync_all_upcoming(session).items():
            print(f"{slug}: {result}")
    finally:
        session.close()


if __name__ == "__main__":
    main()

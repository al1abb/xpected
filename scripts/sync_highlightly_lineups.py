"""Lightweight, frequent lineup-only sync for Champions League + Europa
League via Highlightly — the companion to scripts/sync_upcoming_lineups.py
(bigballsdata.com), covering the two competitions that source doesn't.

Deliberately narrow, same principle as close_out_finished_matches.py: no
model refit, no full backfill — just "is a lineup available yet for
something kicking off soon or recently underway." Meant to run every ~15
minutes via .github/workflows/close-out-finished.yml, piggybacking on that
job's existing schedule and commit step rather than adding a second
workflow (see ingest/highlightly.py's docstring for the 100/day budget
reasoning this depends on).

Usage: python scripts/sync_highlightly_lineups.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.db import SessionLocal, init_db
from ingest.highlightly import sync_all_upcoming


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

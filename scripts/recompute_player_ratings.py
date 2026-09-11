"""Recomputes and persists PlayerRating rows from data/appearances.sqlite.

This has to run inside .github/workflows/backfill-appearances.yml, not
scripts/refresh.py's daily-refresh.yml — that's the only workflow whose
data/appearances.sqlite has ever seen real appearance rows (restored from
its own actions/cache, keyed appearances-db-*; see that workflow's own
comments). daily-refresh.yml checks out a fresh repo with no such cache, so
the same step running there computes against an empty file every time and
persists nothing — scripts/refresh.py still calls
model.player_elo.refresh_player_ratings() too, for a local run where the
file exists on disk directly, but that call is a real no-op in CI.

Usage: python scripts/recompute_player_ratings.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.db import SessionLocal, init_db
from model.player_elo import refresh_player_ratings


def main() -> None:
    init_db()
    session = SessionLocal()
    try:
        print(refresh_player_ratings(session))
    finally:
        session.close()


if __name__ == "__main__":
    main()

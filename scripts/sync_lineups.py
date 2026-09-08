"""Manual, on-demand sync of lineups + per-match player stats from
bigballsdata.com — the automatic version runs daily via scripts/refresh.py.
Use this to force a fresh pull without waiting for the schedule, or to sync
a larger `limit` than the daily job bothers with. See ingest/bigballs.py's
docstring for coverage (5 competitions, current season + 1 prior only) and
the known rough edges (7% null player_id, thin-but-dense-within-window
lineup coverage).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.db import SessionLocal, init_db
from ingest.bigballs import sync_all


def main() -> None:
    init_db()
    session = SessionLocal()
    try:
        for slug, result in sync_all(session).items():
            print(f"{slug}: {result}")
    finally:
        session.close()


if __name__ == "__main__":
    main()

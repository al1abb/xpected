"""One-off/occasional sync of team crests from football-data.org (crests
essentially never change, so this doesn't need to run on every refresh —
API-Football-sourced teams already get their logo captured automatically
during normal fixture ingest, see ingest/api_football.py)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.db import SessionLocal, init_db
from ingest.football_data_org import sync_team_crests


def main() -> None:
    init_db()
    session = SessionLocal()
    try:
        count = sync_team_crests(session)
        print(f"Team crests set: {count}")
    finally:
        session.close()


if __name__ == "__main__":
    main()

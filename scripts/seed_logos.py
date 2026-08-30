"""One-off/occasional sync of team crests from football-data.org (crests
essentially never change, so this doesn't need to run on every refresh —
API-Football-sourced teams already get their logo captured automatically
during normal fixture ingest, see ingest/api_football.py)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.db import SessionLocal, init_db
from ingest.api_football import sync_league_team_crests
from ingest.football_data_org import sync_team_crests


def main() -> None:
    init_db()
    session = SessionLocal()
    try:
        count = sync_team_crests(session)
        print(f"Team crests set (football-data.org): {count}")

        # Süper Lig gets its match data free from football-data.co.uk (no
        # logos there) and isn't covered by football-data.org's crest sync
        # either — one API-Football call for the whole league is worth
        # spending budget on since it's a genuine one-off, not an ongoing cost.
        # The free plan only allows /teams for 2022-2024 (same restriction as
        # current-season fixtures elsewhere in this codebase) — crests don't
        # change season to season, so an older allowed season still works for
        # every club that hasn't been promoted/relegated since.
        af_count = sync_league_team_crests(session, "super-lig", 2024)
        print(f"Team crests set (API-Football, super-lig 2024 roster): {af_count}")
    finally:
        session.close()


if __name__ == "__main__":
    main()

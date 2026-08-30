"""Occasional sync of top scorers/assists for every competition — one
API-Football request per league per category (~24 requests total across all
competitions today), well inside the daily cap. Not part of the regular
fixture-refresh loop; re-run weekly or whenever the tables look stale."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import COMPETITIONS
from app.db import SessionLocal, init_db
from ingest.api_football import sync_top_assists, sync_top_scorers
from ingest.seasons import current_season_start_year


# The free API-Football plan only allows seasons 2022-2024 (confirmed via the
# actual API error: "Free plans do not have access to this season, try from
# 2022 to 2024") — the same wall the Süper Lig crest sync works around in
# scripts/seed_logos.py. 2024 is the newest allowed season, so it's the last
# fallback tried after the current and prior season both fail.
_OLDEST_ALLOWED_SEASON_FALLBACK = 2024


def _sync_with_fallback(session, sync_fn, slug, season):
    candidates = [season, season - 1, _OLDEST_ALLOWED_SEASON_FALLBACK]
    for candidate in dict.fromkeys(candidates):  # de-dupe, keep order
        result = sync_fn(session, slug, candidate)
        if result is not None:
            return result
    return None


def main() -> None:
    init_db()
    session = SessionLocal()
    try:
        season = current_season_start_year()
        for comp in COMPETITIONS:
            slug = comp["slug"]
            scorers = _sync_with_fallback(session, sync_top_scorers, slug, season)
            assists = _sync_with_fallback(session, sync_top_assists, slug, season)
            print(
                f"{slug}: top scorers={scorers if scorers is not None else 'skipped'}, "
                f"top assists={assists if assists is not None else 'skipped'}"
            )
    finally:
        session.close()


if __name__ == "__main__":
    main()

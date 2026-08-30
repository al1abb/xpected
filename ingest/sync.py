"""Ingest orchestrator. `run_free_sync` covers the zero-API-cost sources
(football-data.co.uk + ClubElo); `run_api_sync` (added in phase 3) covers the
competitions those can't supply.
"""

from __future__ import annotations

from app.config import COMPETITIONS
from app.db import SessionLocal, init_db
from ingest import api_football, clubelo, football_data_org, fixturedownload, footballdata_csv
from ingest.known_aliases import apply_known_aliases
from ingest.seasons import af_seasons_back

# Competitions football-data.co.uk/ClubElo can't supply — these go through API-Football.
API_ONLY_SLUGS = [c["slug"] for c in COMPETITIONS if not c["fd_code"]]


def run_free_sync(seasons_back: int = 4) -> dict:
    init_db()
    session = SessionLocal()
    try:
        apply_known_aliases(session)
        results_count = footballdata_csv.ingest_results(session, seasons_back=seasons_back)
        fixtures_count = footballdata_csv.ingest_fixtures(session)
        elo_count = clubelo.ingest_current_ratings(session)
        return {
            "football_data_results": results_count,
            "football_data_fixtures": fixtures_count,
            "clubelo_ratings": elo_count,
        }
    finally:
        session.close()


def run_api_sync(seasons_back: int = 3) -> dict:
    init_db()
    session = SessionLocal()
    summary: dict[str, int] = {}
    try:
        apply_known_aliases(session)
        seasons = af_seasons_back(seasons_back)
        for slug in API_ONLY_SLUGS:
            total = 0
            for season in seasons:
                total += api_football.ingest_competition_season(session, slug, season)
            summary[slug] = total
        return summary
    finally:
        session.close()


def run_current_season_fixture_sync() -> dict:
    """Full current-season fixtures (not just the next few days) via
    fixturedownload.com's whole-season JSON feed — originally just the 3 UEFA
    competitions, now every competition in `fixturedownload.FD_SLUGS`
    (the 8 domestic leagues too, since football-data.co.uk's fixtures.csv
    turned out to only cover a rolling ~4-day window — see
    ingest/fixturedownload.py's module docstring). Champions League also gets
    a second, independent source (football-data.org) alongside fixturedownload.
    """
    init_db()
    session = SessionLocal()
    try:
        apply_known_aliases(session)
        summary = {
            f"fixturedownload_{slug}": fixturedownload.ingest_competition(session, slug)
            for slug in fixturedownload.FD_SLUGS
        }
        summary["football_data_org_champions_league"] = football_data_org.ingest_champions_league(session)
        return summary
    finally:
        session.close()


if __name__ == "__main__":
    summary = run_free_sync()
    for key, value in summary.items():
        print(f"{key}: {value}")

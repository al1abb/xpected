"""Backfill historical data.

--free             : football-data.co.uk + ClubElo only, zero API requests (phase 2)
--api              : API-Football, for the competitions the free sources can't cover (phase 3)
--current-fixtures : full current-season fixtures via fixturedownload.com (all 8 domestic
                     leagues + 3 UEFA competitions) + football-data.org (Champions League
                     second source) — needed because football-data.co.uk's own fixtures.csv
                     only covers a rolling ~4-day window, not the whole season.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ingest.sync import run_api_sync, run_current_season_fixture_sync, run_free_sync


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--free", action="store_true", help="football-data.co.uk + ClubElo")
    parser.add_argument("--api", action="store_true", help="API-Football (phase 3)")
    parser.add_argument("--current-fixtures", action="store_true", help="full current-season fixtures, all leagues")
    parser.add_argument("--seasons-back", type=int, default=4)
    args = parser.parse_args()

    if not args.free and not args.api and not args.current_fixtures:
        parser.error("pass --free, --api, and/or --current-fixtures")

    if args.free:
        print("=== Free sources (football-data.co.uk + ClubElo) ===")
        summary = run_free_sync(seasons_back=args.seasons_back)
        for key, value in summary.items():
            print(f"{key}: {value}")

    if args.api:
        print("=== API-Football (UEFA competitions + Azerbaijan) ===")
        summary = run_api_sync(seasons_back=args.seasons_back)
        for key, value in summary.items():
            print(f"{key}: {value}")

    if args.current_fixtures:
        print("=== Current-season fixtures (fixturedownload.com, all leagues + football-data.org) ===")
        summary = run_current_season_fixture_sync()
        for key, value in summary.items():
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()

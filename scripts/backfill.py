"""Backfill historical data.

--free         : football-data.co.uk + ClubElo only, zero API requests (phase 2)
--api          : API-Football, for the competitions the free sources can't cover (phase 3)
--uefa-current : current-season UEFA fixtures via fixturedownload.com + football-data.org
                 (API-Football's free tier blocks the current season for these competitions)
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ingest.sync import run_api_sync, run_free_sync, run_uefa_current_sync


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--free", action="store_true", help="football-data.co.uk + ClubElo")
    parser.add_argument("--api", action="store_true", help="API-Football (phase 3)")
    parser.add_argument("--uefa-current", action="store_true", help="current-season UEFA fixtures")
    parser.add_argument("--seasons-back", type=int, default=4)
    args = parser.parse_args()

    if not args.free and not args.api and not args.uefa_current:
        parser.error("pass --free, --api, and/or --uefa-current")

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

    if args.uefa_current:
        print("=== Current-season UEFA fixtures (fixturedownload.com + football-data.org) ===")
        summary = run_uefa_current_sync()
        for key, value in summary.items():
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()

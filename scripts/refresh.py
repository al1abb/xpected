"""Daily refresh: pull new results/fixtures from every source, refit the
model, and regenerate predictions. Intended to run on a schedule (Windows Task
Scheduler — see README note in the plan). Safe to run as often as you like:
every step is idempotent and budget-aware.

Season rollover is automatic — ingest/seasons.py derives the current season
from today's date, so this script needs no yearly maintenance; it just starts
picking up the new season's fixtures once July rolls around.

Pass --backtest to also refresh data/backtest_results.json (skipped by
default since a full walk-forward backtest takes a few minutes; run it
weekly, not daily).
"""

import argparse
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import BASE_DIR
from app.db import SessionLocal, init_db
from ingest.football_data_org_players import sync_scorers, sync_squads
from ingest.news import sync_news
from ingest.sync import run_api_sync, run_current_season_fixture_sync, run_free_sync
from model.elo import compute_ratings, persist_ratings
from model.predict import generate_predictions


def _step(name: str, fn) -> None:
    print(f"--- {name} ---")
    try:
        result = fn()
        print(result)
    except Exception:
        # One source failing must not stop the rest of the refresh — this
        # loudly logs the failure instead of silently doing nothing, which is
        # exactly the "silent staleness" failure mode this script exists to avoid.
        print(f"FAILED: {name}")
        traceback.print_exc()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backtest", action="store_true", help="also refresh data/backtest_results.json")
    args = parser.parse_args()

    init_db()

    _step("free sources (football-data.co.uk + ClubElo)", lambda: run_free_sync(seasons_back=1))
    _step("current-season fixtures (fixturedownload, all leagues + UEFA)", run_current_season_fixture_sync)
    _step("API-Football (UEFA competitions + Azerbaijan)", lambda: run_api_sync(seasons_back=1))

    session = SessionLocal()
    try:
        count = generate_predictions(session, notes="scheduled refresh via scripts/refresh.py")
        print(f"--- predictions regenerated: {count} ---")

        # Persist the blended Elo ratings so the web app can read them instead
        # of recomputing. compute_ratings costs 8-12s and makes a live ClubElo
        # request — acceptable here, unacceptable inside a serverless request
        # (see app/main.py::_cached_ratings). Doing it after ingest means the
        # stored ratings reflect the results pulled in above.
        _step(
            "persist Elo ratings for the web app",
            lambda: f"{persist_ratings(session, compute_ratings(session))} team ratings stored",
        )

        # Current-season squads + scorers (football-data.org) — the fix for
        # PlayerStat being stuck on 2024/25 data, since API-Football's free
        # tier walls off every season after that. Covers only its TIER_ONE
        # competitions (see ingest/football_data_org_players.py); app/main.py
        # shows an explicit empty state for the rest rather than stale data.
        _step("squads (football-data.org)", lambda: sync_squads(session))
        _step("top scorers/assists (football-data.org)", lambda: sync_scorers(session))
        _step("football news (RSS)", lambda: sync_news(session))
    finally:
        session.close()

    if args.backtest:
        import json

        from model.backtest import run_backtest

        session = SessionLocal()
        try:
            result = run_backtest(session)
            (BASE_DIR / "data" / "backtest_results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
            print("--- backtest results refreshed ---")
        finally:
            session.close()


if __name__ == "__main__":
    main()

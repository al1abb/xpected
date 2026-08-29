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
from ingest.sync import run_api_sync, run_free_sync, run_uefa_current_sync
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
    _step("current-season UEFA fixtures", run_uefa_current_sync)
    _step("API-Football (UEFA competitions + Azerbaijan)", lambda: run_api_sync(seasons_back=1))

    session = SessionLocal()
    try:
        count = generate_predictions(session, notes="scheduled refresh via scripts/refresh.py")
        print(f"--- predictions regenerated: {count} ---")
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

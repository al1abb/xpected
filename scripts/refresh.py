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
from ingest.bigballs import sync_all as sync_lineups
from ingest.football_data_org_players import sync_scorers, sync_squads
from ingest.news import sync_news
from ingest.resolve_players import seed_players_from_squads
from ingest.sync import run_api_sync, run_current_season_fixture_sync, run_free_sync
from model.elo import compute_ratings, persist_ratings
from model.player_elo import compute_player_ratings, persist_player_ratings, prune_player_ratings
from model.predict import generate_predictions


def _refresh_player_ratings(session) -> str:
    ratings, appearance_counts = compute_player_ratings(session)
    stored = persist_player_ratings(session, ratings, appearance_counts)
    pruned = prune_player_ratings(session)
    return f"{stored} player ratings stored, {pruned} stale snapshot(s) pruned"


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

        # Player-level Elo, replayed from data/appearances.sqlite (see
        # ingest/bigballs_history.py + model/player_elo.py). Pruned in the
        # same step it's persisted — see PlayerRating's own docstring for
        # why this can't wait, unlike team elo_ratings.
        _step("persist player Elo ratings for the web app", lambda: _refresh_player_ratings(session))

        # Current-season squads + scorers (football-data.org) — the fix for
        # PlayerStat being stuck on 2024/25 data, since API-Football's free
        # tier walls off every season after that. Covers only its TIER_ONE
        # competitions (see ingest/football_data_org_players.py); app/main.py
        # shows an explicit empty state for the rest rather than stale data.
        _step("squads (football-data.org)", lambda: sync_squads(session))
        # Resolves any new squad player to a Player + backfills position onto
        # existing ones (see that function's docstring) — after squads sync,
        # so it sees today's roster, not yesterday's.
        _step(
            "seed player identities from squads",
            lambda: f"{seed_players_from_squads(session)} new players seeded",
        )
        _step("top scorers/assists (football-data.org)", lambda: sync_scorers(session))

        # Lineups + per-match player stats (bigballsdata.com) — display only,
        # never a model input (see app/models.py: Lineup, PlayerMatchStat).
        # Covers 5 competitions, current season + 1 prior only; skips itself
        # cleanly if BIGBALLS_API_KEY isn't set. `limit=20` keeps this to
        # ~200 requests/day across the 5 leagues, well inside the free plan's
        # 2000/day (see ingest/bigballs.py).
        _step("lineups + player stats (bigballsdata.com)", lambda: sync_lineups(session))

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

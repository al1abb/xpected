"""One-time (and resumable) historical backfill of per-player match stats
from bigballsdata.com — the data source for model/player_elo.py's rating
replay.

Confirmed live while planning this feature: `/v1/stored/matches/{id}/stats`
is NOT walled to "current season + 1 prior" the way `/v1/stored/matches/{id}/lineups`
is (ingest/bigballs.py's own documented limit) — it returns real per-player
minutes and ratings back to 2019/20, the full span of this app's match
history. `/v1/matches?date=` — undocumented in our own code before this
module, but real and confirmed live — lets this enumerate a specific
historical date instead of only "most recent N finished matches".

Deliberately writes to its own gitignored SQLite store
(data/appearances.sqlite), NEVER app.db. app.db is git-linked to Vercel
(every commit redeploys), and Lineup+PlayerMatchStat's own cost — confirmed
from git history — was ~31KB/match; persisting that for ~12,600 historical
matches would add hundreds of MB to a repo whose committed database is
currently ~10MB. Only two things ever reach app.db from this process: the
Player/PlayerAlias identity rows (ingest/resolve_players.py — real,
durable, and small: one row per real person, not per appearance) and,
eventually, the derived PlayerRating rows model/player_elo.py persists —
never the raw per-match minutes/rating numbers this module collects.

Resumable and idempotent by design: enumerates one (league, date) at a
time — dates come from the caller, which should pass only dates this app's
own `matches` table says that league actually played, so no request is
wasted on an empty date — checks `backfilled_dates` first, and skips a date
already done. A killed run just re-checks a few already-done dates on
restart, for free (those checks cost nothing: they never call the API).

Manual, multi-day operation against the free 2000/day budget — NOT part of
any GitHub Actions workflow, and never triggered by scripts/refresh.py. Run
via scripts/backfill_appearances.py.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from sqlalchemy.orm import Session

from app.models import Competition, Match
from ingest.bigballs import (
    LEAGUE_CODES,
    _extract_stat,
    _float_or_none,
    _int_or_none,
    _MINUTES_KEYS,
    _paced_fetch_json,
    _RATING_KEYS,
    _resolve_match,
)
from ingest.resolve import build_alias_pool
from ingest.resolve_players import resolve_or_create_player

APPEARANCES_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "appearances.sqlite"

# Historical dates are permanently settled — cache effectively forever, same
# reasoning as ingest/clubelo.py's _PAST_DATE_MAX_AGE_HOURS. Keeps a
# restarted run from re-fetching anything it already has on disk, let alone
# already recorded in backfilled_dates.
_PERMANENT_MAX_AGE_HOURS = 24 * 365 * 10


def _connect(path: Path = APPEARANCES_DB_PATH) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute(
        """CREATE TABLE IF NOT EXISTS appearances (
            match_id INTEGER NOT NULL,
            player_id INTEGER NOT NULL,
            team_id INTEGER NOT NULL,
            minutes INTEGER,
            rating REAL,
            PRIMARY KEY (match_id, player_id)
        )"""
    )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS backfilled_dates (
            league_key TEXT NOT NULL,
            date TEXT NOT NULL,
            PRIMARY KEY (league_key, date)
        )"""
    )
    return conn


def _already_backfilled(conn: sqlite3.Connection, league_key: str, date: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM backfilled_dates WHERE league_key = ? AND date = ?", (league_key, date)
    ).fetchone()
    return row is not None


def _mark_backfilled(conn: sqlite3.Connection, league_key: str, date: str) -> None:
    conn.execute("INSERT OR IGNORE INTO backfilled_dates (league_key, date) VALUES (?, ?)", (league_key, date))
    conn.commit()


def _backfill_match_appearances(
    session: Session,
    conn: sqlite3.Connection,
    match: Match,
    bbs_match_id: str,
    home_bbs_name: str,
    away_bbs_name: str,
) -> int:
    already = conn.execute("SELECT 1 FROM appearances WHERE match_id = ? LIMIT 1", (match.id,)).fetchone()
    if already is not None:
        return 0

    data = _paced_fetch_json(f"v1/stored/matches/{bbs_match_id}/stats", max_age_hours=_PERMANENT_MAX_AGE_HOURS)
    players = (data.get("data") or {}).get("players") or []
    if not players:
        return 0

    # Same join-key reasoning as ingest/bigballs.py::_sync_match_player_stats:
    # keyed on bigballsdata's OWN name strings from the /matches listing, not
    # our Team.canonical_name, which can legitimately differ.
    team_by_name = {home_bbs_name: match.home_team_id, away_bbs_name: match.away_team_id}

    written = 0
    for player in players:
        name = player.get("name")
        team_id = team_by_name.get(player.get("team_name"))
        if not name or team_id is None:
            continue
        stats = player.get("stats") or {}
        player_row = resolve_or_create_player(
            session,
            name,
            "bigballs",
            team_id=team_id,
            source_player_id=player.get("id"),
            context=f"bigballs history backfill, match_id={match.id}",
        )
        session.commit()
        conn.execute(
            "INSERT OR IGNORE INTO appearances (match_id, player_id, team_id, minutes, rating) VALUES (?, ?, ?, ?, ?)",
            (
                match.id,
                player_row.id,
                team_id,
                _int_or_none(_extract_stat(stats, _MINUTES_KEYS)),
                _float_or_none(_extract_stat(stats, _RATING_KEYS)),
            ),
        )
        written += 1
    conn.commit()
    return written


def backfill_competition(
    session: Session,
    conn: sqlite3.Connection,
    competition_slug: str,
    dates: list[str],
    *,
    max_requests: int | None = None,
) -> dict[str, int | str | bool]:
    """`dates`: 'YYYY-MM-DD' strings, from the caller's own query against this
    app's `matches` table for when this league actually played — never
    enumerated blindly, so a date with nothing scheduled never costs a
    request.

    `max_requests`: stop cleanly once this many API calls have been spent
    (one per date listing, one per match's /stats fetch) — this is meant to
    run as one bounded slice of a multi-day backfill against the free
    2000/day budget, not to exhaust it in a single invocation. A date is
    only ever marked done in `backfilled_dates` once every match in it has
    been fully processed, so stopping mid-date is always safe to resume:
    the next run re-lists that same date (one extra request) and picks up
    where this one left off, never double-counting or skipping a match.
    """
    league_key = LEAGUE_CODES.get(competition_slug)
    if league_key is None:
        return {"skipped": "not covered by bigballsdata.com"}

    competition = session.query(Competition).filter_by(slug=competition_slug).one()
    pool = build_alias_pool(session, exclude_source="bigballs")

    matched = appearances_written = dates_skipped = requests_spent = 0
    budget_exhausted = False
    for date in dates:
        if _already_backfilled(conn, league_key, date):
            dates_skipped += 1
            continue
        if max_requests is not None and requests_spent >= max_requests:
            budget_exhausted = True
            break

        data = _paced_fetch_json(
            "v1/matches",
            params={"sport": "football", "league": league_key, "date": date},
            max_age_hours=_PERMANENT_MAX_AGE_HOURS,
        )
        requests_spent += 1

        rows = data.get("data", [])
        date_fully_processed = True
        for row in rows:
            if max_requests is not None and requests_spent >= max_requests:
                date_fully_processed = False
                budget_exhausted = True
                break
            match = _resolve_match(session, competition, row, pool)
            if match is None:
                continue
            matched += 1
            already = conn.execute(
                "SELECT 1 FROM appearances WHERE match_id = ? LIMIT 1", (match.id,)
            ).fetchone()
            if already is None:
                requests_spent += 1  # /stats call inside _backfill_match_appearances
            appearances_written += _backfill_match_appearances(
                session, conn, match, row["id"], row["home"]["name"], row["away"]["name"]
            )

        if date_fully_processed:
            _mark_backfilled(conn, league_key, date)
        if budget_exhausted:
            break

    return {
        "matched": matched,
        "appearances_written": appearances_written,
        "dates_skipped_already_done": dates_skipped,
        "requests_spent": requests_spent,
        "budget_exhausted": budget_exhausted,
    }

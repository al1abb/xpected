"""Highlightly ingest — lineups for Champions League + Europa League, the two
of this app's 12 competitions bigballsdata.com (ingest/bigballs.py) doesn't
cover at all (confirmed via its own /v1/coverage: 0% on every data type for
Champions League, and Europa League isn't in its league list).

Confirmed live (Sept 2026) against real, in-progress Champions League
matches: real starting XIs, correctly grouped by formation row (their
/lineups response nests players by row — GK row first, then one row per
formation number — rather than the flat list + formation string
bigballsdata.com gives). Their own match-status tracking was also noticeably
more accurate for this competition than bigballsdata.com's: a match still
shown "scheduled" by bigballsdata.com ~20 minutes after kickoff was
correctly "Half time" here.

Lineups only — no player-match-stats sync. Their /players/{id} endpoint has
real profile depth (an exact position like "Left-Back", not just the coarse
Goalkeeper/Defender/Midfielder/Forward /lineups itself returns) but every
player checked had `logo: null` (no headshot), and fetching it per-player
would burn through the daily budget fast regardless — not evaluated,
because the exact position isn't needed: feeding this source's rows through
the SAME position-code + order_index + formation-string shape
bigballsdata.com uses means app/main.py's existing pitch-layout code (row
grouping, _split_midfield_rows) works unchanged, no per-source branching
there at all. See _POSITION_MAP.

Budget is the defining constraint here, not season walls or coverage: 100
requests/day, confirmed via response headers, no documented per-minute cap.
A naive "check every ~15 minutes across a several-hour pre-kickoff window"
design burns through that alone — a single 8-match Champions League night
polled every 15 minutes for 3+ hours before kickoff is >100 requests before
counting a single actual lineup fetch. Two things keep this affordable:
  1. Every external call is gated behind a free, LOCAL check first — this
     app's own Match table (already populated by football-data.org /
     fixturedownload.com) says whether a covered competition even has a
     match in the target window, at zero Highlightly cost, before ever
     calling out.
  2. Once a lineup is captured for a match, it's never re-fetched — later
     runs skip any match that already has Highlightly-sourced Lineup rows.

Team/match resolution reuses the exact same alias-pool machinery and
enrichment-only stance as ingest/bigballs.py (see that module for why): this
resolves against an EXISTING Match row and never invents one.
"""

from __future__ import annotations

import datetime as dt
import json
import time

from sqlalchemy.orm import Session

from app.config import HIGHLIGHTLY_API_BASE, settings
from app.models import Competition, IngestLog, Lineup, Match
from ingest.cache import fetch_text
from ingest.resolve import build_alias_pool, resolve_existing_team
from ingest.seasons import current_season_start_year

SOURCE = "highlightly"

LEAGUE_IDS: dict[str, int] = {
    "champions-league": 2486,
    "europa-league": 3337,
}

# bigballsdata.com's own G/D/M/F convention — reused (not because it's a
# standard, just so app/main.py's row-grouping code needs no per-source
# branching). Anything unrecognized falls back to the first letter of
# whatever Highlightly sent, uppercased, rather than dropping the player.
_POSITION_MAP = {"Goalkeeper": "G", "Defender": "D", "Midfielder": "M", "Forward": "F"}

# Confirmed live via response headers: 100/day, no stated per-minute limit.
# Paced anyway, defensively, matching every other source in this codebase.
_MIN_SECONDS_BETWEEN_REQUESTS = 0.5
_last_request_at: float | None = None


def _headers() -> dict:
    if not settings.highlightly_api_key:
        raise RuntimeError("HIGHLIGHTLY_API_KEY is not set in .env")
    return {"x-rapidapi-key": settings.highlightly_api_key}


def _paced_fetch_json(path: str, *, params: dict | None = None, max_age_hours: float):
    global _last_request_at
    if _last_request_at is not None:
        elapsed = time.monotonic() - _last_request_at
        if elapsed < _MIN_SECONDS_BETWEEN_REQUESTS:
            time.sleep(_MIN_SECONDS_BETWEEN_REQUESTS - elapsed)
    _last_request_at = time.monotonic()

    text = fetch_text(
        f"{HIGHLIGHTLY_API_BASE}/{path.lstrip('/')}",
        subdir="highlightly",
        max_age_hours=max_age_hours,
        headers=_headers(),
        params=params,
    )
    return json.loads(text)


def _sync_match_lineups(session: Session, match: Match, hl_match_id: int) -> bool:
    data = _paced_fetch_json(f"lineups/{hl_match_id}", max_age_hours=0.2)
    home_data = data.get("homeTeam") or {}
    away_data = data.get("awayTeam") or {}
    home_rows = home_data.get("initialLineup") or []
    away_rows = away_data.get("initialLineup") or []
    if not home_rows and not away_rows:
        # Confirmed live: not-yet-published returns {"formation": "Unknown",
        # "substitutes": [], "initialLineup": []} for both sides, cleanly —
        # never an error, never a wrong/partial row. Nothing to write.
        return False

    home_formation = home_data.get("formation")
    away_formation = away_data.get("formation")
    match.home_formation = home_formation if home_formation and home_formation != "Unknown" else None
    match.away_formation = away_formation if away_formation and away_formation != "Unknown" else None
    match.highlightly_match_id = hl_match_id

    session.query(Lineup).filter_by(match_id=match.id).delete()
    for team_id, rows, subs in (
        (match.home_team_id, home_rows, home_data.get("substitutes") or []),
        (match.away_team_id, away_rows, away_data.get("substitutes") or []),
    ):
        order_index = 0
        # `rows` is already grouped by formation row (GK row first, then one
        # row per formation number) — order_index just needs to preserve
        # that row-by-row, left-to-right order; app/main.py's _build_rows
        # re-derives the row structure itself from position code + this
        # index + the formation string, same as it does for bigballsdata.com,
        # so no separate code path is needed to use Highlightly's grouping
        # directly even though it's already given to us.
        for formation_row in rows:
            for player in formation_row:
                name = player.get("name")
                if not name:
                    continue
                raw_position = player.get("position") or ""
                position = _POSITION_MAP.get(raw_position, (raw_position[:1].upper() or None))
                session.add(
                    Lineup(
                        match_id=match.id,
                        team_id=team_id,
                        player_name=name,
                        position=position,
                        jersey_number=player.get("number"),
                        starter=True,
                        order_index=order_index,
                        bbs_player_id=str(player["id"]) if player.get("id") is not None else None,
                        synced_at=dt.datetime.utcnow(),
                    )
                )
                order_index += 1
        for player in subs:
            name = player.get("name")
            if not name:
                continue
            raw_position = player.get("position") or ""
            position = _POSITION_MAP.get(raw_position, (raw_position[:1].upper() or None))
            session.add(
                Lineup(
                    match_id=match.id,
                    team_id=team_id,
                    player_name=name,
                    position=position,
                    jersey_number=player.get("number"),
                    starter=False,
                    order_index=order_index,
                    bbs_player_id=str(player["id"]) if player.get("id") is not None else None,
                    synced_at=dt.datetime.utcnow(),
                )
            )
            order_index += 1
    return True


def sync_upcoming_lineups(
    session: Session, competition_slug: str, *, hours_ahead: float = 1.5, hours_behind: float = 2.0
) -> dict[str, int]:
    """Lineups for matches kicking off within `hours_ahead`, or that kicked
    off within the last `hours_behind` (their own docs: lineups stay
    available until 120 minutes post-kickoff). 1.5h ahead is deliberately
    generous against their documented "40 minutes before" publish point —
    enough headroom to catch it on an early check without polling for
    hours before a real chance of it existing."""
    league_id = LEAGUE_IDS.get(competition_slug)
    if league_id is None:
        return {"skipped": "not covered by Highlightly"}
    if not settings.highlightly_api_key:
        return {"skipped": "HIGHLIGHTLY_API_KEY not set"}

    competition = session.query(Competition).filter_by(slug=competition_slug).one()
    now = dt.datetime.utcnow()
    window_start = now - dt.timedelta(hours=hours_behind)
    window_end = now + dt.timedelta(hours=hours_ahead)

    # Free, local-only check FIRST — this app's own Match rows already know
    # the schedule (see module docstring: this is what keeps a 100/day
    # budget viable). Zero Highlightly calls happen below this point unless
    # there's a real, not-yet-captured candidate.
    in_window = (
        session.query(Match)
        .filter(Match.competition_id == competition.id, Match.utc_kickoff >= window_start, Match.utc_kickoff <= window_end)
        .all()
    )
    if not in_window:
        return {"matched": 0, "published": 0, "unresolved": 0}

    already_captured = {
        row[0]
        for row in session.query(Lineup.match_id)
        .filter(Lineup.match_id.in_([m.id for m in in_window]))
        .distinct()
    }
    candidate_matches = [m for m in in_window if m.id not in already_captured]
    if not candidate_matches:
        return {"matched": 0, "published": 0, "unresolved": 0}

    started = dt.datetime.utcnow()
    season = current_season_start_year()
    pool = build_alias_pool(session, exclude_source=SOURCE)

    # One list call per distinct date covers every match that date for this
    # league — cheap relative to the per-match lineup calls that follow, and
    # is how we learn Highlightly's own integer id for each match.
    hl_rows_by_date: dict[str, list[dict]] = {}
    for date_str in {m.utc_kickoff.date().isoformat() for m in candidate_matches}:
        data = _paced_fetch_json(
            "matches", params={"leagueId": league_id, "date": date_str, "season": season}, max_age_hours=0.2
        )
        hl_rows_by_date[date_str] = data.get("data", [])

    matched = published = unresolved = 0
    for match in candidate_matches:
        date_str = match.utc_kickoff.date().isoformat()
        hl_match_id = None
        for row in hl_rows_by_date.get(date_str, []):
            home = resolve_existing_team(
                session, row["homeTeam"]["name"], SOURCE, context=f"highlightly {competition_slug}", pool=pool
            )
            away = resolve_existing_team(
                session, row["awayTeam"]["name"], SOURCE, context=f"highlightly {competition_slug}", pool=pool
            )
            if home is not None and away is not None and home.id == match.home_team_id and away.id == match.away_team_id:
                hl_match_id = row["id"]
                break

        if hl_match_id is None:
            unresolved += 1
            continue

        matched += 1
        if _sync_match_lineups(session, match, hl_match_id):
            published += 1
        session.commit()

    if published:
        session.add(
            IngestLog(
                source=SOURCE,
                competition_id=competition.id,
                started_at=started,
                finished_at=dt.datetime.utcnow(),
                status="ok",
                rows_ingested=published,
                message=f"upcoming lineups: matched={matched}, published={published}, unresolved={unresolved}",
            )
        )
        session.commit()
    return {"matched": matched, "published": published, "unresolved": unresolved}


def sync_all_upcoming(session: Session, *, hours_ahead: float = 1.5, hours_behind: float = 2.0) -> dict[str, dict]:
    """sync_upcoming_lineups for every competition LEAGUE_IDS covers."""
    return {
        slug: sync_upcoming_lineups(session, slug, hours_ahead=hours_ahead, hours_behind=hours_behind)
        for slug in LEAGUE_IDS
    }

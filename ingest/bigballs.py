"""bigballsdata.com ingest — lineups + per-match player stats.

Investigated Sept 2026 as a candidate to close the gap future-plans.md
documents under "Player-level data": football-data.org's `/scorers` only
covers the ~49 goal/assist scorers per league, and the project's own rule was
"everyone or nobody" — no partial roster. Spot-checked live (see session
notes, not repeated here): lineups return real starters+bench with formation,
and `/v1/stored/matches/{id}/stats` returns a full per-match box score for
EVERY player who featured, not just scorers. That clears "everyone" for
players who play. It does NOT close the gap for the six competitions this
app tracks that aren't in their coverage at all (see LEAGUE_CODES below).

Confirmed live rough edges, priced into this module rather than ignored:
- `player_id` is null on ~7% of lineup/stat entries — kept (still a real,
  displayable name) but never joinable, so never used to resolve a team.
- The free plan is walled to "current season + 1 prior" — a `history_not_included`
  error names the exact wall, so this never bothers retrying further back.
- `/v1/players/{id}/stats` (season aggregates) returned nothing live despite
  `/v1/coverage` claiming ~92% coverage — not used here. Only the two
  per-MATCH endpoints (lineups, stats) are wired up; a season rollup would
  have to be built by summing PlayerMatchStat rows ourselves.

Enrichment only, same stance as ingest/api_football.py's crest sync: this
resolves against an EXISTING Match row via the natural key and never creates
one. A match bigballsdata.com doesn't have, or whose teams don't resolve,
simply gets no Lineup/PlayerMatchStat rows — never a wrong or partial one.

NOT wired into scripts/refresh.py. Run scripts/sync_lineups.py manually until
a decision is made to add it to the daily loop.
"""

from __future__ import annotations

import datetime as dt
import json
import time

from sqlalchemy.orm import Session

from app.config import BIGBALLS_API_BASE, settings
from app.models import Competition, IngestLog, Lineup, Match, PlayerMatchStat, Team
from ingest.cache import fetch_text
from ingest.resolve import build_alias_pool, resolve_existing_team

SOURCE = "bigballs"

# Confirmed live (Sept 2026): only these 5 of this app's 12 competitions have
# any lineup/player-stat coverage at all. UEFA Champions League is listed by
# bigballsdata.com but showed 0% coverage on every data type when checked via
# /v1/coverage — deliberately excluded rather than wired up to sync nothing.
# Eredivisie, Primeira Liga, Süper Lig, Azerbaijan Premyer Liqa, Europa League
# and Conference League aren't in their league list at all.
LEAGUE_CODES: dict[str, str] = {
    "premier-league": "epl",
    "la-liga": "laliga",
    "serie-a": "seriea",
    "bundesliga": "bundesliga",
    "ligue-1": "ligue1",
}

# Free plan: 100/min, 2000/day (confirmed via X-RateLimit-* response headers).
# Generous next to football-data.org's 10/min, but still paced rather than
# fired in a tight loop — a sync touching ~20 matches x 5 leagues x 2 calls
# (lineups + stats) is ~200 requests in one run.
_MIN_SECONDS_BETWEEN_REQUESTS = 0.65
_last_request_at: float | None = None


def _headers() -> dict:
    if not settings.bigballs_api_key:
        raise RuntimeError("BIGBALLS_API_KEY is not set in .env — get a free key at https://bigballsdata.com/dashboard")
    return {"Authorization": f"Bearer {settings.bigballs_api_key}"}


def _paced_fetch_json(path: str, *, params: dict | None = None, max_age_hours: float) -> dict:
    global _last_request_at
    if _last_request_at is not None:
        elapsed = time.monotonic() - _last_request_at
        if elapsed < _MIN_SECONDS_BETWEEN_REQUESTS:
            time.sleep(_MIN_SECONDS_BETWEEN_REQUESTS - elapsed)
    _last_request_at = time.monotonic()

    text = fetch_text(
        f"{BIGBALLS_API_BASE}/{path.lstrip('/')}",
        subdir="bigballs",
        max_age_hours=max_age_hours,
        headers=_headers(),
        params=params,
    )
    return json.loads(text)


# bigballsdata's per-match stat dict is a flexible {key: {value, label}} bag,
# not a fixed schema (confirmed live) — pull only what a future availability
# feature would need into typed columns; everything else rides along in `extra`.
_MINUTES_KEYS = ("minutes",)
_GOALS_KEYS = ("goals",)
_ASSISTS_KEYS = ("assists",)
_RATING_KEYS = ("rating", "match_rating")


def _extract_stat(stats: dict, keys: tuple[str, ...]) -> str | None:
    for key in keys:
        entry = stats.get(key)
        if entry and entry.get("value") not in (None, ""):
            return entry["value"]
    return None


def _sync_match_lineups(session: Session, match: Match, bbs_match_id: str) -> bool:
    data = _paced_fetch_json(f"v1/stored/matches/{bbs_match_id}/lineups", max_age_hours=24 * 7)
    if not data.get("meta", {}).get("available"):
        return False

    formation = data.get("meta", {}).get("formation") or {}
    match.home_formation = formation.get("home")
    match.away_formation = formation.get("away")

    session.query(Lineup).filter_by(match_id=match.id).delete()
    for side, team_id in (("home", match.home_team_id), ("away", match.away_team_id)):
        # `order_index` preserves the source's own list order per side — the
        # only signal available for where a player sits tactically (e.g.
        # which of 4 defenders plays left back), since the API gives no
        # explicit slot. See app/main.py:_match_lineups for how it's used.
        for order_index, entry in enumerate(data["data"].get(side, [])):
            name = entry.get("name")
            if not name:
                continue
            session.add(
                Lineup(
                    match_id=match.id,
                    team_id=team_id,
                    player_name=name,
                    position=entry.get("position"),
                    jersey_number=entry.get("jersey_number"),
                    starter=bool(entry.get("starter")),
                    order_index=order_index,
                    bbs_player_id=entry.get("player_id"),
                    synced_at=dt.datetime.utcnow(),
                )
            )
    return True


def _sync_match_player_stats(
    session: Session, match: Match, bbs_match_id: str, home_bbs_name: str, away_bbs_name: str
) -> int:
    data = _paced_fetch_json(f"v1/stored/matches/{bbs_match_id}/stats", max_age_hours=24 * 7)
    players = (data.get("data") or {}).get("players") or []
    if not players:
        return 0

    session.query(PlayerMatchStat).filter_by(match_id=match.id).delete()

    # Players carry their own team_id/team_name from bigballsdata, not a
    # home/away flag — map by name against the two teams for THIS match. Keyed
    # on bigballsdata's OWN name strings (from the /matches listing), not our
    # Team.canonical_name: confirmed live that they can differ ("Coventry" in
    # our DB vs "Coventry City" in their /stats payload), which silently
    # dropped every one of that team's players when this keyed on our name
    # instead — not a bad-data problem, a join-key mismatch.
    #
    # Confirmed live: a single /stats response can also carry a handful of
    # players from unrelated fixtures mixed in (e.g. Lille, Chelsea players
    # appearing in a Man City vs Coventry response) — harmless here since
    # anyone whose team_name isn't one of these two exact strings is already
    # filtered out below, not because it's rare.
    team_by_name = {home_bbs_name: match.home_team_id, away_bbs_name: match.away_team_id}

    written = 0
    for player in players:
        name = player.get("name")
        team_id = team_by_name.get(player.get("team_name"))
        if not name or team_id is None:
            continue
        stats = player.get("stats") or {}
        session.add(
            PlayerMatchStat(
                match_id=match.id,
                team_id=team_id,
                player_name=name,
                bbs_player_id=player.get("id"),
                headshot_url=player.get("headshot_url"),
                minutes=_int_or_none(_extract_stat(stats, _MINUTES_KEYS)),
                goals=_int_or_none(_extract_stat(stats, _GOALS_KEYS)),
                assists=_int_or_none(_extract_stat(stats, _ASSISTS_KEYS)),
                rating=_float_or_none(_extract_stat(stats, _RATING_KEYS)),
                extra={k: v.get("value") for k, v in stats.items()},
                synced_at=dt.datetime.utcnow(),
            )
        )
        written += 1
    return written


def _int_or_none(value: str | None) -> int | None:
    try:
        return int(value) if value is not None else None
    except ValueError:
        return None


def _float_or_none(value: str | None) -> float | None:
    try:
        return float(value) if value is not None else None
    except ValueError:
        return None


def sync_competition(session: Session, competition_slug: str, *, limit: int = 20) -> dict[str, int]:
    """Sync lineups + player stats for a competition's most recent finished
    matches. Returns counts for the caller (scripts/sync_lineups.py) to
    print — never raises on a single unmatched/uncovered match, only on
    outright fetch failure for the whole competition."""
    league_key = LEAGUE_CODES.get(competition_slug)
    if league_key is None:
        return {"skipped": "not covered by bigballsdata.com"}
    if not settings.bigballs_api_key:
        return {"skipped": "BIGBALLS_API_KEY not set"}

    competition = session.query(Competition).filter_by(slug=competition_slug).one()
    started = dt.datetime.utcnow()

    data = _paced_fetch_json(
        "v1/matches",
        params={"sport": "football", "league": league_key, "status": "finished", "limit": limit},
        max_age_hours=6,
    )

    pool = build_alias_pool(session, exclude_source=SOURCE)
    matched = lineups_written = stats_written = unresolved = 0

    for row in data.get("data", []):
        home = resolve_existing_team(session, row["home"]["name"], SOURCE, context=f"bigballs {competition_slug}", pool=pool)
        away = resolve_existing_team(session, row["away"]["name"], SOURCE, context=f"bigballs {competition_slug}", pool=pool)
        if home is None or away is None:
            unresolved += 1
            continue

        try:
            kickoff = dt.datetime.fromisoformat(row["kickoff_utc"].replace("Z", "+00:00"))
            kickoff = kickoff.astimezone(dt.timezone.utc).replace(tzinfo=None)
        except (KeyError, ValueError):
            continue

        match = (
            session.query(Match)
            .filter_by(competition_id=competition.id, utc_kickoff=kickoff, home_team_id=home.id, away_team_id=away.id)
            .one_or_none()
        )
        if match is None:
            # Confirmed live: bigballsdata.com's Bundesliga kickoffs run ~2h off
            # true UTC (looks like a missed local->UTC conversion on their end;
            # EPL/La Liga/Serie A/Ligue 1 didn't show this). An exact-timestamp
            # match silently drops those rather than risk mismatching, so fall
            # back to same-day + same-teams — still effectively unique (two
            # fixtures between the same two clubs on the same calendar date
            # doesn't happen), and skip on the rare case of >1 hit rather than
            # guess.
            day_start = kickoff.replace(hour=0, minute=0, second=0, microsecond=0)
            day_end = day_start + dt.timedelta(days=1)
            candidates = (
                session.query(Match)
                .filter(
                    Match.competition_id == competition.id,
                    Match.home_team_id == home.id,
                    Match.away_team_id == away.id,
                    Match.utc_kickoff >= day_start,
                    Match.utc_kickoff < day_end,
                )
                .all()
            )
            match = candidates[0] if len(candidates) == 1 else None
        if match is None:
            unresolved += 1
            continue

        match.bbs_match_id = row["id"]
        matched += 1

        if _sync_match_lineups(session, match, row["id"]):
            lineups_written += 1
        wrote_stats = _sync_match_player_stats(session, match, row["id"], row["home"]["name"], row["away"]["name"])
        stats_written += 1 if wrote_stats else 0
        session.commit()

    session.add(
        IngestLog(
            source=SOURCE,
            competition_id=competition.id,
            started_at=started,
            finished_at=dt.datetime.utcnow(),
            status="ok",
            rows_ingested=matched,
            message=f"matched={matched}, lineups={lineups_written}, stats={stats_written}, unresolved={unresolved}",
        )
    )
    session.commit()
    return {"matched": matched, "lineups": lineups_written, "stats": stats_written, "unresolved": unresolved}


def sync_all(session: Session, *, limit: int = 20) -> dict[str, dict]:
    """Sync every competition LEAGUE_CODES covers. Entry point for
    scripts/refresh.py's daily loop and scripts/sync_lineups.py alike, so
    the two never drift on which competitions get synced."""
    return {slug: sync_competition(session, slug, limit=limit) for slug in LEAGUE_CODES}

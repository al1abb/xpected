"""Current-season squads and top scorers/assists via football-data.org's free
TIER_ONE plan — the fix for a real staleness bug: `PlayerStat`'s only feeder
until now was ingest/api_football.py, whose free tier is walled off from any
season after 2024 (confirmed via the API's own error message; see
scripts/sync_player_stats.py's `_OLDEST_ALLOWED_SEASON_FALLBACK`). Every
competition's top-scorer table was therefore stuck on 2024/25 data —
app/main.py already detects this and correctly blanks the panel rather than
show it, so this replaces the feed rather than fixing a visible bug.

Coverage matches `CREST_COMPETITION_CODES` in ingest/football_data_org.py —
Premier League, La Liga, Serie A, Bundesliga, Ligue 1, Eredivisie, Primeira
Liga, Champions League. Süper Lig and Azerbaijan Premyer Liqa get nothing
here (not on any football-data.org tier); app/main.py renders an explicit
empty state for those rather than a blank panel.

One request per competition (`/competitions/{code}/teams`) covers BOTH crew:
that endpoint embeds each team's full squad directly, so no per-team
follow-up call is needed. It's the exact same endpoint sync_team_crests
already calls — sharing the disk cache (ingest/cache.py keys purely on
URL+params, not caller), so running both in the same window costs one
request per league, not two.
"""

from __future__ import annotations

import datetime as dt
import json
import time

from sqlalchemy.orm import Session

from app.config import settings
from app.models import Competition, IngestLog, PlayerStat, SquadPlayer
from ingest.cache import cache_age_hours, fetch_text
from ingest.football_data_org import BASE, CREST_COMPETITION_CODES, SOURCE, resolve_team
from ingest.resolve import build_alias_pool
from ingest.seasons import current_season_start_year

# football-data.org's /scorers endpoint is sorted by goals — there's no
# separate current-season assists endpoint on the free tier, so the assists
# table here is "assists among the top scorers", not a true assists
# leaderboard (a high-assist, low-goal playmaker outside the goals top 50
# won't appear). Documented rather than silently passed off as complete.
_SCORERS_LIMIT = 50

# football-data.org's free tier caps at 10 requests/minute. sync_squads (8
# competitions) followed by sync_scorers (8 more) in the same refresh fires
# 16 requests in quick succession — confirmed live to actually 429 partway
# through (Serie A and Bundesliga scorers both failed this way before this
# pacing existed). Spacing real requests 6.5s apart keeps every run under
# the limit; a warm daily refresh pays nothing extra since a cache hit never
# sleeps.
_MIN_SECONDS_BETWEEN_REQUESTS = 6.5
_last_request_at: float | None = None


def _paced_fetch_text(url: str, *, subdir: str, max_age_hours: float, **kwargs) -> str:
    global _last_request_at
    age = cache_age_hours(url, subdir, params=kwargs.get("params"))
    real_request_coming = age is None or age >= max_age_hours
    if real_request_coming and _last_request_at is not None:
        elapsed = time.monotonic() - _last_request_at
        if elapsed < _MIN_SECONDS_BETWEEN_REQUESTS:
            time.sleep(_MIN_SECONDS_BETWEEN_REQUESTS - elapsed)
    if real_request_coming:
        _last_request_at = time.monotonic()
    return fetch_text(url, subdir=subdir, max_age_hours=max_age_hours, **kwargs)


def _current_season_label(as_of: dt.date | None = None) -> str:
    start_year = current_season_start_year(as_of)
    return f"{start_year}/{str(start_year + 1)[2:]}"


def _sync_squad(session: Session, competition: Competition, team_data: dict, pool: dict) -> int:
    team = resolve_team(
        session, team_data.get("name") or "", context=f"squad sync {competition.slug}", pool=pool
    )
    if team is None:
        return 0

    squad = team_data.get("squad") or []
    if not squad:
        # An empty squad here is the source not having published one yet for
        # THIS competition (confirmed live: the Champions League league phase
        # returns 36 teams with squad=[] before its season starts) — not "this
        # team now has no players". A team can appear under more than one
        # covered competition (e.g. a Premier League side also in the CL
        # teams list), so treating empty-here as truth would wipe out a
        # perfectly good squad written by an earlier competition in this same
        # sync pass. Skip entirely rather than delete.
        return 0

    # Full replace for this team, not upsert-by-id: departed players must
    # disappear, same reasoning ingest/api_football.py's scorer sync uses.
    session.query(SquadPlayer).filter_by(team_id=team.id).delete()

    written = 0
    for player in squad:
        name = player.get("name")
        if not name:
            continue
        dob = None
        if player.get("dateOfBirth"):
            try:
                dob = dt.date.fromisoformat(player["dateOfBirth"])
            except ValueError:
                dob = None
        session.add(
            SquadPlayer(
                team_id=team.id,
                name=name,
                position=player.get("position"),
                date_of_birth=dob,
                nationality=player.get("nationality"),
                fd_person_id=player.get("id"),
                synced_at=dt.datetime.utcnow(),
            )
        )
        written += 1
    return written


def sync_squads(session: Session) -> dict[str, int | None]:
    """Squad lists for every TIER_ONE-covered competition. Returns
    {slug: players_written} — None for a competition that failed to fetch,
    so the caller (scripts/refresh.py) can report it without stopping the
    others (same fail-soft pattern as every other ingest step there)."""
    if not settings.football_data_org_token:
        return {}

    pool = build_alias_pool(session, exclude_source=SOURCE)
    results: dict[str, int | None] = {}

    for slug, code in CREST_COMPETITION_CODES.items():
        competition = session.query(Competition).filter_by(slug=slug).one_or_none()
        if competition is None:
            continue
        started = dt.datetime.utcnow()

        url = f"{BASE}/competitions/{code}/teams"
        try:
            text = _paced_fetch_text(
                url,
                subdir="football_data_org",
                max_age_hours=24,
                headers={"X-Auth-Token": settings.football_data_org_token},
            )
        except RuntimeError as exc:
            session.add(
                IngestLog(
                    source=SOURCE,
                    competition_id=competition.id,
                    started_at=started,
                    finished_at=dt.datetime.utcnow(),
                    status="error",
                    message=f"squad sync: {exc}",
                )
            )
            session.commit()
            results[slug] = None
            continue

        data = json.loads(text)
        written = 0
        for team_data in data.get("teams", []):
            written += _sync_squad(session, competition, team_data, pool)

        session.add(
            IngestLog(
                source=SOURCE,
                competition_id=competition.id,
                started_at=started,
                finished_at=dt.datetime.utcnow(),
                status="ok",
                rows_ingested=written,
                message="squad sync",
            )
        )
        session.commit()
        results[slug] = written

    return results


def sync_scorers(session: Session) -> dict[str, int | None]:
    """Current-season goals + assists tables for every TIER_ONE-covered
    competition, replacing whatever ingest/api_football.py last wrote there
    (that source is stuck on 2024/25 — see this module's docstring)."""
    if not settings.football_data_org_token:
        return {}

    season_label = _current_season_label()
    pool = build_alias_pool(session, exclude_source=SOURCE)
    results: dict[str, int | None] = {}

    for slug, code in CREST_COMPETITION_CODES.items():
        competition = session.query(Competition).filter_by(slug=slug).one_or_none()
        if competition is None:
            continue
        started = dt.datetime.utcnow()

        url = f"{BASE}/competitions/{code}/scorers"
        try:
            text = _paced_fetch_text(
                url,
                subdir="football_data_org",
                max_age_hours=24,
                headers={"X-Auth-Token": settings.football_data_org_token},
                params={"limit": _SCORERS_LIMIT},
            )
        except RuntimeError as exc:
            session.add(
                IngestLog(
                    source=SOURCE,
                    competition_id=competition.id,
                    started_at=started,
                    finished_at=dt.datetime.utcnow(),
                    status="error",
                    message=f"scorers sync: {exc}",
                )
            )
            session.commit()
            results[slug] = None
            continue

        data = json.loads(text)
        scorers = data.get("scorers") or []

        # Full replace per category, not upsert-by-rank — same reasoning as
        # ingest/api_football.py's scorer sync: a re-sync should reflect the
        # current table exactly, including players who've dropped out of it.
        session.query(PlayerStat).filter_by(competition_id=competition.id, category="goals").delete()
        session.query(PlayerStat).filter_by(competition_id=competition.id, category="assists").delete()

        written = 0
        by_goals = [s for s in scorers if (s.get("goals") or 0) > 0]
        by_goals.sort(key=lambda s: s["goals"], reverse=True)
        for rank, entry in enumerate(by_goals, start=1):
            written += _write_player_stat(session, competition, pool, entry, "goals", rank, season_label)

        by_assists = [s for s in scorers if (s.get("assists") or 0) > 0]
        by_assists.sort(key=lambda s: s["assists"], reverse=True)
        for rank, entry in enumerate(by_assists, start=1):
            written += _write_player_stat(session, competition, pool, entry, "assists", rank, season_label)

        session.add(
            IngestLog(
                source=SOURCE,
                competition_id=competition.id,
                started_at=started,
                finished_at=dt.datetime.utcnow(),
                status="ok",
                rows_ingested=written,
                message=f"scorers sync ({season_label})",
            )
        )
        session.commit()
        results[slug] = written

    return results


def _write_player_stat(
    session: Session, competition: Competition, pool: dict, entry: dict, category: str, rank: int, season_label: str
) -> int:
    player = entry.get("player") or {}
    value = entry.get(category)
    if value is None:
        return 0
    team_data = entry.get("team") or {}
    team = None
    if team_data.get("name"):
        team = resolve_team(session, team_data["name"], context=f"{category} sync {competition.slug}", pool=pool)
    session.add(
        PlayerStat(
            competition_id=competition.id,
            season_label=season_label,
            category=category,
            rank=rank,
            player_name=player.get("name") or "?",
            team_id=team.id if team else None,
            value=value,
            af_player_id=player.get("id"),
            synced_at=dt.datetime.utcnow(),
        )
    )
    return 1

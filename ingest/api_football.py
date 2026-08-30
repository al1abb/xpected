"""API-Football (api-sports.io) ingest — used only for what football-data.co.uk
and ClubElo can't supply: UEFA Champions/Europa/Conference League and the
Azerbaijan Premyer Liqa.

The free tier is 100 requests/day, so every call goes through `_get`, which:
1. Serves a fresh disk cache with zero API cost.
2. Otherwise reserves one request against `ApiBudget` for today before calling
   out. If the daily cap is already spent, it serves a stale cache if one
   exists rather than making the call, and raises only if there's truly
   nothing to serve.

League ids for the big/UEFA competitions are well-documented and set directly
in app.config.COMPETITIONS. Azerbaijan Premyer Liqa's id is NOT hardcoded —
it's resolved once via /leagues?search=... and cached back onto the
Competition row, because guessing a numeric id with no way to verify it here
would be worse than looking it up.
"""

from __future__ import annotations

import datetime as dt
import json

from sqlalchemy.orm import Session

from app.config import API_FOOTBALL_BASE, settings
from app.models import ApiBudget, Competition, IngestLog, Match, PlayerStat
from ingest.cache import cache_age_hours, fetch_text, read_cached_text
from ingest.resolve import build_alias_pool, get_or_create_team, resolve_existing_team

SOURCE = "api_football"

# api-football fixture.status.short -> our Match.status
_FINISHED = {"FT", "AET", "PEN"}
_POSTPONED = {"PST"}
_CANCELLED = {"CANC", "ABD", "AWD", "WO"}
# everything else (NS, TBD, 1H, HT, 2H, ET, BT, P, SUSP, INT, LIVE) -> "scheduled";
# this app has no live-match view, so in-play is not a status worth modelling separately.


class BudgetExceeded(RuntimeError):
    pass


def _headers() -> dict:
    if not settings.api_football_key:
        raise RuntimeError(
            "API_FOOTBALL_KEY is not set in .env — get a free key at "
            "https://dashboard.api-football.com/register"
        )
    return {"x-apisports-key": settings.api_football_key}


def _reserve_budget(session: Session) -> tuple[bool, int, int]:
    today = dt.date.today()
    row = session.query(ApiBudget).filter_by(date=today).one_or_none()
    if row is None:
        row = ApiBudget(date=today, requests_used=0, cap=settings.api_football_daily_cap)
        session.add(row)
        session.flush()
    if row.requests_used >= row.cap:
        return False, row.requests_used, row.cap
    row.requests_used += 1
    session.commit()
    return True, row.requests_used, row.cap


def _get(session: Session, path: str, params: dict, *, max_age_hours: float = 24) -> list:
    url = f"{API_FOOTBALL_BASE}/{path.lstrip('/')}"
    fresh_age = cache_age_hours(url, subdir="api_football", params=params)

    if fresh_age is None or fresh_age >= max_age_hours:
        ok, used, cap = _reserve_budget(session)
        if not ok:
            stale = read_cached_text(url, subdir="api_football", params=params)
            if stale is not None:
                return json.loads(stale).get("response", [])
            raise BudgetExceeded(f"daily cap reached ({used}/{cap}); no cache for {path} {params}")

    text = fetch_text(url, subdir="api_football", max_age_hours=max_age_hours, headers=_headers(), params=params)
    data = json.loads(text)
    if data.get("errors"):
        raise RuntimeError(f"API-Football error for {path} {params}: {data['errors']}")
    return data.get("response", [])


def resolve_league_id(session: Session, competition: Competition) -> int | None:
    """Look up and persist a competition's api-football league id when it
    wasn't known ahead of time (currently: Azerbaijan Premyer Liqa)."""
    if competition.af_id is not None:
        return competition.af_id

    candidates = _get(session, "leagues", {"search": competition.af_name}, max_age_hours=24 * 30)
    if not candidates:
        session.add(
            IngestLog(
                source=SOURCE,
                competition_id=competition.id,
                started_at=dt.datetime.utcnow(),
                finished_at=dt.datetime.utcnow(),
                status="error",
                message=f"no /leagues result for search={competition.af_name!r}",
            )
        )
        session.commit()
        return None

    exact = [c for c in candidates if c["league"]["name"].lower() == competition.af_name.lower()]
    country_match = [c for c in candidates if c.get("country", {}).get("name") == competition.country]
    chosen = (exact or country_match or candidates)[0]

    competition.af_id = chosen["league"]["id"]
    session.add(
        IngestLog(
            source=SOURCE,
            competition_id=competition.id,
            started_at=dt.datetime.utcnow(),
            finished_at=dt.datetime.utcnow(),
            status="ok" if (exact or country_match) else "partial",
            message=(
                f"resolved af_id={competition.af_id} from {len(candidates)} candidate(s); "
                f"picked {chosen['league']['name']!r} ({chosen.get('country', {}).get('name')})"
            ),
        )
    )
    session.commit()
    return competition.af_id


def _status(short: str) -> str:
    if short in _FINISHED:
        return "finished"
    if short in _POSTPONED:
        return "postponed"
    if short in _CANCELLED:
        return "cancelled"
    return "scheduled"


def _parse_fixture(fixture: dict, competition: Competition) -> dict | None:
    kickoff_raw = fixture["fixture"]["date"]
    try:
        kickoff = dt.datetime.fromisoformat(kickoff_raw).astimezone(dt.timezone.utc).replace(tzinfo=None)
    except ValueError:
        return None

    round_name = fixture.get("league", {}).get("round", "") or ""
    neutral = (
        competition.type == "uefa_cup"
        and "final" in round_name.lower()
        and "semi" not in round_name.lower()
        and "quarter" not in round_name.lower()
    )

    return {
        "af_fixture_id": fixture["fixture"]["id"],
        "kickoff": kickoff,
        "status": _status(fixture["fixture"]["status"]["short"]),
        "round": round_name or None,
        "neutral_venue": neutral,
        "home_name": fixture["teams"]["home"]["name"],
        "away_name": fixture["teams"]["away"]["name"],
        "home_logo": fixture["teams"]["home"].get("logo"),
        "away_logo": fixture["teams"]["away"].get("logo"),
        "home_goals": fixture["goals"]["home"],
        "away_goals": fixture["goals"]["away"],
        "home_goals_ht": (fixture.get("score", {}).get("halftime") or {}).get("home"),
        "away_goals_ht": (fixture.get("score", {}).get("halftime") or {}).get("away"),
    }


def _upsert_fixture(session: Session, competition: Competition, row: dict) -> bool:
    existing = session.query(Match).filter_by(af_fixture_id=row["af_fixture_id"]).one_or_none()

    home_team = get_or_create_team(session, row["home_name"], SOURCE, context=f"competition={competition.slug}")
    away_team = get_or_create_team(session, row["away_name"], SOURCE, context=f"competition={competition.slug}")

    if home_team.logo_url is None and row.get("home_logo"):
        home_team.logo_url = row["home_logo"]
    if away_team.logo_url is None and row.get("away_logo"):
        away_team.logo_url = row["away_logo"]

    fields = dict(
        competition_id=competition.id,
        utc_kickoff=row["kickoff"],
        home_team_id=home_team.id,
        away_team_id=away_team.id,
        status=row["status"],
        round=row["round"],
        neutral_venue=row["neutral_venue"],
        home_goals=row["home_goals"],
        away_goals=row["away_goals"],
        home_goals_ht=row["home_goals_ht"],
        away_goals_ht=row["away_goals_ht"],
        source=SOURCE,
        af_fixture_id=row["af_fixture_id"],
    )

    if existing is None:
        # A prior CSV/other-source row might already represent this exact fixture
        # (same competition/kickoff/teams) — natural-key match avoids a duplicate.
        existing = (
            session.query(Match)
            .filter_by(
                competition_id=competition.id,
                utc_kickoff=row["kickoff"],
                home_team_id=home_team.id,
                away_team_id=away_team.id,
            )
            .one_or_none()
        )

    if existing is None:
        session.add(Match(**fields))
        return True

    if existing.status == "finished" and row["status"] != "finished":
        return False
    for key, value in fields.items():
        if value is not None:
            setattr(existing, key, value)
    return False


def ingest_competition_season(session: Session, competition_slug: str, af_season: int) -> int:
    competition = session.query(Competition).filter_by(slug=competition_slug).one()
    started = dt.datetime.utcnow()

    league_id = resolve_league_id(session, competition)
    if league_id is None:
        return 0

    try:
        fixtures = _get(session, "fixtures", {"league": league_id, "season": af_season}, max_age_hours=12)
    except (BudgetExceeded, RuntimeError) as exc:
        session.add(
            IngestLog(
                source=SOURCE,
                competition_id=competition.id,
                started_at=started,
                finished_at=dt.datetime.utcnow(),
                status="error",
                message=str(exc),
            )
        )
        session.commit()
        return 0

    new_count = 0
    for fixture in fixtures:
        row = _parse_fixture(fixture, competition)
        if row is None:
            continue
        if _upsert_fixture(session, competition, row):
            new_count += 1

    session.add(
        IngestLog(
            source=SOURCE,
            competition_id=competition.id,
            started_at=started,
            finished_at=dt.datetime.utcnow(),
            status="ok",
            rows_ingested=new_count,
            message=f"season={af_season}, fixtures_seen={len(fixtures)}",
        )
    )
    session.commit()
    return new_count


def sync_league_team_crests(session: Session, competition_slug: str, af_season: int) -> int:
    """One-off crest backfill for a competition whose match data already
    comes free from football-data.co.uk (so its fixtures are never pulled
    through here, and `_upsert_fixture`'s logo-backfill-on-any-upsert never
    gets a chance to run) but whose teams still have no crest source: no
    football-data.org coverage (Turkish Süper Lig isn't in
    ingest/football_data_org.py's CREST_COMPETITION_CODES), and
    football-data.co.uk's CSVs never carry logos at all.

    API-Football's `/teams` endpoint returns a whole league's crests in one
    request, so this is a single call against the daily budget rather than an
    ongoing cost — worth it for one league; not worth wiring into every
    refresh for every competition already covered elsewhere.

    Resolve-only (see ingest/clubelo.py for why): this only attaches a crest
    to a team we already track from the free source, it never invents one.
    """
    competition = session.query(Competition).filter_by(slug=competition_slug).one()
    started = dt.datetime.utcnow()

    league_id = resolve_league_id(session, competition)
    if league_id is None:
        return 0

    try:
        teams_response = _get(session, "teams", {"league": league_id, "season": af_season}, max_age_hours=24 * 30)
    except (BudgetExceeded, RuntimeError) as exc:
        session.add(
            IngestLog(
                source=SOURCE,
                competition_id=competition.id,
                started_at=started,
                finished_at=dt.datetime.utcnow(),
                status="error",
                message=f"crest sync: {exc}",
            )
        )
        session.commit()
        return 0

    pool = build_alias_pool(session, exclude_source=SOURCE)
    updated = 0
    for entry in teams_response:
        team_data = entry.get("team") or {}
        name, crest = team_data.get("name"), team_data.get("logo")
        if not name or not crest:
            continue
        team = resolve_existing_team(session, name, SOURCE, context=f"crest sync {competition_slug}", pool=pool)
        if team is not None and team.logo_url is None:
            team.logo_url = crest
            updated += 1

    session.add(
        IngestLog(
            source=SOURCE,
            competition_id=competition.id,
            started_at=started,
            finished_at=dt.datetime.utcnow(),
            status="ok",
            rows_ingested=updated,
            message=f"crest sync season={af_season}",
        )
    )
    session.commit()
    return updated


def _sync_player_stat(
    session: Session, competition_slug: str, af_season: int, category: str, endpoint: str
) -> int | None:
    """Shared implementation for sync_top_scorers/sync_top_assists. Returns
    None (rather than raising) when the season is blocked/errors out, so the
    caller (scripts/sync_player_stats.py) can retry an earlier season the
    same way the Süper Lig crest sync falls back to 2024 — one call per
    league per category, cheap enough not to need caching finesse beyond the
    normal week-long disk cache.
    """
    competition = session.query(Competition).filter_by(slug=competition_slug).one()
    started = dt.datetime.utcnow()

    league_id = resolve_league_id(session, competition)
    if league_id is None:
        return None

    try:
        response = _get(session, endpoint, {"league": league_id, "season": af_season}, max_age_hours=24 * 7)
    except (BudgetExceeded, RuntimeError) as exc:
        session.add(
            IngestLog(
                source=SOURCE,
                competition_id=competition.id,
                started_at=started,
                finished_at=dt.datetime.utcnow(),
                status="error",
                message=f"{category} sync season={af_season}: {exc}",
            )
        )
        session.commit()
        return None

    pool = build_alias_pool(session, exclude_source=SOURCE)
    season_label = f"{af_season}/{str(af_season + 1)[2:]}"

    # Full replace, not upsert-by-rank: a re-sync should reflect the current
    # top 10 exactly, including players who've dropped out of it.
    session.query(PlayerStat).filter_by(competition_id=competition.id, category=category).delete()

    written = 0
    for entry in response[:10]:
        player = entry.get("player") or {}
        stats_list = entry.get("statistics") or []
        if not stats_list:
            continue
        stat = stats_list[0]
        goals = stat.get("goals") or {}
        value = goals.get("total") if category == "goals" else goals.get("assists")
        if value is None:
            continue
        team_data = stat.get("team") or {}
        team = None
        if team_data.get("name"):
            team = resolve_existing_team(
                session, team_data["name"], SOURCE, context=f"{category} sync {competition_slug}", pool=pool
            )
        written += 1
        session.add(
            PlayerStat(
                competition_id=competition.id,
                season_label=season_label,
                category=category,
                rank=written,
                player_name=player.get("name") or "?",
                team_id=team.id if team else None,
                value=value,
                af_player_id=player.get("id"),
                synced_at=dt.datetime.utcnow(),
            )
        )

    session.add(
        IngestLog(
            source=SOURCE,
            competition_id=competition.id,
            started_at=started,
            finished_at=dt.datetime.utcnow(),
            status="ok",
            rows_ingested=written,
            message=f"{category} sync season={af_season} ({season_label})",
        )
    )
    session.commit()
    return written


def sync_top_scorers(session: Session, competition_slug: str, af_season: int) -> int | None:
    return _sync_player_stat(session, competition_slug, af_season, "goals", "players/topscorers")


def sync_top_assists(session: Session, competition_slug: str, af_season: int) -> int | None:
    return _sync_player_stat(session, competition_slug, af_season, "assists", "players/topassists")

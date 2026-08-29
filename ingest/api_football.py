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
from app.models import ApiBudget, Competition, IngestLog, Match
from ingest.cache import cache_age_hours, fetch_text, read_cached_text
from ingest.resolve import get_or_create_team

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

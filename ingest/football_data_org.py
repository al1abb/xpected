"""football-data.org ingest — free TIER_ONE plan covers UEFA Champions League
with current-season access (unlike API-Football's free tier, which walls off
the current season for this exact competition). Used as a second, independent
source for Champions League fixtures alongside fixturedownload.com: if either
one's format changes or goes down, the other keeps the data flowing.

Confirmed live (2026-08-29) via GET /v4/competitions with no auth: 189
competitions total, with per-competition `plan` tiers — CL is TIER_ONE (free),
Europa League is TIER_TWO, Conference League is TIER_FOUR, and Azerbaijan
Premyer Liqa isn't listed under any tier. Only Champions League is usable here.
"""

from __future__ import annotations

import datetime as dt
import json

from sqlalchemy.orm import Session

from app.config import settings
from app.models import Competition, IngestLog, Match
from ingest.cache import fetch_text
from ingest.resolve import get_or_create_team

SOURCE = "football_data_org"
BASE = "https://api.football-data.org/v4"

# football-data.org status -> our Match.status
_FINISHED = {"FINISHED"}
_POSTPONED = {"POSTPONED"}
_CANCELLED = {"CANCELLED", "SUSPENDED"}
# SCHEDULED/TIMED/IN_PLAY/PAUSED -> "scheduled" (no live view in v1)


def _status(raw: str) -> str:
    if raw in _FINISHED:
        return "finished"
    if raw in _POSTPONED:
        return "postponed"
    if raw in _CANCELLED:
        return "cancelled"
    return "scheduled"


def _parse_match(match: dict) -> dict | None:
    try:
        kickoff = dt.datetime.fromisoformat(match["utcDate"].replace("Z", "+00:00"))
        kickoff = kickoff.astimezone(dt.timezone.utc).replace(tzinfo=None)
    except (KeyError, ValueError):
        return None

    full_time = (match.get("score") or {}).get("fullTime") or {}
    return {
        "kickoff": kickoff,
        "status": _status(match.get("status", "")),
        "round": match.get("stage") or None,
        "home_name": match["homeTeam"]["name"],
        "away_name": match["awayTeam"]["name"],
        "home_goals": full_time.get("home"),
        "away_goals": full_time.get("away"),
    }


def _upsert(session: Session, competition: Competition, row: dict) -> bool:
    home_team = get_or_create_team(session, row["home_name"], SOURCE, context=f"competition={competition.slug}")
    away_team = get_or_create_team(session, row["away_name"], SOURCE, context=f"competition={competition.slug}")

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

    fields = dict(
        competition_id=competition.id,
        utc_kickoff=row["kickoff"],
        home_team_id=home_team.id,
        away_team_id=away_team.id,
        status=row["status"],
        round=row["round"],
        home_goals=row["home_goals"],
        away_goals=row["away_goals"],
        source=SOURCE,
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


def ingest_champions_league(session: Session) -> int:
    competition = session.query(Competition).filter_by(slug="champions-league").one()
    started = dt.datetime.utcnow()

    if not settings.football_data_org_token:
        session.add(
            IngestLog(
                source=SOURCE,
                competition_id=competition.id,
                started_at=started,
                finished_at=dt.datetime.utcnow(),
                status="error",
                message="FOOTBALL_DATA_ORG_TOKEN not set in .env",
            )
        )
        session.commit()
        return 0

    url = f"{BASE}/competitions/CL/matches"
    try:
        text = fetch_text(
            url, subdir="football_data_org", max_age_hours=12, headers={"X-Auth-Token": settings.football_data_org_token}
        )
    except RuntimeError as exc:
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

    data = json.loads(text)
    matches = data.get("matches", [])
    new_count = 0
    for match in matches:
        row = _parse_match(match)
        if row is None:
            continue
        if _upsert(session, competition, row):
            new_count += 1

    session.add(
        IngestLog(
            source=SOURCE,
            competition_id=competition.id,
            started_at=started,
            finished_at=dt.datetime.utcnow(),
            status="ok",
            rows_ingested=new_count,
            message=f"matches_seen={len(matches)}",
        )
    )
    session.commit()
    return new_count


# football-data.org competition code for every domestic league it covers at
# TIER_ONE (free), plus Champions League — used only to pull team crests, not
# match data (Süper Lig and Azerbaijan Premyer Liqa aren't covered here at all).
CREST_COMPETITION_CODES = {
    "premier-league": "PL",
    "la-liga": "PD",
    "serie-a": "SA",
    "bundesliga": "BL1",
    "ligue-1": "FL1",
    "eredivisie": "DED",
    "primeira-liga": "PPL",
    "champions-league": "CL",
}


def sync_team_crests(session: Session) -> int:
    """One-off/occasional sync — crests essentially never change, so this
    doesn't need to run on every refresh. Never overwrites a logo_url that's
    already set (e.g. from API-Football), only fills gaps."""
    started = dt.datetime.utcnow()
    if not settings.football_data_org_token:
        return 0

    updated = 0
    for slug, code in CREST_COMPETITION_CODES.items():
        competition = session.query(Competition).filter_by(slug=slug).one_or_none()
        if competition is None:
            continue

        url = f"{BASE}/competitions/{code}/teams"
        try:
            text = fetch_text(
                url, subdir="football_data_org", max_age_hours=24 * 30, headers={"X-Auth-Token": settings.football_data_org_token}
            )
        except RuntimeError as exc:
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
            continue

        data = json.loads(text)
        for team_data in data.get("teams", []):
            crest = team_data.get("crest")
            name = team_data.get("name")
            if not crest or not name:
                continue
            team = get_or_create_team(session, name, SOURCE, context=f"crest sync {slug}")
            if team.logo_url is None:
                team.logo_url = crest
                updated += 1

    session.add(
        IngestLog(
            source=SOURCE,
            started_at=started,
            finished_at=dt.datetime.utcnow(),
            status="ok",
            rows_ingested=updated,
            message="team crest sync",
        )
    )
    session.commit()
    return updated

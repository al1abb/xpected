"""fixturedownload.com ingest: free, no-key, structured JSON fixture feeds —
originally added for the UEFA club competitions (closing the "show upcoming
UEFA fixtures" gap that API-Football's free tier blocks for those), then
extended to the 8 domestic leagues too, because football-data.co.uk's
fixtures.csv (ingest/footballdata_csv.py) turned out to be a rolling ~4-day
window, not a season-long feed — confirmed live by inspecting its cached
file, which had exactly 4 distinct dates. fixturedownload.com's whole-season
JSON, by contrast, already covers the full 2026/27 season for every domestic
league mapped below (verified live for all 8 — team names spot-checked per
league, e.g. Galatasaray/Fenerbahçe/Beşiktaş confirm `super-lig` is genuinely
the Turkish league, not a coincidental slug hit).

Verified directly: /feed/json/champions-league-2026 returns real 2026/27
fixtures including unplayed ones (null scores). Europa League and Conference
League's 2026/27 pages don't exist yet as of this writing — same season-start
lag seen in openfootball's repo — so a 404 here is treated as "not published
yet," not an error, and ingest simply retries next refresh. No coverage for
the Azerbaijan Premyer Liqa exists on this site at all (its one Azerbaijan
entry is the national team's Nations League fixtures, a different thing
entirely), and API-Football's free tier blocks its 2026/27 season too
(confirmed live: "Free plans do not have access to this season, try from
2022 to 2024") — that one gap remains genuinely unsolved after checking
every available source; see future-plans.md rather than a silent claim of
being fixed.
"""

from __future__ import annotations

import datetime as dt
import json

from sqlalchemy.orm import Session

from app.models import Competition, IngestLog, Match
from ingest.cache import fetch_text
from ingest.resolve import get_or_create_team
from ingest.seasons import current_season_start_year

SOURCE = "fixturedownload"

# our competition slug -> fixturedownload.com's slug (only competitions covered there).
# Azerbaijan Premyer Liqa deliberately absent — see module docstring.
FD_SLUGS = {
    "champions-league": "champions-league",
    "europa-league": "europa-league",
    "conference-league": "conference-league",
    "premier-league": "epl",
    "la-liga": "la-liga",
    "serie-a": "serie-a",
    "bundesliga": "bundesliga",
    "ligue-1": "ligue-1",
    "eredivisie": "eredivisie",
    "primeira-liga": "primeira-liga",
    "super-lig": "super-lig",
}


def _parse_kickoff(date_utc: str) -> dt.datetime | None:
    try:
        return dt.datetime.strptime(date_utc, "%Y-%m-%d %H:%M:%SZ")
    except ValueError:
        return None


def parse_feed(text: str) -> list[dict]:
    raw = json.loads(text)
    rows = []
    for item in raw:
        kickoff = _parse_kickoff(item.get("DateUtc", ""))
        if kickoff is None or not item.get("HomeTeam") or not item.get("AwayTeam"):
            continue
        home_goals, away_goals = item.get("HomeTeamScore"), item.get("AwayTeamScore")
        rows.append(
            {
                "kickoff": kickoff,
                "home_name": item["HomeTeam"].strip(),
                "away_name": item["AwayTeam"].strip(),
                "status": "finished" if home_goals is not None and away_goals is not None else "scheduled",
                "home_goals": home_goals,
                "away_goals": away_goals,
                "round": item.get("Group") or (f"Round {item['RoundNumber']}" if item.get("RoundNumber") else None),
            }
        )
    return rows


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


def ingest_competition(session: Session, competition_slug: str, start_year: int | None = None) -> int:
    fd_slug = FD_SLUGS.get(competition_slug)
    if fd_slug is None:
        return 0

    competition = session.query(Competition).filter_by(slug=competition_slug).one()
    start_year = start_year or current_season_start_year()
    started = dt.datetime.utcnow()
    url = f"https://fixturedownload.com/feed/json/{fd_slug}-{start_year}"

    try:
        text = fetch_text(url, subdir="fixturedownload", max_age_hours=12)
    except RuntimeError as exc:
        not_yet_published = "404" in str(exc)
        session.add(
            IngestLog(
                source=SOURCE,
                competition_id=competition.id,
                started_at=started,
                finished_at=dt.datetime.utcnow(),
                status="partial" if not_yet_published else "error",
                message=(
                    f"season {start_year} not yet published at {fd_slug}" if not_yet_published else str(exc)
                ),
            )
        )
        session.commit()
        return 0

    rows = parse_feed(text)
    new_count = sum(1 for row in rows if _upsert(session, competition, row))

    session.add(
        IngestLog(
            source=SOURCE,
            competition_id=competition.id,
            started_at=started,
            finished_at=dt.datetime.utcnow(),
            status="ok",
            rows_ingested=new_count,
            message=f"season={start_year}, fixtures_seen={len(rows)}",
        )
    )
    session.commit()
    return new_count

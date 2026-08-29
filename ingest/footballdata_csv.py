"""football-data.co.uk ingest: results (with shots/cards/closing odds) for the
8 main leagues we cover there, plus the combined upcoming-fixtures file.

Zero API cost — this is the bulk of the training data. See ingest/cache.py for
why every fetch goes through a disk cache with backoff.
"""

from __future__ import annotations

import csv
import datetime as dt
import io

from sqlalchemy.orm import Session

from app.config import COMPETITIONS, FOOTBALL_DATA_CSV_BASE
from app.models import Competition, IngestLog, Match, OddsSnapshot
from ingest.cache import fetch_text
from ingest.resolve import get_or_create_team
from ingest.seasons import fd_season_codes_back

SOURCE = "football_data"

MAIN_LEAGUE_COMPETITIONS = [c for c in COMPETITIONS if c["fd_code"]]

REQUIRED_RESULTS_COLUMNS = {"Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"}
REQUIRED_FIXTURES_COLUMNS = {"Div", "Date", "HomeTeam", "AwayTeam"}


class SchemaDriftError(RuntimeError):
    """Raised when a source's column set no longer matches what we parse for —
    a signal the site changed format, not that this week has no matches."""


def _check_columns(fieldnames: list[str] | None, required: set[str], context: str) -> None:
    present = set(fieldnames or [])
    missing = required - present
    if missing:
        raise SchemaDriftError(f"{context}: missing expected columns {sorted(missing)} — source format may have changed")


def _parse_date(date_str: str, time_str: str) -> dt.datetime | None:
    date_str = (date_str or "").strip()
    if not date_str:
        return None
    for fmt in ("%d/%m/%Y", "%d/%m/%y"):
        try:
            d = dt.datetime.strptime(date_str, fmt).date()
            break
        except ValueError:
            continue
    else:
        return None

    time_str = (time_str or "").strip() or "15:00"
    try:
        t = dt.datetime.strptime(time_str, "%H:%M").time()
    except ValueError:
        t = dt.time(15, 0)

    # football-data.co.uk times are UK local (not UTC); treated as UTC here since
    # day-level resolution is what the model actually depends on.
    return dt.datetime.combine(d, t)


def _int_or_none(value: str) -> int | None:
    value = (value or "").strip()
    if not value:
        return None
    try:
        return int(float(value))
    except ValueError:
        return None


def _float_or_none(value: str) -> float | None:
    value = (value or "").strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def parse_results_csv(text: str) -> list[dict]:
    reader = csv.DictReader(io.StringIO(text))
    _check_columns(reader.fieldnames, REQUIRED_RESULTS_COLUMNS, "football-data.co.uk results CSV")
    rows = []
    for row in reader:
        if not row.get("HomeTeam") or not row.get("AwayTeam"):
            continue
        kickoff = _parse_date(row.get("Date", ""), row.get("Time", ""))
        if kickoff is None:
            continue
        home_goals = _int_or_none(row.get("FTHG", ""))
        away_goals = _int_or_none(row.get("FTAG", ""))
        # Closing average odds (AvgC*) when present, else pre-match average (Avg*).
        odds_h = _float_or_none(row.get("AvgCH", "")) or _float_or_none(row.get("AvgH", ""))
        odds_d = _float_or_none(row.get("AvgCD", "")) or _float_or_none(row.get("AvgD", ""))
        odds_a = _float_or_none(row.get("AvgCA", "")) or _float_or_none(row.get("AvgA", ""))
        rows.append(
            {
                "kickoff": kickoff,
                "home_name": row["HomeTeam"].strip(),
                "away_name": row["AwayTeam"].strip(),
                "status": "finished" if home_goals is not None and away_goals is not None else "scheduled",
                "home_goals": home_goals,
                "away_goals": away_goals,
                "home_goals_ht": _int_or_none(row.get("HTHG", "")),
                "away_goals_ht": _int_or_none(row.get("HTAG", "")),
                "home_shots": _int_or_none(row.get("HS", "")),
                "away_shots": _int_or_none(row.get("AS", "")),
                "home_shots_on_target": _int_or_none(row.get("HST", "")),
                "away_shots_on_target": _int_or_none(row.get("AST", "")),
                "home_corners": _int_or_none(row.get("HC", "")),
                "away_corners": _int_or_none(row.get("AC", "")),
                "home_yellow": _int_or_none(row.get("HY", "")),
                "away_yellow": _int_or_none(row.get("AY", "")),
                "home_red": _int_or_none(row.get("HR", "")),
                "away_red": _int_or_none(row.get("AR", "")),
                "odds_home": odds_h,
                "odds_draw": odds_d,
                "odds_away": odds_a,
            }
        )
    return rows


def parse_fixtures_csv(text: str, fd_codes: set[str]) -> list[dict]:
    reader = csv.DictReader(io.StringIO(text))
    _check_columns(reader.fieldnames, REQUIRED_FIXTURES_COLUMNS, "football-data.co.uk fixtures.csv")
    rows = []
    for row in reader:
        div = (row.get("Div") or "").strip()
        if div not in fd_codes:
            continue
        if not row.get("HomeTeam") or not row.get("AwayTeam"):
            continue
        kickoff = _parse_date(row.get("Date", ""), row.get("Time", ""))
        if kickoff is None:
            continue
        rows.append(
            {
                "fd_code": div,
                "kickoff": kickoff,
                "home_name": row["HomeTeam"].strip(),
                "away_name": row["AwayTeam"].strip(),
                "status": "scheduled",
            }
        )
    return rows


def _upsert_match(session: Session, competition_id: int, row: dict) -> bool:
    home_team = get_or_create_team(
        session, row["home_name"], SOURCE, context=f"competition_id={competition_id}"
    )
    away_team = get_or_create_team(
        session, row["away_name"], SOURCE, context=f"competition_id={competition_id}"
    )

    existing = (
        session.query(Match)
        .filter_by(
            competition_id=competition_id,
            utc_kickoff=row["kickoff"],
            home_team_id=home_team.id,
            away_team_id=away_team.id,
        )
        .one_or_none()
    )

    fields = dict(
        competition_id=competition_id,
        utc_kickoff=row["kickoff"],
        home_team_id=home_team.id,
        away_team_id=away_team.id,
        status=row["status"],
        home_goals=row.get("home_goals"),
        away_goals=row.get("away_goals"),
        home_goals_ht=row.get("home_goals_ht"),
        away_goals_ht=row.get("away_goals_ht"),
        home_shots=row.get("home_shots"),
        away_shots=row.get("away_shots"),
        home_shots_on_target=row.get("home_shots_on_target"),
        away_shots_on_target=row.get("away_shots_on_target"),
        home_corners=row.get("home_corners"),
        away_corners=row.get("away_corners"),
        home_yellow=row.get("home_yellow"),
        away_yellow=row.get("away_yellow"),
        home_red=row.get("home_red"),
        away_red=row.get("away_red"),
        source=SOURCE,
    )

    is_new = False
    if existing is None:
        match = Match(**fields)
        session.add(match)
        session.flush()
        is_new = True
    else:
        match = existing
        # Never let a scheduled-fixture re-fetch stomp a result that's already finished.
        if not (match.status == "finished" and row["status"] != "finished"):
            for key, value in fields.items():
                if value is not None:
                    setattr(match, key, value)

    odds_home, odds_draw, odds_away = row.get("odds_home"), row.get("odds_draw"), row.get("odds_away")
    if odds_home and odds_draw and odds_away:
        existing_odds = (
            session.query(OddsSnapshot)
            .filter_by(match_id=match.id, source=SOURCE, bookmaker="closing_avg")
            .one_or_none()
        )
        if existing_odds is None:
            session.add(
                OddsSnapshot(
                    match_id=match.id,
                    bookmaker="closing_avg",
                    home_odds=odds_home,
                    draw_odds=odds_draw,
                    away_odds=odds_away,
                    source=SOURCE,
                )
            )
        else:
            existing_odds.home_odds, existing_odds.draw_odds, existing_odds.away_odds = (
                odds_home,
                odds_draw,
                odds_away,
            )

    return is_new


def ingest_results(session: Session, seasons_back: int = 4) -> int:
    started = dt.datetime.utcnow()
    total_new = 0
    codes = fd_season_codes_back(seasons_back)
    for comp in MAIN_LEAGUE_COMPETITIONS:
        competition = session.query(Competition).filter_by(slug=comp["slug"]).one()
        for season_code in codes:
            url = f"{FOOTBALL_DATA_CSV_BASE}/mmz4281/{season_code}/{comp['fd_code']}.csv"
            try:
                text = fetch_text(url, subdir="footballdata")
                rows = parse_results_csv(text)
            except (RuntimeError, SchemaDriftError) as exc:
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
                continue
            for row in rows:
                if _upsert_match(session, competition.id, row):
                    total_new += 1
            session.flush()
    session.add(
        IngestLog(
            source=SOURCE,
            competition_id=None,
            started_at=started,
            finished_at=dt.datetime.utcnow(),
            status="ok",
            rows_ingested=total_new,
            message=f"seasons={codes}",
        )
    )
    session.commit()
    return total_new


def ingest_fixtures(session: Session) -> int:
    started = dt.datetime.utcnow()
    fd_code_to_competition = {
        comp["fd_code"]: session.query(Competition).filter_by(slug=comp["slug"]).one()
        for comp in MAIN_LEAGUE_COMPETITIONS
    }
    url = f"{FOOTBALL_DATA_CSV_BASE}/fixtures.csv"
    try:
        text = fetch_text(url, subdir="footballdata", max_age_hours=6)
        rows = parse_fixtures_csv(text, set(fd_code_to_competition))
    except (RuntimeError, SchemaDriftError) as exc:
        session.add(
            IngestLog(
                source=SOURCE,
                started_at=started,
                finished_at=dt.datetime.utcnow(),
                status="error",
                message=str(exc),
            )
        )
        session.commit()
        return 0
    total_new = 0
    for row in rows:
        competition = fd_code_to_competition[row["fd_code"]]
        if _upsert_match(session, competition.id, row):
            total_new += 1
    session.add(
        IngestLog(
            source=SOURCE,
            started_at=started,
            finished_at=dt.datetime.utcnow(),
            status="ok",
            rows_ingested=total_new,
        )
    )
    session.commit()
    return total_new

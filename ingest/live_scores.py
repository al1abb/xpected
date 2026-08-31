"""Real live scores for the subset of competitions football-data.org's free
TIER_ONE plan covers: Premier League, La Liga, Serie A, Bundesliga, Ligue 1,
Eredivisie, Primeira Liga, and Champions League — confirmed live via
GET /v4/competitions (see ingest/football_data_org.py's docstring for the
same check). Europa League (TIER_TWO), Conference League (TIER_FOUR), Süper
Lig and Azerbaijan Premyer Liqa (not listed at any tier) aren't covered —
those competitions keep the wall-clock "Live" estimate in app/main.py with
no real score.

Unlike ingest/football_data_org.py (which only ever ingests Champions League
match rows), fetch_live_matches only *reads* the live-matches endpoint — it
never creates or updates a `Match` row, so it's safe to call from a
request-serving route with no ingest side effects. fetch_recently_finished
below is read-only in the same sense — the actual DB write lives in
scripts/close_out_finished_matches.py, which is the only place either
function's output gets persisted.

GET /v4/matches?status=LIVE returns every live match across every
competition in a single call, so polling this endpoint costs one request
regardless of how many matches are live at once — a very different budget
profile than API-Football's free tier (see app/main.py::_live_state's
docstring for why that source was ruled out for this).
"""

from __future__ import annotations

import datetime as dt
import json

from app.config import settings
from ingest.cache import fetch_text

BASE = "https://api.football-data.org/v4"

# football-data.org competition `code` -> our Competition.slug — only the
# TIER_ONE (free) competitions we track are listed here.
FD_ORG_COMPETITION_CODES: dict[str, str] = {
    "PL": "premier-league",
    "PD": "la-liga",
    "SA": "serie-a",
    "BL1": "bundesliga",
    "FL1": "ligue-1",
    "DED": "eredivisie",
    "PPL": "primeira-liga",
    "CL": "champions-league",
}

# Cache the live-matches response briefly so several concurrent page
# requests (or one page polling frequently) coalesce into one upstream call
# per window rather than one each — well within the free tier's 10 req/min,
# but no reason to spend budget you don't need to.
_CACHE_SECONDS = 15


def fetch_live_matches() -> list[dict]:
    """Every currently in-play match in our tracked TIER_ONE competitions.

    Returns [] on any failure (missing token, network error, rate limit) —
    a live score is a nice-to-have overlay on top of the estimate badge that
    already exists, never something worth breaking a page render over."""
    if not settings.football_data_org_token:
        return []

    try:
        text = fetch_text(
            f"{BASE}/matches",
            subdir="live_scores",
            max_age_hours=_CACHE_SECONDS / 3600,
            params={"status": "LIVE"},
            headers={"X-Auth-Token": settings.football_data_org_token},
        )
        raw = json.loads(text)
    except Exception:
        return []

    rows = []
    for match in raw.get("matches", []):
        slug = FD_ORG_COMPETITION_CODES.get((match.get("competition") or {}).get("code"))
        if slug is None:
            continue
        home_team = match.get("homeTeam") or {}
        away_team = match.get("awayTeam") or {}
        home_name, away_name = home_team.get("name"), away_team.get("name")
        if not home_name or not away_name:
            continue
        full_time = (match.get("score") or {}).get("fullTime") or {}
        rows.append(
            {
                "competition_slug": slug,
                "home_name": home_name,
                "away_name": away_name,
                "home_tla": home_team.get("tla"),
                "away_tla": away_team.get("tla"),
                "home_goals": full_time.get("home"),
                "away_goals": full_time.get("away"),
                "minute": match.get("minute"),
                "status": match.get("status"),  # "IN_PLAY" | "PAUSED"
            }
        )
    return rows


def fetch_recently_finished(*, lookback_days: int = 3) -> list[dict]:
    """Every match football-data.org already confirms FINISHED in the last
    `lookback_days`, for the same TIER_ONE competitions as fetch_live_matches.

    Backs scripts/close_out_finished_matches.py: football-data.co.uk and
    fixturedownload.com (our primary result sources — see
    ingest/footballdata_csv.py, ingest/fixturedownload.py) can lag by up to a
    day before posting a confirmed final score, even though football-data.org
    typically has it within minutes of full time (it's the same source that
    powers the live-score overlay). This lets a frequent, lightweight job
    close out a match immediately instead of waiting for the next full daily
    refresh — without needing to track live->finished transitions across
    runs itself, since re-querying a few days back is cheap and stateless.

    Returns [] on any failure — same fail-soft contract as fetch_live_matches.
    """
    if not settings.football_data_org_token:
        return []

    today = dt.date.today()
    date_from = today - dt.timedelta(days=lookback_days)

    try:
        text = fetch_text(
            f"{BASE}/matches",
            subdir="live_scores",
            max_age_hours=_CACHE_SECONDS / 3600,
            params={"status": "FINISHED", "dateFrom": date_from.isoformat(), "dateTo": today.isoformat()},
            headers={"X-Auth-Token": settings.football_data_org_token},
        )
        raw = json.loads(text)
    except Exception:
        return []

    rows = []
    for match in raw.get("matches", []):
        slug = FD_ORG_COMPETITION_CODES.get((match.get("competition") or {}).get("code"))
        if slug is None:
            continue
        home_name = (match.get("homeTeam") or {}).get("name")
        away_name = (match.get("awayTeam") or {}).get("name")
        if not home_name or not away_name:
            continue
        full_time = (match.get("score") or {}).get("fullTime") or {}
        if full_time.get("home") is None or full_time.get("away") is None:
            continue
        rows.append(
            {
                "competition_slug": slug,
                "home_name": home_name,
                "away_name": away_name,
                "home_goals": full_time["home"],
                "away_goals": full_time["away"],
            }
        )
    return rows

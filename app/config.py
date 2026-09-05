"""App settings and the competition registry.

The registry is the single source of truth for which competitions exist, how to
find their data in each source, and how to display them. Everything else
(ingest, model, UI nav) reads this list rather than hardcoding competition
names.
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=BASE_DIR / ".env", extra="ignore")

    api_football_key: str = ""
    api_football_daily_cap: int = 90
    football_data_org_token: str = ""
    database_url: str = f"sqlite:///{(BASE_DIR / 'data' / 'app.db').as_posix()}"
    stale_after_hours: int = 36
    # Vercel sets this env var automatically in its build/runtime environment —
    # not something we set ourselves. Used to switch to a writable /tmp copy
    # of the DB and cache dir, since Vercel's filesystem is read-only except
    # /tmp (see app/db.py and the RAW_DATA_DIR override below).
    vercel: str = ""


settings = Settings()

RAW_DATA_DIR = Path("/tmp/raw") if settings.vercel else (BASE_DIR / "data" / "raw")
FOOTBALL_DATA_CSV_BASE = "https://www.football-data.co.uk"
CLUBELO_API_BASE = "http://api.clubelo.com"
API_FOOTBALL_BASE = "https://v3.football.api-sports.io"

COMPETITION_TYPE_LEAGUE = "league"
COMPETITION_TYPE_UEFA_CUP = "uefa_cup"

# fd_code: football-data.co.uk division code (see Notes.txt), None if not covered there.
# af_id: api-sports.io v3 league id. Big-five + UEFA ids below are the well-documented,
#   stable ids widely used against the v3 API. af_id=None means "resolve by name at
#   ingest time via GET /leagues?search=..." and cache the result — used where the id
#   is not confidently known ahead of time (see ingest/api_football.py: resolve_league_id).
# neutral_venue: True for competitions where a fixture may be played at a neutral venue
#   (finals) — checked per-fixture from the API, this just flags "possible" for the UI.
COMPETITIONS = [
    {
        "slug": "premier-league",
        "name": "Premier League",
        "country": "England",
        "type": COMPETITION_TYPE_LEAGUE,
        "fd_code": "E0",
        "af_id": 39,
        "af_name": "Premier League",
        "tier": 1,
        "sort_order": 10,
    },
    {
        "slug": "la-liga",
        "name": "La Liga",
        "country": "Spain",
        "type": COMPETITION_TYPE_LEAGUE,
        "fd_code": "SP1",
        "af_id": 140,
        "af_name": "La Liga",
        "tier": 1,
        "sort_order": 20,
    },
    {
        "slug": "serie-a",
        "name": "Serie A",
        "country": "Italy",
        "type": COMPETITION_TYPE_LEAGUE,
        "fd_code": "I1",
        "af_id": 135,
        "af_name": "Serie A",
        "tier": 1,
        "sort_order": 30,
    },
    {
        "slug": "bundesliga",
        "name": "Bundesliga",
        "country": "Germany",
        "type": COMPETITION_TYPE_LEAGUE,
        "fd_code": "D1",
        "af_id": 78,
        "af_name": "Bundesliga",
        "tier": 1,
        "sort_order": 40,
    },
    {
        "slug": "ligue-1",
        "name": "Ligue 1",
        "country": "France",
        "type": COMPETITION_TYPE_LEAGUE,
        "fd_code": "F1",
        "af_id": 61,
        "af_name": "Ligue 1",
        "tier": 1,
        "sort_order": 50,
    },
    {
        "slug": "eredivisie",
        "name": "Eredivisie",
        "country": "Netherlands",
        "type": COMPETITION_TYPE_LEAGUE,
        "fd_code": "N1",
        "af_id": 88,
        "af_name": "Eredivisie",
        "tier": 2,
        "sort_order": 60,
    },
    {
        "slug": "primeira-liga",
        "name": "Primeira Liga",
        "country": "Portugal",
        "type": COMPETITION_TYPE_LEAGUE,
        "fd_code": "P1",
        "af_id": 94,
        "af_name": "Primeira Liga",
        "tier": 2,
        "sort_order": 70,
    },
    {
        "slug": "super-lig",
        "name": "Süper Lig",
        "country": "Turkey",
        "type": COMPETITION_TYPE_LEAGUE,
        "fd_code": "T1",
        "af_id": 203,
        "af_name": "Super Lig",
        "tier": 2,
        "sort_order": 80,
    },
    {
        "slug": "azerbaijan-premyer-liqa",
        "name": "Premyer Liqa",
        "country": "Azerbaijan",
        "type": COMPETITION_TYPE_LEAGUE,
        "fd_code": None,
        "af_id": None,  # resolved at ingest time, see ingest/api_football.py
        "af_name": "Premyer Liqa",
        "tier": 3,
        "sort_order": 90,
    },
    {
        "slug": "champions-league",
        "name": "UEFA Champions League",
        "country": "Europe",
        "type": COMPETITION_TYPE_UEFA_CUP,
        "fd_code": None,
        "af_id": 2,
        "af_name": "UEFA Champions League",
        "tier": 1,
        "sort_order": 1,
        "neutral_venue": True,
    },
    {
        "slug": "europa-league",
        "name": "UEFA Europa League",
        "country": "Europe",
        "type": COMPETITION_TYPE_UEFA_CUP,
        "fd_code": None,
        "af_id": 3,
        "af_name": "UEFA Europa League",
        "tier": 1,
        "sort_order": 2,
        "neutral_venue": True,
    },
    {
        "slug": "conference-league",
        "name": "UEFA Europa Conference League",
        "country": "Europe",
        "type": COMPETITION_TYPE_UEFA_CUP,
        "fd_code": None,
        "af_id": None,  # resolved at ingest time, see ingest/api_football.py
        "af_name": "UEFA Europa Conference League",
        "tier": 1,
        "sort_order": 3,
        "neutral_venue": True,
    },
]

for _c in COMPETITIONS:
    _c.setdefault("neutral_venue", False)


# Bottom-of-table zones per domestic league, for the standings banding on
# /competition/<slug>. Counted from the BOTTOM (position > teams - n), which
# is what makes this robust: relegation is always defined from the foot of the
# table, so the rule survives a league changing size.
#
# `teams` is a guard, not a layout input. The table is only banded when the
# computed standings have exactly this many rows; otherwise it renders
# unbanded. That matters because banding the wrong rows is worse than not
# banding at all, and this is not hypothetical — before the Sept 2026 team
# merge, four of these leagues showed 19-22 rows because clubs were split
# across duplicate Team rows (see scripts/find_duplicate_teams.py), which
# would have coloured safe mid-table clubs as relegated.
#
# `playoff` is the relegation play-off place(s) directly above the automatic
# drop, where a league has them — shown in amber rather than red because those
# clubs are not down, they have another game to save themselves.
#
# UEFA competitions are deliberately absent: the Champions League league phase
# has a completely different zone structure (top 8 direct to the R16, 9-24 to a
# knockout play-off, 25-36 eliminated) and is not a relegation table at all.
LEAGUE_ZONES: dict[str, dict[str, int]] = {
    "premier-league": {"teams": 20, "relegation": 3, "playoff": 0},
    "la-liga": {"teams": 20, "relegation": 3, "playoff": 0},
    "serie-a": {"teams": 20, "relegation": 3, "playoff": 0},
    # 16th plays a two-legged play-off against 2. Bundesliga's 3rd.
    "bundesliga": {"teams": 18, "relegation": 2, "playoff": 1},
    # 16th plays a play-off against a Ligue 2 side.
    "ligue-1": {"teams": 18, "relegation": 2, "playoff": 1},
    # Only 18th goes down automatically; 16th and 17th enter the play-offs.
    "eredivisie": {"teams": 18, "relegation": 1, "playoff": 2},
    "primeira-liga": {"teams": 18, "relegation": 2, "playoff": 0},
    "super-lig": {"teams": 18, "relegation": 3, "playoff": 0},
    # Small league; format has moved around, so no zones claimed until it can
    # be confirmed against a real season (its fixtures are also still missing —
    # see future-plans.md).
    "azerbaijan-premyer-liqa": {"teams": 0, "relegation": 0, "playoff": 0},
}


def league_zone_for(slug: str, position: int, row_count: int) -> str | None:
    """'relegation' | 'playoff' | None for a standings row.

    Returns None for any competition without configured zones, and for every
    row when `row_count` disagrees with the configured team count — see the
    guard note on LEAGUE_ZONES.
    """
    zones = LEAGUE_ZONES.get(slug)
    if not zones or not zones["teams"] or row_count != zones["teams"]:
        return None
    relegation_from = row_count - zones["relegation"]
    if position > relegation_from:
        return "relegation"
    playoff_from = relegation_from - zones["playoff"]
    if zones["playoff"] and position > playoff_from:
        return "playoff"
    return None


# Football news feeds, ingested (never fetched live — see ingest/news.py and
# NewsItem's docstring). Every entry here was hand-verified returning current,
# genuinely distinct items in the Sept 2026 research pass behind this feature.
#
# Dead ends deliberately NOT included, so they aren't re-tried later:
# Goal.com (404), UEFA's own feed (connection failed), Guardian "europaleague"
# and BBC "european-football" (both 404), Sky Sports' /sports.xml (general
# Sky News sport — returned Formula 1 content, not football).
#
# Coverage gap: no dedicated feed exists for Eredivisie, Primeira Liga,
# Süper Lig, Azerbaijan Premyer Liqa, Europa League or Conference League —
# those competitions only get items via team-name matching against the
# general feeds below, which will surface far fewer. The UI shows an
# explicit empty state for a competition page with zero items, not a blank
# panel silently passed off as "no news".
NEWS_FEEDS = [
    # General backbone — every item goes through team-name matching only.
    {"url": "https://feeds.bbci.co.uk/sport/football/rss.xml", "source": "bbc", "competition_slug": None},
    {"url": "https://www.theguardian.com/football/rss", "source": "guardian", "competition_slug": None},
    {"url": "https://www.espn.com/espn/rss/soccer/news", "source": "espn", "competition_slug": None},
    {"url": "https://www.90min.com/posts.rss", "source": "90min", "competition_slug": None},
    # Competition-scoped — every item is tagged with its competition directly,
    # independent of whether a team name is detected in the headline.
    {
        "url": "https://www.theguardian.com/football/premierleague/rss",
        "source": "guardian",
        "competition_slug": "premier-league",
    },
    {
        "url": "https://feeds.bbci.co.uk/sport/football/premier-league/rss.xml",
        "source": "bbc",
        "competition_slug": "premier-league",
    },
    {"url": "https://www.theguardian.com/football/laligafootball/rss", "source": "guardian", "competition_slug": "la-liga"},
    {"url": "https://www.theguardian.com/football/serieafootball/rss", "source": "guardian", "competition_slug": "serie-a"},
    {
        "url": "https://www.theguardian.com/football/bundesligafootball/rss",
        "source": "guardian",
        "competition_slug": "bundesliga",
    },
    {"url": "https://www.theguardian.com/football/ligue1football/rss", "source": "guardian", "competition_slug": "ligue-1"},
    {
        "url": "https://www.theguardian.com/football/championsleague/rss",
        "source": "guardian",
        "competition_slug": "champions-league",
    },
    {
        "url": "https://feeds.bbci.co.uk/sport/football/champions-league/rss.xml",
        "source": "bbc",
        "competition_slug": "champions-league",
    },
]

# How long a news item stays visible before scripts/refresh.py prunes it —
# bounds growth the same way predictions were bounded last round, since this
# table would otherwise accumulate roughly 250-300 items/day forever.
NEWS_RETENTION_DAYS = 14

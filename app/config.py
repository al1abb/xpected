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

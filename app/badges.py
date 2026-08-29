"""Competition logos. Team logos live on Team.logo_url (populated by ingest —
see ingest/football_data_org.py's crest sync and ingest/api_football.py's
per-fixture logo capture); competitions have no such ingest path of their own,
so this is a small curated lookup instead of a schema field.

10 of 12 competitions have a real, verified emblem URL (football-data.org's
crest CDN — confirmed live, including Europa/Conference League, whose emblem
metadata is visible even though their match data sits behind a paid plan).
Süper Lig and the Azerbaijan Premyer Liqa aren't covered by that source at any
tier, and rather than guess at a URL, they fall back to a flag emoji — no
external dependency, nothing that can 404.
"""

from __future__ import annotations

COMPETITION_LOGOS: dict[str, str] = {
    "premier-league": "https://crests.football-data.org/PL.png",
    "la-liga": "https://crests.football-data.org/laliga.png",
    "serie-a": "https://crests.football-data.org/c111.png",
    "bundesliga": "https://crests.football-data.org/BL1.png",
    "ligue-1": "https://crests.football-data.org/FL1.png",
    "eredivisie": "https://crests.football-data.org/ED.png",
    "primeira-liga": "https://crests.football-data.org/PPL.png",
    "champions-league": "https://crests.football-data.org/CL.png",
    "europa-league": "https://crests.football-data.org/EL.png",
    "conference-league": "https://crests.football-data.org/UCL.png",
}

COUNTRY_FLAG_EMOJI: dict[str, str] = {
    "Turkey": "🇹🇷",
    "Azerbaijan": "🇦🇿",
}


def competition_logo(slug: str) -> str | None:
    return COMPETITION_LOGOS.get(slug)


def country_flag(country: str) -> str | None:
    return COUNTRY_FLAG_EMOJI.get(country)

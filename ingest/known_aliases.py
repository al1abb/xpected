"""Curated cross-source name mappings for cases plain fuzzy matching cannot
bridge — transliteration differences (Qarabağ / Karabakh Agdam / Qarabag FK)
and English-football abbreviations that diverge too far from the full name
for difflib to score above even the review threshold ("Man United" vs
"Manchester United" scores 0.74; "Karabakh Agdam" vs "Qarabag FK" scores 0.50,
*below* the 0.60 floor that would even flag it for review).

This list is deliberately not exhaustive — it covers the clubs most likely to
collide across football-data.co.uk (abbreviated), API-Football (full name)
and ClubElo (sometimes transliterated). Anything not listed here still goes
through algorithmic fuzzy matching, and failing that, the unresolved-alias
review queue. Extend this list when a genuine duplicate is spotted there.

Applied once, before any other source ingest, via apply_known_aliases().
"""

from __future__ import annotations

from sqlalchemy.orm import Session

from app.models import Team, TeamAlias

SEED_SOURCE = "manual"

KNOWN_ALIASES: list[dict] = [
    {"canonical": "Qarabag FK", "aliases": ["Qarabag", "Qarabağ", "Qarabağ FK", "Karabakh Agdam", "FK Qarabag"]},
    # England
    {"canonical": "Manchester United", "aliases": ["Man United", "Man Utd", "Manchester Utd"]},
    {"canonical": "Manchester City", "aliases": ["Man City"]},
    {"canonical": "Tottenham Hotspur", "aliases": ["Tottenham", "Spurs"]},
    {"canonical": "Nottingham Forest", "aliases": ["Nott'm Forest", "Notts Forest"]},
    {"canonical": "Wolverhampton Wanderers", "aliases": ["Wolves"]},
    {"canonical": "West Ham United", "aliases": ["West Ham"]},
    {"canonical": "West Bromwich Albion", "aliases": ["West Brom"]},
    {"canonical": "Sheffield United", "aliases": ["Sheffield Utd"]},
    {"canonical": "Sheffield Wednesday", "aliases": ["Sheffield Weds"]},
    {"canonical": "Newcastle United", "aliases": ["Newcastle"]},
    {"canonical": "Leicester City", "aliases": ["Leicester"]},
    {"canonical": "Brighton & Hove Albion", "aliases": ["Brighton"]},
    # Spain
    {"canonical": "Atletico Madrid", "aliases": ["Ath Madrid", "Atlético Madrid"]},
    {"canonical": "Real Sociedad", "aliases": ["Sociedad"]},
    {"canonical": "Real Betis", "aliases": ["Betis"]},
    {"canonical": "Espanyol", "aliases": ["Espanol"]},
    {"canonical": "Athletic Bilbao", "aliases": ["Ath Bilbao"]},
    {"canonical": "Rayo Vallecano", "aliases": ["Vallecano"]},
    {"canonical": "Deportivo Alaves", "aliases": ["Alaves"]},
    {"canonical": "Celta Vigo", "aliases": ["Celta"]},
    # Italy
    {"canonical": "Inter Milan", "aliases": ["Inter", "Internazionale"]},
    {"canonical": "AC Milan", "aliases": ["Milan"]},
    {"canonical": "AS Roma", "aliases": ["Roma"]},
    {"canonical": "Hellas Verona", "aliases": ["Verona"]},
    # Germany
    {"canonical": "Eintracht Frankfurt", "aliases": ["Ein Frankfurt"]},
    {"canonical": "Borussia Monchengladbach", "aliases": ["M'gladbach", "Gladbach", "Borussia M.Gladbach"]},
    {"canonical": "1. FC Koln", "aliases": ["FC Koln", "Koln", "Cologne"]},
    {"canonical": "1. FC Nurnberg", "aliases": ["Nurnberg"]},
    # France
    {"canonical": "Paris Saint Germain", "aliases": ["Paris SG", "PSG"]},
    {"canonical": "Saint-Etienne", "aliases": ["St Etienne"]},
    # Confirmed genuine duplicates surfaced by the unresolved-alias review queue
    # while backfilling UEFA competitions (api_football/fixturedownload naming
    # diverges further from football-data.co.uk than the domestic-only seed
    # above anticipated) — each of these scored 0.62-0.82, below the 0.84
    # auto-confirm cutoff, so without a seed entry they'd sit as two separate
    # Team rows splitting one club's real history.
    {"canonical": "Borussia Dortmund", "aliases": ["Dortmund"]},
    {"canonical": "Bayer Leverkusen", "aliases": ["Leverkusen"]},
    {"canonical": "VfB Stuttgart", "aliases": ["Stuttgart"]},
    {"canonical": "RB Leipzig", "aliases": ["Leipzig"]},
    {"canonical": "TSG 1899 Hoffenheim", "aliases": ["Hoffenheim", "1899 Hoffenheim"]},
    {"canonical": "Sporting Braga", "aliases": ["SC Braga", "Sp Braga", "Braga"]},
    {"canonical": "Athletic Bilbao", "aliases": ["Athletic Club"]},  # extends existing entry above
    {"canonical": "Club Brugge KV", "aliases": ["Club Brugge"]},
    {"canonical": "LASK Linz", "aliases": ["LASK"]},
    {"canonical": "Slovan Bratislava", "aliases": ["S. Bratislava"]},
    {"canonical": "Shakhtar Donetsk", "aliases": ["Shakhtar"]},
]


def apply_known_aliases(session: Session) -> int:
    """Idempotent: safe to call on every ingest run. Returns count of new aliases added."""
    added = 0
    for entry in KNOWN_ALIASES:
        canonical = entry["canonical"]
        team = session.query(Team).filter_by(canonical_name=canonical).one_or_none()
        if team is None:
            team = Team(canonical_name=canonical)
            session.add(team)
            session.flush()

        for name in [canonical, *entry["aliases"]]:
            exists = session.query(TeamAlias).filter_by(alias=name, source=SEED_SOURCE).one_or_none()
            if exists is None:
                session.add(TeamAlias(team_id=team.id, alias=name, source=SEED_SOURCE))
                added += 1
    session.commit()
    return added

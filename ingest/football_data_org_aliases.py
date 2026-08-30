"""Curated football-data.org name -> our canonical name overrides.

football-data.org's team-crest endpoint returns full legal/formal names
("FC Internazionale Milano", "Real Sociedad de Futbol") that the general
fuzzy matcher in ingest/resolve.py correctly declines to auto-confirm —
short-name components ("Inter", "Sociedad") against a long formal name score
well below the confirm cutoff, and a few near-miss scores land on the wrong
club entirely (e.g. "AZ" nearest-matches "Araz" by raw character overlap, not
"AZ Alkmaar"). Hand-verified once here rather than guessed on ingest.

Built while diagnosing why 45% of teams with upcoming fixtures had no crest:
`sync_team_crests` was using `get_or_create_team` (the create path), so every
one of these misses silently became a brand-new, duplicate `Team` row holding
only a crest and no match history, rather than attaching the crest to the
real team. See scripts/merge_teams.py for the cleanup of the ~50 duplicates
that produced, and ingest/football_data_org.py for the resolve-only fix.
"""

from __future__ import annotations

# football-data.org's team "name" field -> our canonical Team.canonical_name
FD_ORG_TO_CANONICAL: dict[str, str] = {
    "1. FSV Mainz 05": "Mainz",
    "ACF Fiorentina": "Fiorentina",
    "ADO Den Haag": "Den Haag",
    "AJ Auxerre": "Auxerre",
    "AZ": "AZ Alkmaar",
    "Angers SCO": "Angers",
    "Bologna FC 1909": "Bologna",
    "CA Osasuna": "Osasuna",
    "Cagliari Calcio": "Cagliari",
    "Como 1907": "Como",
    "Coventry City FC": "Coventry",
    "ES Troyes AC": "Troyes",
    "FC Internazionale Milano": "Inter Milan",
    "FC Twente '65": "Twente",
    "Feyenoord Rotterdam": "Feyenoord",
    "Frosinone Calcio": "Frosinone",
    "GD Estoril Praia": "Estoril",
    "Hamburger SV": "Hamburg",
    "Hull City AFC": "Hull",
    "Ipswich Town FC": "Ipswich",
    "Leeds United FC": "Leeds",
    "Levante UD": "Levante",
    "Lille OSC": "Lille",
    "NEC": "Nijmegen",
    "OGC Nice": "Nice",
    "Olympique Lyonnais": "Lyon",
    "Olympique de Marseille": "Marseille",
    "PAE AEK": "AEK Athens FC",
    "PEC Zwolle": "Zwolle",
    "Parma Calcio 1913": "Parma",
    "RC Deportivo La Coruña": "La Coruna",
    "RC Strasbourg Alsace": "Strasbourg",
    "RCD Espanyol de Barcelona": "Espanyol",
    "Racing Club de Lens": "Lens",
    "Rayo Vallecano de Madrid": "Rayo Vallecano",
    "Real Betis Balompié": "Real Betis",
    "Real Racing Club de Santander": "Santander",
    "Real Sociedad de Fútbol": "Real Sociedad",
    "SBV Excelsior": "Excelsior",
    "SC Cambuur-Leeuwarden": "Cambuur",
    "SS Lazio": "Lazio",
    "SSC Napoli": "Napoli",
    "SV 07 Elversberg": "Elversberg",
    "Sport Lisboa e Benfica": "Benfica",
    "Sporting Clube de Braga": "Sporting Braga",
    "Sporting Clube de Portugal": "Sporting CP",
    "Stade Brestois 29": "Brest",
    "Stade Rennais FC 1901": "Rennes",
    "Telstar 1963": "Telstar",
    "US Lecce": "Lecce",
    "US Sassuolo Calcio": "Sassuolo",
    "Udinese Calcio": "Udinese",
    "Willem II Tilburg": "Willem II",
}

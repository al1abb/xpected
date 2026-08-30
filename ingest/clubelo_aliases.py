"""Curated ClubElo name -> our canonical name overrides.

ClubElo abbreviates aggressively ("Bayern", "Man City", "Sociedad") in ways
the general fuzzy matcher in ingest/resolve.py correctly refuses to
auto-confirm (short strings, low character-overlap ratio) — that caution is
right for arbitrary sources, but here the mapping is small and worth hand-
verifying once. Built by cross-checking every club in our tracked
competitions against a live ClubElo snapshot; see the plan/commit history for
the verification queries.

Only covers cases the fuzzy matcher in `ingest/resolve.py` would otherwise
send to manual review or drop. Anything not listed here either resolves
automatically (exact or high-confidence fuzzy match) or is genuinely absent
from ClubElo (smaller/newly-promoted clubs) and falls back to the internal,
scale-mapped rating in model/elo.py.
"""

from __future__ import annotations

# our canonical Team.canonical_name -> ClubElo's "Club" column
CANONICAL_TO_CLUBELO: dict[str, str] = {
    "1. FC Koln": "Koeln",
    "AZ Alkmaar": "Alkmaar",
    "AEK Athens FC": "AEK",
    "Athletic Bilbao": "Bilbao",
    "Atletico Madrid": "Atletico",
    "Bayer Leverkusen": "Leverkusen",
    "Bayern Munich": "Bayern",
    "Borussia Dortmund": "Dortmund",
    "Brighton & Hove Albion": "Brighton",
    "Club Brugge KV": "Brugge",
    "Celta Vigo": "Celta",
    "Deportivo Alaves": "Alaves",
    "Eintracht Frankfurt": "Frankfurt",
    "Estrela": "Estrela Amadora",
    "For Sittard": "Sittard",
    "Inter Milan": "Inter",
    "LASK Linz": "LASK",
    "La Coruna": "Depor",
    "Manchester City": "Man City",
    "Manchester United": "Man United",
    "Newcastle United": "Newcastle",
    "Nottingham Forest": "Forest",
    "PSV Eindhoven": "PSV",
    "Paris Saint Germain": "Paris SG",
    "Real Betis": "Betis",
    "Real Sociedad": "Sociedad",
    "Schalke 04": "Schalke",
    "Shakhtar Donetsk": "Shakhtar",
    "Sp Lisbon": "Sporting",
    "Sporting Braga": "Braga",
    "TSG 1899 Hoffenheim": "Hoffenheim",
    "Tottenham Hotspur": "Tottenham",
    "VfB Stuttgart": "Stuttgart",
    "Werder Bremen": "Werder",
}

# Reverse lookup, built once: ClubElo name -> our canonical name.
CLUBELO_TO_CANONICAL: dict[str, str] = {v: k for k, v in CANONICAL_TO_CLUBELO.items()}

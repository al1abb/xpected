"""Team colours for the probability bar. A curated set of real club colours
for the clubs anyone would recognise; everything else gets a deterministic
colour derived from the team's name, so the same team always renders the same
way without needing manual upkeep for all ~400 teams in the database.
"""

from __future__ import annotations

import colorsys
import hashlib

# canonical_name -> (primary hex, secondary hex or None)
CURATED_COLORS: dict[str, tuple[str, str | None]] = {
    # England
    "Manchester United": ("#DA291C", "#FBE122"),
    "Manchester City": ("#6CABDD", "#1C2C5B"),
    "Liverpool": ("#C8102E", "#00B2A9"),
    "Arsenal": ("#EF0107", "#023474"),
    "Chelsea": ("#034694", "#EE242C"),
    "Tottenham Hotspur": ("#132257", "#FFFFFF"),
    "Newcastle United": ("#241F20", "#FFFFFF"),
    "Aston Villa": ("#670E36", "#95BFE5"),
    "West Ham United": ("#7A263A", "#1BB1E7"),
    "Everton": ("#003399", "#FFFFFF"),
    "Leicester City": ("#003090", "#FDBE11"),
    "Brighton & Hove Albion": ("#0057B8", "#FFCD00"),
    "Wolverhampton Wanderers": ("#FDB913", "#231F20"),
    "Nottingham Forest": ("#DD0000", "#FFFFFF"),
    "Crystal Palace": ("#1B458F", "#C4122E"),
    "Fulham": ("#000000", "#FFFFFF"),
    "Brentford": ("#D20000", "#FFFFFF"),
    "Bournemouth": ("#DA291C", "#000000"),
    # Spain
    "Real Madrid": ("#FEBE10", "#00529F"),
    "Barcelona": ("#A50044", "#004D98"),
    "Atletico Madrid": ("#CB3524", "#272E61"),
    "Real Sociedad": ("#0067B1", "#FFFFFF"),
    "Athletic Bilbao": ("#EE2523", "#FFFFFF"),
    "Real Betis": ("#00954C", "#FFFFFF"),
    "Sevilla": ("#D8112B", "#FFFFFF"),
    "Villarreal": ("#FFE667", "#005187"),
    "Valencia": ("#EE3524", "#FFFFFF"),
    "Espanyol": ("#00529F", "#FFFFFF"),
    "Celta Vigo": ("#8AC3EE", "#FFFFFF"),
    "Rayo Vallecano": ("#E01A2B", "#FFFFFF"),
    "Deportivo Alaves": ("#1C4F9C", "#FFFFFF"),
    # Italy
    "Juventus": ("#000000", "#FFFFFF"),
    "Inter Milan": ("#0068A8", "#000000"),
    "AC Milan": ("#FB090B", "#000000"),
    "Napoli": ("#12A0D7", "#FFFFFF"),
    "AS Roma": ("#8E1F2F", "#F0BC42"),
    "Lazio": ("#87D8F7", "#FFFFFF"),
    "Atalanta": ("#1C1C4E", "#000000"),
    "Fiorentina": ("#592C82", "#FFFFFF"),
    "Bologna": ("#943126", "#08316E"),
    "Torino": ("#881D23", "#000000"),
    "Hellas Verona": ("#0A3A6E", "#FFE300"),
    # Germany
    "Bayern Munich": ("#DC052D", "#0066B2"),
    "Borussia Dortmund": ("#FDE100", "#000000"),
    "RB Leipzig": ("#DD0741", "#FFFFFF"),
    "Bayer Leverkusen": ("#E32221", "#000000"),
    "Eintracht Frankfurt": ("#E1000F", "#000000"),
    "VfB Stuttgart": ("#E32219", "#FFFFFF"),
    "Borussia Monchengladbach": ("#000000", "#FFFFFF"),
    "TSG 1899 Hoffenheim": ("#1C63B7", "#FFFFFF"),
    "Werder Bremen": ("#1D9053", "#FFFFFF"),
    "SC Freiburg": ("#000000", "#FF0000"),
    "1. FC Union Berlin": ("#EB1923", "#FFFFFF"),
    # France
    "Paris Saint Germain": ("#004170", "#DA291C"),
    "Olympique de Marseille": ("#2FAEE0", "#FFFFFF"),
    "Olympique Lyonnais": ("#003087", "#DA291C"),
    "AS Monaco": ("#E51A23", "#FFFFFF"),
    "Lille": ("#E2001A", "#0033A0"),
    "Stade Rennais": ("#E62128", "#000000"),
    "OGC Nice": ("#CC0000", "#000000"),
    "RC Lens": ("#FFD100", "#E2001A"),
    "Stade de Reims": ("#E2001A", "#FFFFFF"),
    "Saint-Etienne": ("#00953B", "#FFFFFF"),
    # Netherlands / Portugal / Turkey
    "Ajax": ("#D2122E", "#FFFFFF"),
    "PSV Eindhoven": ("#ED1C24", "#FFFFFF"),
    "Feyenoord": ("#00A650", "#ED1C24"),
    "AZ Alkmaar": ("#D2122E", "#FFFFFF"),
    "Benfica": ("#E52620", "#FFFFFF"),
    "Sporting CP": ("#00744A", "#FFFFFF"),
    "FC Porto": ("#00447C", "#FFFFFF"),
    "Sporting Braga": ("#B10D24", "#FFFFFF"),
    "Galatasaray": ("#A90432", "#FDB913"),
    "Fenerbahce": ("#0033A0", "#FFE800"),
    "Besiktas": ("#000000", "#FFFFFF"),
    "Trabzonspor": ("#7B122B", "#0F2F5F"),
    # UEFA regulars / Azerbaijan
    "Club Brugge KV": ("#0057A8", "#000000"),
    "Shakhtar Donetsk": ("#FF6600", "#000000"),
    "Dinamo Zagreb": ("#00549F", "#FFFFFF"),
    "Red Bull Salzburg": ("#DB021E", "#FFFFFF"),
    "Slavia Praha": ("#DA232D", "#FFFFFF"),
    "Qarabag FK": ("#000000", "#FFDA44"),
}

DEFAULT_SATURATION = 0.55
DEFAULT_LIGHTNESS = 0.42


def fallback_color(name: str) -> str:
    """Deterministic hex colour from a team name, so the same team always
    renders identically without a curated entry."""
    digest = hashlib.sha256(name.encode()).hexdigest()
    hue = int(digest[:8], 16) / 0xFFFFFFFF
    r, g, b = colorsys.hls_to_rgb(hue, DEFAULT_LIGHTNESS, DEFAULT_SATURATION)
    return f"#{int(r * 255):02X}{int(g * 255):02X}{int(b * 255):02X}"


def team_colors(canonical_name: str) -> tuple[str, str | None]:
    if canonical_name in CURATED_COLORS:
        return CURATED_COLORS[canonical_name]
    return fallback_color(canonical_name), None


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _rgb_distance(a: str, b: str) -> float:
    ar, ag, ab = _hex_to_rgb(a)
    br, bg, bb = _hex_to_rgb(b)
    return ((ar - br) ** 2 + (ag - bg) ** 2 + (ab - bb) ** 2) ** 0.5


def _rotate_hue(hex_color: str, degrees: float = 0.5) -> str:
    r, g, b = (c / 255 for c in _hex_to_rgb(hex_color))
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    h = (h + degrees) % 1.0
    r2, g2, b2 = colorsys.hls_to_rgb(h, l, s)
    return f"#{int(r2 * 255):02X}{int(g2 * 255):02X}{int(b2 * 255):02X}"


COLLISION_THRESHOLD = 60.0  # RGB Euclidean distance below which two colours read as "the same" at a glance


def resolve_match_colors(
    home_primary: str, away_primary: str, away_secondary: str | None
) -> tuple[str, str]:
    """If home and away colours are too close to tell apart, prefer the away
    team's secondary colour; if that's unavailable or still too close,
    deterministically rotate the away colour's hue as a last resort."""
    if _rgb_distance(home_primary, away_primary) >= COLLISION_THRESHOLD:
        return home_primary, away_primary

    if away_secondary and _rgb_distance(home_primary, away_secondary) >= COLLISION_THRESHOLD:
        return home_primary, away_secondary

    return home_primary, _rotate_hue(away_primary)

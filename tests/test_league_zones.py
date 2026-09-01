"""Tests for app/config.py::league_zone_for — the relegation/play-off banding
rule behind the standings table.

The guard behaviour is the important part. Colouring the wrong club as
relegated is a worse failure than showing no colour at all, so a table whose
row count disagrees with the configured league size must come back unbanded.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from app.config import COMPETITIONS, LEAGUE_ZONES, league_zone_for


def _zones(slug, row_count):
    return [league_zone_for(slug, pos, row_count) for pos in range(1, row_count + 1)]


def test_premier_league_bottom_three_are_relegation():
    zones = _zones("premier-league", 20)
    assert zones[:17] == [None] * 17
    assert zones[17:] == ["relegation"] * 3


def test_bundesliga_has_two_down_and_one_playoff():
    zones = _zones("bundesliga", 18)
    assert zones[16:] == ["relegation", "relegation"]  # 17th, 18th
    assert zones[15] == "playoff"  # 16th
    assert zones[14] is None  # 15th is safe


def test_eredivisie_has_one_down_and_two_playoff():
    """Inverted relative to Bundesliga — only the last club goes automatically,
    so the amber band is larger than the red one."""
    zones = _zones("eredivisie", 18)
    assert zones[17] == "relegation"  # 18th only
    assert zones[15:17] == ["playoff", "playoff"]  # 16th, 17th
    assert zones[14] is None


def test_no_banding_when_row_count_disagrees_with_configured_size():
    """A 22-row Ligue 1 table means the data is wrong (this really happened —
    duplicate Team rows). Band nothing rather than colour the wrong clubs."""
    assert _zones("ligue-1", 22) == [None] * 22
    assert _zones("ligue-1", 17) == [None] * 17
    # ...and the correct size still bands.
    assert _zones("ligue-1", 18)[17] == "relegation"


def test_unknown_or_unzoned_competitions_are_never_banded():
    assert _zones("champions-league", 36) == [None] * 36
    assert _zones("europa-league", 36) == [None] * 36
    assert _zones("not-a-real-league", 20) == [None] * 20
    # Configured but deliberately zero'd out pending a confirmed format.
    assert _zones("azerbaijan-premyer-liqa", 10) == [None] * 10


def test_zones_never_overlap_or_exceed_the_table():
    for slug, cfg in LEAGUE_ZONES.items():
        if not cfg["teams"]:
            continue
        zones = _zones(slug, cfg["teams"])
        assert zones.count("relegation") == cfg["relegation"], slug
        assert zones.count("playoff") == cfg["playoff"], slug
        # Bands must be contiguous and sit at the foot of the table.
        banded = [i for i, z in enumerate(zones) if z is not None]
        assert banded == list(range(len(zones) - len(banded), len(zones))), slug


@pytest.mark.parametrize("slug", [c["slug"] for c in COMPETITIONS if c["type"] == "league"])
def test_every_domestic_league_has_a_zone_entry(slug):
    """A new league added to COMPETITIONS without a LEAGUE_ZONES entry would
    silently render unbanded forever; fail loudly instead."""
    assert slug in LEAGUE_ZONES

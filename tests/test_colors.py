import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.colors import fallback_color, resolve_match_colors, team_colors


def test_fallback_color_deterministic():
    assert fallback_color("Some Random FC") == fallback_color("Some Random FC")


def test_fallback_color_is_valid_hex():
    c = fallback_color("Anything")
    assert len(c) == 7 and c.startswith("#")
    int(c[1:], 16)  # raises if not valid hex


def test_curated_club_returns_known_colors():
    primary, secondary = team_colors("Liverpool")
    assert primary == "#C8102E"
    assert secondary == "#00B2A9"


def test_uncurated_club_uses_fallback():
    primary, secondary = team_colors("Some Obscure Lower League Club")
    assert secondary is None
    assert primary == fallback_color("Some Obscure Lower League Club")


def test_distinct_colors_pass_through_unchanged():
    home, away = resolve_match_colors("#FF0000", "#0000FF", None)
    assert (home, away) == ("#FF0000", "#0000FF")


def test_identical_colors_fall_back_to_secondary():
    home, away = resolve_match_colors("#DA291C", "#DA291C", "#FBE122")
    assert home == "#DA291C"
    assert away == "#FBE122"


def test_identical_colors_with_no_secondary_rotates_hue():
    home, away = resolve_match_colors("#DA291C", "#DA291C", None)
    assert away != "#DA291C"


def test_identical_colors_with_similar_secondary_still_rotates():
    # secondary is also red-ish and would still collide — must not return it
    home, away = resolve_match_colors("#DA291C", "#DA291C", "#D8112B")
    assert away not in ("#DA291C", "#D8112B")

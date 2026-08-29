import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.badges import COMPETITION_LOGOS, competition_logo, country_flag
from app.config import COMPETITIONS


def test_every_competition_has_a_logo_or_flag_fallback():
    for c in COMPETITIONS:
        has_logo = competition_logo(c["slug"]) is not None
        has_flag = country_flag(c["country"]) is not None
        assert has_logo or has_flag, f"{c['slug']} has neither a logo nor a flag fallback"


def test_unknown_competition_returns_none():
    assert competition_logo("not-a-real-competition") is None


def test_unknown_country_returns_none():
    assert country_flag("Nowhereland") is None


def test_logo_urls_use_https():
    for url in COMPETITION_LOGOS.values():
        assert url.startswith("https://")

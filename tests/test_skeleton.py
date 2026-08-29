import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import COMPETITIONS


def test_competition_registry_slugs_unique():
    slugs = [c["slug"] for c in COMPETITIONS]
    assert len(slugs) == len(set(slugs))


def test_competition_registry_has_twelve():
    assert len(COMPETITIONS) == 12


def test_every_competition_has_a_data_source():
    for c in COMPETITIONS:
        assert c["fd_code"] is not None or c["af_id"] is not None or c.get("af_name"), (
            f"{c['slug']} has no way to be ingested"
        )

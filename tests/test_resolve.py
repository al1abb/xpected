import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.models import Base, Team, UnresolvedAlias
from ingest.known_aliases import apply_known_aliases
from ingest.resolve import get_or_create_team, normalize


@pytest.fixture()
def session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    s = Session()
    yield s
    s.close()


def test_same_source_repeat_reuses_team(session):
    t1 = get_or_create_team(session, "Real Madrid", "football_data")
    t2 = get_or_create_team(session, "Real Madrid", "football_data")
    assert t1.id == t2.id
    assert session.query(Team).count() == 1


def test_same_source_distinct_short_names_do_not_collide(session):
    """Regression: 'Real Madrid' vs 'Ath Madrid' scored 0.76 similarity before
    same-source exclusion was added, and would have wrongly merged."""
    get_or_create_team(session, "Real Madrid", "football_data")
    get_or_create_team(session, "Ath Madrid", "football_data")
    assert session.query(Team).count() == 2
    assert session.query(UnresolvedAlias).count() == 0


def test_known_alias_seed_bridges_qarabag_transliterations(session):
    apply_known_aliases(session)
    from_clubelo = get_or_create_team(session, "Karabakh Agdam", "clubelo")
    from_api = get_or_create_team(session, "Qarabag FK", "api_football")
    assert from_clubelo.id == from_api.id
    assert session.query(Team).filter_by(canonical_name="Qarabag FK").count() == 1


def test_unseeded_transliteration_gap_is_the_known_limitation(session):
    """Without the seed, 'Karabakh Agdam' vs 'Qarabag FK' scores ~0.50 —
    below even the review threshold — so plain fuzzy matching alone creates
    two teams silently. This test documents why known_aliases.py exists."""
    a = get_or_create_team(session, "Karabakh Agdam", "clubelo")
    b = get_or_create_team(session, "Qarabag FK", "api_football")
    assert a.id != b.id
    assert session.query(UnresolvedAlias).count() == 0  # silently missed, not even flagged


def test_moderate_similarity_is_flagged_not_guessed(session):
    """'Man United' vs 'Manchester United' scores ~0.74: below the 0.84
    auto-confirm cutoff but above the 0.60 floor, so it must create a new
    team AND log it for manual review rather than silently merging OR
    silently missing it."""
    get_or_create_team(session, "Man United", "football_data")
    get_or_create_team(session, "Manchester United", "api_football")
    assert session.query(Team).count() == 2
    assert session.query(UnresolvedAlias).count() == 1


def test_normalize_strips_common_suffixes():
    assert normalize("Qarabag FK") == normalize("FK Qarabag")


def test_known_alias_seed_bridges_uefa_naming_variants(session):
    """Regression: these scored 0.62-0.82 in the live UEFA backfill — below the
    0.84 auto-confirm cutoff — and would otherwise split one club's history
    across two Team rows."""
    apply_known_aliases(session)
    dortmund_full = get_or_create_team(session, "Borussia Dortmund", "api_football")
    dortmund_short = get_or_create_team(session, "Dortmund", "fixturedownload")
    assert dortmund_full.id == dortmund_short.id

    bilbao_alt_name = get_or_create_team(session, "Athletic Club", "api_football")
    bilbao_abbrev = get_or_create_team(session, "Ath Bilbao", "football_data")
    assert bilbao_alt_name.id == bilbao_abbrev.id


def test_unrelated_short_names_not_flagged_despite_character_overlap(session):
    """Regression: 'Everton' vs 'Hellas Verona' scores 0.77 and 'Mainz' vs
    'AC Milan' scores 0.60 on pure character ratio despite zero real
    relationship — these must not reach the review queue at all."""
    apply_known_aliases(session)  # populates a 'manual'-source pool to fuzzy-match against
    get_or_create_team(session, "Everton", "football_data")
    get_or_create_team(session, "Mainz", "football_data")
    assert session.query(UnresolvedAlias).count() == 0


def test_same_team_seen_twice_before_commit_does_not_crash(session):
    """Regression: within one uncommitted ingest run, the same raw_name from the
    same source appearing twice (a team playing two matches) must reuse the
    same Team both times, not attempt a second INSERT with the same
    canonical_name and hit the unique constraint."""
    first = get_or_create_team(session, "Sabail FK", "api_football")
    second = get_or_create_team(session, "Sabail FK", "api_football")
    session.commit()
    assert first.id == second.id
    assert session.query(Team).filter_by(canonical_name="Sabail FK").count() == 1


def test_shared_word_ambiguity_still_flagged(session):
    """'Real Madrid' vs 'Atletico Madrid' share the token 'madrid' — a cheap,
    legitimate thing to ask a human to glance at, unlike pure noise."""
    get_or_create_team(session, "Atletico Madrid", "football_data")
    get_or_create_team(session, "Real Madrid", "api_football")
    assert session.query(UnresolvedAlias).count() == 1

"""Regression tests for scripts/merge_teams.py.

The bug class this guards against: the same real club ingested under two
names becomes two `Team` rows, so its history/upcoming fixtures/Elo rating
split in half instead of accumulating on one (confirmed in practice — see the
module docstring in scripts/merge_teams.py). These tests exercise the merge
logic itself in isolation, on a synthetic DB, rather than asserting facts
about the live dataset (which legitimately contains real zero-history teams —
newly promoted/newly-entering-Europe clubs — so a hardcoded "no zero-history
team" check on real data would be both fragile and wrong).
"""

import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.models import Base, Competition, EloRating, Match, Prediction, Team, TeamAlias
from scripts.merge_teams import merge_team


@pytest.fixture()
def session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    s = Session()
    yield s
    s.close()


def _comp(session, slug="league"):
    comp = Competition(slug=slug, name=slug, country="X", type="league", fd_code="X0")
    session.add(comp)
    session.flush()
    return comp


def test_merge_reassigns_matches_and_aliases(session):
    comp = _comp(session)
    survivor = Team(canonical_name="Real Name")
    duplicate = Team(canonical_name="Short Name")
    opponent = Team(canonical_name="Opponent")
    session.add_all([survivor, duplicate, opponent])
    session.flush()

    finished = Match(competition_id=comp.id, utc_kickoff=dt.datetime(2026, 1, 1), status="finished", home_team_id=duplicate.id, away_team_id=opponent.id, home_goals=1, away_goals=0, source="test")
    upcoming = Match(competition_id=comp.id, utc_kickoff=dt.datetime(2026, 6, 1), status="scheduled", home_team_id=opponent.id, away_team_id=duplicate.id, source="test")
    session.add_all([finished, upcoming])
    session.add(TeamAlias(team_id=duplicate.id, alias="Short Name", source="fixturedownload"))
    session.add(EloRating(team_id=duplicate.id, as_of_date=dt.date(2026, 1, 1), elo=1550.0, source="clubelo"))
    session.commit()

    survivor_id, duplicate_id = survivor.id, duplicate.id
    stats = merge_team(session, duplicate, survivor)
    session.commit()

    assert stats["matches_reassigned"] == 2
    assert stats["matches_dropped"] == 0
    assert stats["aliases_moved"] == 1
    assert stats["elo_ratings_moved"] == 1

    assert session.get(Match, finished.id).home_team_id == survivor_id
    assert session.get(Match, upcoming.id).away_team_id == survivor_id
    assert session.query(TeamAlias).filter_by(team_id=survivor_id, alias="Short Name").one_or_none() is not None
    assert session.query(EloRating).filter_by(team_id=survivor_id).one_or_none() is not None
    assert session.get(Team, duplicate_id) is None


def test_merge_resolves_match_collision_keeping_the_finished_one(session):
    """If both the duplicate and the survivor already have a row for what
    turns out to be the same real fixture (same competition/kickoff/
    opponent), the merge must not violate the uniqueness constraint — it
    should keep the one with a real result and drop the other, including that
    dropped match's own predictions/odds."""
    comp = _comp(session)
    survivor = Team(canonical_name="Real Name")
    duplicate = Team(canonical_name="Short Name")
    opponent = Team(canonical_name="Opponent")
    session.add_all([survivor, duplicate, opponent])
    session.flush()

    kickoff = dt.datetime(2026, 3, 1)
    # Same fixture, arrived from two sources: one still shows "scheduled"
    # (pre-match ingest), the other has since been updated with the result.
    stale_scheduled = Match(competition_id=comp.id, utc_kickoff=kickoff, status="scheduled", home_team_id=duplicate.id, away_team_id=opponent.id, source="test")
    real_result = Match(competition_id=comp.id, utc_kickoff=kickoff, status="finished", home_team_id=survivor.id, away_team_id=opponent.id, home_goals=2, away_goals=1, source="test")
    session.add_all([stale_scheduled, real_result])
    session.flush()
    session.add(Prediction(match_id=stale_scheduled.id, model_run_id=1, home_win_prob=0.4, draw_prob=0.3, away_win_prob=0.3))
    session.commit()

    real_result_id = real_result.id
    stale_id = stale_scheduled.id
    stats = merge_team(session, duplicate, survivor)
    session.commit()

    assert stats["matches_dropped"] == 1
    assert session.get(Match, real_result_id) is not None
    assert session.get(Match, stale_id) is None
    assert session.query(Prediction).filter_by(match_id=stale_id).count() == 0

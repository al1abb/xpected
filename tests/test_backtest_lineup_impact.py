import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.models import Base, Competition, Lineup, Match, Team
from model.player_elo import MIN_STARTERS_FOR_LIVE_STRENGTH
from scripts.backtest_lineup_impact import _rescored, qualifying_match_ids


@pytest.fixture()
def session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    s = Session()
    yield s
    s.close()


def _team(session, name):
    t = Team(canonical_name=name)
    session.add(t)
    session.flush()
    return t


def _match(session, home, away, *, status="finished"):
    comp = session.query(Competition).first()
    if comp is None:
        comp = Competition(slug="premier-league", name="EPL", country="England", type="league")
        session.add(comp)
        session.flush()
    m = Match(competition_id=comp.id, utc_kickoff=dt.datetime(2026, 1, 1), status=status, home_team_id=home.id, away_team_id=away.id, home_goals=1, away_goals=0, source="test")
    session.add(m)
    session.flush()
    return m


def _lineup(session, match_id, team_id, player_ids, *, starter=True):
    for pid in player_ids:
        session.add(Lineup(match_id=match_id, team_id=team_id, player_name=f"P{pid}", starter=starter, player_id=pid))
    session.commit()


# ---------- qualifying_match_ids ----------


def test_qualifying_match_ids_includes_match_with_enough_starters_both_sides(session):
    home, away = _team(session, "Home"), _team(session, "Away")
    match = _match(session, home, away)
    _lineup(session, match.id, home.id, range(1, 1 + MIN_STARTERS_FOR_LIVE_STRENGTH))
    _lineup(session, match.id, away.id, range(101, 101 + MIN_STARTERS_FOR_LIVE_STRENGTH))

    assert qualifying_match_ids(session) == {match.id}


def test_qualifying_match_ids_excludes_match_below_threshold_on_one_side(session):
    home, away = _team(session, "Home"), _team(session, "Away")
    match = _match(session, home, away)
    _lineup(session, match.id, home.id, range(1, 1 + MIN_STARTERS_FOR_LIVE_STRENGTH - 1))
    _lineup(session, match.id, away.id, range(101, 101 + MIN_STARTERS_FOR_LIVE_STRENGTH))

    assert qualifying_match_ids(session) == set()


def test_qualifying_match_ids_excludes_scheduled_match(session):
    """Only finished matches can be scored by a backtest at all."""
    home, away = _team(session, "Home"), _team(session, "Away")
    match = _match(session, home, away, status="scheduled")
    _lineup(session, match.id, home.id, range(1, 1 + MIN_STARTERS_FOR_LIVE_STRENGTH))
    _lineup(session, match.id, away.id, range(101, 101 + MIN_STARTERS_FOR_LIVE_STRENGTH))

    assert qualifying_match_ids(session) == set()


def test_qualifying_match_ids_ignores_unresolved_starters(session):
    """A starter row with no player_id (unresolved name) doesn't count
    toward the threshold -- matches live_team_strength's own bar."""
    home, away = _team(session, "Home"), _team(session, "Away")
    match = _match(session, home, away)
    session.add(Lineup(match_id=match.id, team_id=home.id, player_name="Unresolved", starter=True, player_id=None))
    session.commit()
    _lineup(session, match.id, home.id, range(1, MIN_STARTERS_FOR_LIVE_STRENGTH))  # one short + the unresolved row above
    _lineup(session, match.id, away.id, range(101, 101 + MIN_STARTERS_FOR_LIVE_STRENGTH))

    assert qualifying_match_ids(session) == set()


def test_qualifying_match_ids_ignores_bench_rows(session):
    home, away = _team(session, "Home"), _team(session, "Away")
    match = _match(session, home, away)
    _lineup(session, match.id, home.id, range(1, 1 + MIN_STARTERS_FOR_LIVE_STRENGTH))
    _lineup(session, match.id, home.id, [999], starter=False)  # bench, should not count toward the threshold
    _lineup(session, match.id, away.id, range(101, 101 + MIN_STARTERS_FOR_LIVE_STRENGTH))

    assert qualifying_match_ids(session) == {match.id}


# ---------- _rescored ----------


def test_rescored_filters_to_given_match_ids():
    raw = [
        {"match_id": 1, "probs": (0.6, 0.2, 0.2), "actual": 0},
        {"match_id": 2, "probs": (0.2, 0.2, 0.6), "actual": 2},
        {"match_id": 3, "probs": (0.3, 0.3, 0.4), "actual": 1},
    ]
    result = _rescored(raw, {1, 2})
    assert result["n"] == 2


def test_rescored_none_means_no_filtering():
    raw = [
        {"match_id": 1, "probs": (0.6, 0.2, 0.2), "actual": 0},
        {"match_id": 2, "probs": (0.2, 0.2, 0.6), "actual": 2},
    ]
    result = _rescored(raw, None)
    assert result["n"] == 2


def test_rescored_empty_subset_returns_zero_n():
    raw = [{"match_id": 1, "probs": (0.6, 0.2, 0.2), "actual": 0}]
    result = _rescored(raw, set())
    assert result["n"] == 0

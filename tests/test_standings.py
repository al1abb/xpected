"""Tests for model/standings.py — point totals, tie-break ordering, and
partial-season robustness (teams with different match counts)."""

import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.models import Base, Competition, Match, Team
from model.standings import compute_standings


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


def _match(comp, home, away, hg, ag, kickoff, status="finished"):
    return Match(
        competition_id=comp.id,
        utc_kickoff=kickoff,
        status=status,
        home_team_id=home.id,
        away_team_id=away.id,
        home_goals=hg,
        away_goals=ag,
        source="test",
    )


def test_points_and_basic_ordering(session):
    comp = _comp(session)
    a = Team(canonical_name="Alpha")
    b = Team(canonical_name="Beta")
    c = Team(canonical_name="Gamma")
    session.add_all([a, b, c])
    session.flush()

    season_start = dt.date(2026, 7, 1)
    matches = [
        _match(comp, a, b, 2, 0, dt.datetime(2026, 8, 1)),  # A win
        _match(comp, c, a, 1, 1, dt.datetime(2026, 8, 8)),  # draw
        _match(comp, b, c, 0, 3, dt.datetime(2026, 8, 15)),  # C win
    ]
    session.add_all(matches)
    session.commit()

    rows = compute_standings(session, comp.id, season_start)
    by_name = {r["team"].canonical_name: r for r in rows}

    assert by_name["Alpha"]["points"] == 4  # win + draw
    assert by_name["Gamma"]["points"] == 4  # win + draw
    assert by_name["Beta"]["points"] == 0

    # Alpha and Gamma tie on points; Gamma has better GD (+3 vs Alpha's +1)
    # so Gamma ranks above Alpha.
    assert [r["team"].canonical_name for r in rows] == ["Gamma", "Alpha", "Beta"]
    assert rows[0]["position"] == 1
    assert rows[1]["position"] == 2
    assert rows[2]["position"] == 3


def test_tie_break_falls_through_to_goals_for_then_name(session):
    comp = _comp(session)
    a = Team(canonical_name="Alpha")
    b = Team(canonical_name="Beta")
    opp1 = Team(canonical_name="Opp1")
    opp2 = Team(canonical_name="Opp2")
    session.add_all([a, b, opp1, opp2])
    session.flush()

    season_start = dt.date(2026, 7, 1)
    # Both Alpha and Beta: 1 win, same GD (+2), but Alpha scored more goals.
    matches = [
        _match(comp, a, opp1, 3, 1, dt.datetime(2026, 8, 1)),  # GD +2, GF 3
        _match(comp, b, opp2, 2, 0, dt.datetime(2026, 8, 1)),  # GD +2, GF 2
    ]
    session.add_all(matches)
    session.commit()

    rows = compute_standings(session, comp.id, season_start)
    names = [r["team"].canonical_name for r in rows if r["team"].canonical_name in ("Alpha", "Beta")]
    assert names.index("Alpha") < names.index("Beta")


def test_partial_season_unequal_matches_played_does_not_crash(session):
    comp = _comp(session)
    a = Team(canonical_name="Alpha")
    b = Team(canonical_name="Beta")
    c = Team(canonical_name="Gamma")
    session.add_all([a, b, c])
    session.flush()

    season_start = dt.date(2026, 7, 1)
    # Alpha has played twice, Gamma only once (partial season in progress).
    matches = [
        _match(comp, a, b, 1, 0, dt.datetime(2026, 8, 1)),
        _match(comp, a, c, 1, 1, dt.datetime(2026, 8, 8)),
    ]
    session.add_all(matches)
    session.commit()

    rows = compute_standings(session, comp.id, season_start)
    by_name = {r["team"].canonical_name: r for r in rows}
    assert by_name["Alpha"]["played"] == 2
    assert by_name["Gamma"]["played"] == 1
    assert by_name["Beta"]["played"] == 1


def test_empty_season_returns_empty_list(session):
    comp = _comp(session)
    rows = compute_standings(session, comp.id, dt.date(2026, 7, 1))
    assert rows == []


def test_matches_before_season_start_excluded(session):
    comp = _comp(session)
    a = Team(canonical_name="Alpha")
    b = Team(canonical_name="Beta")
    session.add_all([a, b])
    session.flush()

    # Last season's match — must not leak into the new season's table.
    session.add(_match(comp, a, b, 5, 0, dt.datetime(2025, 8, 1)))
    session.commit()

    rows = compute_standings(session, comp.id, dt.date(2026, 7, 1))
    assert rows == []

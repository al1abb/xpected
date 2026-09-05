import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.models import Base, Match, Team
from model.form import form_grade


@pytest.fixture()
def session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    s = Session()
    yield s
    s.close()


def _match(comp_id, home_id, away_id, kickoff, home_goals, away_goals):
    return Match(
        competition_id=comp_id,
        utc_kickoff=kickoff,
        status="finished",
        home_team_id=home_id,
        away_team_id=away_id,
        home_goals=home_goals,
        away_goals=away_goals,
        source="test",
    )


def _teams(session):
    a, b = Team(canonical_name="A"), Team(canonical_name="B")
    session.add_all([a, b])
    session.flush()
    return a, b


def test_form_grade_withheld_with_too_few_season_matches(session):
    a, b = _teams(session)
    season_start = dt.datetime(2026, 7, 1)
    for i in range(3):  # below MIN_SEASON_MATCHES_FOR_GRADE
        session.add(_match(1, a.id, b.id, season_start + dt.timedelta(days=i), 1, 0))
    session.commit()

    assert form_grade(session, a.id, as_of=dt.datetime(2026, 8, 1)) is None


def test_form_grade_withheld_when_no_matches_at_all(session):
    a, b = _teams(session)
    assert form_grade(session, a.id) is None


def test_form_grade_normal_when_recent_matches_season_rate(session):
    a, b = _teams(session)
    season_start = dt.datetime(2026, 7, 1)
    # 8 wins in a row -> recent PPG == season PPG exactly -> ratio 1.0 -> "C".
    for i in range(8):
        session.add(_match(1, a.id, b.id, season_start + dt.timedelta(days=i), 1, 0))
    session.commit()

    grade = form_grade(session, a.id, as_of=dt.datetime(2026, 9, 1))
    assert grade["letter"] == "C"
    assert grade["arrow"] == "→"
    assert grade["ratio"] == pytest.approx(1.0)


def test_form_grade_improves_on_a_hot_streak(session):
    a, b = _teams(session)
    season_start = dt.datetime(2026, 7, 1)
    # Poor start: 5 defeats.
    for i in range(5):
        session.add(_match(1, a.id, b.id, season_start + dt.timedelta(days=i), 0, 1))
    # Then a hot streak of 6 straight wins.
    for i in range(6):
        session.add(_match(1, a.id, b.id, season_start + dt.timedelta(days=10 + i), 3, 0))
    session.commit()

    grade = form_grade(session, a.id, as_of=dt.datetime(2026, 9, 1))
    # season PPG is dragged down by the 5 defeats; recent PPG is a perfect 3.0
    # -> ratio well above 1 -> best grade.
    assert grade["letter"] == "A"
    assert grade["arrow"] == "↑"
    assert grade["recent_ppg"] == pytest.approx(3.0)


def test_form_grade_worsens_on_a_cold_streak(session):
    a, b = _teams(session)
    season_start = dt.datetime(2026, 7, 1)
    # Strong start: 6 wins.
    for i in range(6):
        session.add(_match(1, a.id, b.id, season_start + dt.timedelta(days=i), 3, 0))
    # Then a cold streak of 6 straight defeats.
    for i in range(6):
        session.add(_match(1, a.id, b.id, season_start + dt.timedelta(days=10 + i), 0, 1))
    session.commit()

    grade = form_grade(session, a.id, as_of=dt.datetime(2026, 9, 1))
    assert grade["letter"] == "E"
    assert grade["arrow"] == "↓"
    assert grade["recent_ppg"] == pytest.approx(0.0)


def test_form_grade_uses_prior_season_tail_when_new_season_is_young(session):
    """A team 5 games into a new season (all draws, so its season PPG is
    exactly 1.0) still gets a real 'recent' window that reaches back across
    the season boundary into last season's form, rather than being compared
    to an identical subset of itself."""
    a, b = _teams(session)
    season_start = dt.datetime(2026, 7, 1)
    for i in range(5):
        session.add(_match(1, a.id, b.id, season_start + dt.timedelta(days=i), 1, 1))  # all draws
    last_season_start = dt.datetime(2025, 7, 1)
    for i in range(3):
        session.add(_match(1, a.id, b.id, last_season_start + dt.timedelta(days=i), 3, 0))  # wins
    session.commit()

    grade = form_grade(session, a.id, as_of=dt.datetime(2026, 7, 10))
    assert grade is not None
    # recent (last 6, any season) = 5 draws + 1 win from last season's tail,
    # NOT identical to the 5-draw season-only subset -> ratio != 1.0.
    assert grade["ratio"] != pytest.approx(1.0)


def test_form_grade_withheld_on_a_scoreless_season(session):
    """Every match a 0-0 draw -> season PPG is 1.0, not 0 -> this is NOT the
    degenerate all-losses case; only a true 0 season_ppg (never happens with
    draws counting 1pt, but exercised directly here) should short-circuit."""
    a, b = _teams(session)
    season_start = dt.datetime(2026, 7, 1)
    for i in range(6):
        session.add(_match(1, a.id, b.id, season_start + dt.timedelta(days=i), 0, 0))
    session.commit()
    grade = form_grade(session, a.id, as_of=dt.datetime(2026, 9, 1))
    assert grade is not None
    assert grade["season_ppg"] == pytest.approx(1.0)


def test_form_grade_withheld_when_season_ppg_is_zero(session):
    a, b = _teams(session)
    season_start = dt.datetime(2026, 7, 1)
    for i in range(6):  # 6 straight defeats -> season PPG exactly 0
        session.add(_match(1, a.id, b.id, season_start + dt.timedelta(days=i), 0, 1))
    session.commit()
    assert form_grade(session, a.id, as_of=dt.datetime(2026, 9, 1)) is None

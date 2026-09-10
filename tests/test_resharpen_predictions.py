import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.models import Base, Competition, Lineup, Match, ModelRun, Prediction, Team
from model import elo
from model.player_elo import MIN_STARTERS_FOR_LIVE_STRENGTH
from scripts.resharpen_predictions import find_candidates, resharpen


@pytest.fixture()
def session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    s = Session()
    yield s
    s.close()


@pytest.fixture(autouse=True)
def _no_network_clubelo(monkeypatch):
    monkeypatch.setattr(elo.clubelo, "fetch_snapshot", lambda session, on_date: {})


@pytest.fixture(autouse=True)
def _isolated_appearances_db(monkeypatch, tmp_path):
    from ingest.bigballs_history import _connect as connect_appearances
    from model import player_elo

    monkeypatch.setattr(player_elo, "_connect", lambda: connect_appearances(tmp_path / "appearances.sqlite"))


def _setup_match(session, *, lineup_based=False, starters_home=0, starters_away=0):
    comp = Competition(slug="premier-league", name="EPL", country="England", type="league", fd_code="E0")
    session.add(comp)
    session.flush()
    home, away = Team(canonical_name="Home"), Team(canonical_name="Away")
    session.add_all([home, away])
    session.flush()
    kickoff = dt.datetime.utcnow() + dt.timedelta(hours=1)
    match = Match(competition_id=comp.id, utc_kickoff=kickoff, status="scheduled", home_team_id=home.id, away_team_id=away.id, source="test")
    session.add(match)
    session.flush()

    model_run = ModelRun(params={})
    session.add(model_run)
    session.flush()
    session.add(
        Prediction(
            match_id=match.id, model_run_id=model_run.id,
            home_win_prob=0.4, draw_prob=0.3, away_win_prob=0.3,
            lineup_based=lineup_based,
        )
    )
    for i in range(starters_home):
        session.add(Lineup(match_id=match.id, team_id=home.id, player_name=f"H{i}", starter=True, player_id=i + 1))
    for i in range(starters_away):
        session.add(Lineup(match_id=match.id, team_id=away.id, player_name=f"A{i}", starter=True, player_id=100 + i))
    session.commit()
    return match, model_run


def test_find_candidates_empty_when_no_model_run(session):
    model_run, candidates = find_candidates(session)
    assert model_run is None
    assert candidates == []


def test_find_candidates_skips_match_already_lineup_based(session):
    _setup_match(session, lineup_based=True, starters_home=11, starters_away=11)
    _, candidates = find_candidates(session)
    assert candidates == []


def test_find_candidates_skips_match_without_enough_starters(session):
    _setup_match(session, lineup_based=False, starters_home=MIN_STARTERS_FOR_LIVE_STRENGTH - 1, starters_away=11)
    _, candidates = find_candidates(session)
    assert candidates == []


def test_find_candidates_includes_match_with_confirmed_lineup_not_yet_sharpened(session):
    match, _ = _setup_match(session, lineup_based=False, starters_home=11, starters_away=11)
    _, candidates = find_candidates(session)
    assert [m.id for m in candidates] == [match.id]


def test_resharpen_updates_existing_prediction_in_place_without_new_model_run(session):
    match, model_run = _setup_match(session, lineup_based=False, starters_home=11, starters_away=11)
    before_count = session.query(Prediction).count()
    before_run_count = session.query(ModelRun).count()

    updated = resharpen(session, model_run, [match])
    assert updated == 1

    after_count = session.query(Prediction).count()
    after_run_count = session.query(ModelRun).count()
    assert after_count == before_count  # updated in place, not a new row
    assert after_run_count == before_run_count  # no new ModelRun created

    prediction = session.query(Prediction).filter_by(match_id=match.id, model_run_id=model_run.id).one()
    assert prediction.lineup_based is True

"""Tests for ingest/api_football.py's top-scorers/top-assists sync — parsing
a representative API-Football /players/topscorers-shaped response into
PlayerStat rows, resolving teams via existing aliases, and full-replace
behaviour on re-sync."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import ingest.api_football as api_football
from app.models import Base, Competition, PlayerStat, Team, TeamAlias
from ingest.api_football import sync_top_assists, sync_top_scorers


@pytest.fixture()
def session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    s = Session()
    yield s
    s.close()


def _competition(session, af_id=39):
    comp = Competition(
        slug="premier-league",
        name="Premier League",
        country="England",
        type="league",
        af_id=af_id,
        af_name="Premier League",
    )
    session.add(comp)
    session.flush()
    return comp


def _topscorers_response():
    return [
        {
            "player": {"id": 1, "name": "Erling Haaland"},
            "statistics": [{"team": {"id": 50, "name": "Manchester City"}, "goals": {"total": 20, "assists": 3}}],
        },
        {
            "player": {"id": 2, "name": "Mohamed Salah"},
            "statistics": [{"team": {"id": 40, "name": "Liverpool"}, "goals": {"total": 18, "assists": 10}}],
        },
    ]


def test_sync_top_scorers_writes_ranked_rows(session, monkeypatch):
    comp = _competition(session)
    monkeypatch.setattr(api_football, "_get", lambda *a, **k: _topscorers_response())

    written = sync_top_scorers(session, "premier-league", 2026)

    assert written == 2
    rows = (
        session.query(PlayerStat)
        .filter_by(competition_id=comp.id, category="goals")
        .order_by(PlayerStat.rank)
        .all()
    )
    assert [r.player_name for r in rows] == ["Erling Haaland", "Mohamed Salah"]
    assert [r.value for r in rows] == [20, 18]
    assert rows[0].rank == 1
    assert rows[0].season_label == "2026/27"


def test_sync_top_assists_uses_assists_value_not_goals(session, monkeypatch):
    comp = _competition(session)
    monkeypatch.setattr(api_football, "_get", lambda *a, **k: _topscorers_response())

    written = sync_top_assists(session, "premier-league", 2026)

    assert written == 2
    rows = (
        session.query(PlayerStat)
        .filter_by(competition_id=comp.id, category="assists")
        .order_by(PlayerStat.rank)
        .all()
    )
    assert [r.value for r in rows] == [3, 10]  # from goals.assists, not goals.total


def test_sync_resolves_team_via_existing_alias(session, monkeypatch):
    _competition(session)
    team = Team(canonical_name="Manchester City")
    session.add(team)
    session.flush()
    session.add(TeamAlias(team_id=team.id, alias="Manchester City", source="football_data"))
    session.commit()

    monkeypatch.setattr(api_football, "_get", lambda *a, **k: _topscorers_response())
    sync_top_scorers(session, "premier-league", 2026)

    row = session.query(PlayerStat).filter_by(player_name="Erling Haaland").one()
    assert row.team_id == team.id


def test_sync_replaces_stale_rows_on_rerun(session, monkeypatch):
    comp = _competition(session)
    monkeypatch.setattr(api_football, "_get", lambda *a, **k: _topscorers_response())
    sync_top_scorers(session, "premier-league", 2026)

    # Second sync with only one player now in the top 10 — the dropped
    # player's stale row must not linger.
    monkeypatch.setattr(api_football, "_get", lambda *a, **k: _topscorers_response()[:1])
    sync_top_scorers(session, "premier-league", 2026)

    rows = session.query(PlayerStat).filter_by(competition_id=comp.id, category="goals").all()
    assert len(rows) == 1
    assert rows[0].player_name == "Erling Haaland"


def test_sync_returns_none_on_api_error_without_writing_rows(session, monkeypatch):
    _competition(session)

    def _raise(*a, **k):
        raise RuntimeError("season blocked on free plan")

    monkeypatch.setattr(api_football, "_get", _raise)
    result = sync_top_scorers(session, "premier-league", 2026)

    assert result is None
    assert session.query(PlayerStat).count() == 0

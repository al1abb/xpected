import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.config import settings
from app.models import Base, Competition, PlayerStat, SquadPlayer, Team, TeamAlias
from ingest.football_data_org_players import sync_scorers, sync_squads


@pytest.fixture()
def session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    s = Session()
    yield s
    s.close()


@pytest.fixture(autouse=True)
def _has_token(monkeypatch):
    monkeypatch.setattr(settings, "football_data_org_token", "test-token")


@pytest.fixture(autouse=True)
def _no_real_pacing(monkeypatch):
    """sync_squads/sync_scorers go through _paced_fetch_text, which checks
    real on-disk cache age and can sleep to respect football-data.org's
    10 req/min free-tier limit (see its docstring — confirmed live to 429
    without this). Tests fake fetch_text entirely, so pacing decisions based
    on real cache files would make tests slow and dependent on ambient
    filesystem state; force the "always a fresh request, never sleep" path
    instead, deterministically."""
    monkeypatch.setattr("ingest.football_data_org_players.cache_age_hours", lambda *a, **k: None)
    monkeypatch.setattr("ingest.football_data_org_players.time.sleep", lambda *a, **k: None)


def _competition(session, slug="premier-league"):
    comp = Competition(slug=slug, name="Premier League", country="England", type="league")
    session.add(comp)
    session.flush()
    return comp


def _team_with_alias(session, canonical_name, *, alias=None, source="football_data"):
    """resolve_existing_team only matches against TeamAlias rows (see
    ingest/resolve.py's _alias_pool), never a bare Team.canonical_name — in
    production every team always has at least one alias from whichever
    source first created it, so tests need one too."""
    team = Team(canonical_name=canonical_name)
    session.add(team)
    session.flush()
    session.add(TeamAlias(team_id=team.id, alias=alias or canonical_name, source=source))
    session.commit()
    return team


def _teams_payload(*teams):
    return json.dumps({"teams": list(teams)})


def _team(name, squad):
    return {"name": name, "squad": squad}


def _player(id, name, position="Midfield", dob="1998-04-12", nationality="England"):
    return {"id": id, "name": name, "position": position, "dateOfBirth": dob, "nationality": nationality}


# ---------- sync_squads ----------


def test_sync_squads_no_token_returns_empty(session, monkeypatch):
    monkeypatch.setattr(settings, "football_data_org_token", None)
    assert sync_squads(session) == {}


def test_sync_squads_writes_players_for_resolved_team(session, monkeypatch):
    _competition(session)
    team = _team_with_alias(session, "Arsenal")

    payload = _teams_payload(_team("Arsenal FC", [_player(1, "Bukayo Saka"), _player(2, "Declan Rice")]))
    monkeypatch.setattr("ingest.football_data_org_players.fetch_text", lambda *a, **k: payload)

    results = sync_squads(session)
    assert results["premier-league"] == 2

    rows = session.query(SquadPlayer).filter_by(team_id=team.id).all()
    assert {r.name for r in rows} == {"Bukayo Saka", "Declan Rice"}
    assert all(r.date_of_birth is not None for r in rows)


def test_sync_squads_skips_unresolvable_team_without_crashing(session, monkeypatch):
    _competition(session)
    # No matching Team in the DB at all.
    payload = _teams_payload(_team("Some Obscure FC", [_player(1, "Nobody")]))
    monkeypatch.setattr("ingest.football_data_org_players.fetch_text", lambda *a, **k: payload)

    results = sync_squads(session)
    assert results["premier-league"] == 0
    assert session.query(SquadPlayer).count() == 0


def test_sync_squads_replaces_departed_players_on_resync(session, monkeypatch):
    """A player who left the club must not linger after a re-sync — full
    replace per team, same reasoning as the existing scorer sync."""
    _competition(session)
    team = _team_with_alias(session, "Arsenal")

    first = _teams_payload(_team("Arsenal FC", [_player(1, "Old Player")]))
    monkeypatch.setattr("ingest.football_data_org_players.fetch_text", lambda *a, **k: first)
    sync_squads(session)
    assert {r.name for r in session.query(SquadPlayer).filter_by(team_id=team.id)} == {"Old Player"}

    second = _teams_payload(_team("Arsenal FC", [_player(2, "New Signing")]))
    monkeypatch.setattr("ingest.football_data_org_players.fetch_text", lambda *a, **k: second)
    sync_squads(session)
    names = {r.name for r in session.query(SquadPlayer).filter_by(team_id=team.id)}
    assert names == {"New Signing"}  # "Old Player" is gone, not just added-to


def test_sync_squads_empty_squad_from_one_competition_does_not_wipe_another(session, monkeypatch):
    """Regression: a team can appear in more than one covered competition's
    teams list in the SAME sync pass (e.g. a Premier League side that's also
    in the Champions League's 36-team list). Confirmed live: the Champions
    League league phase returns squad=[] for every team before its season
    starts. CREST_COMPETITION_CODES processes 'champions-league' AFTER
    'premier-league' — if that later, empty response were allowed to delete-
    then-not-replace, it would wipe out the real squad the Premier League
    pass had just written for the same team."""
    _competition(session, slug="premier-league")
    _competition(session, slug="champions-league")
    team = _team_with_alias(session, "Arsenal")

    pl_payload = _teams_payload(_team("Arsenal FC", [_player(1, "Bukayo Saka"), _player(2, "Declan Rice")]))
    cl_payload = _teams_payload(_team("Arsenal FC", []))  # source hasn't published CL squads yet

    def _fetch(url, **kwargs):
        return cl_payload if "CL" in url else pl_payload

    monkeypatch.setattr("ingest.football_data_org_players.fetch_text", _fetch)

    results = sync_squads(session)
    assert results["premier-league"] == 2
    assert results["champions-league"] == 0

    names = {r.name for r in session.query(SquadPlayer).filter_by(team_id=team.id)}
    assert names == {"Bukayo Saka", "Declan Rice"}  # NOT wiped by the later empty CL response


def test_sync_squads_handles_fetch_failure_per_competition(session, monkeypatch):
    _competition(session, slug="premier-league")
    _competition(session, slug="la-liga")

    def _boom(url, **kwargs):
        if "PL" in url:
            raise RuntimeError("boom")
        return _teams_payload()

    monkeypatch.setattr("ingest.football_data_org_players.fetch_text", _boom)
    results = sync_squads(session)
    assert results["premier-league"] is None
    assert results["la-liga"] == 0  # the OTHER competition still ran


# ---------- sync_scorers ----------


def _scorers_payload(*scorers):
    return json.dumps({"scorers": list(scorers)})


def _scorer(name, team_name, goals, assists=None):
    return {"player": {"name": name}, "team": {"name": team_name}, "goals": goals, "assists": assists}


def test_sync_scorers_no_token_returns_empty(session, monkeypatch):
    monkeypatch.setattr(settings, "football_data_org_token", None)
    assert sync_scorers(session) == {}


def test_sync_scorers_ranks_goals_and_assists_independently(session, monkeypatch):
    comp = _competition(session)
    _team_with_alias(session, "Arsenal")

    payload = _scorers_payload(
        _scorer("A", "Arsenal FC", goals=10, assists=1),
        _scorer("B", "Arsenal FC", goals=5, assists=8),  # fewer goals, more assists
    )
    monkeypatch.setattr("ingest.football_data_org_players.fetch_text", lambda *a, **k: payload)

    sync_scorers(session)

    goals_table = session.query(PlayerStat).filter_by(competition_id=comp.id, category="goals").order_by(PlayerStat.rank).all()
    assert [r.player_name for r in goals_table] == ["A", "B"]

    assists_table = (
        session.query(PlayerStat).filter_by(competition_id=comp.id, category="assists").order_by(PlayerStat.rank).all()
    )
    # B has more assists than A -> ranked first, despite ranking second on goals.
    assert [r.player_name for r in assists_table] == ["B", "A"]


def test_sync_scorers_excludes_zero_or_missing_assists(session, monkeypatch):
    comp = _competition(session)
    payload = _scorers_payload(
        _scorer("A", "Arsenal FC", goals=3, assists=0),
        _scorer("B", "Arsenal FC", goals=2, assists=None),
    )
    monkeypatch.setattr("ingest.football_data_org_players.fetch_text", lambda *a, **k: payload)

    sync_scorers(session)
    assists_table = session.query(PlayerStat).filter_by(competition_id=comp.id, category="assists").all()
    assert assists_table == []


def test_sync_scorers_replaces_stale_table_on_resync(session, monkeypatch):
    comp = _competition(session)
    first = _scorers_payload(_scorer("Old Leader", "Arsenal FC", goals=20))
    monkeypatch.setattr("ingest.football_data_org_players.fetch_text", lambda *a, **k: first)
    sync_scorers(session)
    assert session.query(PlayerStat).filter_by(competition_id=comp.id, category="goals").count() == 1

    second = _scorers_payload(_scorer("New Leader", "Arsenal FC", goals=1))
    monkeypatch.setattr("ingest.football_data_org_players.fetch_text", lambda *a, **k: second)
    sync_scorers(session)
    rows = session.query(PlayerStat).filter_by(competition_id=comp.id, category="goals").all()
    assert [r.player_name for r in rows] == ["New Leader"]

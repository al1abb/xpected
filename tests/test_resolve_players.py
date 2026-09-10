import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import datetime as dt

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.models import Base, Player, PlayerAlias, SquadPlayer, Team, UnresolvedPlayerAlias
from ingest.resolve_players import resolve_or_create_player, seed_players_from_squads, split_surname_initial


@pytest.fixture()
def session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    s = Session()
    yield s
    s.close()


@pytest.fixture()
def team(session):
    t = Team(canonical_name="Leicester City")
    session.add(t)
    session.flush()
    return t


# ---------- split_surname_initial ----------


def test_split_surname_initial_abbreviated_name():
    assert split_surname_initial("M. Hermansen") == ("hermansen", "m")


def test_split_surname_initial_full_name_matches_abbreviated_pair():
    # Same (surname, initial) as the abbreviated form above — the whole
    # point is that full and abbreviated spellings agree on this key.
    assert split_surname_initial("Mads Hermansen") == ("hermansen", "m")


def test_split_surname_initial_single_word_name_has_no_initial():
    assert split_surname_initial("Neymar") == ("neymar", None)


# ---------- resolve_or_create_player: strong id channel ----------


def test_strong_source_id_resolves_across_different_teams():
    # A transfer: same bbs_player_id, new team_id. The strong-id channel
    # must keep returning the same Player — this is the exact advantage a
    # team-scoped key alone could never give.
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    s = sessionmaker(bind=engine)()
    team_a = Team(canonical_name="Team A")
    team_b = Team(canonical_name="Team B")
    s.add_all([team_a, team_b])
    s.flush()

    p1 = resolve_or_create_player(s, "E. Haaland", "bigballs", team_id=team_a.id, source_player_id="bbs-9001")
    p2 = resolve_or_create_player(s, "E. Haaland", "bigballs", team_id=team_b.id, source_player_id="bbs-9001")

    assert p1.id == p2.id
    assert s.query(Player).count() == 1
    s.close()


def test_repeat_same_team_reuses_player(session, team):
    p1 = resolve_or_create_player(session, "M. Hermansen", "bigballs", team_id=team.id, source_player_id="bbs-1")
    p2 = resolve_or_create_player(session, "M. Hermansen", "bigballs", team_id=team.id, source_player_id="bbs-1")
    assert p1.id == p2.id
    assert session.query(Player).count() == 1


# ---------- resolve_or_create_player: name-based channel ----------


def test_abbreviated_name_resolves_to_existing_full_name_on_same_team(session, team):
    # Full name arrives first (e.g. seeded from SquadPlayer), no strong id.
    full = resolve_or_create_player(session, "Mads Hermansen", "football_data_org", team_id=team.id)
    # Later, an abbreviated historical box-score entry for the same team,
    # also with no id (the 59%-null case this module exists for).
    abbreviated = resolve_or_create_player(session, "M. Hermansen", "bigballs", team_id=team.id)

    assert full.id == abbreviated.id
    assert session.query(Player).count() == 1


def test_same_surname_initial_different_teams_do_not_collide(session):
    team_a = Team(canonical_name="Team A")
    team_b = Team(canonical_name="Team B")
    session.add_all([team_a, team_b])
    session.flush()

    p1 = resolve_or_create_player(session, "M. Silva", "bigballs", team_id=team_a.id)
    p2 = resolve_or_create_player(session, "M. Silva", "bigballs", team_id=team_b.id)

    # Unrelated players on different rosters who merely share initials —
    # must never be merged just because the name string matches.
    assert p1.id != p2.id
    assert session.query(Player).count() == 2


def test_two_same_surname_squadmates_do_not_silently_merge(session, team):
    # Two genuinely different players on the SAME roster sharing a surname
    # and first initial — e.g. two "M. Silva"s. Neither the full names nor
    # the abbreviated one should ever cause an automatic merge.
    resolve_or_create_player(session, "Marcus Silva", "football_data_org", team_id=team.id)
    resolve_or_create_player(session, "Mateus Silva", "football_data_org", team_id=team.id)
    assert session.query(Player).count() == 2

    # A later abbreviated sighting of "M. Silva" for this team is genuinely
    # ambiguous between the two — must be flagged, not guessed, and must
    # not merge into either existing player.
    ambiguous = resolve_or_create_player(session, "M. Silva", "bigballs", team_id=team.id, context="test")

    assert session.query(UnresolvedPlayerAlias).count() == 1
    unresolved = session.query(UnresolvedPlayerAlias).one()
    assert unresolved.raw_name == "M. Silva"
    assert unresolved.team_id == team.id
    # A third Player row is created so the appearance has somewhere to
    # attach — same stance as team resolution on an ambiguous match.
    assert session.query(Player).count() == 3
    assert ambiguous.id not in {p.id for p in session.query(Player).filter(Player.canonical_name.in_(["Marcus Silva", "Mateus Silva"]))}


def test_no_team_id_still_creates_a_player(session):
    # Without a team scope, the name channel has nothing to match against —
    # still must not raise, and should not fabricate an unresolved-alias
    # entry (there's no roster to be ambiguous against).
    player = resolve_or_create_player(session, "Unknown Player", "manual")
    assert player.canonical_name == "Unknown Player"
    assert session.query(UnresolvedPlayerAlias).count() == 0


def test_empty_name_raises(session):
    with pytest.raises(ValueError):
        resolve_or_create_player(session, "   ", "bigballs", team_id=None)


# ---------- seed_players_from_squads ----------


def test_seed_players_from_squads_creates_one_player_per_squad_row(session, team):
    session.add(
        SquadPlayer(
            team_id=team.id,
            name="Mads Hermansen",
            position="Goalkeeper",
            date_of_birth=dt.date(2002, 3, 22),
            fd_person_id=555,
        )
    )
    session.flush()

    created = seed_players_from_squads(session)

    assert created == 1
    player = session.query(Player).one()
    assert player.canonical_name == "Mads Hermansen"
    assert player.date_of_birth == dt.date(2002, 3, 22)
    assert player.normalized_surname == "hermansen"


def test_seed_players_from_squads_is_idempotent(session, team):
    session.add(SquadPlayer(team_id=team.id, name="Mads Hermansen", fd_person_id=555))
    session.flush()

    first = seed_players_from_squads(session)
    second = seed_players_from_squads(session)

    assert first == 1
    assert second == 0
    assert session.query(Player).count() == 1


def test_seeded_full_name_then_abbreviated_lineup_name_resolve_together(session, team):
    session.add(SquadPlayer(team_id=team.id, name="Mads Hermansen", fd_person_id=555))
    session.flush()
    seed_players_from_squads(session)

    seeded = session.query(Player).one()
    lineup_sighting = resolve_or_create_player(session, "M. Hermansen", "bigballs", team_id=team.id)

    assert lineup_sighting.id == seeded.id
    assert session.query(Player).count() == 1
    assert session.query(PlayerAlias).filter_by(source="bigballs").count() == 1

import datetime as dt
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.models import Base, Lineup, Match, PlayerRating, Team
from ingest.bigballs_history import _connect
from model.player_elo import (
    BASE_RATING,
    MIN_STARTERS_FOR_LIVE_STRENGTH,
    SHRINKAGE_APPEARANCES_THRESHOLD,
    SOURCE_INTERNAL,
    _replay_player_elo,
    compute_ear,
    live_team_strength,
    load_persisted_player_ratings,
    persist_player_ratings,
    player_strength,
    prune_player_ratings,
    team_strength,
)


@pytest.fixture()
def session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    s = Session()
    yield s
    s.close()


@pytest.fixture()
def appearances_conn(tmp_path):
    conn = _connect(tmp_path / "appearances.sqlite")
    yield conn
    conn.close()


def _team(session, name):
    t = Team(canonical_name=name)
    session.add(t)
    session.flush()
    return t


def _match(session, home, away, home_goals, away_goals, kickoff):
    m = Match(
        competition_id=1,
        utc_kickoff=kickoff,
        status="finished",
        home_team_id=home.id,
        away_team_id=away.id,
        home_goals=home_goals,
        away_goals=away_goals,
        source="test",
    )
    session.add(m)
    session.flush()
    return m


def _appear(conn, match_id, player_id, team_id, minutes):
    conn.execute(
        "INSERT INTO appearances (match_id, player_id, team_id, minutes, rating) VALUES (?, ?, ?, ?, NULL)",
        (match_id, player_id, team_id, minutes),
    )
    conn.commit()


# ---------- player_strength / team_strength ----------


def test_player_strength_new_player_is_exactly_base_rating():
    assert player_strength(999, ratings={}, appearance_counts={}) == BASE_RATING


def test_player_strength_established_player_uses_own_rating_in_full():
    ratings, counts = {1: 1700.0}, {1: SHRINKAGE_APPEARANCES_THRESHOLD}
    assert player_strength(1, ratings, counts) == pytest.approx(1700.0)


def test_player_strength_thin_history_shrinks_toward_base_rating():
    ratings, counts = {1: 1700.0}, {1: SHRINKAGE_APPEARANCES_THRESHOLD // 2}
    strength = player_strength(1, ratings, counts)
    assert BASE_RATING < strength < 1700.0


def test_team_strength_none_when_no_usable_weight():
    assert team_strength([(1, 0), (2, 0)], ratings={}, appearance_counts={}) is None


def test_team_strength_is_weighted_mean():
    ratings = {1: 1600.0, 2: 1400.0}
    counts = {1: SHRINKAGE_APPEARANCES_THRESHOLD, 2: SHRINKAGE_APPEARANCES_THRESHOLD}
    assert team_strength([(1, 90), (2, 90)], ratings, counts) == pytest.approx(1500.0)
    assert team_strength([(1, 90), (2, 10)], ratings, counts) > 1500.0


# ---------- _replay_player_elo ----------


def test_replay_winner_players_gain_loser_players_lose(session, appearances_conn):
    home, away = _team(session, "Home"), _team(session, "Away")
    match = _match(session, home, away, home_goals=2, away_goals=0, kickoff=dt.datetime(2026, 1, 1))
    _appear(appearances_conn, match.id, 1, home.id, 90)
    _appear(appearances_conn, match.id, 2, away.id, 90)

    ratings, counts = _replay_player_elo(session, appearances_conn)
    assert ratings[1] > BASE_RATING
    assert ratings[2] < BASE_RATING
    assert counts[1] == counts[2] == 1


def test_replay_no_usable_data_leaves_both_sides_unrated(session, appearances_conn):
    """A match with only one side's appearances recorded (the other team not
    yet covered by this source) must not be scored at all -- there's no
    opponent strength to compute an expectation against."""
    home, away = _team(session, "Home"), _team(session, "Away")
    match = _match(session, home, away, home_goals=2, away_goals=0, kickoff=dt.datetime(2026, 1, 1))
    _appear(appearances_conn, match.id, 1, home.id, 90)
    # away side: no appearance rows at all

    ratings, counts = _replay_player_elo(session, appearances_conn)
    assert ratings == {}
    assert counts == {}


def test_replay_upset_win_moves_rating_more_than_expected_win():
    """Mirrors test_model.py's test_bigger_margin_moves_rating_more: build a
    fresh in-memory scenario per call, comparing the WINNER's rating gain in
    a final 1-0 match once one side has already been established as clearly
    stronger. An upset win should move the winner's rating more than an
    expected win of the same scoreline."""

    def winner_gain(favourite_wins: bool) -> float:
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)
        s = sessionmaker(bind=engine)()
        conn = sqlite3.connect(":memory:")
        conn.execute(
            "CREATE TABLE appearances (match_id INTEGER, player_id INTEGER, team_id INTEGER, "
            "minutes INTEGER, rating REAL, PRIMARY KEY (match_id, player_id))"
        )

        fav, dog, filler1, filler2 = (
            _team(s, "Fav"),
            _team(s, "Dog"),
            _team(s, "Filler1"),
            _team(s, "Filler2"),
        )
        FAV_PLAYER, DOG_PLAYER = 1, 2
        base = dt.datetime(2025, 1, 1)
        for i in range(4):
            m1 = _match(s, fav, filler1, home_goals=4, away_goals=0, kickoff=base + dt.timedelta(days=i))
            _appear(conn, m1.id, FAV_PLAYER, fav.id, 90)
            _appear(conn, m1.id, 100 + i, filler1.id, 90)

            m2 = _match(s, filler2, dog, home_goals=4, away_goals=0, kickoff=base + dt.timedelta(days=i))
            _appear(conn, m2.id, 200 + i, filler2.id, 90)
            _appear(conn, m2.id, DOG_PLAYER, dog.id, 90)

        ratings_before, _ = _replay_player_elo(s, conn)
        pre_rating = ratings_before[FAV_PLAYER if favourite_wins else DOG_PLAYER]
        assert ratings_before[FAV_PLAYER] > ratings_before[DOG_PLAYER]  # sanity: the gap is real

        final_kickoff = base + dt.timedelta(days=100)
        if favourite_wins:
            final = _match(s, fav, dog, home_goals=1, away_goals=0, kickoff=final_kickoff)
        else:
            final = _match(s, dog, fav, home_goals=1, away_goals=0, kickoff=final_kickoff)
        _appear(conn, final.id, FAV_PLAYER, fav.id, 90)
        _appear(conn, final.id, DOG_PLAYER, dog.id, 90)

        ratings_after, _ = _replay_player_elo(s, conn)
        winner_id = FAV_PLAYER if favourite_wins else DOG_PLAYER
        gain = ratings_after[winner_id] - pre_rating
        s.close()
        conn.close()
        return gain

    assert winner_gain(favourite_wins=False) > winner_gain(favourite_wins=True)


def test_replay_more_minutes_moves_rating_more(session, appearances_conn):
    """Two independent single-match scenarios, identical result and margin,
    differing only in how many minutes the winning player was on for."""

    def gain_for_minutes(minutes: int) -> float:
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)
        s = sessionmaker(bind=engine)()
        conn = sqlite3.connect(":memory:")
        conn.execute(
            "CREATE TABLE appearances (match_id INTEGER, player_id INTEGER, team_id INTEGER, "
            "minutes INTEGER, rating REAL, PRIMARY KEY (match_id, player_id))"
        )
        home, away = _team(s, "Home"), _team(s, "Away")
        match = _match(s, home, away, home_goals=2, away_goals=0, kickoff=dt.datetime(2026, 1, 1))
        _appear(conn, match.id, 1, home.id, minutes)
        _appear(conn, match.id, 2, away.id, 90)
        ratings, _ = _replay_player_elo(s, conn)
        s.close()
        conn.close()
        return ratings[1] - BASE_RATING

    assert gain_for_minutes(90) > gain_for_minutes(10)


def test_replay_rating_transfers_across_a_club_change(session, appearances_conn):
    """The core advantage over team-level Elo: a player's rating is keyed on
    the player alone, so it carries forward unchanged in identity (only
    updated by performance) when they appear for a different team."""
    club_a, club_b, opponent = _team(session, "Club A"), _team(session, "Club B"), _team(session, "Opponent")
    PLAYER = 1

    match1 = _match(session, club_a, opponent, home_goals=1, away_goals=0, kickoff=dt.datetime(2025, 1, 1))
    _appear(appearances_conn, match1.id, PLAYER, club_a.id, 90)
    _appear(appearances_conn, match1.id, 50, opponent.id, 90)

    ratings_after_a, counts_after_a = _replay_player_elo(session, appearances_conn)
    rating_at_club_a = ratings_after_a[PLAYER]
    assert rating_at_club_a != BASE_RATING  # actually moved from the match at Club A

    # Same player, now appearing for Club B (a transfer), in a later match.
    match2 = _match(session, club_b, opponent, home_goals=1, away_goals=0, kickoff=dt.datetime(2025, 6, 1))
    _appear(appearances_conn, match2.id, PLAYER, club_b.id, 90)
    _appear(appearances_conn, match2.id, 51, opponent.id, 90)

    ratings_after_b, counts_after_b = _replay_player_elo(session, appearances_conn)
    # The rating continued updating from where it left off at Club A -- it
    # was not reset to BASE_RATING just because the team_id changed.
    assert counts_after_b[PLAYER] == 2
    assert ratings_after_b[PLAYER] != rating_at_club_a  # it moved again (second win)
    assert ratings_after_b[PLAYER] > BASE_RATING  # still net positive after two wins


def test_replay_no_lookahead(session, appearances_conn):
    home, away = _team(session, "Home"), _team(session, "Away")
    early = _match(session, home, away, home_goals=1, away_goals=0, kickoff=dt.datetime(2026, 1, 1))
    _appear(appearances_conn, early.id, 1, home.id, 90)
    _appear(appearances_conn, early.id, 2, away.id, 90)

    late = _match(session, home, away, home_goals=5, away_goals=0, kickoff=dt.datetime(2026, 6, 1))
    _appear(appearances_conn, late.id, 1, home.id, 90)
    _appear(appearances_conn, late.id, 2, away.id, 90)

    ratings_full, counts_full = _replay_player_elo(session, appearances_conn)
    ratings_cutoff, counts_cutoff = _replay_player_elo(session, appearances_conn, as_of=dt.datetime(2026, 3, 1))

    assert counts_cutoff[1] == 1  # only the early match counted
    assert ratings_cutoff[1] != ratings_full[1]  # the later blowout hasn't been applied yet


def test_replay_exclude_match_id_omits_that_match(session, appearances_conn):
    home, away = _team(session, "Home"), _team(session, "Away")
    match = _match(session, home, away, home_goals=1, away_goals=0, kickoff=dt.datetime(2026, 1, 1))
    _appear(appearances_conn, match.id, 1, home.id, 90)
    _appear(appearances_conn, match.id, 2, away.id, 90)

    ratings, counts = _replay_player_elo(session, appearances_conn, exclude_match_id=match.id)
    assert ratings == {}
    assert counts == {}


# ---------- live_team_strength ----------


def _lineup_row(session, match_id, team_id, player_id, *, starter=True):
    row = Lineup(
        match_id=match_id,
        team_id=team_id,
        player_name=f"Player {player_id}",
        starter=starter,
        player_id=player_id,
    )
    session.add(row)
    session.flush()
    return row


def test_live_team_strength_none_below_min_starters(session):
    home, away = _team(session, "Home"), _team(session, "Away")
    match = _match(session, home, away, home_goals=0, away_goals=0, kickoff=dt.datetime(2026, 1, 1))
    for i in range(MIN_STARTERS_FOR_LIVE_STRENGTH - 1):
        _lineup_row(session, match.id, home.id, player_id=i + 1)

    assert live_team_strength(session, match.id, home.id, ratings={}, appearance_counts={}) is None


def test_live_team_strength_computed_at_min_starters(session):
    home, away = _team(session, "Home"), _team(session, "Away")
    match = _match(session, home, away, home_goals=0, away_goals=0, kickoff=dt.datetime(2026, 1, 1))
    for i in range(MIN_STARTERS_FOR_LIVE_STRENGTH):
        _lineup_row(session, match.id, home.id, player_id=i + 1)

    strength = live_team_strength(session, match.id, home.id, ratings={}, appearance_counts={})
    assert strength == pytest.approx(BASE_RATING)  # no ratings on file yet -> every starter defaults to BASE_RATING


def test_live_team_strength_ignores_bench_and_unresolved_names(session):
    home, away = _team(session, "Home"), _team(session, "Away")
    match = _match(session, home, away, home_goals=0, away_goals=0, kickoff=dt.datetime(2026, 1, 1))
    for i in range(MIN_STARTERS_FOR_LIVE_STRENGTH):
        _lineup_row(session, match.id, home.id, player_id=i + 1)
    _lineup_row(session, match.id, home.id, player_id=999, starter=False)  # bench: shouldn't count
    session.add(Lineup(match_id=match.id, team_id=home.id, player_name="Unresolved", starter=True, player_id=None))
    session.flush()

    ratings = {1: 1700.0}
    strength = live_team_strength(session, match.id, home.id, ratings=ratings, appearance_counts={1: SHRINKAGE_APPEARANCES_THRESHOLD})
    # Only the MIN_STARTERS_FOR_LIVE_STRENGTH resolved starters count -> player
    # 1's 1700 pulls the mean above BASE_RATING, proving bench/unresolved rows
    # were excluded rather than silently dragging it back toward BASE_RATING.
    assert strength > BASE_RATING


def test_live_team_strength_reflects_stronger_starters(session):
    home, away = _team(session, "Strong XI"), _team(session, "Weak XI")
    match = _match(session, home, away, home_goals=0, away_goals=0, kickoff=dt.datetime(2026, 1, 1))
    for i in range(MIN_STARTERS_FOR_LIVE_STRENGTH):
        _lineup_row(session, match.id, home.id, player_id=i + 1)
        _lineup_row(session, match.id, away.id, player_id=100 + i)

    ratings = {i + 1: 1800.0 for i in range(MIN_STARTERS_FOR_LIVE_STRENGTH)}
    ratings.update({100 + i: 1300.0 for i in range(MIN_STARTERS_FOR_LIVE_STRENGTH)})
    counts = {pid: SHRINKAGE_APPEARANCES_THRESHOLD for pid in ratings}

    home_strength = live_team_strength(session, match.id, home.id, ratings, counts)
    away_strength = live_team_strength(session, match.id, away.id, ratings, counts)
    assert home_strength > away_strength


# ---------- persistence ----------


def test_persist_player_ratings_is_idempotent_per_day(session):
    day = dt.date(2026, 1, 1)
    persist_player_ratings(session, {1: 1600.0}, {1: 5}, as_of=day)
    persist_player_ratings(session, {1: 1650.0}, {1: 6}, as_of=day)

    rows = session.query(PlayerRating).filter_by(player_id=1, as_of_date=day, source=SOURCE_INTERNAL).all()
    assert len(rows) == 1
    assert rows[0].elo == pytest.approx(1650.0)
    assert rows[0].appearances == 6


def test_load_persisted_player_ratings_returns_latest_snapshot(session):
    persist_player_ratings(session, {1: 1500.0}, {1: 1}, as_of=dt.date(2026, 1, 1))
    persist_player_ratings(session, {1: 1520.0}, {1: 2}, as_of=dt.date(2026, 1, 5))

    ratings, counts = load_persisted_player_ratings(session)
    assert ratings[1] == pytest.approx(1520.0)
    assert counts[1] == 2


def test_load_persisted_player_ratings_empty_database_returns_empty_dicts(session):
    ratings, counts = load_persisted_player_ratings(session)
    assert ratings == {}
    assert counts == {}


def test_prune_player_ratings_keeps_only_latest_snapshot_per_player(session):
    persist_player_ratings(session, {1: 1500.0}, {1: 1}, as_of=dt.date(2026, 1, 1))
    persist_player_ratings(session, {1: 1520.0}, {1: 2}, as_of=dt.date(2026, 1, 5))
    persist_player_ratings(session, {2: 1400.0}, {2: 3}, as_of=dt.date(2026, 1, 5))

    deleted = prune_player_ratings(session)
    assert deleted == 1  # only player 1's 2026-01-01 row was stale

    remaining = session.query(PlayerRating).all()
    assert len(remaining) == 2
    by_player = {row.player_id: row for row in remaining}
    assert by_player[1].as_of_date == dt.date(2026, 1, 5)
    assert by_player[1].elo == pytest.approx(1520.0)
    assert by_player[2].as_of_date == dt.date(2026, 1, 5)


# ---------- compute_ear ----------


def test_compute_ear_zero_for_median_player_in_a_three_player_bucket():
    ratings = {1: 1600.0, 2: 1500.0, 3: 1400.0}
    position = {1: "M", 2: "M", 3: "M"}
    competition = {1: 10, 2: 10, 3: 10}
    ear = compute_ear(ratings, position, competition)
    assert ear[2] == pytest.approx(0.0)  # the median IS the replacement level
    assert ear[1] > 0
    assert ear[3] < 0


def test_compute_ear_separates_positions_within_the_same_competition():
    """A weak defender shouldn't be judged against strong midfielders just
    because they share a competition -- the replacement level is per
    position, not just per competition."""
    ratings = {1: 1700.0, 2: 1650.0, 3: 1300.0}
    position = {1: "M", 2: "M", 3: "D"}
    competition = {1: 10, 2: 10, 3: 10}
    ear = compute_ear(ratings, position, competition)
    # Player 3 is the ONLY defender in this competition -- their own rating
    # IS the (single-player) replacement level for that bucket.
    assert ear[3] == pytest.approx(0.0)


def test_compute_ear_separates_competitions_at_the_same_position():
    """The same raw rating should score differently as EAR depending on
    which competition's positional baseline it's compared against."""
    ratings = {1: 1600.0, 2: 1400.0, 3: 1600.0}
    position = {1: "M", 2: "M", 3: "M"}
    competition = {1: 10, 2: 10, 3: 20}  # player 3 is alone in competition 20
    ear = compute_ear(ratings, position, competition)
    assert ear[3] == pytest.approx(0.0)  # sole player in their bucket -> IS the baseline
    assert ear[1] != ear[3]  # same raw rating, different bucket -> different EAR


def test_compute_ear_omits_players_missing_position_or_competition():
    ratings = {1: 1600.0, 2: 1500.0, 3: 1400.0}
    position = {1: "M", 2: None, 3: "M"}  # player 2 has no known position
    competition = {1: 10, 2: 10, 3: None}  # player 3 has no known current competition
    ear = compute_ear(ratings, position, competition)
    assert set(ear) == {1}
    assert 2 not in ear
    assert 3 not in ear

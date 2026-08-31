"""Tests for the pure/DB-backed helpers added to app/main.py this session:
group_by_day's enriched fields + group_by_month, the live/pending match
state estimate, and the team-search query's dedupe behavior."""

import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.main import (
    ESTIMATED_MATCH_DURATION,
    SOON_WINDOW,
    _live_state,
    group_by_day,
    group_by_month,
    search_teams_query,
    with_today_marker,
)
from app.models import Base, Competition, Match, Team, TeamAlias


def _card(kickoff):
    m = Match(
        competition_id=1,
        utc_kickoff=kickoff,
        status="scheduled",
        home_team_id=1,
        away_team_id=2,
        source="test",
    )
    return {"match": m}


def test_group_by_day_enriches_fields_and_sorts_within_day():
    today = dt.datetime.utcnow().date()
    tomorrow = today + dt.timedelta(days=1)
    cards = [
        _card(dt.datetime.combine(tomorrow, dt.time(20, 0))),
        _card(dt.datetime.combine(tomorrow, dt.time(12, 0))),
        _card(dt.datetime.combine(today, dt.time(15, 0))),
    ]

    days = group_by_day(cards)

    assert [d["date"] for d in days] == [today, tomorrow]
    today_group = days[0]
    assert today_group["iso"] == today.isoformat()
    assert today_group["anchor"] == f"d-{today.isoformat()}"
    assert today_group["label"] == "Today"
    assert today_group["is_today"] is True
    assert today_group["days_ahead"] == 0
    assert today_group["count"] == 1

    tomorrow_group = days[1]
    assert tomorrow_group["label"] == "Tomorrow"
    assert tomorrow_group["is_today"] is False
    # within a day, cards sort chronologically ascending regardless of input order
    assert [c["match"].utc_kickoff.hour for c in tomorrow_group["cards"]] == [12, 20]


def test_group_by_day_open_threshold_beyond_always_open_groups():
    today = dt.datetime.utcnow().date()
    # 4 distinct near-term days (exceeds _ALWAYS_OPEN_GROUPS=3), then one
    # within the 13-day horizon, then one far beyond it.
    early = [today + dt.timedelta(days=i) for i in (1, 2, 3, 4)]
    within_horizon = today + dt.timedelta(days=10)
    far = today + dt.timedelta(days=90)
    dates = early + [within_horizon, far]
    cards = [_card(dt.datetime.combine(d, dt.time(12, 0))) for d in dates]

    days = group_by_day(cards)
    by_date = {d["date"]: d for d in days}

    # First 3 matchdays are always open regardless of how far out they are.
    assert by_date[early[0]]["open"] is True
    assert by_date[early[2]]["open"] is True
    # 4th matchday (index 3, past _ALWAYS_OPEN_GROUPS) is still within the
    # 13-day horizon, so it's open on that basis instead.
    assert by_date[early[3]]["open"] is True
    assert by_date[within_horizon]["open"] is True
    # Genuinely far out and past the always-open count: closed by default.
    assert by_date[far]["open"] is False


def test_group_by_day_reverse_orders_days_newest_first():
    today = dt.datetime.utcnow().date()
    d1, d2, d3 = today, today + dt.timedelta(days=1), today + dt.timedelta(days=2)
    cards = [_card(dt.datetime.combine(d, dt.time(12, 0))) for d in (d1, d2, d3)]

    ascending = group_by_day(cards)
    descending = group_by_day(cards, reverse=True)

    assert [d["date"] for d in ascending] == [d1, d2, d3]
    assert [d["date"] for d in descending] == [d3, d2, d1]


def test_group_by_month_buckets_and_counts():
    days = [
        {"month_key": (2026, 9), "month_label": "September 2026", "month_short": "Sep", "count": 2},
        {"month_key": (2026, 9), "month_label": "September 2026", "month_short": "Sep", "count": 1},
        {"month_key": (2026, 10), "month_label": "October 2026", "month_short": "Oct", "count": 3},
    ]

    months = group_by_month(days)

    assert [m["key"] for m in months] == [(2026, 9), (2026, 10)]
    assert months[0]["count"] == 3
    assert len(months[0]["days"]) == 2
    assert months[1]["count"] == 3
    assert len(months[1]["days"]) == 1


def _match(status, kickoff):
    return Match(
        competition_id=1,
        utc_kickoff=kickoff,
        status=status,
        home_team_id=1,
        away_team_id=2,
        source="test",
    )


def test_live_state_none_for_distant_future_and_finished():
    now = dt.datetime(2026, 9, 1, 12, 0)
    distant_future = _match("scheduled", now + SOON_WINDOW + dt.timedelta(minutes=1))
    finished = _match("finished", now - dt.timedelta(hours=5))
    assert _live_state(distant_future, now) is None
    assert _live_state(finished, now) is None


def test_live_state_soon_within_soon_window():
    now = dt.datetime(2026, 9, 1, 12, 0)
    about_to_kick_off = _match("scheduled", now + dt.timedelta(minutes=1))
    edge_of_window = _match("scheduled", now + SOON_WINDOW)
    assert _live_state(about_to_kick_off, now) == "soon"
    assert _live_state(edge_of_window, now) == "soon"


def test_live_state_live_within_estimated_duration():
    now = dt.datetime(2026, 9, 1, 12, 0)
    just_kicked_off = _match("scheduled", now - dt.timedelta(minutes=10))
    near_end = _match("scheduled", now - ESTIMATED_MATCH_DURATION + dt.timedelta(minutes=1))
    assert _live_state(just_kicked_off, now) == "live"
    assert _live_state(near_end, now) == "live"


def test_live_state_pending_after_estimated_duration():
    now = dt.datetime(2026, 9, 1, 12, 0)
    overdue = _match("scheduled", now - ESTIMATED_MATCH_DURATION - dt.timedelta(minutes=1))
    assert _live_state(overdue, now) == "pending"


@pytest.fixture()
def session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    s = Session()
    yield s
    s.close()


def test_search_teams_query_dedupes_multi_alias_matches(session):
    team = Team(canonical_name="Manchester United")
    session.add(team)
    session.flush()
    session.add_all(
        [
            TeamAlias(team_id=team.id, alias="Manchester United", source="football_data"),
            TeamAlias(team_id=team.id, alias="Man Utd", source="api_football"),
            TeamAlias(team_id=team.id, alias="Man United", source="clubelo"),
        ]
    )
    session.commit()

    results = search_teams_query(session, "man")

    matches = [t for t in results if t.id == team.id]
    assert len(matches) == 1


def test_search_teams_query_matches_alias_not_just_canonical_name(session):
    team = Team(canonical_name="Qarabag FK")
    session.add(team)
    session.flush()
    session.add(TeamAlias(team_id=team.id, alias="Karabakh Agdam", source="clubelo"))
    session.commit()

    results = search_teams_query(session, "karabakh")

    assert any(t.id == team.id for t in results)


def test_search_teams_query_below_min_chars_returns_empty(session):
    session.add(Team(canonical_name="Arsenal"))
    session.commit()
    assert search_teams_query(session, "a") == []


def test_search_teams_query_prefix_match_ranks_first(session):
    session.add_all([Team(canonical_name="Manchester United"), Team(canonical_name="AFC Mansfield")])
    session.commit()

    results = search_teams_query(session, "man")

    assert results[0].canonical_name == "Manchester United"


def test_with_today_marker_inserts_when_today_has_no_match():
    today = dt.datetime.utcnow().date()
    future = today + dt.timedelta(days=9)
    days = group_by_day([_card(dt.datetime.combine(future, dt.time(12, 0)))])
    assert not any(d["date"] == today for d in days)  # sanity: today truly absent

    with_marker = with_today_marker(days, today)

    assert [d["date"] for d in with_marker] == [today, future]
    marker = with_marker[0]
    assert marker["is_today"] is True
    assert marker["anchor"] is None
    assert marker["count"] == 0


def test_with_today_marker_noop_when_today_already_present():
    today = dt.datetime.utcnow().date()
    days = group_by_day([_card(dt.datetime.combine(today, dt.time(12, 0)))])

    result = with_today_marker(days, today)

    assert len(result) == 1
    assert result[0]["anchor"] is not None  # the real matchday, not a marker
    assert result[0]["count"] == 1

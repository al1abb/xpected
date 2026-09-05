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
    _prune_live_match_state,
    _resolve_live_clock,
    group_by_day,
    group_by_month,
    search_teams_query,
    with_today_marker,
)
from app.models import Base, Competition, LiveMatchState, Match, Team, TeamAlias


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


def test_resolve_live_clock_prefers_real_minute_when_source_provides_one(session):
    clock, estimated = _resolve_live_clock(
        session, 1, "IN_PLAY", 51, dt.datetime.utcnow() - dt.timedelta(minutes=51), dt.datetime.utcnow()
    )
    assert clock == "51'"
    assert estimated is False


def test_resolve_live_clock_paused_returns_ht_and_records_state(session):
    now = dt.datetime.utcnow()
    kickoff = now - dt.timedelta(minutes=48)
    clock, estimated = _resolve_live_clock(session, 1, "PAUSED", None, kickoff, now)
    session.commit()

    assert clock == "HT"
    assert estimated is False
    state = session.get(LiveMatchState, 1)
    assert state.last_status == "PAUSED"
    assert state.paused_since == now
    assert state.total_paused_minutes == 0.0


def test_resolve_live_clock_counts_real_playing_time_across_half_time(session):
    kickoff = dt.datetime.utcnow() - dt.timedelta(hours=2)
    paused_at = kickoff + dt.timedelta(minutes=45)
    _resolve_live_clock(session, 1, "PAUSED", None, kickoff, paused_at)
    session.commit()

    resumed_at = paused_at + dt.timedelta(minutes=15)
    clock, estimated = _resolve_live_clock(session, 1, "IN_PLAY", None, kickoff, resumed_at)
    session.commit()
    assert clock == "~45'"
    assert estimated is True

    six_minutes_later = resumed_at + dt.timedelta(minutes=6)
    clock, estimated = _resolve_live_clock(session, 1, "IN_PLAY", None, kickoff, six_minutes_later)
    assert clock == "~51'"
    assert estimated is True


def test_resolve_live_clock_accounts_for_a_second_mid_match_stoppage(session):
    # Not every PAUSED is half-time — a later weather/medical/VAR stoppage
    # also freezes the real clock, and must add to the paused ledger same
    # as half-time does, or the estimate inflates by however long it lasted.
    kickoff = dt.datetime.utcnow() - dt.timedelta(hours=2)
    ht_start = kickoff + dt.timedelta(minutes=45)
    _resolve_live_clock(session, 1, "PAUSED", None, kickoff, ht_start)
    session.commit()
    ht_end = ht_start + dt.timedelta(minutes=15)
    _resolve_live_clock(session, 1, "IN_PLAY", None, kickoff, ht_end)
    session.commit()

    second_stoppage_start = ht_end + dt.timedelta(minutes=20)  # real minute ~65
    clock, _ = _resolve_live_clock(session, 1, "PAUSED", None, kickoff, second_stoppage_start)
    session.commit()
    assert clock == "HT"  # labelled the same as any pause — see model docstring

    second_stoppage_end = second_stoppage_start + dt.timedelta(minutes=25)
    clock, estimated = _resolve_live_clock(session, 1, "IN_PLAY", None, kickoff, second_stoppage_end)
    session.commit()
    assert clock == "~65'"  # real playing time recovered despite the extra 25min stoppage
    assert estimated is True

    six_more_minutes = second_stoppage_end + dt.timedelta(minutes=6)
    clock, estimated = _resolve_live_clock(session, 1, "IN_PLAY", None, kickoff, six_more_minutes)
    assert clock == "~71'"
    assert estimated is True


def test_resolve_live_clock_falls_back_to_wall_clock_when_never_observed_half_time(session):
    # First-ever poll for this match already lands well past 45' — we missed
    # the real transition (e.g. app cold-started mid-second-half) — so this
    # is a best-effort constant-gap guess, not a real anchor.
    kickoff = dt.datetime.utcnow() - dt.timedelta(minutes=85)
    now = dt.datetime.utcnow()
    clock, estimated = _resolve_live_clock(session, 1, "IN_PLAY", None, kickoff, now)
    assert clock == "~70'"
    assert estimated is True


def test_resolve_live_clock_shows_added_time_past_90_instead_of_freezing(session):
    kickoff = dt.datetime.utcnow() - dt.timedelta(hours=2)
    paused_at = kickoff + dt.timedelta(minutes=48)
    _resolve_live_clock(session, 1, "PAUSED", None, kickoff, paused_at)
    session.commit()

    resumed_at = paused_at + dt.timedelta(minutes=15)
    _resolve_live_clock(session, 1, "IN_PLAY", None, kickoff, resumed_at)
    session.commit()

    # 48min first half (real, including its own stoppage) + 52min into the
    # second half = 100 real playing minutes, still within the give-up
    # ceiling (115min since kickoff).
    deep_stoppage = resumed_at + dt.timedelta(minutes=52)
    clock, estimated = _resolve_live_clock(session, 1, "IN_PLAY", None, kickoff, deep_stoppage)
    assert clock == "~90+10'"
    assert estimated is True


def test_resolve_live_clock_gives_up_once_a_match_has_run_implausibly_long(session):
    # Regression test: a real match (Man City vs Coventry) sat reporting
    # IN_PLAY well after full time before the source's FINISHED flip caught
    # up, and the old constant-gap formula projected "90+20" forever in the
    # meantime. Past ESTIMATED_MATCH_DURATION since kickoff, stop guessing.
    kickoff = dt.datetime.utcnow() - dt.timedelta(hours=3)
    paused_at = kickoff + dt.timedelta(minutes=48)
    _resolve_live_clock(session, 1, "PAUSED", None, kickoff, paused_at)
    session.commit()

    resumed_at = paused_at + dt.timedelta(minutes=15)
    _resolve_live_clock(session, 1, "IN_PLAY", None, kickoff, resumed_at)
    session.commit()

    stuck = kickoff + ESTIMATED_MATCH_DURATION + dt.timedelta(minutes=1)
    clock, estimated = _resolve_live_clock(session, 1, "IN_PLAY", None, kickoff, stuck)
    assert clock is None
    assert estimated is False


def test_resolve_live_clock_first_half_counts_up_from_kickoff(session):
    kickoff = dt.datetime.utcnow() - dt.timedelta(minutes=20)
    now = dt.datetime.utcnow()
    clock, estimated = _resolve_live_clock(session, 1, "IN_PLAY", None, kickoff, now)
    assert clock == "~20'"
    assert estimated is True


def test_resolve_live_clock_unresolvable_status_returns_none(session):
    clock, estimated = _resolve_live_clock(session, 1, "SUSPENDED", None, dt.datetime.utcnow(), dt.datetime.utcnow())
    assert clock is None
    assert estimated is False


def test_prune_live_match_state_removes_only_stale_rows(session):
    now = dt.datetime.utcnow()
    session.add_all(
        [
            LiveMatchState(match_id=1, last_status="IN_PLAY", updated_at=now - dt.timedelta(hours=5)),
            LiveMatchState(match_id=2, last_status="IN_PLAY", updated_at=now - dt.timedelta(minutes=5)),
        ]
    )
    session.commit()

    _prune_live_match_state(session, now)
    session.commit()

    remaining = {s.match_id for s in session.query(LiveMatchState).all()}
    assert remaining == {2}

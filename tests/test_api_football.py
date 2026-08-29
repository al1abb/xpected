import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.models import Competition
from ingest.api_football import _parse_fixture, _status


def _fixture(status_short="FT", round_name="Group Stage - 3", home_goals=2, away_goals=1):
    return {
        "fixture": {
            "id": 12345,
            "date": "2026-09-16T19:00:00+00:00",
            "status": {"short": status_short},
        },
        "league": {"round": round_name},
        "teams": {"home": {"name": "Qarabag FK"}, "away": {"name": "Real Madrid"}},
        "goals": {"home": home_goals, "away": away_goals},
        "score": {"halftime": {"home": 1, "away": 0}},
    }


def _competition(**kwargs):
    defaults = dict(slug="champions-league", name="UCL", country="Europe", type="uefa_cup")
    defaults.update(kwargs)
    return Competition(**defaults)


def test_status_mapping():
    assert _status("FT") == "finished"
    assert _status("AET") == "finished"
    assert _status("PEN") == "finished"
    assert _status("NS") == "scheduled"
    assert _status("PST") == "postponed"
    assert _status("CANC") == "cancelled"
    assert _status("1H") == "scheduled"  # in-play collapses to scheduled — no live view in v1


def test_parse_fixture_basic_fields():
    row = _parse_fixture(_fixture(), _competition())
    assert row["af_fixture_id"] == 12345
    assert row["status"] == "finished"
    assert row["home_goals"] == 2
    assert row["away_goals"] == 1
    assert row["home_goals_ht"] == 1
    assert row["away_goals_ht"] == 0
    assert row["kickoff"].isoformat() == "2026-09-16T19:00:00"  # UTC, tz-naive


def test_neutral_venue_only_for_uefa_final_not_semi_or_quarter():
    final = _parse_fixture(_fixture(round_name="Final"), _competition(type="uefa_cup"))
    semi = _parse_fixture(_fixture(round_name="Semi-finals"), _competition(type="uefa_cup"))
    quarter = _parse_fixture(_fixture(round_name="Quarter-finals"), _competition(type="uefa_cup"))
    league_final_round = _parse_fixture(_fixture(round_name="Final"), _competition(type="league"))

    assert final["neutral_venue"] is True
    assert semi["neutral_venue"] is False
    assert quarter["neutral_venue"] is False
    assert league_final_round["neutral_venue"] is False


def test_parse_fixture_captures_team_logos():
    fixture = _fixture()
    fixture["teams"]["home"]["logo"] = "https://media.api-sports.io/football/teams/1.png"
    fixture["teams"]["away"]["logo"] = "https://media.api-sports.io/football/teams/2.png"
    row = _parse_fixture(fixture, _competition())
    assert row["home_logo"] == "https://media.api-sports.io/football/teams/1.png"
    assert row["away_logo"] == "https://media.api-sports.io/football/teams/2.png"


def test_parse_fixture_handles_missing_logo_field():
    row = _parse_fixture(_fixture(), _competition())
    assert row["home_logo"] is None
    assert row["away_logo"] is None


def test_missing_halftime_score_does_not_crash():
    fixture = _fixture()
    fixture["score"]["halftime"] = None
    row = _parse_fixture(fixture, _competition())
    assert row["home_goals_ht"] is None
    assert row["away_goals_ht"] is None

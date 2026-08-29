import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from ingest.fixturedownload import parse_feed


def test_parse_feed_upcoming_and_finished():
    feed = json.dumps(
        [
            {
                "MatchNumber": 1,
                "RoundNumber": 1,
                "DateUtc": "2026-09-08 16:45:00Z",
                "Location": "Jan Breydelstadion",
                "HomeTeam": "Club Brugge",
                "AwayTeam": "Aston Villa",
                "Group": None,
                "HomeTeamScore": None,
                "AwayTeamScore": None,
                "Winner": "",
            },
            {
                "MatchNumber": 2,
                "RoundNumber": 1,
                "DateUtc": "2026-09-08 19:00:00Z",
                "Location": "Somewhere",
                "HomeTeam": "Real Madrid",
                "AwayTeam": "Marseille",
                "Group": None,
                "HomeTeamScore": 3,
                "AwayTeamScore": 1,
                "Winner": "Real Madrid",
            },
        ]
    )
    rows = parse_feed(feed)
    assert len(rows) == 2
    assert rows[0]["status"] == "scheduled"
    assert rows[0]["home_goals"] is None
    assert rows[1]["status"] == "finished"
    assert rows[1]["home_goals"] == 3
    assert rows[1]["kickoff"].isoformat() == "2026-09-08T19:00:00"


def test_parse_feed_skips_rows_with_bad_date():
    feed = json.dumps([{"DateUtc": "not-a-date", "HomeTeam": "A", "AwayTeam": "B"}])
    assert parse_feed(feed) == []


def test_footballdata_csv_raises_on_missing_columns():
    from ingest.footballdata_csv import SchemaDriftError, parse_results_csv

    with pytest.raises(SchemaDriftError):
        parse_results_csv("SomeColumn,OtherColumn\nfoo,bar\n")


def test_footballdata_csv_fixtures_raises_on_missing_columns():
    from ingest.footballdata_csv import SchemaDriftError, parse_fixtures_csv

    with pytest.raises(SchemaDriftError):
        parse_fixtures_csv("SomeColumn,OtherColumn\nfoo,bar\n", {"E0"})


def test_footballdata_csv_accepts_real_shaped_header():
    from ingest.footballdata_csv import parse_results_csv

    rows = parse_results_csv("Div,Date,Time,HomeTeam,AwayTeam,FTHG,FTAG\nE0,16/08/2024,20:00,Man United,Fulham,1,0\n")
    assert len(rows) == 1


def test_footballdata_csv_fixtures_captures_odds():
    """Regression: upcoming fixtures carry live pre-match odds in the same
    file, but parsing used to drop them entirely — meaning the match page's
    market-odds comparison would only ever show up on already-finished games."""
    from ingest.footballdata_csv import parse_fixtures_csv

    header = "Div,Date,Time,HomeTeam,AwayTeam,AvgH,AvgD,AvgA"
    row = "E0,28/08/2026,19:45,Arsenal,Chelsea,2.1,3.4,3.2"
    rows = parse_fixtures_csv(f"{header}\n{row}\n", {"E0"})
    assert len(rows) == 1
    assert rows[0]["odds_home"] == pytest.approx(2.1)
    assert rows[0]["odds_draw"] == pytest.approx(3.4)
    assert rows[0]["odds_away"] == pytest.approx(3.2)

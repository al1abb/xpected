import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import ingest.bigballs_history as history
import ingest.cache as cache
from app.models import Base, Competition
from ingest.bigballs_history import MAX_REFUSALS_PER_DATE, _connect, backfill_competition
from ingest.cache import FetchError, fetch_text

DATES = ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04", "2020-01-05"]


@pytest.fixture()
def session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    s = sessionmaker(bind=engine)()
    s.add(Competition(slug="bundesliga", name="Bundesliga", country="Germany", type="league"))
    s.commit()
    yield s
    s.close()


@pytest.fixture()
def conn(tmp_path):
    c = _connect(tmp_path / "appearances.sqlite")
    yield c
    c.close()


def _fake_api(monkeypatch, status_by_date: dict[str, int]):
    """Every /v1/matches listing succeeds with no matches, except the dates
    in `status_by_date`, which fail with that HTTP status."""
    calls = []

    def fake_fetch(path, *, params=None, max_age_hours):
        date = params["date"]
        calls.append(date)
        if date in status_by_date:
            raise FetchError("https://example/v1/matches", status_by_date[date], '{"error":"history_not_included"}')
        return {"data": []}

    monkeypatch.setattr(history, "_paced_fetch_json", fake_fetch)
    return calls


def _done(conn):
    return {d for (d,) in conn.execute("SELECT date FROM backfilled_dates")}


def test_refused_date_is_skipped_and_later_dates_still_backfilled(session, conn, monkeypatch):
    _fake_api(monkeypatch, {"2020-01-02": 403})

    result = backfill_competition(session, conn, "bundesliga", DATES)

    assert _done(conn) == set(DATES) - {"2020-01-02"}
    assert result["dates_refused"] == 1
    assert not result["rate_limited"]
    assert "history_not_included" in result["errors"][0]


def test_date_is_given_up_after_repeated_refusals(session, conn, monkeypatch):
    calls = _fake_api(monkeypatch, {"2020-01-02": 403})
    for _ in range(MAX_REFUSALS_PER_DATE):
        backfill_competition(session, conn, "bundesliga", DATES)
    calls.clear()

    result = backfill_competition(session, conn, "bundesliga", DATES)

    assert calls == []
    assert result["dates_given_up"] == 1
    assert result["requests_spent"] == 0


def test_league_stops_after_consecutive_refusals(session, conn, monkeypatch):
    calls = _fake_api(monkeypatch, {d: 403 for d in DATES})

    result = backfill_competition(session, conn, "bundesliga", DATES)

    assert len(calls) == history.MAX_CONSECUTIVE_REFUSALS_PER_RUN
    assert result["dates_refused"] == history.MAX_CONSECUTIVE_REFUSALS_PER_RUN


def test_rate_limit_stops_the_run_without_recording_a_refusal(session, conn, monkeypatch):
    calls = _fake_api(monkeypatch, {"2020-01-02": 429})

    result = backfill_competition(session, conn, "bundesliga", DATES)

    assert calls == ["2020-01-01", "2020-01-02"]
    assert result["rate_limited"]
    assert _done(conn) == {"2020-01-01"}
    assert conn.execute("SELECT COUNT(*) FROM refused_dates").fetchone()[0] == 0


def test_fetch_error_carries_the_servers_explanation(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "RAW_DATA_DIR", tmp_path)
    monkeypatch.setattr(
        cache.httpx,
        "get",
        lambda *a, **kw: httpx.Response(403, text='{"error": "history_not_included"}'),
    )

    with pytest.raises(FetchError) as excinfo:
        fetch_text("https://api.example/v1/matches", subdir="t")

    assert excinfo.value.status_code == 403
    assert "HTTP 403" in str(excinfo.value)
    assert "history_not_included" in str(excinfo.value)

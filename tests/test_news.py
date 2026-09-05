import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.config import NEWS_RETENTION_DAYS
from app.models import Base, Competition, NewsItem, Team
from ingest.news import _team_candidates, match_teams, parse_rss, sync_news


@pytest.fixture()
def session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    s = Session()
    yield s
    s.close()


# ---------- parse_rss ----------

RSS_ITEM = """<item>
  <title>{title}</title>
  <link>{link}</link>
  <pubDate>{pub_date}</pubDate>
</item>"""

RSS_WRAPPER = "<rss version=\"2.0\"><channel>{items}</channel></rss>"


def _feed(*items):
    return RSS_WRAPPER.format(items="".join(items))


def test_parse_rss_extracts_title_link_and_date():
    xml = _feed(RSS_ITEM.format(title="Test Headline", link="https://example.com/a", pub_date="Fri, 04 Sep 2026 22:37:16 GMT"))
    items = parse_rss(xml)
    assert len(items) == 1
    assert items[0]["title"] == "Test Headline"
    assert items[0]["url"] == "https://example.com/a"
    assert items[0]["published_at"] == dt.datetime(2026, 9, 4, 22, 37, 16)


def test_parse_rss_handles_cdata_title():
    xml = _feed(
        "<item><title><![CDATA[CDATA Headline]]></title><link>https://example.com/b</link></item>"
    )
    items = parse_rss(xml)
    assert items[0]["title"] == "CDATA Headline"


def test_parse_rss_skips_item_missing_title_or_link():
    xml = _feed(
        "<item><link>https://example.com/no-title</link></item>",
        RSS_ITEM.format(title="Has Both", link="https://example.com/c", pub_date="Fri, 04 Sep 2026 10:00:00 GMT"),
        "<item><title>No Link</title></item>",
    )
    items = parse_rss(xml)
    assert len(items) == 1
    assert items[0]["title"] == "Has Both"


def test_parse_rss_missing_pubdate_is_none_not_a_crash():
    xml = _feed("<item><title>No Date</title><link>https://example.com/d</link></item>")
    items = parse_rss(xml)
    assert items[0]["published_at"] is None


def test_parse_rss_malformed_xml_returns_empty_not_raises():
    assert parse_rss("<not><valid xml") == []


def test_parse_rss_empty_feed_returns_empty():
    assert parse_rss(_feed()) == []


# ---------- match_teams ----------


def _team(session, name):
    t = Team(canonical_name=name)
    session.add(t)
    session.flush()
    return t


def test_match_teams_finds_full_name_in_headline(session):
    liverpool = _team(session, "Liverpool")
    candidates = _team_candidates(session)
    assert match_teams("Isak finally arrives as Gakpo proves value to Liverpool", candidates) == [liverpool.id]


def test_match_teams_disambiguates_two_clubs_sharing_a_city_word(session):
    """The exact scenario the plan flagged: 'Milan' alone must not resolve to
    either club, but headlines using each club's full disambiguated name
    must resolve correctly and independently."""
    ac = _team(session, "AC Milan")
    inter = _team(session, "Inter Milan")
    candidates = _team_candidates(session)

    assert set(match_teams("AC Milan sign new coach ahead of derby with Inter Milan", candidates)) == {ac.id, inter.id}
    assert match_teams("Milan derby preview: what to expect", candidates) == []
    assert match_teams("AC Milan crash out of the cup", candidates) == [ac.id]


def test_match_teams_word_boundary_rejects_partial_word_match(session):
    arsenal = _team(session, "Arsenal")
    candidates = _team_candidates(session)
    # "Arsenalist" contains "Arsenal" as a substring but is not a word-boundary match.
    assert match_teams("The Arsenalist fan podcast previews the weekend", candidates) == []
    assert match_teams("Arsenal win again", candidates) == [arsenal.id]


def test_match_teams_no_match_returns_empty(session):
    _team(session, "Liverpool")
    candidates = _team_candidates(session)
    assert match_teams("Completely unrelated headline about tennis", candidates) == []


def test_match_teams_case_insensitive(session):
    liverpool = _team(session, "Liverpool")
    candidates = _team_candidates(session)
    assert match_teams("LIVERPOOL win the league", candidates) == [liverpool.id]


# ---------- sync_news ----------


def _competition(session, slug="premier-league"):
    comp = Competition(slug=slug, name="Premier League", country="England", type="league")
    session.add(comp)
    session.flush()
    return comp


def _patch_feeds(monkeypatch, feeds, fetch_map):
    monkeypatch.setattr("ingest.news.NEWS_FEEDS", feeds)
    monkeypatch.setattr("ingest.news.fetch_text", lambda url, **kwargs: fetch_map[url])


def test_sync_news_writes_items_and_tags_competition(session, monkeypatch):
    comp = _competition(session)
    liverpool = _team(session, "Liverpool")

    feeds = [{"url": "https://feed.test/pl", "source": "test", "competition_slug": "premier-league"}]
    payload = _feed(
        RSS_ITEM.format(title="Liverpool win big", link="https://example.com/1", pub_date="Fri, 04 Sep 2026 10:00:00 GMT")
    )
    _patch_feeds(monkeypatch, feeds, {"https://feed.test/pl": payload})

    result = sync_news(session)
    assert result["written"] == 1

    item = session.query(NewsItem).one()
    assert item.competition_id == comp.id
    assert [t.id for t in item.teams] == [liverpool.id]


def test_sync_news_dedupes_same_url_across_feeds(session, monkeypatch):
    """The same article legitimately appears in multiple feeds — must be
    stored once, not once per feed it was seen in."""
    feeds = [
        {"url": "https://feed.test/a", "source": "a", "competition_slug": None},
        {"url": "https://feed.test/b", "source": "b", "competition_slug": None},
    ]
    payload = _feed(
        RSS_ITEM.format(title="Same story everywhere", link="https://example.com/dupe", pub_date="Fri, 04 Sep 2026 10:00:00 GMT")
    )
    _patch_feeds(monkeypatch, feeds, {"https://feed.test/a": payload, "https://feed.test/b": payload})

    result = sync_news(session)
    assert result["written"] == 1
    assert session.query(NewsItem).count() == 1


def test_sync_news_skips_already_stored_url_on_resync(session, monkeypatch):
    feeds = [{"url": "https://feed.test/pl", "source": "test", "competition_slug": None}]
    payload = _feed(
        RSS_ITEM.format(title="Old story", link="https://example.com/old", pub_date="Fri, 04 Sep 2026 10:00:00 GMT")
    )
    _patch_feeds(monkeypatch, feeds, {"https://feed.test/pl": payload})
    sync_news(session)
    assert session.query(NewsItem).count() == 1

    result = sync_news(session)  # same feed content again
    assert result["written"] == 0
    assert session.query(NewsItem).count() == 1


def test_sync_news_prunes_items_older_than_retention_window(session, monkeypatch):
    stale = NewsItem(
        url="https://example.com/stale",
        title="Old news",
        source="test",
        fetched_at=dt.datetime.utcnow() - dt.timedelta(days=NEWS_RETENTION_DAYS + 1),
    )
    fresh = NewsItem(url="https://example.com/fresh", title="Fresh news", source="test", fetched_at=dt.datetime.utcnow())
    session.add_all([stale, fresh])
    session.commit()

    monkeypatch.setattr("ingest.news.NEWS_FEEDS", [])
    result = sync_news(session)
    assert result["pruned"] == 1
    remaining = {r.url for r in session.query(NewsItem).all()}
    assert remaining == {"https://example.com/fresh"}


def test_sync_news_continues_after_one_feed_fails(session, monkeypatch):
    feeds = [
        {"url": "https://feed.test/broken", "source": "broken", "competition_slug": None},
        {"url": "https://feed.test/ok", "source": "ok", "competition_slug": None},
    ]
    payload = _feed(
        RSS_ITEM.format(title="Still works", link="https://example.com/ok", pub_date="Fri, 04 Sep 2026 10:00:00 GMT")
    )

    def _fetch(url, **kwargs):
        if "broken" in url:
            raise RuntimeError("boom")
        return payload

    monkeypatch.setattr("ingest.news.NEWS_FEEDS", feeds)
    monkeypatch.setattr("ingest.news.fetch_text", _fetch)

    result = sync_news(session)
    assert result["written"] == 1
    assert len(result["errors"]) == 1
    assert session.query(NewsItem).count() == 1

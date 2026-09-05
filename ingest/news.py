"""Football news ingest — RSS feeds only, parsed with the stdlib
(`xml.etree.ElementTree` + `email.utils` for RFC 822 dates), no new
dependency. Feed list lives in app.config.NEWS_FEEDS, hand-verified live
during the Sept 2026 research pass (see its comment for what was tried and
rejected).

Ingested during scripts/refresh.py only — a match/team page never makes a
network call to fetch this, same rule ClubElo was moved off the request path
for last round. Items are pruned after NEWS_RETENTION_DAYS so this table
stays bounded, same reasoning as the prediction-pruning work.

Team matching: a headline is checked against every team's CANONICAL name
(not the wider, noisier alias universe) as a single whole-word/phrase match,
case-insensitive and accent-insensitive. This is deliberately narrower than
ingest/resolve.py's fuzzy matcher — those team names are already the
disambiguated form ("AC Milan", "Inter Milan", not bare "Milan"), so a
whole-phrase match can't cross-hit between them the way a bare token like
"Milan" could. A headline that only says "Milan derby" without either club's
full name simply matches neither, which is the safe failure direction.
"""

from __future__ import annotations

import datetime as dt
import re
import unicodedata
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime

from sqlalchemy.orm import Session

from app.config import NEWS_FEEDS, NEWS_RETENTION_DAYS
from app.models import Competition, IngestLog, NewsItem, Team
from ingest.cache import fetch_text

SOURCE = "news"


def _strip_accents(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    return "".join(c for c in text if not unicodedata.combining(c))


def _parse_pub_date(raw: str | None) -> dt.datetime | None:
    if not raw:
        return None
    try:
        parsed = parsedate_to_datetime(raw)
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(dt.timezone.utc).replace(tzinfo=None)
    return parsed


def parse_rss(xml_text: str) -> list[dict]:
    """Every `<item>` in an RSS 2.0 feed -> {title, url, published_at}.
    Skips silently anything missing a title or a link — a malformed single
    item must not take down the whole feed's ingest."""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return []

    items = []
    for item in root.iter("item"):
        title_el = item.find("title")
        link_el = item.find("link")
        if title_el is None or link_el is None or not (title_el.text and link_el.text):
            continue
        pub_el = item.find("pubDate")
        items.append(
            {
                "title": title_el.text.strip(),
                "url": link_el.text.strip(),
                "published_at": _parse_pub_date(pub_el.text if pub_el is not None else None),
            }
        )
    return items


def _team_candidates(session: Session) -> list[tuple[re.Pattern, int]]:
    """One compiled whole-phrase regex per team, longest name first so a
    longer match (rare, but e.g. a reserve-side naming collision) wins over
    a shorter one contained within it."""
    teams = session.query(Team).filter(Team.canonical_name.isnot(None)).all()
    candidates = []
    for team in teams:
        name = _strip_accents(team.canonical_name)
        pattern = re.compile(r"(?<![A-Za-z0-9])" + re.escape(name) + r"(?![A-Za-z0-9])", re.IGNORECASE)
        candidates.append((pattern, team.id, len(name)))
    candidates.sort(key=lambda c: c[2], reverse=True)
    return [(pattern, team_id) for pattern, team_id, _ in candidates]


def match_teams(title: str, candidates: list[tuple[re.Pattern, int]]) -> list[int]:
    title = _strip_accents(title)
    return [team_id for pattern, team_id in candidates if pattern.search(title)]


def sync_news(session: Session) -> dict:
    """Fetch every feed in NEWS_FEEDS, write new items (deduped on url — the
    same article legitimately appears in several feeds), then prune anything
    older than NEWS_RETENTION_DAYS. Fails soft per feed, same pattern as
    every other ingest step in scripts/refresh.py."""
    started = dt.datetime.utcnow()
    candidates = _team_candidates(session)
    competition_by_slug = {c.slug: c.id for c in session.query(Competition).all()}
    existing_urls = {row[0] for row in session.query(NewsItem.url).all()}

    written = 0
    errors = []
    for feed in NEWS_FEEDS:
        try:
            text = fetch_text(feed["url"], subdir="news", max_age_hours=1)
        except RuntimeError as exc:
            errors.append(f"{feed['url']}: {exc}")
            continue

        competition_id = competition_by_slug.get(feed["competition_slug"]) if feed["competition_slug"] else None
        for entry in parse_rss(text):
            if entry["url"] in existing_urls:
                continue
            existing_urls.add(entry["url"])  # same article can appear in >1 feed in this same pass

            news_item = NewsItem(
                url=entry["url"],
                title=entry["title"],
                source=feed["source"],
                competition_id=competition_id,
                published_at=entry["published_at"],
                fetched_at=dt.datetime.utcnow(),
            )
            team_ids = match_teams(entry["title"], candidates)
            if team_ids:
                news_item.teams = session.query(Team).filter(Team.id.in_(team_ids)).all()
            session.add(news_item)
            written += 1

    cutoff = dt.datetime.utcnow() - dt.timedelta(days=NEWS_RETENTION_DAYS)
    pruned = session.query(NewsItem).filter(NewsItem.fetched_at < cutoff).delete(synchronize_session=False)

    session.add(
        IngestLog(
            source=SOURCE,
            started_at=started,
            finished_at=dt.datetime.utcnow(),
            status="ok" if not errors else "partial",
            rows_ingested=written,
            message=f"pruned={pruned}" + (f"; errors={errors}" if errors else ""),
        )
    )
    session.commit()
    return {"written": written, "pruned": pruned, "errors": errors}

"""Shared disk-cache + backoff HTTP fetch for the free (unauthenticated) sources.

football-data.co.uk returned HTTP 429 during development against this exact
project, so caching and backoff here are load-bearing, not defensive
boilerplate: re-running ingest without them will get the app's IP throttled.
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path

import httpx

from app.config import RAW_DATA_DIR

DEFAULT_MAX_AGE_HOURS = 12
RETRY_STATUSES = {429, 502, 503, 504}
MAX_RETRIES = 4


def _cache_path(url: str, subdir: str) -> Path:
    digest = hashlib.sha256(url.encode()).hexdigest()[:16]
    safe_name = url.rsplit("/", 1)[-1] or "index"
    safe_name = "".join(c for c in safe_name if c.isalnum() or c in ".-_")
    out_dir = RAW_DATA_DIR / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    return Path(out_dir) / f"{digest}_{safe_name}"


def fetch_text(
    url: str,
    *,
    subdir: str,
    max_age_hours: float = DEFAULT_MAX_AGE_HOURS,
    force: bool = False,
    headers: dict | None = None,
    params: dict | None = None,
) -> str:
    """GET url, transparently cached to disk. Retries with exponential backoff on
    429/5xx. Returns decoded text (utf-8-sig, so a leading BOM never leaks into data).
    """
    full_url = str(httpx.URL(url, params=params)) if params else url
    path = _cache_path(full_url, subdir)

    if not force and path.exists():
        age_hours = (time.time() - path.stat().st_mtime) / 3600
        if age_hours < max_age_hours:
            return path.read_text(encoding="utf-8-sig")

    last_error: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = httpx.get(url, params=params, headers=headers, timeout=30, follow_redirects=True)
        except httpx.TransportError as exc:
            last_error = exc
            time.sleep(2**attempt)
            continue

        if resp.status_code in RETRY_STATUSES:
            last_error = RuntimeError(f"{url} -> HTTP {resp.status_code}")
            time.sleep(2**attempt * 2)
            continue

        if resp.status_code != 200:
            if path.exists():
                # Serve stale cache rather than fail the whole ingest run over one bad fetch.
                return path.read_text(encoding="utf-8-sig")
            raise RuntimeError(f"{url} -> HTTP {resp.status_code}")

        resp.encoding = "utf-8-sig"
        text = resp.text
        path.write_text(text, encoding="utf-8-sig")
        return text

    if path.exists():
        return path.read_text(encoding="utf-8-sig")
    raise RuntimeError(f"Failed to fetch {url} after {MAX_RETRIES} attempts: {last_error}")


def cache_age_hours(url: str, subdir: str, *, params: dict | None = None) -> float | None:
    full_url = str(httpx.URL(url, params=params)) if params else url
    path = _cache_path(full_url, subdir)
    if not path.exists():
        return None
    return (time.time() - path.stat().st_mtime) / 3600


def read_cached_text(url: str, subdir: str, *, params: dict | None = None) -> str | None:
    """Read whatever is on disk regardless of age, without ever attempting a
    network call. Used when the daily request budget is exhausted: serving a
    stale response is strictly better than serving nothing."""
    full_url = str(httpx.URL(url, params=params)) if params else url
    path = _cache_path(full_url, subdir)
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8-sig")

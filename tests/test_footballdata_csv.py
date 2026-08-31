"""Tests for ingest/footballdata_csv.py's date/time parsing — specifically the
UK-local-to-UTC conversion (see _parse_date's docstring for the BST bug this
guards against)."""

import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ingest.footballdata_csv import _parse_date


def test_parse_date_converts_bst_to_utc():
    # 15:00 UK local in August (BST, UTC+1) -> 14:00 UTC.
    assert _parse_date("31/08/2026", "15:00") == dt.datetime(2026, 8, 31, 14, 0)


def test_parse_date_gmt_matches_utc():
    # 15:00 UK local in January (GMT, UTC+0) -> unchanged.
    assert _parse_date("15/01/2026", "15:00") == dt.datetime(2026, 1, 15, 15, 0)


def test_parse_date_defaults_time_when_missing():
    assert _parse_date("15/01/2026", "") == dt.datetime(2026, 1, 15, 15, 0)


def test_parse_date_returns_none_for_missing_date():
    assert _parse_date("", "15:00") is None

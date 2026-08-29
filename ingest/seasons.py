"""Season-year math shared by football-data.co.uk (code like '2627') and
API-Football (int like 2026), both for the same 2026/27 season. Centralised so
season rollover (phase 7) only has to be correct in one place.
"""

from __future__ import annotations

import datetime as dt


def current_season_start_year(today: dt.date | None = None) -> int:
    """European club season runs Jul/Aug -> May/Jun. Before July, we're still
    in the season that started the previous calendar year."""
    today = today or dt.date.today()
    return today.year if today.month >= 7 else today.year - 1


def fd_season_code(start_year: int) -> str:
    return f"{str(start_year)[2:]}{str(start_year + 1)[2:]}"


def fd_season_codes_back(n: int, today: dt.date | None = None) -> list[str]:
    start = current_season_start_year(today)
    return [fd_season_code(start - i) for i in range(n)]


def af_seasons_back(n: int, today: dt.date | None = None) -> list[int]:
    start = current_season_start_year(today)
    return [start - i for i in range(n)]

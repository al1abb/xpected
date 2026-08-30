"""Manual corrections for crests that are wrong at the source, not wrong
because of a linking/dedup bug. Every regular crest writer (api_football.py,
football_data_org.py) is guarded `if team.logo_url is None` — fill-gaps-only,
so nothing else can ever correct an already-set bad URL. This is the
deliberate escape hatch: one dict entry + a re-run of
scripts/apply_logo_overrides.py.
"""

from __future__ import annotations

TEAM_LOGO_OVERRIDES: dict[str, str] = {
    # football-data.org's crest for this club (id 10233) is wrong at the
    # source — verified against API-Football's independently-sourced crest
    # for the same real club in the Azerbaijan Premyer Liqa cache.
    "Sabah FA": "https://media.api-sports.io/football/teams/13976.png",
}

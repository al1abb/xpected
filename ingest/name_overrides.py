"""Manual corrections for a team's canonical display name when the source
that first created it used a wrong or ambiguous name. `Team.canonical_name`
is only ever set once, at row creation (see
ingest/resolve.py::get_or_create_team) — never overwritten by later syncs —
so a bad name from the ingest source persists forever unless corrected here.
This is the escape hatch, mirroring ingest/logo_overrides.py: one dict entry
(wrong name -> right name) + a re-run of scripts/apply_name_overrides.py.
"""

from __future__ import annotations

TEAM_NAME_OVERRIDES: dict[str, str] = {
    # API-Football names this Azerbaijan Premyer Liqa club (id 13976) "Sabah
    # FA". The real club's official name is "Sabah FK" (Azerbaijani: Sabah
    # Futbol Klubu) — confirmed by football-data.org's own alias for the same
    # team (see TeamAlias source=football_data_org). "Sabah FA" isn't just a
    # cosmetic slip: it's the actual name of an unrelated Malaysian Super
    # League club, so the wrong name collides with a real different club's
    # real name. The crest itself is already correctly sourced from the
    # Azerbaijan cache (see logo_overrides.py) — this fixes the name only.
    "Sabah FA": "Sabah FK",
}

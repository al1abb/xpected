"""Apply ingest/logo_overrides.py's manual crest corrections. Unlike the
regular crest syncs (fill-gaps-only, `if team.logo_url is None`), this
unconditionally overwrites — that's the point, since it exists specifically
to fix an already-set wrong URL.

Run this *before* scripts/apply_name_overrides.py on a freshly rebuilt DB:
this is keyed by the source's original (wrong) name, so renaming first would
make this lookup miss."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.db import SessionLocal, init_db
from app.models import Team
from ingest.logo_overrides import TEAM_LOGO_OVERRIDES


def main() -> None:
    init_db()
    session = SessionLocal()
    try:
        applied = 0
        for canonical_name, logo_url in TEAM_LOGO_OVERRIDES.items():
            team = session.query(Team).filter_by(canonical_name=canonical_name).one_or_none()
            if team is None:
                print(f"Skipped (no such team): {canonical_name}")
                continue
            team.logo_url = logo_url
            applied += 1
            print(f"Set logo for {canonical_name}: {logo_url}")
        session.commit()
        print(f"Applied {applied}/{len(TEAM_LOGO_OVERRIDES)} overrides.")
    finally:
        session.close()


if __name__ == "__main__":
    main()

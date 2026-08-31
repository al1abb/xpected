"""Apply ingest/name_overrides.py's manual canonical-name corrections.

Run this *before* scripts/apply_logo_overrides.py on a freshly rebuilt DB:
logo overrides are keyed by the source's original (wrong) name, so renaming
first would make that lookup miss."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.db import SessionLocal, init_db
from app.models import Team
from ingest.name_overrides import TEAM_NAME_OVERRIDES


def main() -> None:
    init_db()
    session = SessionLocal()
    try:
        applied = 0
        for wrong_name, right_name in TEAM_NAME_OVERRIDES.items():
            team = session.query(Team).filter_by(canonical_name=wrong_name).one_or_none()
            if team is None:
                print(f"Skipped (no such team): {wrong_name}")
                continue
            team.canonical_name = right_name
            applied += 1
            print(f"Renamed {wrong_name!r} -> {right_name!r}")
        session.commit()
        print(f"Applied {applied}/{len(TEAM_NAME_OVERRIDES)} overrides.")
    finally:
        session.close()


if __name__ == "__main__":
    main()

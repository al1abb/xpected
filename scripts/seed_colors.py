"""Populate primary/secondary colors for every team that doesn't have one yet.
Idempotent — never overwrites a color that's already set (curated or manually
edited), so re-running after adding new curated entries only fills gaps."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.colors import team_colors
from app.db import SessionLocal, init_db
from app.models import Team


def main() -> None:
    init_db()
    session = SessionLocal()
    try:
        updated = 0
        for team in session.query(Team).filter(Team.primary_color.is_(None)).all():
            primary, secondary = team_colors(team.canonical_name)
            team.primary_color = primary
            team.secondary_color = secondary
            updated += 1
        session.commit()
        print(f"Colors set for {updated} teams.")
    finally:
        session.close()


if __name__ == "__main__":
    main()

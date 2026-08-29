"""Seed the competitions table from app.config.COMPETITIONS. Idempotent — safe to re-run."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import COMPETITIONS
from app.db import SessionLocal, init_db
from app.models import Competition


def seed_competitions() -> None:
    init_db()
    session = SessionLocal()
    try:
        created, updated = 0, 0
        for c in COMPETITIONS:
            existing = session.query(Competition).filter_by(slug=c["slug"]).one_or_none()
            if existing is None:
                session.add(Competition(**c))
                created += 1
            else:
                for key, value in c.items():
                    setattr(existing, key, value)
                updated += 1
        session.commit()
        print(f"Competitions seeded: {created} created, {updated} updated, {len(COMPETITIONS)} total.")
    finally:
        session.close()


if __name__ == "__main__":
    seed_competitions()

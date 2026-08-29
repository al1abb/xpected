"""Fit the model against all finished matches and predict every currently
scheduled fixture. Run after each ingest refresh (see phase 7)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.db import SessionLocal, init_db
from model.predict import generate_predictions


def main() -> None:
    init_db()
    session = SessionLocal()
    try:
        count = generate_predictions(session, notes="manual run via scripts/fit.py")
        print(f"Predictions generated: {count}")
    finally:
        session.close()


if __name__ == "__main__":
    main()

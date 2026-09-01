"""Prune superseded prediction rows.

model/predict.py's generate_predictions creates a new ModelRun on every call
and writes one Prediction per *currently scheduled* match. Nothing has ever
deleted them, and the scheduled pool is the whole rest of the season (~2,800
fixtures out to next May, not just this week's), so each daily refresh added
~2,800 rows — about 1.15 MB/day, ~420 MB/year, growing without bound. Since
data/app.db is committed on every refresh, that cost compounds in git too.

Retention keeps exactly what the app and the accuracy pages actually read:

  1. Every prediction belonging to the newest ModelRun — what the site
     currently displays (app/main.py's model-run lookups).
  2. Per match, the most recent prediction made BEFORE that match kicked off —
     the genuinely out-of-sample record that model/backtest.py's
     _latest_pre_kickoff_predictions scores, and that a finished match's page
     shows. This is what makes the accuracy history permanent, so pruning must
     never touch it.

Everything else is a prediction that was superseded before its match was
played and is displayed nowhere. Deleting it changes no page and no metric.

Growth after this is bounded by real fixtures rather than by refresh count:
roughly the scheduled pool, plus one retained row per finished match.

Usage:
  python scripts/prune_predictions.py --dry-run   # report only, no writes
  python scripts/prune_predictions.py             # prune, then VACUUM
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import text
from sqlalchemy.orm import Session

from app.db import SessionLocal, engine
from app.models import Match, ModelRun, Prediction

# SQLite's default host-parameter ceiling is 999; stay comfortably under it.
DELETE_CHUNK = 500


def ids_to_keep(session: Session) -> set[int]:
    keep: set[int] = set()

    latest_run = session.query(ModelRun).order_by(ModelRun.id.desc()).first()
    if latest_run is not None:
        keep.update(
            pid for (pid,) in session.query(Prediction.id).filter(Prediction.model_run_id == latest_run.id)
        )

    # Most recent pre-kickoff prediction per match. Deliberately not restricted
    # to finished matches: a match that has kicked off but has no result posted
    # yet will become finished later, and its pre-kickoff prediction has to
    # still be here when it does.
    best: dict[int, tuple[int, object]] = {}
    rows = (
        session.query(Prediction.id, Prediction.match_id, Prediction.created_at)
        .join(Match, Prediction.match_id == Match.id)
        .filter(Prediction.created_at < Match.utc_kickoff)
        .all()
    )
    for pred_id, match_id, created_at in rows:
        current = best.get(match_id)
        if current is None or created_at > current[1]:
            best[match_id] = (pred_id, created_at)
    keep.update(pred_id for pred_id, _ in best.values())

    return keep


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="report what would be deleted, change nothing")
    args = parser.parse_args()

    session = SessionLocal()
    try:
        total = session.query(Prediction).count()
        keep = ids_to_keep(session)
        doomed = [pid for (pid,) in session.query(Prediction.id) if pid not in keep]

        print(f"predictions total : {total}")
        print(f"  keep            : {len(keep)}")
        print(f"  delete          : {len(doomed)}")

        if args.dry_run:
            print("\n--dry-run: nothing was changed.")
            return
        if not doomed:
            print("\nNothing to prune.")
            return

        for start in range(0, len(doomed), DELETE_CHUNK):
            chunk = doomed[start : start + DELETE_CHUNK]
            session.query(Prediction).filter(Prediction.id.in_(chunk)).delete(synchronize_session=False)
        session.commit()
        remaining = session.query(Prediction).count()
        print(f"\ndeleted {len(doomed)}; {remaining} remain")
    finally:
        session.close()

    # VACUUM cannot run inside a transaction, and without it SQLite keeps the
    # freed pages in the file — which would defeat the point here, since the
    # file size is what gets committed to git on every refresh.
    with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        conn.execute(text("VACUUM"))
    print("vacuumed")


if __name__ == "__main__":
    main()

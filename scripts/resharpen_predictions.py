"""Re-predicts a scheduled match's EXISTING Prediction row, in place, once a
confirmed starting lineup lands for it — the companion job that makes
Predictor._team_strength_for's lineup preference (model/predict.py) actually
observable on the site. Without this, no match would ever show a lineup-
based prediction in practice: the daily refresh (scripts/refresh.py) runs
once at 06:00 UTC, hours before any given day's lineups are confirmed
(~1h pre-kickoff), so every prediction it writes is necessarily the team-
level fallback.

Deliberately NOT a full scripts/refresh.py-style generate_predictions() call
— that creates a brand new ModelRun and rewrites all ~2,800 scheduled
predictions on every call, which run every 15 minutes (this is meant to
piggyback on the existing close-out-finished.yml job, same reasoning
ingest/bigballs.py's lineup syncs already established) would dirty
data/app.db and trigger a Vercel redeploy on nearly every tick regardless of
whether anything real happened. Instead: find the handful of matches (often
zero) whose lineup just became usable, and update ONLY those matches'
existing Prediction rows for the CURRENT ModelRun — cheap, and a genuine
no-op (no DB touch) when there's nothing to do, matching the no-op-commit
discipline documented at ingest/bigballs.py's lineup-sync functions.

Usage: python scripts/resharpen_predictions.py
"""

import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.db import SessionLocal, init_db
from app.models import Lineup, Match, ModelRun, Prediction
from model.player_elo import MIN_STARTERS_FOR_LIVE_STRENGTH
from model.predict import Predictor

# Generous next to lineups' own ~1h-before-kickoff publish point (same
# reasoning as ingest/bigballs.py::sync_upcoming_lineups' hours_ahead) —
# enough headroom to catch a lineup on an early check, not a claim that
# lineups usually exist this far out.
CANDIDATE_WINDOW_HOURS = 2.0


def _confirmed_starter_count(session, match_id: int, team_id: int) -> int:
    return (
        session.query(Lineup)
        .filter_by(match_id=match_id, team_id=team_id, starter=True)
        .filter(Lineup.player_id.isnot(None))
        .count()
    )


def find_candidates(session) -> tuple[ModelRun | None, list[Match]]:
    """Scheduled matches, kicking off soon, whose CURRENT prediction hasn't
    used a lineup yet but now has a confirmed one on file for both sides.
    Pure reads — no session.commit() anywhere in this function, so calling
    it costs nothing even when (as on most 15-minute ticks) it finds
    nothing."""
    model_run = session.query(ModelRun).order_by(ModelRun.id.desc()).first()
    if model_run is None:
        return None, []

    window_end = dt.datetime.utcnow() + dt.timedelta(hours=CANDIDATE_WINDOW_HOURS)
    scheduled = (
        session.query(Match).filter(Match.status == "scheduled", Match.utc_kickoff <= window_end).all()
    )

    candidates = []
    for match in scheduled:
        prediction = session.query(Prediction).filter_by(match_id=match.id, model_run_id=model_run.id).one_or_none()
        if prediction is None or prediction.lineup_based:
            continue
        home_n = _confirmed_starter_count(session, match.id, match.home_team_id)
        away_n = _confirmed_starter_count(session, match.id, match.away_team_id)
        if home_n >= MIN_STARTERS_FOR_LIVE_STRENGTH and away_n >= MIN_STARTERS_FOR_LIVE_STRENGTH:
            candidates.append(match)
    return model_run, candidates


def resharpen(session, model_run: ModelRun, matches: list[Match]) -> int:
    """Builds ONE Predictor (the expensive part — full Dixon-Coles/Elo/
    player-Elo refit) and updates each candidate's existing Prediction row
    in place. Only ever called with a non-empty `matches`, so this cost is
    paid exactly when there's real new work, never on a no-op tick."""
    predictor = Predictor(session)
    updated = 0
    for match in matches:
        summary = predictor.predict_match(match)
        prediction = session.query(Prediction).filter_by(match_id=match.id, model_run_id=model_run.id).one()
        prediction.home_win_prob = summary["home_win_prob"]
        prediction.draw_prob = summary["draw_prob"]
        prediction.away_win_prob = summary["away_win_prob"]
        prediction.home_xg_pred = summary["home_xg_pred"]
        prediction.away_xg_pred = summary["away_xg_pred"]
        prediction.over_2_5_prob = summary["over_2_5_prob"]
        prediction.btts_prob = summary["btts_prob"]
        prediction.top_scorelines = summary["top_scorelines"]
        prediction.confidence = summary["confidence"]
        prediction.lineup_based = summary["lineup_based"]
        updated += 1
    session.commit()
    return updated


def main() -> None:
    init_db()
    session = SessionLocal()
    try:
        model_run, candidates = find_candidates(session)
        if not candidates:
            print("no matches ready to resharpen")
            return
        updated = resharpen(session, model_run, candidates)
        print(f"resharpened {updated} prediction(s): {[m.id for m in candidates]}")
    finally:
        session.close()


if __name__ == "__main__":
    main()

"""Tests for scripts/prune_predictions.py.

The one property that actually matters: pruning must not change anything the
accuracy pages report. The retained set is defined by what the read paths
query, so the strongest possible assertion is to compute the live-tracking
summary before and after and require it to be identical — not merely similar.
"""

import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.models import Base, Competition, Match, ModelRun, Prediction, Team
from model.backtest import live_tracking_summary
from scripts.prune_predictions import ids_to_keep


@pytest.fixture()
def session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    s = sessionmaker(bind=engine)()
    yield s
    s.close()


def _setup(session):
    comp = Competition(slug="l", name="L", country="X", type="league", fd_code="X0")
    session.add(comp)
    session.flush()
    home = Team(canonical_name="Home", country="X")
    away = Team(canonical_name="Away", country="X")
    session.add_all([home, away])
    session.flush()
    return comp, home, away


def _match(comp, home, away, kickoff, *, status="finished", hg=2, ag=0):
    return Match(
        competition_id=comp.id,
        utc_kickoff=kickoff,
        status=status,
        home_team_id=home.id,
        away_team_id=away.id,
        home_goals=hg if status == "finished" else None,
        away_goals=ag if status == "finished" else None,
        source="test",
    )


def _pred(match, run, created_at, probs=(0.6, 0.25, 0.15)):
    return Prediction(
        match_id=match.id,
        model_run_id=run.id,
        created_at=created_at,
        home_win_prob=probs[0],
        draw_prob=probs[1],
        away_win_prob=probs[2],
        top_scorelines=[],
    )


def test_prune_keeps_latest_pre_kickoff_prediction_and_leaves_metrics_identical(session):
    comp, home, away = _setup(session)
    kickoff = dt.datetime(2026, 3, 1, 15, 0)
    match = _match(comp, home, away, kickoff)
    session.add(match)
    session.flush()

    # One prediction per (match, run) — `predictions` is unique on that pair,
    # so successive re-predictions of the same fixture each need their own run,
    # exactly as generate_predictions does it.
    run_a = ModelRun(run_at=dt.datetime(2026, 1, 1), params={})
    run_b = ModelRun(run_at=dt.datetime(2026, 2, 1), params={})
    run_c = ModelRun(run_at=dt.datetime(2026, 2, 15), params={})
    newest_run = ModelRun(run_at=dt.datetime(2026, 4, 1), params={})
    session.add_all([run_a, run_b, run_c, newest_run])
    session.flush()

    # Three pre-kickoff predictions with different probabilities; only the
    # most recent one is what the site actually showed before kickoff.
    superseded_a = _pred(match, run_a, dt.datetime(2026, 1, 2), (0.9, 0.05, 0.05))
    superseded_b = _pred(match, run_b, dt.datetime(2026, 2, 2), (0.8, 0.1, 0.1))
    latest_pre = _pred(match, run_c, dt.datetime(2026, 2, 20), (0.6, 0.25, 0.15))
    session.add_all([superseded_a, superseded_b, latest_pre])
    session.commit()

    before = live_tracking_summary(session)
    assert before["n"] == 1

    keep = ids_to_keep(session)
    assert latest_pre.id in keep
    assert superseded_a.id not in keep
    assert superseded_b.id not in keep

    session.query(Prediction).filter(Prediction.id.notin_(keep)).delete(synchronize_session=False)
    session.commit()

    assert live_tracking_summary(session) == before


def test_prune_keeps_every_prediction_from_the_newest_run(session):
    """Scheduled matches have no pre-kickoff-vs-finished record to preserve,
    so the newest run is the only thing keeping their displayed prediction
    alive."""
    comp, home, away = _setup(session)
    upcoming = _match(comp, home, away, dt.datetime(2026, 6, 1, 15, 0), status="scheduled")
    session.add(upcoming)
    session.flush()

    old_run = ModelRun(run_at=dt.datetime(2026, 1, 1), params={})
    new_run = ModelRun(run_at=dt.datetime(2026, 5, 1), params={})
    session.add_all([old_run, new_run])
    session.flush()

    stale = _pred(upcoming, old_run, dt.datetime(2026, 1, 2))
    current = _pred(upcoming, new_run, dt.datetime(2026, 5, 2))
    session.add_all([stale, current])
    session.commit()

    keep = ids_to_keep(session)
    assert current.id in keep
    # `stale` is also pre-kickoff for this match, but `current` is more recent,
    # so only the latter survives on both counts.
    assert stale.id not in keep


def test_prune_retains_pre_kickoff_row_for_a_match_not_yet_marked_finished(session):
    """A match that has kicked off but whose result has not been posted yet
    will become finished later; its pre-kickoff prediction has to still be
    there when that happens, or the accuracy history loses it permanently."""
    comp, home, away = _setup(session)
    played = _match(comp, home, away, dt.datetime(2026, 3, 1, 15, 0), status="scheduled")
    session.add(played)
    session.flush()

    old_run = ModelRun(run_at=dt.datetime(2026, 1, 1), params={})
    new_run = ModelRun(run_at=dt.datetime(2026, 5, 1), params={})
    session.add_all([old_run, new_run])
    session.flush()

    pre_kickoff = _pred(played, old_run, dt.datetime(2026, 2, 25))
    # A later run re-predicted it *after* kickoff — not a valid tracked record.
    post_kickoff = _pred(played, new_run, dt.datetime(2026, 5, 2))
    session.add_all([pre_kickoff, post_kickoff])
    session.commit()

    keep = ids_to_keep(session)
    assert pre_kickoff.id in keep, "pre-kickoff row must survive until the result lands"
    assert post_kickoff.id in keep, "newest run's row is what the page displays"

import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.models import Base, Match, ModelRun, Prediction, Team
from model.backtest import brier, live_tracking_summary, log_loss, rps, tracked_predictions_page, wilson_interval


@pytest.fixture()
def session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    s = Session()
    yield s
    s.close()


def test_rps_perfect_prediction_is_zero():
    assert rps((1.0, 0.0, 0.0), actual=0) == pytest.approx(0.0)


def test_rps_maximally_wrong_confident_prediction_is_one():
    # Predicted certain home win, away actually won — outcomes are 2 apart on
    # the ordinal [home, draw, away] scale, the worst case RPS can score.
    assert rps((1.0, 0.0, 0.0), actual=2) == pytest.approx(1.0)


def test_rps_penalizes_draw_miss_less_than_full_miss():
    # Predicted certain home win; actual is a draw (adjacent) vs actual is
    # away win (two away) — RPS must treat the draw miss as strictly cheaper,
    # which is the entire point of using RPS over a flat multi-class score.
    rps_draw_miss = rps((1.0, 0.0, 0.0), actual=1)
    rps_away_miss = rps((1.0, 0.0, 0.0), actual=2)
    assert 0 < rps_draw_miss < rps_away_miss


def test_brier_perfect_prediction_is_zero():
    assert brier((1.0, 0.0, 0.0), actual=0) == pytest.approx(0.0)


def test_brier_does_not_distinguish_adjacency():
    # Unlike RPS, Brier treats every wrong class the same regardless of
    # ordinal distance — draw-miss and away-miss score identically.
    assert brier((1.0, 0.0, 0.0), actual=1) == pytest.approx(brier((1.0, 0.0, 0.0), actual=2))


def test_log_loss_perfect_prediction_near_zero():
    assert log_loss((0.999999, 0.0000005, 0.0000005), actual=0) < 0.01


def test_log_loss_confident_wrong_prediction_is_large():
    assert log_loss((0.99, 0.005, 0.005), actual=2) > 4.0


def test_log_loss_handles_zero_probability_without_crashing():
    # A model that assigns literally 0% to the outcome that happened must not
    # raise (math domain error) — it should be heavily penalized instead.
    assert log_loss((1.0, 0.0, 0.0), actual=2) > 10


# ---------- wilson_interval ----------


def test_wilson_interval_zero_n_returns_zero_zero():
    assert wilson_interval(0, 0) == (0.0, 0.0)


def test_wilson_interval_contains_the_observed_rate():
    low, high = wilson_interval(30, 44)
    assert low < 30 / 44 < high


def test_wilson_interval_widens_at_smaller_sample_size():
    # Same observed rate, less data behind it -> less certainty -> wider band.
    low_small, high_small = wilson_interval(3, 10)
    low_large, high_large = wilson_interval(300, 1000)
    assert (high_small - low_small) > (high_large - low_large)


# ---------- live_tracking_summary ----------


def _finished_match(comp_id, home_id, away_id, kickoff, home_goals, away_goals):
    return Match(
        competition_id=comp_id,
        utc_kickoff=kickoff,
        status="finished",
        home_team_id=home_id,
        away_team_id=away_id,
        home_goals=home_goals,
        away_goals=away_goals,
        source="test",
    )


def test_live_tracking_ignores_predictions_made_after_kickoff(session):
    home, away = Team(canonical_name="Home"), Team(canonical_name="Away")
    session.add_all([home, away])
    session.flush()
    match = _finished_match(1, home.id, away.id, dt.datetime(2026, 1, 10), 2, 0)
    session.add(match)
    session.flush()
    run = ModelRun(params={})
    session.add(run)
    session.flush()
    # Made AFTER kickoff — hindsight, must not count as a genuine tracked prediction.
    session.add(
        Prediction(
            match_id=match.id, model_run_id=run.id, home_win_prob=0.9, draw_prob=0.05, away_win_prob=0.05,
            created_at=dt.datetime(2026, 1, 11),
        )
    )
    session.commit()

    assert live_tracking_summary(session) == {"n": 0}


def test_live_tracking_uses_latest_pre_kickoff_prediction(session):
    """A match re-predicted twice before kickoff should be scored once, using
    the most recent (most representative of what the site actually showed
    right before the game) — not double-counted."""
    home, away = Team(canonical_name="Home"), Team(canonical_name="Away")
    session.add_all([home, away])
    session.flush()
    kickoff = dt.datetime(2026, 1, 10)
    match = _finished_match(1, home.id, away.id, kickoff, 2, 0)  # home win, actual=0
    session.add(match)
    session.flush()
    # Two separate model runs, as generate_predictions() would produce on two
    # different refits — each creates its own Prediction for this match.
    run1, run2 = ModelRun(params={}), ModelRun(params={})
    session.add_all([run1, run2])
    session.flush()
    # Earlier, wrong-leaning prediction...
    session.add(
        Prediction(
            match_id=match.id, model_run_id=run1.id, home_win_prob=0.2, draw_prob=0.3, away_win_prob=0.5,
            created_at=kickoff - dt.timedelta(days=5),
        )
    )
    # ...superseded by a later, correct-leaning one, still before kickoff.
    session.add(
        Prediction(
            match_id=match.id, model_run_id=run2.id, home_win_prob=0.7, draw_prob=0.2, away_win_prob=0.1,
            created_at=kickoff - dt.timedelta(days=1),
        )
    )
    session.commit()

    result = live_tracking_summary(session)
    assert result["n"] == 1
    assert result["accuracy"] == pytest.approx(1.0)  # scored on the later (correct) prediction only


def test_live_tracking_calibration_table_buckets_by_confidence(session):
    home, away = Team(canonical_name="Home"), Team(canonical_name="Away")
    session.add_all([home, away])
    session.flush()
    run = ModelRun(params={})
    session.add(run)
    session.flush()

    kickoff = dt.datetime(2026, 1, 10)
    match = _finished_match(1, home.id, away.id, kickoff, 1, 0)  # home win
    session.add(match)
    session.flush()
    session.add(
        Prediction(
            match_id=match.id, model_run_id=run.id, home_win_prob=0.75, draw_prob=0.15, away_win_prob=0.10,
            created_at=kickoff - dt.timedelta(days=1),
        )
    )
    session.commit()

    result = live_tracking_summary(session)
    assert result["n"] == 1
    band = result["calibration_table"][0]
    assert band["band_label"] == "70-80%"
    assert band["n"] == 1
    assert band["realized_rate"] == pytest.approx(1.0)
    assert result["recent"][0]["hit"] is True
    assert result["recent"][0]["predicted_pick"] == "home"
    # New: expected accuracy is exactly the top-pick's own confidence (n=1);
    # the Wilson interval on a single hit spans wide but must contain it.
    assert result["expected_accuracy"] == pytest.approx(0.75)
    assert result["ci_low"] <= result["accuracy"] <= result["ci_high"]
    # No draws in this fixture -> ceiling is the full 100%.
    assert result["draw_actual"] == 0
    assert result["decidable_ceiling"] == pytest.approx(1.0)
    assert result["decidable_accuracy"] == pytest.approx(result["accuracy"])


def test_live_tracking_decidable_ceiling_accounts_for_actual_draws(session):
    """One draw among two tracked matches -> ceiling is 50%, and decidable
    accuracy rescales the raw hit rate by that ceiling rather than reporting
    a number that can never reach 100% by construction."""
    home, away = Team(canonical_name="Home"), Team(canonical_name="Away")
    session.add_all([home, away])
    session.flush()
    run = ModelRun(params={})
    session.add(run)
    session.flush()

    kickoff = dt.datetime(2026, 1, 10)
    # Home win, correctly picked home.
    m1 = _finished_match(1, home.id, away.id, kickoff, 2, 0)
    session.add(m1)
    session.flush()
    session.add(
        Prediction(
            match_id=m1.id, model_run_id=run.id, home_win_prob=0.7, draw_prob=0.2, away_win_prob=0.1,
            created_at=kickoff - dt.timedelta(days=1),
        )
    )
    # A draw the model still (correctly, per Answer 1) ranked home ahead of.
    m2 = _finished_match(1, home.id, away.id, kickoff + dt.timedelta(days=1), 1, 1)
    session.add(m2)
    session.flush()
    session.add(
        Prediction(
            match_id=m2.id, model_run_id=run.id, home_win_prob=0.5, draw_prob=0.3, away_win_prob=0.2,
            created_at=kickoff,
        )
    )
    session.commit()

    result = live_tracking_summary(session)
    assert result["n"] == 2
    assert result["correct"] == 1  # the draw was necessarily a miss on top-pick
    assert result["draw_actual"] == 1
    assert result["decidable_ceiling"] == pytest.approx(0.5)
    # 1 correct out of (2 matches * 50% ceiling) = 1 non-draw match -> 100%.
    assert result["decidable_accuracy"] == pytest.approx(1.0)


def test_live_tracking_decidable_accuracy_never_exceeds_100_percent(session):
    """Regression: a correctly-picked draw must NOT stay in the numerator
    while every draw (including that one) is removed from the denominator —
    that combination is unbounded above 100%. Here: 2/2 non-draws correct
    PLUS a correctly-picked draw. The naive (buggy) formula would give
    3 correct / (3 * (2/3) ceiling) = 3/2 = 150%. The fix must cap at 100%:
    (3 correct - 1 draw hit) / 2 non-draw matches = 100%, not more."""
    home, away = Team(canonical_name="Home"), Team(canonical_name="Away")
    session.add_all([home, away])
    session.flush()
    run = ModelRun(params={})
    session.add(run)
    session.flush()
    kickoff = dt.datetime(2026, 1, 10)

    for i, (hg, ag) in enumerate([(2, 0), (0, 1)]):  # two non-draws, both correctly picked
        m = _finished_match(1, home.id, away.id, kickoff + dt.timedelta(days=i), hg, ag)
        session.add(m)
        session.flush()
        home_p, away_p = (0.7, 0.1) if hg > ag else (0.1, 0.7)
        session.add(
            Prediction(
                match_id=m.id, model_run_id=run.id, home_win_prob=home_p, draw_prob=0.2, away_win_prob=away_p,
                created_at=m.utc_kickoff - dt.timedelta(hours=1),
            )
        )
    # A draw the model correctly ranked as its top pick.
    m_draw = _finished_match(1, home.id, away.id, kickoff + dt.timedelta(days=2), 1, 1)
    session.add(m_draw)
    session.flush()
    session.add(
        Prediction(
            match_id=m_draw.id, model_run_id=run.id, home_win_prob=0.2, draw_prob=0.6, away_win_prob=0.2,
            created_at=m_draw.utc_kickoff - dt.timedelta(hours=1),
        )
    )
    session.commit()

    result = live_tracking_summary(session)
    assert result["n"] == 3
    assert result["correct"] == 3  # all three, including the draw, were hits
    assert result["draw_actual"] == 1
    assert result["decidable_accuracy"] == pytest.approx(1.0)
    assert result["decidable_accuracy"] <= 1.0


# ---------- tracked_predictions_page ----------


def _tracked_match_with_prediction(session, run, comp_id, home_id, away_id, kickoff, home_goals, away_goals):
    match = _finished_match(comp_id, home_id, away_id, kickoff, home_goals, away_goals)
    session.add(match)
    session.flush()
    session.add(
        Prediction(
            match_id=match.id, model_run_id=run.id, home_win_prob=0.6, draw_prob=0.25, away_win_prob=0.15,
            created_at=kickoff - dt.timedelta(days=1),
        )
    )
    return match


def test_tracked_predictions_page_orders_newest_first_and_paginates(session):
    home, away = Team(canonical_name="Home"), Team(canonical_name="Away")
    session.add_all([home, away])
    session.flush()
    run = ModelRun(params={})
    session.add(run)
    session.flush()

    base = dt.datetime(2026, 1, 1)
    for i in range(5):
        _tracked_match_with_prediction(session, run, 1, home.id, away.id, base + dt.timedelta(days=i), 1, 0)
    session.commit()

    page1 = tracked_predictions_page(session, page=1, page_size=2)
    assert page1["total"] == 5
    assert page1["total_pages"] == 3
    assert len(page1["rows"]) == 2
    # Newest kickoff (day 4) first.
    assert page1["rows"][0]["match"].utc_kickoff == base + dt.timedelta(days=4)

    page2 = tracked_predictions_page(session, page=2, page_size=2)
    assert page2["rows"][0]["match"].utc_kickoff == base + dt.timedelta(days=2)


def test_tracked_predictions_page_clamps_out_of_range_page(session):
    home, away = Team(canonical_name="Home"), Team(canonical_name="Away")
    session.add_all([home, away])
    session.flush()
    run = ModelRun(params={})
    session.add(run)
    session.flush()
    _tracked_match_with_prediction(session, run, 1, home.id, away.id, dt.datetime(2026, 1, 1), 1, 0)
    session.commit()

    result = tracked_predictions_page(session, page=99, page_size=25)
    assert result["page"] == 1
    assert result["total_pages"] == 1
    assert len(result["rows"]) == 1


def test_tracked_predictions_page_empty(session):
    result = tracked_predictions_page(session)
    assert result == {
        "rows": [], "total": 0, "correct": 0, "wrong": 0, "page": 1, "page_size": 25, "total_pages": 1,
    }


def test_tracked_predictions_page_counts_correct_and_wrong_across_full_history(session):
    """correct/wrong must reflect ALL tracked predictions, not just the
    current page — the page-1 slice below only shows 2 of 3."""
    home, away = Team(canonical_name="Home"), Team(canonical_name="Away")
    session.add_all([home, away])
    session.flush()
    run = ModelRun(params={})
    session.add(run)
    session.flush()

    base = dt.datetime(2026, 1, 1)
    # Predicted home (0.6 highest) twice, actual home win both times — hits.
    _tracked_match_with_prediction(session, run, 1, home.id, away.id, base, 1, 0)
    _tracked_match_with_prediction(session, run, 1, home.id, away.id, base + dt.timedelta(days=1), 2, 0)
    # Same prediction (still favors home), but away actually won — a miss.
    _tracked_match_with_prediction(session, run, 1, home.id, away.id, base + dt.timedelta(days=2), 0, 1)
    session.commit()

    result = tracked_predictions_page(session, page=1, page_size=2)
    assert result["total"] == 3
    assert result["correct"] == 2
    assert result["wrong"] == 1
    assert len(result["rows"]) == 2  # page size still limits the visible rows

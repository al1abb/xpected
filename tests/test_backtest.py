import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from model.backtest import brier, log_loss, rps


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

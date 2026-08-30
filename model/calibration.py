"""Temperature scaling: the standard, minimal fix for a model that is
systematically over- or under-confident — exactly the failure mode an
MLE-fit model like Dixon-Coles is prone to, and exactly what RPS/log-loss
punish directly (see model/backtest.py's docstring on why calibration, not
raw hit-rate, is the real success criterion for this app).

Fit once, out-of-sample: `fit_temperature` takes the raw (probs, actual)
pairs a walk-forward backtest produced (never predictions scored against
their own training fit — that would just measure how confident the fit
already believes itself to be, not whether it's *right* to). The result is a
single scalar T, hardcoded into model/predict.py once tuned — same pattern as
XI_PER_DAY, SHOTS_BLEND_WEIGHT, REST_MAX_PENALTY.

T > 1 means the raw model was overconfident (temperature scaling pulls
predictions toward uniform); T < 1 means underconfident; T == 1 is a no-op.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar

Probs = tuple[float, float, float]


def _scale(probs: Probs, temperature: float) -> np.ndarray:
    p = np.clip(np.array(probs, dtype=float), 1e-10, 1.0)
    logits = np.log(p) / temperature
    logits -= logits.max()  # numerical stability before exponentiating
    exp = np.exp(logits)
    return exp / exp.sum()


def apply_temperature(probs: Probs, temperature: float) -> Probs:
    if temperature == 1.0:
        return probs
    q = _scale(probs, temperature)
    return float(q[0]), float(q[1]), float(q[2])


def _mean_log_loss(temperature: float, records: list[dict]) -> float:
    total = 0.0
    for r in records:
        q = _scale(r["probs"], temperature)
        total += -np.log(max(q[r["actual"]], 1e-10))
    return total / len(records)


def fit_temperature(records: list[dict]) -> float:
    """records: [{"probs": (h, d, a), "actual": 0|1|2}, ...] — the
    "raw_predictions" list from model.backtest.run_backtest(...,
    collect_predictions=True). Returns the log-loss-minimising temperature."""
    if not records:
        return 1.0
    result = minimize_scalar(lambda t: _mean_log_loss(t, records), bounds=(0.2, 5.0), method="bounded")
    return float(result.x)

"""One-shot accuracy tuning pass: decay half-life sweep (7a), calibration
temperature fit (7b), and ordinal-ensemble validation (7c), run back-to-back
in a single process so results are directly comparable and the whole thing
only needs one invocation. Each stage is validated against the walk-forward
backtest and reported honestly — nothing here gets silently assumed to help.

This intentionally runs full backtests sequentially rather than as separate
script invocations: each backtest run is expensive, and consolidating avoids
juggling multiple long-lived background processes against the same SQLite
file (a real problem hit earlier — concurrent writers caused lock contention
and cross-contaminated the ClubElo disk cache with mismatched date grains).
"""

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.db import SessionLocal, init_db
from model import calibration
from model.backtest import run_backtest

HALF_LIVES_DAYS = [180, 365]  # 108d ~= Dixon & Coles' own published value
ENSEMBLE_WEIGHT_CANDIDATE = 0.35

# 108d already ran to completion in a prior invocation of this script before
# it was interrupted (see git history / session notes): rps=0.20089,
# accuracy=0.5229. Recorded here so the sweep report and "best" comparison
# stay complete without paying to refit it — only its scoreboard summary was
# captured, not its raw_predictions, so it's excluded from ever being picked
# as the calibration/ensemble base (immaterial here since 180d already beats
# it on RPS).
KNOWN_RESULTS = [{"half_life_days": 108, "xi_per_day": 0.006418, "rps": 0.20089, "accuracy": 0.5229}]


def _rps(result: dict) -> float | None:
    return result.get("model", {}).get("rps")


def main() -> None:
    init_db()
    session = SessionLocal()
    report = {"decay_sweep": [], "ensemble": {}, "calibration": {}}
    try:
        print("=== 7a: decay half-life sweep ===", flush=True)
        report["decay_sweep"].extend(KNOWN_RESULTS)
        for k in KNOWN_RESULTS:
            print(f"(known) half_life={k['half_life_days']:4d}d  xi={k['xi_per_day']:.6f}  rps={k['rps']}  accuracy={k['accuracy']}", flush=True)
        best = None
        for half_life in HALF_LIVES_DAYS:
            xi = math.log(2) / half_life
            print(f"\n--- half_life={half_life}d (xi={xi:.6f}) ---", flush=True)
            result = run_backtest(session, xi_per_day=xi, collect_predictions=True, verbose=True)
            rps = _rps(result)
            acc = result.get("model", {}).get("accuracy")
            print(f"half_life={half_life:4d}d  xi={xi:.6f}  rps={rps}  accuracy={acc}", flush=True)
            report["decay_sweep"].append({"half_life_days": half_life, "xi_per_day": xi, "rps": rps, "accuracy": acc})
            if rps is not None and (best is None or rps < best["rps"]):
                best = {"half_life": half_life, "xi": xi, "rps": rps, "result": result}

        if best is None:
            print("No usable backtest result — aborting.", flush=True)
            return

        print(f"\nBest decay: half_life={best['half_life']}d (xi={best['xi']:.6f}), rps={best['rps']:.4f}", flush=True)

        print("\n=== 7c: ordinal ensemble validation at best decay ===", flush=True)
        ensemble_result = run_backtest(
            session,
            xi_per_day=best["xi"],
            ensemble_weight=ENSEMBLE_WEIGHT_CANDIDATE,
            collect_predictions=True,
            verbose=True,
        )
        ensemble_rps = _rps(ensemble_result)
        print(f"ensemble_weight={ENSEMBLE_WEIGHT_CANDIDATE}  rps={ensemble_rps}  (vs {best['rps']:.4f} without)", flush=True)
        ensemble_helps = ensemble_rps is not None and ensemble_rps < best["rps"]
        report["ensemble"] = {
            "weight_tried": ENSEMBLE_WEIGHT_CANDIDATE,
            "rps_with_ensemble": ensemble_rps,
            "rps_without_ensemble": best["rps"],
            "helps": ensemble_helps,
        }

        final_result = ensemble_result if ensemble_helps else best["result"]
        final_label = "with ensemble" if ensemble_helps else "without ensemble (no improvement found)"
        print(f"\nFinal configuration: half_life={best['half_life']}d, {final_label}", flush=True)

        print("\n=== 7b: calibration temperature ===", flush=True)
        records = final_result.get("raw_predictions", [])
        temperature = calibration.fit_temperature(records)
        baseline_loss = calibration._mean_log_loss(1.0, records) if records else None
        corrected_loss = calibration._mean_log_loss(temperature, records) if records else None
        print(f"n={len(records)}  temperature={temperature:.4f}  log_loss 1.0->{baseline_loss}  T->{corrected_loss}", flush=True)
        report["calibration"] = {
            "temperature": temperature,
            "n": len(records),
            "log_loss_uncalibrated": baseline_loss,
            "log_loss_calibrated": corrected_loss,
        }

        out_path = Path(__file__).resolve().parent.parent / "data" / "tune_accuracy_results.json"
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nWritten to {out_path}")
    finally:
        session.close()


if __name__ == "__main__":
    main()

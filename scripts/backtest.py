"""Walk-forward backtest against real data. Prints RPS/Brier/log-loss/accuracy
for the model vs. the home-advantage baseline and (where odds exist) the
market baseline, overall and per competition.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import BASE_DIR
from app.db import SessionLocal, init_db
from model.backtest import run_backtest


def main() -> None:
    init_db()
    session = SessionLocal()
    try:
        result = run_backtest(session)
        print(json.dumps(result, indent=2))
        out_path = BASE_DIR / "data" / "backtest_results.json"
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"\nWritten to {out_path}")
    finally:
        session.close()


if __name__ == "__main__":
    main()

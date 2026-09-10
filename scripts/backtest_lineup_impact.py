"""Phase 5: does the lineup-derived team strength (model/predict.py's
Predictor._team_strength_for, added in Phase 4) actually improve
predictions, measured rather than assumed?

Runs model/backtest.py's existing walk-forward run_backtest() TWICE, once
with Predictor.use_lineup_strength on and once forced off, over the SAME
recent window and the SAME matches — then re-scores both runs restricted to
just the "qualifying" subset: finished matches with a confirmed lineup on
file for both sides (MIN_STARTERS_FOR_LIVE_STRENGTH+ resolved starters —
the same bar model/player_elo.py::live_team_strength itself uses). Matches
outside that subset are identical between the two runs by construction (the
toggle only ever changes anything when a qualifying lineup exists), so
including them would only dilute any real signal — this script also prints
the full-window numbers as a sanity check that they really are identical,
which would fail loudly if the toggle were wired wrong.

Deliberately NOT a full-history backtest: run_backtest's default
burn_in_days=365 walks forward from one year after the dataset's first
match, but the app has only been syncing confirmed lineups for a few weeks
(since Aug 2026) and only started resolving them to real player identities
in Phase 4 (Sept 2026) — the other ~99% of a full walk-forward's periods
would have zero qualifying matches and cost real time for nothing (a single
Predictor construction now costs ~200s with the player-Elo replay folded
in). burn_in_days is instead computed to start the walk-forward a fixed
number of days before today, long enough to cover the whole lineup-covered
window without wasting periods on history that can't possibly differ.

Usage: python scripts/backtest_lineup_impact.py
"""

import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.db import SessionLocal, init_db
from app.models import Lineup, Match
from model.backtest import Scoreboard, run_backtest
from model.player_elo import MIN_STARTERS_FOR_LIVE_STRENGTH

# How far back to walk forward from today — generous next to the real
# lineup-coverage window (a few weeks as of Sept 2026) so it's never the
# limiting factor, without re-scoring years of matches that can't possibly
# have a qualifying lineup.
RECENT_WINDOW_DAYS = 45
# Below this many qualifying matches, a comparison is noise, not a
# measurement — print the numbers anyway (transparency), but say so plainly
# rather than implying a conclusion the sample size can't support.
MIN_QUALIFYING_FOR_A_VERDICT = 30


def qualifying_match_ids(session) -> set[int]:
    """Finished matches with >=MIN_STARTERS_FOR_LIVE_STRENGTH resolved
    starters on BOTH sides — the same bar live_team_strength applies live.
    A match below this bar behaves identically whether use_lineup_strength
    is on or off, so it doesn't belong in either subset comparison."""
    counts: dict[int, dict[int, int]] = {}
    starter_rows = (
        session.query(Lineup.match_id, Lineup.team_id, Lineup.player_id)
        .join(Match, Match.id == Lineup.match_id)
        .filter(Match.status == "finished", Lineup.starter.is_(True), Lineup.player_id.isnot(None))
        .all()
    )
    for match_id, team_id, _player_id in starter_rows:
        counts.setdefault(match_id, {}).setdefault(team_id, 0)
        counts[match_id][team_id] += 1

    return {
        match_id
        for match_id, by_team in counts.items()
        if len(by_team) == 2 and min(by_team.values()) >= MIN_STARTERS_FOR_LIVE_STRENGTH
    }


def _rescored(raw_predictions: list[dict], match_ids: set[int] | None) -> dict:
    from model.backtest import brier, log_loss, rps

    board = Scoreboard()
    for row in raw_predictions:
        if match_ids is not None and row["match_id"] not in match_ids:
            continue
        board.rps_sum += rps(row["probs"], row["actual"])
        board.brier_sum += brier(row["probs"], row["actual"])
        board.log_loss_sum += log_loss(row["probs"], row["actual"])
        board.correct_top_pick += int(row["probs"].index(max(row["probs"])) == row["actual"])
        board.n += 1
    return board.summary()


def main() -> None:
    init_db()
    session = SessionLocal()
    try:
        qualifying = qualifying_match_ids(session)
        print(f"qualifying matches (finished, confirmed lineup both sides): {len(qualifying)}")
        if len(qualifying) < MIN_QUALIFYING_FOR_A_VERDICT:
            print(
                f"Below {MIN_QUALIFYING_FOR_A_VERDICT} matches — running the comparison anyway for "
                "visibility, but treat any difference below as noise, not a verdict. Lineup coverage "
                "is still shallow (see ingest/bigballs_history.py); re-run this after more matches "
                "with a resolved lineup have finished."
            )

        earliest_finished = session.query(Match.utc_kickoff).filter(Match.status == "finished").order_by(Match.utc_kickoff.asc()).first()
        if earliest_finished is None:
            print("no finished matches on record")
            return
        burn_in_days = max(1, (dt.datetime.utcnow() - earliest_finished[0]).days - RECENT_WINDOW_DAYS)

        print(f"\nrunning walk-forward backtest, burn_in_days={burn_in_days} (recent {RECENT_WINDOW_DAYS}d window)...")
        with_lineup = run_backtest(session, burn_in_days=burn_in_days, use_lineup_strength=True, collect_predictions=True)
        without_lineup = run_backtest(session, burn_in_days=burn_in_days, use_lineup_strength=False, collect_predictions=True)

        for label, result in (("WITH lineup strength", with_lineup), ("WITHOUT (team Elo only)", without_lineup)):
            if result.get("error"):
                print(f"{label}: {result['error']}")
                return

        print("\n--- full recent window (sanity check: should match closely/exactly) ---")
        for label, result in (("with", with_lineup), ("without", without_lineup)):
            m = result["model"]
            print(f"  {label:8s}: n={m['n']:5d}  rps={m['rps']:.4f}  brier={m['brier']:.4f}  accuracy={m['accuracy']*100:.1f}%")

        print(f"\n--- qualifying subset only (n={len(qualifying)}) ---")
        with_sub = _rescored(with_lineup["raw_predictions"], qualifying)
        without_sub = _rescored(without_lineup["raw_predictions"], qualifying)
        for label, s in (("with lineup", with_sub), ("without (team Elo)", without_sub)):
            if s.get("n"):
                print(f"  {label:20s}: n={s['n']:4d}  rps={s['rps']:.4f}  brier={s['brier']:.4f}  accuracy={s['accuracy']*100:.1f}%")
            else:
                print(f"  {label:20s}: n=0")

        if with_sub.get("n") and without_sub.get("n"):
            rps_delta = without_sub["rps"] - with_sub["rps"]
            print(f"\nRPS delta (positive = lineup strength helps): {rps_delta:+.4f}")
    finally:
        session.close()


if __name__ == "__main__":
    main()

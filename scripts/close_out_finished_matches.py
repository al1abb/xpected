"""Lightweight, frequent close-out for matches football-data.org already
confirms FINISHED but our own DB still shows `scheduled` — either because
kickoff just passed, or because our primary result sources
(football-data.co.uk, fixturedownload.com) haven't posted the confirmed
score yet, which can lag by up to a day. This closes that gap in minutes
instead, using a source we already know is fast (it's what powers the live
score overlay).

Deliberately narrow: no model refit, no prediction regeneration, no ingest
of new fixtures — just "did a match we're tracking actually finish, and if
so, record its final score." Meant to run every ~10-15 minutes via
.github/workflows/close-out-finished.yml; cheap enough to run that often
(one football-data.org request, well within its 10 req/min free-tier limit
— see ingest/live_scores.py).

Only closes matches whose kickoff falls within the same lookback window
used to query football-data.org, as a defensive bound against ever touching
an unrelated future fixture between the same two teams (e.g. a reverse leg).

Usage: python scripts/close_out_finished_matches.py
"""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.db import SessionLocal, init_db
from app.models import Competition, Match, Team
from ingest.football_data_org_aliases import FD_ORG_TO_CANONICAL
from ingest.live_scores import fetch_recently_finished
from ingest.resolve import normalize

LOOKBACK_DAYS = 3


def main() -> int:
    init_db()
    session = SessionLocal()
    try:
        finished = fetch_recently_finished(lookback_days=LOOKBACK_DAYS)
        if not finished:
            print("No recently-finished matches from football-data.org.")
            return 0

        teams_by_norm = {normalize(name): team_id for team_id, name in session.query(Team.id, Team.canonical_name)}
        kickoff_floor = dt.datetime.combine(
            dt.date.today() - dt.timedelta(days=LOOKBACK_DAYS + 1), dt.time.min
        )
        kickoff_ceiling = dt.datetime.utcnow()

        closed = 0
        for row in finished:
            competition = session.query(Competition).filter_by(slug=row["competition_slug"]).one_or_none()
            if competition is None:
                continue
            home_id = teams_by_norm.get(normalize(FD_ORG_TO_CANONICAL.get(row["home_name"], row["home_name"])))
            away_id = teams_by_norm.get(normalize(FD_ORG_TO_CANONICAL.get(row["away_name"], row["away_name"])))
            if home_id is None or away_id is None:
                continue

            candidates = (
                session.query(Match)
                .filter(
                    Match.competition_id == competition.id,
                    Match.home_team_id == home_id,
                    Match.away_team_id == away_id,
                    Match.status == "scheduled",
                    Match.utc_kickoff >= kickoff_floor,
                    Match.utc_kickoff <= kickoff_ceiling,
                )
                .all()
            )
            for match in candidates:
                match.status = "finished"
                match.home_goals = row["home_goals"]
                match.away_goals = row["away_goals"]
                closed += 1
                print(
                    f"  closed: match {match.id} — {row['home_name']} {row['home_goals']}-{row['away_goals']} "
                    f"{row['away_name']} ({row['competition_slug']})"
                )

        session.commit()
        print(f"Closed out {closed} match row(s).")
        return closed
    finally:
        session.close()


if __name__ == "__main__":
    main()

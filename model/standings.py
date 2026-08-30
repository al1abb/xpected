"""League table computation — derived entirely from `matches`, no external
ingest needed. Applies to the 8 domestic leagues and the Champions League
(whose current league-phase format is a single 36-team table, structurally
the same problem as a domestic table).
"""

from __future__ import annotations

import datetime as dt

from sqlalchemy.orm import Session

from app.models import Match, Team


def compute_standings(session: Session, competition_id: int, season_start: dt.date) -> list[dict]:
    """Standings for `competition_id` from finished matches on/after `season_start`.

    Returns rows sorted by the standard football tie-break order: points,
    then goal difference, then goals for, then name — each tagged with a
    1-based `position`. Empty list if nothing's been played yet.
    """
    matches = (
        session.query(Match)
        .filter(
            Match.competition_id == competition_id,
            Match.status == "finished",
            Match.utc_kickoff >= dt.datetime.combine(season_start, dt.time.min),
            Match.home_goals.isnot(None),
            Match.away_goals.isnot(None),
        )
        .all()
    )
    if not matches:
        return []

    stats: dict[int, dict] = {}

    def _row(team_id: int) -> dict:
        return stats.setdefault(
            team_id,
            {"team_id": team_id, "played": 0, "won": 0, "drawn": 0, "lost": 0, "gf": 0, "ga": 0},
        )

    for m in matches:
        home = _row(m.home_team_id)
        away = _row(m.away_team_id)
        home["played"] += 1
        away["played"] += 1
        home["gf"] += m.home_goals
        home["ga"] += m.away_goals
        away["gf"] += m.away_goals
        away["ga"] += m.home_goals
        if m.home_goals > m.away_goals:
            home["won"] += 1
            away["lost"] += 1
        elif m.home_goals < m.away_goals:
            away["won"] += 1
            home["lost"] += 1
        else:
            home["drawn"] += 1
            away["drawn"] += 1

    teams = {t.id: t for t in session.query(Team).filter(Team.id.in_(stats.keys())).all()}

    rows = []
    for team_id, row in stats.items():
        team = teams[team_id]
        row["gd"] = row["gf"] - row["ga"]
        row["points"] = row["won"] * 3 + row["drawn"]
        row["team"] = team
        rows.append(row)

    rows.sort(key=lambda r: (-r["points"], -r["gd"], -r["gf"], r["team"].canonical_name))
    for i, row in enumerate(rows, start=1):
        row["position"] = i

    return rows

"""Player-level Elo, replayed from data/appearances.sqlite (see
ingest/bigballs_history.py) — the foundation for deriving a team's strength
from who is actually on the pitch, rather than from the club's own win/loss
record alone (model/elo.py). See future-plans-era research and this
project's own comparison against playerelo.football: the core advantage
over team Elo is that a player's rating travels with them across a
transfer, and a new signing or an injury changes team strength immediately
rather than only after enough matches accumulate under the new shirt.

Same replay shape as model/elo.py's _replay_internal (chronological,
K-factor, margin-of-victory scaled, reusing elo._goal_diff_multiplier and
elo._expected_home_score directly) with one structural difference:
participation is not binary. A player who starts and plays 90 minutes
should move more than one who comes on for the last 10 — see
_player_delta's minutes scaling — so this is not strictly zero-sum the way
team Elo is (a match's 22 players don't all move by the same magnitude),
though every home player still moves in the same direction and every away
player in the mirrored one.

No cross-league anchor here (contrast model/elo.py's ClubElo anchoring):
data/appearances.sqlite currently covers one country's data source across 5
leagues that mostly don't share players, so there's nothing yet to anchor
against. Revisit if/when coverage reaches leagues with real player overlap
(transfers between covered leagues already carry a rating across for free,
per the module docstring above — this note is about a *missing* cross-pool
anchor, not the transfer case, which already works).

Team strength for a match is a weighted mean of its players' *shrunk*
ratings (see player_strength): a thin-history player counts for less,
converging on their own observed rating by SHRINKAGE_APPEARANCES_THRESHOLD
appearances — same shrinkage-by-sample-size idea as
model/predict.py::SHRINKAGE_MATCH_THRESHOLD, applied per-player instead of
per-league-fit. team_strength() takes a generic list of (player_id, weight)
pairs rather than being hard-wired to "minutes from a played match" so the
same function serves two different callers with two different weight
sources: this module's own historical replay (weight = minutes actually
played, from data/appearances.sqlite) and, later, a live prediction for an
upcoming match with a confirmed-but-not-yet-played lineup (weight from
app.db's Lineup table, where minutes don't exist yet because the match
hasn't been played — see model/predict.py's Phase 4 integration).
"""

from __future__ import annotations

import datetime as dt
import sqlite3

from sqlalchemy import func
from sqlalchemy.orm import Session

from app.models import Match, PlayerRating
from ingest.bigballs_history import _connect
from model.elo import BASE_RATING, _expected_home_score, _goal_diff_multiplier

K_BASE = 20.0

# Appearances below this count have their contribution to team strength
# shrunk toward BASE_RATING — a player's first few appearances are too small
# a sample to trust in full, same reasoning as predict.py's shrinkage weight
# for a thin-history league fit. Reaches full trust (weight 1.0) at exactly
# this many appearances.
SHRINKAGE_APPEARANCES_THRESHOLD = 10

SOURCE_INTERNAL = "internal"


def player_strength(player_id: int, ratings: dict[int, float], appearance_counts: dict[int, int]) -> float:
    """This player's rating, shrunk toward BASE_RATING for a thin appearance
    history. A player never seen contributes exactly BASE_RATING; one at or
    past SHRINKAGE_APPEARANCES_THRESHOLD contributes their own rating in
    full."""
    raw = ratings.get(player_id, BASE_RATING)
    n = appearance_counts.get(player_id, 0)
    weight = min(1.0, n / SHRINKAGE_APPEARANCES_THRESHOLD)
    return weight * raw + (1 - weight) * BASE_RATING


def team_strength(
    weighted_player_ids: list[tuple[int, float]],
    ratings: dict[int, float],
    appearance_counts: dict[int, int],
) -> float | None:
    """Weighted mean of player_strength() across `weighted_player_ids` —
    (player_id, weight) pairs, weight typically minutes played (this
    module's replay) or a flat per-starter weight (a live, not-yet-played
    lineup — see module docstring). None if there's no usable weight at all
    (e.g. a side with zero recorded minutes), so callers can tell "no
    player-derived signal for this side" apart from a genuine low rating."""
    total_weight = sum(w for _, w in weighted_player_ids if w > 0)
    if total_weight <= 0:
        return None
    return (
        sum(player_strength(pid, ratings, appearance_counts) * w for pid, w in weighted_player_ids if w > 0)
        / total_weight
    )


def _load_appearances_by_match(conn: sqlite3.Connection, match_ids: set[int]) -> dict[int, list[tuple[int, int, int]]]:
    """match_id -> [(player_id, team_id, minutes), ...], minutes-having rows
    only — a row with no recorded minutes can't be weighted and would only
    ever contribute a zero-weight no-op to team_strength."""
    if not match_ids:
        return {}
    by_match: dict[int, list[tuple[int, int, int]]] = {}
    placeholders = ",".join("?" * len(match_ids))
    rows = conn.execute(
        f"SELECT match_id, player_id, team_id, minutes FROM appearances "
        f"WHERE match_id IN ({placeholders}) AND minutes IS NOT NULL AND minutes > 0",
        tuple(match_ids),
    ).fetchall()
    for match_id, player_id, team_id, minutes in rows:
        by_match.setdefault(match_id, []).append((player_id, team_id, minutes))
    return by_match


def _replay_player_elo(
    session: Session,
    conn: sqlite3.Connection,
    *,
    as_of: dt.datetime | None = None,
    exclude_match_id: int | None = None,
) -> tuple[dict[int, float], dict[int, int]]:
    """Chronological replay over every finished match that has appearance
    data, up to `as_of`. Returns ({player_id: rating}, {player_id: appearance
    count}) — both are needed downstream (see player_strength's shrinkage),
    so this is the one pass that produces both rather than recomputing
    counts separately.

    No-lookahead by construction: team strength for a match is computed from
    `ratings`/`appearance_counts` as accumulated from every STRICTLY EARLIER
    match only, and this match's own players are updated afterward — the
    same ordering discipline as model/elo.py's _replay_internal."""
    query = session.query(Match).filter(Match.status == "finished")
    if as_of is not None:
        query = query.filter(Match.utc_kickoff <= as_of)
    query = query.order_by(Match.utc_kickoff.asc(), Match.id.asc())
    matches = [m for m in query.all() if m.id != exclude_match_id]

    appearances_by_match = _load_appearances_by_match(conn, {m.id for m in matches})

    ratings: dict[int, float] = {}
    appearance_counts: dict[int, int] = {}

    for match in matches:
        rows = appearances_by_match.get(match.id)
        if not rows or match.home_goals is None or match.away_goals is None:
            continue

        home_players = [(pid, minutes) for pid, team_id, minutes in rows if team_id == match.home_team_id]
        away_players = [(pid, minutes) for pid, team_id, minutes in rows if team_id == match.away_team_id]
        if not home_players or not away_players:
            continue

        strength_home = team_strength(home_players, ratings, appearance_counts)
        strength_away = team_strength(away_players, ratings, appearance_counts)
        if strength_home is None or strength_away is None:
            continue

        goal_diff = match.home_goals - match.away_goals
        if goal_diff > 0:
            actual_home = 1.0
        elif goal_diff == 0:
            actual_home = 0.5
        else:
            actual_home = 0.0
        actual_away = 1.0 - actual_home

        expected_home = _expected_home_score(strength_home, strength_away)
        expected_away = 1.0 - expected_home
        margin_mult = _goal_diff_multiplier(abs(goal_diff))

        for side_players, actual, expected in (
            (home_players, actual_home, expected_home),
            (away_players, actual_away, expected_away),
        ):
            for player_id, minutes in side_players:
                k_effective = K_BASE * margin_mult * (minutes / 90.0)
                current = ratings.get(player_id, BASE_RATING)
                ratings[player_id] = current + k_effective * (actual - expected)
                appearance_counts[player_id] = appearance_counts.get(player_id, 0) + 1

    return ratings, appearance_counts


def compute_player_ratings(
    session: Session,
    *,
    as_of: dt.datetime | None = None,
    exclude_match_id: int | None = None,
) -> tuple[dict[int, float], dict[int, int]]:
    """Thin DB wrapper around _replay_player_elo: opens
    data/appearances.sqlite, runs the replay, closes it. Returns
    ({player_id: rating}, {player_id: appearance count}) — see
    persist_player_ratings for why both need to reach app.db."""
    conn = _connect()
    try:
        return _replay_player_elo(session, conn, as_of=as_of, exclude_match_id=exclude_match_id)
    finally:
        conn.close()


def persist_player_ratings(
    session: Session,
    ratings: dict[int, float],
    appearance_counts: dict[int, int],
    *,
    as_of: dt.date | None = None,
) -> int:
    """Store `ratings` as `player_ratings` rows with source='internal',
    mirroring model/elo.py::persist_ratings exactly — called from the
    offline refresh, never a web request, since compute_player_ratings
    replays the full appearance history from scratch every call. Idempotent
    per day via player_ratings' (player_id, as_of_date, source) uniqueness.
    """
    as_of = as_of or dt.date.today()
    existing = {
        row.player_id: row
        for row in session.query(PlayerRating).filter_by(as_of_date=as_of, source=SOURCE_INTERNAL).all()
    }
    for player_id, elo_value in ratings.items():
        appearances = appearance_counts.get(player_id, 0)
        row = existing.get(player_id)
        if row is None:
            session.add(
                PlayerRating(
                    player_id=player_id,
                    as_of_date=as_of,
                    elo=elo_value,
                    appearances=appearances,
                    source=SOURCE_INTERNAL,
                )
            )
        else:
            row.elo = elo_value
            row.appearances = appearances
    session.commit()
    return len(ratings)


def load_persisted_player_ratings(session: Session) -> tuple[dict[int, float], dict[int, int]]:
    """({player_id: rating}, {player_id: appearance count}) from each
    player's single most recent persisted snapshot — the plain-SELECT read
    path a live request should use instead of recomputing (see
    model/elo.py::load_persisted_ratings for the same pattern on teams).
    Returns ({}, {}) on a database that has never persisted player ratings;
    callers must tolerate that the same way they tolerate an empty
    load_persisted_ratings."""
    latest_dates = (
        session.query(PlayerRating.player_id, func.max(PlayerRating.as_of_date).label("as_of_date"))
        .filter(PlayerRating.source == SOURCE_INTERNAL)
        .group_by(PlayerRating.player_id)
        .subquery()
    )
    rows = (
        session.query(PlayerRating.player_id, PlayerRating.elo, PlayerRating.appearances)
        .join(
            latest_dates,
            (PlayerRating.player_id == latest_dates.c.player_id)
            & (PlayerRating.as_of_date == latest_dates.c.as_of_date),
        )
        .filter(PlayerRating.source == SOURCE_INTERNAL)
        .all()
    )
    ratings = {player_id: elo_value for player_id, elo_value, _ in rows}
    appearance_counts = {player_id: appearances for player_id, _, appearances in rows}
    return ratings, appearance_counts


def prune_player_ratings(session: Session) -> int:
    """Delete every PlayerRating row except each player's single most recent
    as_of_date. Unlike elo_ratings (whole history kept — 4,109 rows total is
    cheap), a player-level equivalent writes one row per resolved player per
    day: thousands now, growing with every backfilled season, in a database
    committed to git on every refresh. Nothing reads historical player-rating
    snapshots yet (no Phase 6 history chart exists), so there's no accuracy
    cost to pruning aggressively from day one — see PlayerRating's own
    docstring. Returns the number of rows deleted."""
    latest_dates = (
        session.query(PlayerRating.player_id, func.max(PlayerRating.as_of_date).label("as_of_date"))
        .filter(PlayerRating.source == SOURCE_INTERNAL)
        .group_by(PlayerRating.player_id)
        .subquery()
    )
    stale = (
        session.query(PlayerRating)
        .outerjoin(
            latest_dates,
            (PlayerRating.player_id == latest_dates.c.player_id)
            & (PlayerRating.as_of_date == latest_dates.c.as_of_date),
        )
        .filter(PlayerRating.source == SOURCE_INTERNAL, latest_dates.c.player_id.is_(None))
        .all()
    )
    deleted = len(stale)
    for row in stale:
        session.delete(row)
    session.commit()
    return deleted

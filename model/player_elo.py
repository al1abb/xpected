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
import statistics

from sqlalchemy import func
from sqlalchemy.orm import Session

from app.models import Competition, Lineup, Match, PlayerRating
from ingest.bigballs_history import _connect
from model.elo import BASE_RATING, _expected_home_score, _goal_diff_multiplier

K_BASE = 20.0

# Appearances below this count have their contribution to team strength
# shrunk toward BASE_RATING — a player's first few appearances are too small
# a sample to trust in full, same reasoning as predict.py's shrinkage weight
# for a thin-history league fit. Reaches full trust (weight 1.0) at exactly
# this many appearances.
SHRINKAGE_APPEARANCES_THRESHOLD = 10

# A "confirmed lineup" for team-strength purposes needs at least this many
# RATED starters — resolved to a real Player AND with at least one
# appearance behind their rating. Not a strict 11, to tolerate the
# occasional data quirk (one unresolved name, a debutant) without silently
# degrading to a near-empty, unrepresentative XI. Below this,
# live_team_strength returns None so the caller falls back to team-level
# Elo rather than trusting a partial lineup.
#
# "Rated", not just "resolved", matters: an unrated player counts as exactly
# BASE_RATING, so a lineup of them reads as a perfectly average side. Until
# Sept 2026 this only checked resolution, and with no ratings reaching the
# live Predictor at all every lineup-based prediction put both teams at
# 1500 — erasing e.g. Manchester United vs Sabah FK's real gap (94% home
# win on team Elo, 67% "lineup-based"). A league with no appearance history
# yet (Ligue 1 on bigballsdata.com's free plan) would do the same.
MIN_STARTERS_FOR_LIVE_STRENGTH = 7

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


def live_team_strength(
    session: Session,
    match_id: int,
    team_id: int,
    ratings: dict[int, float],
    appearance_counts: dict[int, int],
) -> float | None:
    """team_strength() for an upcoming match's CONFIRMED starting XI (see
    app.models.Lineup, written by ingest/bigballs.py and
    ingest/highlightly.py), not a played one — there are no minutes yet
    because the match hasn't kicked off, so every resolved starter counts
    equally (weight 1.0) rather than by minutes played. This is the "who is
    actually on the pitch today" signal model/predict.py prefers over team-
    level Elo whenever it's available (see that module's Predictor).

    None whenever there isn't a trustworthy confirmed lineup for this side:
    no Lineup rows at all (most of a match's life — see
    MIN_STARTERS_FOR_LIVE_STRENGTH), or fewer than that many starters
    resolved to a real Player with a real rating. A starter with no
    resolved player_id is skipped rather than guessed at — see
    Lineup.player_id's docstring for why that can be null."""
    starters = resolved_starter_ids(session, match_id, team_id)
    if rated_count(starters, appearance_counts) < MIN_STARTERS_FOR_LIVE_STRENGTH:
        return None
    # Every resolved starter still counts (an unrated debutant is a real
    # BASE_RATING-ish contributor, via player_strength's shrinkage); the
    # rated-count bar above only decides whether the XI as a whole is
    # known well enough to trust over team Elo.
    return team_strength([(player_id, 1.0) for player_id in starters], ratings, appearance_counts)


def resolved_starter_ids(session: Session, match_id: int, team_id: int) -> list[int]:
    """Player ids of this side's confirmed starters that resolved to a real
    Player — the only ones that could ever carry a rating."""
    return [
        player_id
        for (player_id,) in session.query(Lineup.player_id)
        .filter_by(match_id=match_id, team_id=team_id, starter=True)
        .filter(Lineup.player_id.isnot(None))
        .all()
    ]


def rated_count(player_ids: list[int], appearance_counts: dict[int, int]) -> int:
    """How many of `player_ids` have any appearance history behind their
    rating — compared against MIN_STARTERS_FOR_LIVE_STRENGTH by both
    live_team_strength and scripts/resharpen_predictions.py, so the two can
    never disagree about whether a lineup is usable."""
    return sum(1 for player_id in player_ids if appearance_counts.get(player_id, 0) > 0)


# ---------- lineup adjustment on the team-Elo scale ----------
#
# A lineup's player-derived strength is NOT a team Elo and must never stand
# in for one. Measured on production data (Sept 2026): the spread of home-
# minus-away gaps was 103 points player-derived vs 181 on team Elo, and
# player Elo has no cross-league anchor at all (see the module docstring) —
# substituting one for the other pulled every lineup-based prediction toward
# a coin flip. Instead a confirmed XI now ADJUSTS its own team's Elo, by how
# much stronger or weaker it is than the XIs that same team has actually
# fielded lately, both rated from the same ratings dict. Comparing a team
# with itself cancels any league-level offset in player Elo, and an ordinary
# lineup (the common case) changes nothing: the prediction is exactly the
# team-Elo one until the XI really is unusual (rotation, injuries, a
# weakened cup side).

# How many of a team's most recent XIs make up "the XI it usually fields",
# and how few are too few to call anything usual.
TYPICAL_XI_MATCHES = 10
MIN_TYPICAL_XI_MATCHES = 3
# Lineups older than this don't describe the current squad.
TYPICAL_XI_LOOKBACK_DAYS = 365

# Team-Elo points per point of XI strength, fitted per Predictor by
# elo_per_xi_point(). Clamped so a thin or noisy fit can't produce an
# absurd multiplier; the default stands in when there are too few teams
# to fit at all.
DEFAULT_ELO_PER_XI_POINT = 1.0
MIN_ELO_PER_XI_POINT = 0.5
MAX_ELO_PER_XI_POINT = 3.0
MIN_TEAMS_FOR_ELO_PER_XI_POINT = 10

# No single lineup moves a team more than this many Elo points either way —
# roughly the gap between a title contender and a mid-table side, which is
# as far as even a heavily rotated XI should reasonably drag a club.
MAX_LINEUP_ADJUSTMENT = 150.0


def typical_xi_strengths(
    session: Session,
    ratings: dict[int, float],
    appearance_counts: dict[int, int],
    *,
    before: dt.datetime,
    exclude_match_id: int | None = None,
) -> tuple[dict[int, float], dict[int, int]]:
    """({team_id: mean strength of its recent XIs}, {team_id: its domestic
    league's competition_id}) — the baseline a confirmed lineup is compared
    against. Only finished matches strictly before `before` count, so a
    backtest at a historical cutoff never sees a later lineup. An XI must
    clear the same rated-starter bar as a live one, and a team needs at
    least MIN_TYPICAL_XI_MATCHES such XIs, else it has no baseline and its
    predictions stay on plain team Elo."""
    league_ids = {cid for (cid,) in session.query(Competition.id).filter(Competition.type == "league")}
    query = (
        session.query(Lineup.match_id, Lineup.team_id, Lineup.player_id, Match.utc_kickoff, Match.competition_id)
        .join(Match, Match.id == Lineup.match_id)
        .filter(
            Lineup.starter.is_(True),
            Lineup.player_id.isnot(None),
            Match.status == "finished",
            Match.utc_kickoff < before,
            Match.utc_kickoff >= before - dt.timedelta(days=TYPICAL_XI_LOOKBACK_DAYS),
        )
    )
    if exclude_match_id is not None:
        query = query.filter(Lineup.match_id != exclude_match_id)

    xis: dict[tuple[int, int], list[int]] = {}
    match_info: dict[int, tuple[dt.datetime, int]] = {}
    for match_id, team_id, player_id, kickoff, competition_id in query:
        xis.setdefault((team_id, match_id), []).append(player_id)
        match_info[match_id] = (kickoff, competition_id)

    by_team: dict[int, list[tuple[dt.datetime, int, list[int]]]] = {}
    for (team_id, match_id), players in xis.items():
        if rated_count(players, appearance_counts) < MIN_STARTERS_FOR_LIVE_STRENGTH:
            continue
        kickoff, competition_id = match_info[match_id]
        by_team.setdefault(team_id, []).append((kickoff, competition_id, players))

    strengths: dict[int, float] = {}
    league_of: dict[int, int] = {}
    for team_id, entries in by_team.items():
        recent = sorted(entries, key=lambda e: e[0], reverse=True)[:TYPICAL_XI_MATCHES]
        if len(recent) < MIN_TYPICAL_XI_MATCHES:
            continue
        per_match = [team_strength([(pid, 1.0) for pid in players], ratings, appearance_counts) for _, _, players in recent]
        strengths[team_id] = statistics.fmean(per_match)
        domestic = [cid for _, cid, _ in recent if cid in league_ids]
        if domestic:
            league_of[team_id] = statistics.mode(domestic)
    return strengths, league_of


def elo_per_xi_point(
    typical: dict[int, float],
    league_of: dict[int, int],
    team_elos: dict[int, float],
) -> float:
    """How many team-Elo points one point of XI strength is worth: the
    pooled within-league OLS slope of team Elo on typical-XI strength.
    Within-league (both sides demeaned per league) because player Elo has
    no cross-league anchor — a pooled cross-league slope would mostly
    measure that missing anchor, not squad quality. Falls back to
    DEFAULT_ELO_PER_XI_POINT with too few teams; always clamped."""
    by_league: dict[int, list[tuple[float, float]]] = {}
    for team_id, xi_strength in typical.items():
        league = league_of.get(team_id)
        if league is None or team_id not in team_elos:
            continue
        by_league.setdefault(league, []).append((xi_strength, team_elos[team_id]))

    sxy = sxx = 0.0
    n = 0
    for pairs in by_league.values():
        if len(pairs) < 2:
            continue
        mean_x = statistics.fmean(x for x, _ in pairs)
        mean_y = statistics.fmean(y for _, y in pairs)
        for x, y in pairs:
            sxy += (x - mean_x) * (y - mean_y)
            sxx += (x - mean_x) ** 2
        n += len(pairs)
    if n < MIN_TEAMS_FOR_ELO_PER_XI_POINT or sxx <= 0:
        return DEFAULT_ELO_PER_XI_POINT
    return min(MAX_ELO_PER_XI_POINT, max(MIN_ELO_PER_XI_POINT, sxy / sxx))


def lineup_adjusted_strength(team_elo: float, xi_strength: float, typical_strength: float, per_point: float) -> float:
    """`team_elo`, moved by how far today's XI sits from the team's usual one
    (converted to Elo points, capped at MAX_LINEUP_ADJUSTMENT either way)."""
    adjustment = per_point * (xi_strength - typical_strength)
    return team_elo + max(-MAX_LINEUP_ADJUSTMENT, min(MAX_LINEUP_ADJUSTMENT, adjustment))


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


def refresh_player_ratings(session: Session) -> str:
    """compute_player_ratings -> persist -> prune, in one call. The only
    place this can run for real is a workflow whose data/appearances.sqlite
    actually has appearance rows in it (restored from actions/cache) — see
    .github/workflows/backfill-appearances.yml, which is the sole writer of
    that file's persisted history. Also used by scripts/refresh.py for a
    local run where the file exists on disk directly."""
    ratings, appearance_counts = compute_player_ratings(session)
    stored = persist_player_ratings(session, ratings, appearance_counts)
    pruned = prune_player_ratings(session)
    return f"{stored} player ratings stored, {pruned} stale snapshot(s) pruned"


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


def compute_ear(
    ratings: dict[int, float],
    player_position: dict[int, str | None],
    player_competition: dict[int, int | None],
) -> dict[int, float]:
    """Elo Above Replacement: each player's rating minus the median rating
    among every OTHER rated player sharing their position within their same
    competition — "replacement level" for that specific slot, the same idea
    baseball's WAR is built on. Median rather than mean so one outlier
    superstar doesn't drag the bar up for everyone else at that position.

    Grouped by competition, not globally, for the same reason model/elo.py
    anchors ratings onto ClubElo's cross-league scale rather than comparing
    raw internal numbers across leagues directly: a "replacement-level"
    midfielder in the Premier League and one in the Azerbaijan Premyer Liqa
    are not interchangeable baselines.

    Only meaningful for a player with both a known position
    (Player.primary_position — populated from lineup data and
    ingest/resolve_players.py's squad-position mapping, so absent for
    anyone with neither) and a known current competition (via their
    current team's primary league). Missing either: simply absent from the
    result, never a misleading 0."""
    buckets: dict[tuple[int, str], list[float]] = {}
    for player_id, rating in ratings.items():
        position = player_position.get(player_id)
        competition_id = player_competition.get(player_id)
        if position is None or competition_id is None:
            continue
        buckets.setdefault((competition_id, position), []).append(rating)

    replacement_level = {key: statistics.median(values) for key, values in buckets.items()}

    ear: dict[int, float] = {}
    for player_id, rating in ratings.items():
        position = player_position.get(player_id)
        competition_id = player_competition.get(player_id)
        if position is None or competition_id is None:
            continue
        ear[player_id] = rating - replacement_level[(competition_id, position)]
    return ear

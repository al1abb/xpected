"""Fix split team identities: the same real club ingested under two different
names from two different sources ends up as two `Team` rows, so its history,
upcoming fixtures, and Elo rating are silently split in half instead of
accumulating on one team. This is a real accuracy bug, not cosmetic — a
duplicate with zero recorded history defaults to Elo's BASE_RATING (1500),
which can produce an absurd prediction for its next fixture (confirmed:
"Porto vs PSV" read 88/10/2 before this merge, because "PSV" and
"PSV Eindhoven" were two different teams).

See ingest/resolve.py for why this happens: a name below the auto-merge
similarity cutoff creates a new team rather than guessing, by design — this
script is where a human confirms the ones that really are the same club.

Usage: python scripts/merge_teams.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy.orm import Session

from app.db import SessionLocal
from app.models import EloRating, Match, OddsSnapshot, Prediction, Team, TeamAlias, UnresolvedAlias

# (duplicate canonical_name, survivor canonical_name, rename survivor to | None)
# Survivor is whichever row holds the real match history; `rename` lets the
# nicer/more complete name win even when it wasn't the history-holding row
# (Sp Lisbon has 241 matches under football-data.co.uk's abbreviation; the
# accurately-named "Sporting CP" row has only 10, from being freshly created
# by a source that couldn't confidently match the two).
MERGES: list[tuple[str, str, str | None]] = [
    ("PSV", "PSV Eindhoven", None),
    ("Sabah", "Sabah FA", None),
    ("Sporting CP", "Sp Lisbon", "Sporting CP"),
    ("SV 07 Elversberg", "Elversberg", None),
    ("Stade Brestois 29", "Brest", None),
    # --- 2026-09 batch, found by scripts/find_duplicate_teams.py -------------
    # All 14 are the same shape: football-data.co.uk had the club under its
    # short/abbreviated name with years of history, then fixturedownload's
    # 2026/27 season feed introduced the club's full legal name as a NEW team
    # carrying the whole season's fixtures. Each pair was confirmed by both
    # teams' season opener falling on the identical date, and by the league
    # landing on its real size once merged (e.g. Ligue 1 22 -> 18).
    # Survivor is the row with more lifetime matches, per this file's
    # convention; `rename` is used only where the surviving row's name is
    # genuinely wrong or ambiguous, not merely less formal.
    ("Santander", "R. Racing Club", "Racing Santander"),
    ("La Coruna", "RC Deportivo", "Deportivo La Coruna"),  # vs Deportivo Alaves
    ("Sport-Club Freiburg", "Freiburg", None),
    ("RC Lens", "Lens", None),
    ("LOSC Lille", "Lille", None),
    ("Havre Athletic Club", "Le Havre", None),
    ("Stade Rennais FC", "Rennes", None),
    ("Excelsior Rotterdam", "Excelsior", None),
    ("N.E.C. Nijmegen", "Nijmegen", "NEC Nijmegen"),
    ("SL Benfica", "Benfica", None),
    ("Vitória SC", "Guimaraes", "Vitória Guimarães"),
    ("Academico Viseu", "Académico", "Académico Viseu"),
    ("Çaykur Rizespor", "Rizespor", None),
    # "Buyuksehyr" is football-data.co.uk's mangled abbreviation, so the
    # survivor takes the duplicate's (correct) name here.
    ("Istanbul Basaksehir", "Buyuksehyr", "Istanbul Basaksehir"),
]


def _pick_keeper(a: Match, b: Match) -> tuple[Match, Match]:
    """When a merge causes two Match rows to collide on the same natural key,
    keep whichever already has a final score; if both/neither do, keep the
    lower id (created first, more likely to have accumulated related rows)."""
    a_final = a.status == "finished" and a.home_goals is not None
    b_final = b.status == "finished" and b.home_goals is not None
    if a_final and not b_final:
        return a, b
    if b_final and not a_final:
        return b, a
    return (a, b) if a.id < b.id else (b, a)


def _drop_match(session: Session, match: Match) -> None:
    session.query(Prediction).filter_by(match_id=match.id).delete(synchronize_session=False)
    session.query(OddsSnapshot).filter_by(match_id=match.id).delete(synchronize_session=False)
    session.delete(match)


def _reassign_match_side(session: Session, match: Match, field: str, new_team_id: int) -> bool:
    """Returns True if `match` was updated in place, False if it was dropped
    (because reassigning it collided with an existing equivalent fixture)."""
    other_field = "away_team_id" if field == "home_team_id" else "home_team_id"
    other_id = getattr(match, other_field)
    collision = (
        session.query(Match)
        .filter_by(
            competition_id=match.competition_id,
            utc_kickoff=match.utc_kickoff,
            **{field: new_team_id, other_field: other_id},
        )
        .filter(Match.id != match.id)
        .one_or_none()
    )
    if collision is None:
        setattr(match, field, new_team_id)
        return True

    keep, drop = _pick_keeper(match, collision)
    if drop.id == match.id:
        _drop_match(session, match)
        return False

    # Order matters: drop the colliding row and flush that delete BEFORE
    # pointing `keep` at the survivor. Doing it the other way round marks
    # `keep` dirty first, and _drop_match's very next Query.delete() triggers
    # an autoflush that pushes the pending UPDATE out while the colliding row
    # is still present — tripping the matches natural-key unique constraint on
    # a state that was only ever meant to be transient. (Confirmed: this is
    # exactly how the 2026-09 merge batch first failed.)
    _drop_match(session, drop)
    session.flush()
    setattr(keep, field, new_team_id)
    return True


def merge_team(session: Session, duplicate: Team, survivor: Team) -> dict:
    stats = {"matches_reassigned": 0, "matches_dropped": 0, "aliases_moved": 0, "elo_ratings_moved": 0}

    for match in list(session.query(Match).filter_by(home_team_id=duplicate.id)):
        kept = _reassign_match_side(session, match, "home_team_id", survivor.id)
        stats["matches_reassigned" if kept else "matches_dropped"] += 1

    for match in list(session.query(Match).filter_by(away_team_id=duplicate.id)):
        kept = _reassign_match_side(session, match, "away_team_id", survivor.id)
        stats["matches_reassigned" if kept else "matches_dropped"] += 1

    for rating in list(session.query(EloRating).filter_by(team_id=duplicate.id)):
        collision = (
            session.query(EloRating)
            .filter_by(team_id=survivor.id, as_of_date=rating.as_of_date, source=rating.source)
            .one_or_none()
        )
        if collision is None:
            rating.team_id = survivor.id
            stats["elo_ratings_moved"] += 1
        else:
            session.delete(rating)

    for alias in list(duplicate.aliases):
        collision = (
            session.query(TeamAlias)
            .filter_by(alias=alias.alias, source=alias.source)
            .filter(TeamAlias.id != alias.id)
            .one_or_none()
        )
        if collision is None:
            # Reassign via the relationship, not a raw `alias.team_id = ...`
            # write: `Team.aliases` is a tracked bidirectional collection, and
            # a raw FK write leaves `alias` still sitting in `duplicate`'s
            # in-memory collection — so when `duplicate` is deleted below,
            # SQLAlchemy's default cascade nulls the FK back out (confirmed:
            # a raw-FK version of this line silently undid the reassignment
            # the moment the duplicate was deleted, in the same flush).
            alias.team = survivor
            stats["aliases_moved"] += 1
        elif collision.team_id != survivor.id:
            session.delete(alias)

    session.query(UnresolvedAlias).filter_by(resolved_team_id=duplicate.id).update(
        {"resolved_team_id": survivor.id}, synchronize_session=False
    )

    # Delete (and flush it through) before the caller renames the survivor —
    # otherwise a rename onto the duplicate's old name and the duplicate's
    # delete can land in the same flush in the wrong order, tripping
    # teams.canonical_name's unique constraint transiently.
    session.delete(duplicate)
    session.flush()
    return stats


def main() -> None:
    session = SessionLocal()
    for dup_name, survivor_name, rename_to in MERGES:
        duplicate = session.query(Team).filter_by(canonical_name=dup_name).one_or_none()
        survivor = session.query(Team).filter_by(canonical_name=survivor_name).one_or_none()
        if duplicate is None:
            print(f"skip: no team named {dup_name!r} (already merged?)")
            continue
        if survivor is None:
            print(f"skip: survivor {survivor_name!r} not found for duplicate {dup_name!r}")
            continue

        stats = merge_team(session, duplicate, survivor)
        if rename_to and survivor.canonical_name != rename_to:
            survivor.canonical_name = rename_to
        session.commit()
        print(f"merged {dup_name!r} -> {survivor.canonical_name!r}: {stats}")


if __name__ == "__main__":
    main()

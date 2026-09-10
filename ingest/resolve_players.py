"""Player-identity resolution across sources.

Team resolution (ingest/resolve.py) leans on fuzzy string matching because
every source spells out a full club name. Players don't get that: there is
no global player identity anywhere in this schema, and historical rows are
badly degraded — confirmed live against bigballsdata.com while planning this
module, 59% of historical lineup/stat rows carry a null player id, and 92%
of historical names are abbreviated ("M. Hermansen", not "Mads Hermansen").
Fuzzy character-similarity matching (resolve.py's approach) is useless on
"M. Hermansen" vs "Mads Hermansen" — the token overlap is deliberately
sparse by design, not a spelling variant.

Two resolution channels instead, tried in order:

1. Strong id: when a source gives a stable per-player id (bigballsdata's
   bbs_player_id, present on ~93% of current-season rows), that id IS the
   identity. Looked up via PlayerAlias(source, source_player_id) —
   deliberately independent of team, so it keeps resolving to the same
   Player across a transfer, which a team-scoped key could not do.

2. Name-based: when no strong id exists, resolve on (team_id, normalized
   surname, first initial) — the only signal an abbreviated name carries.
   Scoped to ONE team's roster, where that tuple is close to unique. A
   collision (two same-surname, same-initial players on one roster) is
   never silently guessed: it's logged to UnresolvedPlayerAlias for manual
   review, the same stance ingest/resolve.py takes on an ambiguous team
   name, and a new Player row is still created so the appearance has
   somewhere to attach — better than dropping real data.
"""

from __future__ import annotations

import re
import unicodedata

from sqlalchemy.orm import Session

from app.models import Player, PlayerAlias, SquadPlayer, UnresolvedPlayerAlias


def normalize(name: str) -> str:
    name = unicodedata.normalize("NFKD", name)
    name = "".join(c for c in name if not unicodedata.combining(c))
    name = name.lower().strip()
    name = re.sub(r"[^a-z0-9\s.]", " ", name)
    return re.sub(r"\s+", " ", name).strip()


def split_surname_initial(raw_name: str) -> tuple[str, str | None]:
    """'M. Hermansen' -> ('hermansen', 'm'); 'Mads Hermansen' -> ('hermansen',
    'm') — same pair either way, which is the entire point: it's the key
    abbreviated and full names agree on. A bare one-word name (e.g.
    'Neymar') -> (that word, None), since there's no initial to extract."""
    norm = normalize(raw_name)
    parts = [p for p in norm.replace(".", " ").split() if p]
    if not parts:
        return "", None
    surname = parts[-1]
    first_initial = parts[0][0] if len(parts) > 1 else None
    return surname, first_initial


def _first_name_token(raw_name: str) -> str | None:
    norm = normalize(raw_name)
    parts = [p for p in norm.replace(".", " ").split() if p]
    return parts[0] if len(parts) > 1 else None


def _is_abbreviated_first_name(raw_name: str) -> bool:
    token = _first_name_token(raw_name)
    return token is not None and len(token) == 1


def _team_surname_pool(session: Session, team_id: int) -> dict[tuple[str, str | None], list[Player]]:
    player_ids = {
        row[0] for row in session.query(PlayerAlias.player_id).filter_by(team_id=team_id).distinct()
    }
    if not player_ids:
        return {}
    pool: dict[tuple[str, str | None], list[Player]] = {}
    for player in session.query(Player).filter(Player.id.in_(player_ids)).all():
        pool.setdefault((player.normalized_surname, player.first_initial), []).append(player)
    return pool


def _add_player_alias(
    session: Session,
    player: Player,
    raw_name: str,
    source: str,
    *,
    team_id: int | None,
    source_player_id: str | None,
) -> None:
    exists = session.query(PlayerAlias).filter_by(alias=raw_name, source=source, team_id=team_id).one_or_none()
    if exists is None:
        session.add(
            PlayerAlias(
                player_id=player.id,
                alias=raw_name,
                source=source,
                team_id=team_id,
                source_player_id=source_player_id,
            )
        )
        # Explicit flush, same reasoning as ingest/resolve.py's _add_alias:
        # without it, the same raw_name seen twice before the next commit
        # won't find its own just-added alias.
        session.flush()


def resolve_or_create_player(
    session: Session,
    raw_name: str,
    source: str,
    *,
    team_id: int | None = None,
    source_player_id: str | None = None,
    date_of_birth=None,
    context: str = "",
) -> Player:
    raw_name = raw_name.strip()
    if not raw_name:
        raise ValueError("empty player name")

    # Channel 1: strong per-source id, independent of team. Once resolved
    # this way no further write is needed — a second alias row carrying the
    # same source_player_id (e.g. after a transfer to a new team_id) would
    # violate PlayerAlias's (source, source_player_id) uniqueness, and
    # nothing downstream currently needs a per-team history of this alias.
    if source_player_id:
        existing = (
            session.query(PlayerAlias).filter_by(source=source, source_player_id=source_player_id).one_or_none()
        )
        if existing is not None:
            return existing.player

    surname, first_initial = split_surname_initial(raw_name)

    # Channel 2: name-based, scoped to one team's roster.
    if team_id is not None:
        existing_alias = (
            session.query(PlayerAlias).filter_by(alias=raw_name, source=source, team_id=team_id).one_or_none()
        )
        if existing_alias is not None:
            return existing_alias.player

        candidates = _team_surname_pool(session, team_id).get((surname, first_initial), [])

        # A full first name carries more information than the coarse
        # (surname, initial) bucket alone — two different players can share
        # both (e.g. "Marcus Silva" and "Mateus Silva" are both ("silva",
        # "m")). When raw_name isn't itself abbreviated, narrow to
        # candidates whose own known first-name token actually matches;
        # narrowing to zero means "confidently a different, new person",
        # not "give up and treat as ambiguous".
        if candidates and not _is_abbreviated_first_name(raw_name):
            incoming_first = _first_name_token(raw_name)
            candidates = [c for c in candidates if _first_name_token(c.canonical_name) == incoming_first]

        if len(candidates) == 1:
            player = candidates[0]
            _add_player_alias(session, player, raw_name, source, team_id=team_id, source_player_id=source_player_id)
            return player
        if len(candidates) > 1:
            # Ambiguous — never guess which one. Still falls through to
            # create a new Player below, same stance as
            # ingest/resolve.py::get_or_create_team on an ambiguous match:
            # the appearance needs somewhere real to attach.
            session.add(
                UnresolvedPlayerAlias(
                    raw_name=raw_name,
                    source=source,
                    team_id=team_id,
                    context=(
                        f"{context} | {len(candidates)} existing players share surname+initial on this team".strip(
                            " |"
                        )
                    ),
                )
            )
            session.flush()

    # No confident match: a genuinely new player.
    player = Player(
        canonical_name=raw_name,
        normalized_surname=surname,
        first_initial=first_initial,
        date_of_birth=date_of_birth,
    )
    session.add(player)
    session.flush()
    _add_player_alias(session, player, raw_name, source, team_id=team_id, source_player_id=source_player_id)
    return player


def seed_players_from_squads(session: Session) -> int:
    """One-time/idempotent: create a Player + a 'football_data_org'
    PlayerAlias for every SquadPlayer not already aliased, seeding real full
    names and dates of birth so abbreviated lineup names ('M. Hermansen')
    have a full-name anchor to match against on the same team (see
    resolve_or_create_player's name channel). Safe to call repeatedly —
    already-aliased squad players are skipped, not re-created."""
    created = 0
    for squad_player in session.query(SquadPlayer).all():
        existing = (
            session.query(PlayerAlias)
            .filter_by(alias=squad_player.name, source="football_data_org", team_id=squad_player.team_id)
            .one_or_none()
        )
        if existing is not None:
            continue
        resolve_or_create_player(
            session,
            squad_player.name,
            "football_data_org",
            team_id=squad_player.team_id,
            date_of_birth=squad_player.date_of_birth,
            context=f"seeded from squad_players id={squad_player.id}",
        )
        created += 1
    return created

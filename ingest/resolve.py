"""Team-name resolution across sources.

The same club shows up as "Man United" (football-data.co.uk), "Manchester
United" (API-Football) and "Karabakh Agdam" / "Qarabağ FK" (ClubElo vs.
everyone else). Getting this wrong silently splits one team's history into two,
which quietly corrupts the model. The rule here: auto-link only on a
normalized exact match or a high-confidence fuzzy match; anything in between
gets created as a new team but logged to `unresolved_aliases` for manual
review, never silently guessed.
"""

from __future__ import annotations

import difflib
import re
import unicodedata

from sqlalchemy.orm import Session

from app.models import Team, TeamAlias, UnresolvedAlias

# Confirm a fuzzy match automatically above this similarity...
CONFIRM_CUTOFF = 0.84
# ...but only surface it for manual review between this and the confirm cutoff.
# Below this, there's no real evidence of a duplicate — it's just a new team.
MAYBE_CUTOFF = 0.60

_STRIP_WORDS = {"fc", "cf", "afc", "cfc", "sc", "ac", "as", "fk", "club", "the", "de", "u21", "u23"}


def normalize(name: str) -> str:
    name = unicodedata.normalize("NFKD", name)
    name = "".join(c for c in name if not unicodedata.combining(c))
    name = name.lower().strip()
    name = re.sub(r"[^a-z0-9\s]", " ", name)
    tokens = [t for t in name.split() if t not in _STRIP_WORDS]
    return " ".join(tokens) or name.strip()


def _alias_pool(session: Session, exclude_source: str) -> dict[str, Team]:
    """normalized name -> Team, from every alias NOT created by `exclude_source`.

    Excluding the current source is deliberate: a single source is already
    internally consistent (football-data.co.uk never uses two spellings for the
    same club within its own files), so any within-source "fuzzy match" is
    coincidental character overlap, not a real duplicate — e.g. "Real Madrid"
    scoring 0.76 against "Ath Madrid" purely because both contain "Madrid".
    Ambiguity worth flagging only exists *across* sources.
    """
    pool: dict[str, Team] = {}
    for alias in session.query(TeamAlias).filter(TeamAlias.source != exclude_source).all():
        pool.setdefault(normalize(alias.alias), alias.team)
    return pool


def _best_match(norm_target: str, pool: dict[str, Team]) -> tuple[float, str | None]:
    best_score, best_key = 0.0, None
    for key in pool:
        score = difflib.SequenceMatcher(None, norm_target, key).ratio()
        if score > best_score:
            best_score, best_key = score, key
    return best_score, best_key


def _has_real_overlap(norm_target: str, norm_candidate: str) -> bool:
    """Character-level ratio alone is unreliable for short club names — e.g.
    'Everton' vs 'Hellas Verona' scores 0.77, 'Mainz' vs 'AC Milan' scores 0.60,
    with zero actual relationship. Require an actual shared word or a substring
    relationship before treating a match as worth a human's attention; this
    still catches 'Man United' / 'Manchester United' (shared token 'united')
    and allows genuinely ambiguous same-city pairs like 'Real Madrid' /
    'Atletico Madrid' through for a quick manual glance, while rejecting
    coincidental character overlap between unrelated club names.
    """
    if norm_target in norm_candidate or norm_candidate in norm_target:
        return True
    return bool(set(norm_target.split()) & set(norm_candidate.split()))


def get_or_create_team(session: Session, raw_name: str, source: str, *, context: str = "") -> Team:
    raw_name = raw_name.strip()
    if not raw_name:
        raise ValueError("empty team name")

    existing_alias = session.query(TeamAlias).filter_by(alias=raw_name, source=source).one_or_none()
    if existing_alias is not None:
        return existing_alias.team

    norm_target = normalize(raw_name)
    pool = _alias_pool(session, exclude_source=source)

    # Exact normalized match — the common case once a handful of sources have run.
    if norm_target in pool:
        team = pool[norm_target]
        _add_alias(session, team, raw_name, source)
        return team

    score, best_key = _best_match(norm_target, pool) if pool else (0.0, None)

    if score >= CONFIRM_CUTOFF and best_key is not None:
        team = pool[best_key]
        _add_alias(session, team, raw_name, source)
        return team

    # No confident match: it's either a genuinely new team, or an ambiguous one.
    team = Team(canonical_name=raw_name)
    session.add(team)
    session.flush()
    _add_alias(session, team, raw_name, source)

    if score >= MAYBE_CUTOFF and best_key is not None and _has_real_overlap(norm_target, best_key):
        session.add(
            UnresolvedAlias(
                raw_name=raw_name,
                source=source,
                context=f"{context} | closest existing match: {pool[best_key].canonical_name!r} (similarity {score:.2f})".strip(
                    " |"
                ),
            )
        )

    return team


def _add_alias(session: Session, team: Team, raw_name: str, source: str) -> None:
    exists = session.query(TeamAlias).filter_by(alias=raw_name, source=source).one_or_none()
    if exists is None:
        session.add(TeamAlias(team_id=team.id, alias=raw_name, source=source))
        # Flush explicitly rather than rely on session-level autoflush: without this,
        # the same raw_name seen twice before the next commit won't find its own
        # just-added alias and will try to INSERT a second Team with the same
        # canonical_name, violating the unique constraint.
        session.flush()

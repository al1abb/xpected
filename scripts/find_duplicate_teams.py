"""Find split team identities that ingest/resolve.py deliberately did NOT
auto-merge, using a structural signal resolve.py has no access to.

resolve.py matches names globally and only auto-links above CONFIRM_CUTOFF
(0.84), which is correctly conservative — "Everton" vs "Hellas Verona" scores
0.77 with zero relationship, so a looser global cutoff would create real
corruption. The cost is that genuine duplicates whose names differ by a
sponsor/legal prefix ("Lille" vs "LOSC Lille" ≈ 0.67, "Lens" vs "RC Lens"
≈ 0.73) fall through to `unresolved_aliases` and sit there until a human looks.

The signal used here instead: **opponent coverage within one league season.**
A round-robin league fixture list pairs every real club with every other, so a
real club's distinct-opponent count is ~N-1. When one club is split across two
`Team` rows, the fixtures divide between them and *both* rows show a sharply
reduced opponent count. Flagging teams far below their league's median opponent
count therefore finds the split halves directly, without relying on the names
matching at all.

That matters, because name similarity alone both over- and under-fires. It
flags real pairs as duplicates (Man City/Man United, AC Milan/Inter Milan,
Paris FC/PSG all share tokens) while missing real duplicates whose two sources
chose unrelated words for the same club (Rennes/Stade Rennais FC,
Santander/R. Racing Club, Guimaraes/Vitória SC, Buyuksehyr/Istanbul
Basaksehir). Coverage caught all of those; names caught none of them.

Verified on the 2026/27 season: this flagged exactly 14 halves across 6
leagues, and merging them landed every league on its real size (Ligue 1 22->18,
La Liga 22->20, Primeira 21->18, Eredivisie 20->18, Bundesliga 19->18,
Süper Lig 20->18) with no false positives.

Scoped to `Competition.type == 'league'`: in a cup — including the Champions
League's 36-team league phase, where each side plays only 8 others — a low
opponent count is normal and carries no information.

Read-only: prints a report, changes nothing. Confirm by eye, then paste into
scripts/merge_teams.py's MERGES list.

Usage: python scripts/find_duplicate_teams.py [--season-start YYYY-MM-DD]
"""

from __future__ import annotations

import argparse
import datetime as dt
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy.orm import Session

from app.db import SessionLocal
from app.models import Competition, Match, Team
from ingest.resolve import normalize

# A team is a suspected split half below this fraction of its league's median
# opponent count. 0.6 cleanly separated 14 halves (1-4 opponents) from every
# real club (17-19) in the 2026/27 data — the gap is wide, so the exact
# threshold is not delicate.
COVERAGE_RATIO = 0.6


def _tokens(name: str) -> set[str]:
    return set(normalize(name).split())


def _name_relation(a: str, b: str) -> str | None:
    """How two names relate, or None. Used only to *rank* a suspected half's
    candidate partners — never to decide whether something is a duplicate."""
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return None
    if ta <= tb or tb <= ta:
        return "one name's tokens contain the other's"
    if ta & tb:
        return f"shared token(s) {sorted(ta & tb)}"
    na, nb = normalize(a).replace(" ", ""), normalize(b).replace(" ", "")
    if na and nb and (na in nb or nb in na):
        return "substring after normalize"
    return None


def _season_bounds(season_start: dt.date) -> tuple[dt.datetime, dt.datetime]:
    return (
        dt.datetime.combine(season_start, dt.time.min),
        dt.datetime.combine(season_start.replace(year=season_start.year + 1), dt.time.min),
    )


def analyse_league(session: Session, comp: Competition, season_start: dt.date) -> dict | None:
    start, end = _season_bounds(season_start)
    rows = (
        session.query(Match.home_team_id, Match.away_team_id, Match.utc_kickoff)
        .filter(
            Match.competition_id == comp.id,
            Match.utc_kickoff >= start,
            Match.utc_kickoff < end,
        )
        .all()
    )
    if not rows:
        return None

    opponents: dict[int, set[int]] = {}
    opener: dict[int, dt.date] = {}
    for home_id, away_id, kickoff in rows:
        opponents.setdefault(home_id, set()).add(away_id)
        opponents.setdefault(away_id, set()).add(home_id)
        day = kickoff.date()
        for tid in (home_id, away_id):
            if tid not in opener or day < opener[tid]:
                opener[tid] = day

    counts = {tid: len(opps) for tid, opps in opponents.items()}
    median = statistics.median(counts.values())
    threshold = median * COVERAGE_RATIO

    by_id = {t.id: t for t in session.query(Team).filter(Team.id.in_(list(counts))).all()}
    halves = sorted((tid for tid, c in counts.items() if c < threshold), key=lambda t: counts[t])

    return {
        "competition": comp.slug,
        "team_count": len(counts),
        "median_opponents": median,
        "counts": counts,
        "opponents": opponents,
        "opener": opener,
        "by_id": by_id,
        "halves": halves,
        "meets": {frozenset((p[0], p[1])) for p in rows if p[0] != p[1]},
    }


def suggest_partners(analysis: dict, lifetime: dict[int, int]) -> list[dict]:
    """For each suspected half, rank the league's other teams as candidate
    partners.

    Ranking uses two signals — deliberately only two, because a third
    ("the two rows' opponent sets should be disjoint and jointly cover the
    league") was tried and provably never fires: both halves typically face the
    *same* opponent in different fixtures (Lens played Brest in August under one
    source; the other row is scheduled against Brest later), so the sets always
    overlap.

    - **never meet** (hard filter): two rows of the same club are never
      scheduled against each other.
    - **name relation** then **shared season-opener date**: ordering hints
      only. Names are absent for roughly a third of real duplicates
      (Rennes/Stade Rennais FC, Santander/R. Racing Club,
      Guimaraes/Vitória SC, Buyuksehyr/Istanbul Basaksehir), and an opener
      date is shared by a third of the league, so neither decides anything.
      For a half with no name-related candidate, expect to pick the partner by
      hand — the value here is narrowing hundreds of teams to a handful of
      halves, not making the final call.
    """
    out = []
    for half_id in analysis["halves"]:
        half = analysis["by_id"].get(half_id)
        if half is None:
            continue
        candidates = []
        for other_id in analysis["counts"]:
            if other_id == half_id or frozenset((half_id, other_id)) in analysis["meets"]:
                continue
            other = analysis["by_id"].get(other_id)
            if other is None:
                continue
            relation = _name_relation(half.canonical_name, other.canonical_name)
            candidates.append(
                {
                    "name": other.canonical_name,
                    "id": other_id,
                    "lifetime": lifetime.get(other_id, 0),
                    "relation": relation,
                    "same_opener": analysis["opener"].get(half_id) == analysis["opener"].get(other_id),
                }
            )
        candidates.sort(
            key=lambda c: (
                c["relation"] is None,
                not c["same_opener"],
                -c["lifetime"],
            )
        )
        out.append(
            {
                "competition": analysis["competition"],
                "half": half.canonical_name,
                "half_id": half_id,
                "half_lifetime": lifetime.get(half_id, 0),
                "opponents": analysis["counts"][half_id],
                "candidates": candidates[:3],
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--season-start",
        default=None,
        help="Season start date (YYYY-MM-DD). Defaults to 1 July of the current season.",
    )
    args = parser.parse_args()

    if args.season_start:
        season_start = dt.date.fromisoformat(args.season_start)
    else:
        from ingest.seasons import current_season_start_year

        season_start = dt.date(current_season_start_year(dt.date.today()), 7, 1)

    session = SessionLocal()
    try:
        lifetime: dict[int, int] = {}
        for home_id, away_id in session.query(Match.home_team_id, Match.away_team_id).all():
            lifetime[home_id] = lifetime.get(home_id, 0) + 1
            lifetime[away_id] = lifetime.get(away_id, 0) + 1

        leagues = (
            session.query(Competition)
            .filter(Competition.type == "league")
            .order_by(Competition.sort_order)
            .all()
        )
        analyses = [a for a in (analyse_league(session, c, season_start) for c in leagues) if a]
        suggestions = [s for a in analyses for s in suggest_partners(a, lifetime)]
    finally:
        session.close()

    print(f"Season starting {season_start.isoformat()}\n")
    print("League coverage (a healthy league has 0 suspects):")
    for a in analyses:
        print(
            f'  {a["competition"]:24} teams={a["team_count"]:>3}  '
            f'median_opponents={a["median_opponents"]:.0f}  suspects={len(a["halves"])}'
        )

    if not suggestions:
        print("\nNo split team identities found.")
        return

    print(f"\n{len(suggestions)} suspected split half/halves — each is likely one club recorded twice:")
    current = None
    for s in suggestions:
        if s["competition"] != current:
            current = s["competition"]
            print(f"\n{current}")
        print(f'  {s["half"]!r} — only {s["opponents"]} opponents, {s["half_lifetime"]} lifetime matches')
        if not any(c["relation"] for c in s["candidates"]):
            print("      (no name-related candidate — pick the partner by hand)")
        for c in s["candidates"]:
            signals = []
            if c["relation"]:
                signals.append(c["relation"])
            if c["same_opener"]:
                signals.append("same opener date")
            note = " + ".join(signals) or "no shared signal — probably not it"
            # Survivor is the row with more lifetime matches (merge_teams.py's
            # convention: keep whichever row actually holds the history).
            if c["lifetime"] >= s["half_lifetime"]:
                dup, surv = s["half"], c["name"]
            else:
                dup, surv = c["name"], s["half"]
            print(f'      ("{dup}", "{surv}", None),  # {c["lifetime"]} lifetime — {note}')

    print("\nEach half is listed with its top candidate partners. Confirm by eye")
    print("(matching season-opener dates are strong evidence), then paste the")
    print("chosen line into scripts/merge_teams.py's MERGES list.")


if __name__ == "__main__":
    main()

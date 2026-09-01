import datetime as dt
import json

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy import case, func, or_, select
from sqlalchemy.orm import Session

from app.badges import competition_logo, competition_short, country_flag
from app.colors import resolve_match_colors, team_colors
from app.config import BASE_DIR, league_zone_for, settings
from app.db import SessionLocal, init_db
from app.models import Competition, EloRating, IngestLog, Match, ModelRun, PlayerStat, Prediction, Team, TeamAlias
from ingest.football_data_org_aliases import FD_ORG_TO_CANONICAL
from ingest.live_scores import fetch_live_matches
from ingest.resolve import normalize
from ingest.seasons import current_season_start_year
from model import backtest, league_strength
from model.backtest import devigged_market_probs
from model.elo import BASE_RATING, load_persisted_ratings
from model.predict import score_matrix, summarize_matrix
from model.standings import compute_standings

app = FastAPI(title="Xpected")
templates = Jinja2Templates(directory=str(BASE_DIR / "app" / "templates"))
templates.env.globals["competition_logo"] = competition_logo
templates.env.globals["competition_short"] = competition_short
templates.env.globals["country_flag"] = country_flag

# Competitions that get a custom color treatment on their competition page —
# see the CSS variable layer in base.html (.theme-ucl). One dict entry + one
# CSS block adds a future competition theme; no template fork needed.
COMPETITION_THEME_CLASS = {"champions-league": "theme-ucl"}

# Static files live in public/, which Vercel serves straight from its CDN —
# requests for /static/* are answered at the edge and never reach this
# function at all. They used to be mounted from app/static/ and therefore
# served BY the function: every image was a serverless invocation from the
# function's region, returned with `cache-control: max-age=0` so browsers
# re-fetched all of it on every page view.
#
# This mount exists only for local dev, which has no CDN in front of it.
#
# It is mounted CONDITIONALLY, and that is load-bearing: vercel.json excludes
# public/** from the function bundle (the CDN serves those files, so shipping
# them inside the function too is pure weight). StaticFiles validates its
# directory at construction and raises RuntimeError when it is missing, which
# on Vercel happens at import time and takes down every route with
# FUNCTION_INVOCATION_FAILED — not just /static. Guarding the mount keeps the
# app importable wherever the directory legitimately isn't present.
static_dir = BASE_DIR / "public" / "static"
if static_dir.is_dir():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.on_event("startup")
def on_startup() -> None:
    # Deliberately does no warming. An earlier version pre-computed the Elo
    # ratings here, which made sense for a long-lived local server but is
    # backwards on serverless: instances are ephemeral and every close-out
    # commit redeploys, so that 8-12s computation (plus a live ClubElo fetch
    # that can never be cache-warm on a fresh instance) was being paid on
    # *every* cold start, before the first byte of the first page. Ratings are
    # now precomputed offline by scripts/refresh.py and read with a SELECT —
    # see _cached_ratings — so there is nothing left worth warming.
    init_db()


def get_session() -> Session:
    return SessionLocal()


# ---------- shared view-model helpers ----------


def _nav_competitions(session: Session) -> tuple[list[Competition], list[Competition]]:
    comps = session.query(Competition).order_by(Competition.sort_order).all()
    uefa = [c for c in comps if c.type == "uefa_cup"]
    leagues = [c for c in comps if c.type == "league"]
    return uefa, leagues


def _latest_model_run(session: Session) -> ModelRun | None:
    return session.query(ModelRun).order_by(ModelRun.id.desc()).first()


# Elo ratings are READ here, never computed. model.elo.compute_ratings() costs
# 8-12s (a full replay of every finished match plus a ClubElo name-resolution
# pass of ~185,000 fuzzy string comparisons) *and* makes a live ClubElo HTTP
# request. That is fine in the offline refresh, which has no time limit and a
# warm disk cache; it is actively harmful in a serverless request, where every
# cold start begins with an empty /tmp/raw — so the fetch can never hit cache —
# and a slow ClubElo can burn the function's entire time budget before a single
# byte is served. scripts/refresh.py now persists the blended ratings via
# model.elo.persist_ratings, so this is a plain SELECT.
#
# The TTL below is only a memo to avoid re-querying on every request; it is no
# longer protecting an expensive computation. Ratings are therefore as fresh as
# the last refresh (~24h) rather than ~1h — Elo moves negligibly in a day, and
# in practice the old cache rarely survived that long anyway, since every
# close-out commit redeploys and discards all warm instances.
_RATINGS_CACHE: dict = {"computed_at": None, "ratings": {}}
_RATINGS_CACHE_TTL = dt.timedelta(hours=1)


def _cached_ratings(session: Session) -> dict[int, float]:
    now = dt.datetime.utcnow()
    computed_at = _RATINGS_CACHE["computed_at"]
    if computed_at is None or now - computed_at > _RATINGS_CACHE_TTL:
        _RATINGS_CACHE["ratings"] = load_persisted_ratings(session)
        _RATINGS_CACHE["computed_at"] = now
    return _RATINGS_CACHE["ratings"]


# For the team-compare feature: a full model.predict.Predictor() is where a
# genuine head-to-head prediction would come from, but measured at ~160s to
# construct (it fits Dixon-Coles + shots + an ordinal model across every
# domestic league from scratch, with no caching of its own) — far too slow
# for a live request. league_strength.EloCalibration is the same Elo-bridge
# math Predictor already falls back to for any cross-league match (no shared
# domestic pool), and on its own it's cheap: just a linear regression over
# UEFA results, using ratings we've already cached above. Good enough for an
# honestly-labeled comparison estimate; not a substitute for a real
# scheduled match's actual prediction.
_ELO_CALIB_CACHE: dict = {"computed_at": None, "calib": None}
_CROSS_LEAGUE_RHO = -0.05  # typical mild Dixon-Coles rho; only affects low-score correlation


def _cached_elo_calibration(session: Session) -> league_strength.EloCalibration:
    now = dt.datetime.utcnow()
    computed_at = _ELO_CALIB_CACHE["computed_at"]
    if computed_at is None or now - computed_at > _RATINGS_CACHE_TTL:
        _ELO_CALIB_CACHE["calib"] = league_strength.fit(session, _cached_ratings(session))
        _ELO_CALIB_CACHE["computed_at"] = now
    return _ELO_CALIB_CACHE["calib"]


def _elo_bridge_estimate(session: Session, home_id: int, away_id: int) -> dict:
    ratings = _cached_ratings(session)
    calib = _cached_elo_calibration(session)
    elo_home = ratings.get(home_id, BASE_RATING)
    elo_away = ratings.get(away_id, BASE_RATING)
    lambda_home, lambda_away = calib.lambdas(elo_home, elo_away)
    matrix = score_matrix(lambda_home, lambda_away, _CROSS_LEAGUE_RHO)
    summary = summarize_matrix(matrix)
    summary["home_xg_pred"] = lambda_home
    summary["away_xg_pred"] = lambda_away
    return summary


def _data_freshness(session: Session) -> tuple[str | None, str | None]:
    """Returns (freshness_display, stale_warning)."""
    last_ok = (
        session.query(IngestLog)
        .filter(IngestLog.status == "ok")
        .order_by(IngestLog.finished_at.desc())
        .first()
    )
    if last_ok is None or last_ok.finished_at is None:
        return None, "No successful data refresh has been recorded yet."

    age = dt.datetime.utcnow() - last_ok.finished_at
    hours = age.total_seconds() / 3600
    display = f"{int(hours)}h ago" if hours < 48 else f"{int(hours / 24)}d ago"
    warning = None
    if hours > settings.stale_after_hours:
        warning = f"Data hasn't refreshed in {display}. Predictions may be based on outdated form."
    return display, warning


def _kickoff_display(kickoff: dt.datetime) -> str:
    return kickoff.strftime("%a %d %b, %H:%M")


def _day_label(d: dt.date, today: dt.date) -> str:
    if d == today:
        return "Today"
    if d == today + dt.timedelta(days=1):
        return "Tomorrow"
    return d.strftime("%A %d %B")


# No ingest source this app uses gives real in-play state (API-Football's own
# 1H/HT/2H/ET/LIVE statuses already collapse to plain "scheduled" — see
# ingest/api_football.py::_status()), so "live" here is a wall-clock estimate,
# not a real score/clock feed: kickoff has passed, status hasn't flipped to
# finished yet, and it's within a plausible match-plus-stoppage window.
ESTIMATED_MATCH_DURATION = dt.timedelta(hours=2, minutes=15)
# "Soon" is a pure pre-kickoff heads-up, not an estimate of anything —
# just "this one's coming up shortly."
SOON_WINDOW = dt.timedelta(hours=1)


def _live_state(match: Match, now: dt.datetime) -> str | None:
    if match.status != "scheduled":
        return None
    if match.utc_kickoff > now:
        return "soon" if match.utc_kickoff - now <= SOON_WINDOW else None
    if now - match.utc_kickoff <= ESTIMATED_MATCH_DURATION:
        return "live"
    return "pending"


def build_match_card(session: Session, match: Match, model_run_id: int | None) -> dict:
    home = session.get(Team, match.home_team_id)
    away = session.get(Team, match.away_team_id)
    competition = session.get(Competition, match.competition_id)

    home_primary = home.primary_color or team_colors(home.canonical_name)[0]
    away_primary = away.primary_color or team_colors(away.canonical_name)[0]
    away_secondary = away.secondary_color
    home_color, away_color = resolve_match_colors(home_primary, away_primary, away_secondary)

    prediction = None
    if model_run_id is not None:
        prediction = (
            session.query(Prediction).filter_by(match_id=match.id, model_run_id=model_run_id).one_or_none()
        )

    return {
        "match": match,
        "competition": competition,
        "competition_name": competition.name if competition else "",
        "round": match.round,
        "kickoff_display": _kickoff_display(match.utc_kickoff),
        "home_name": home.canonical_name,
        "away_name": away.canonical_name,
        "home_logo": home.logo_url,
        "away_logo": away.logo_url,
        "home_color": home_color,
        "away_color": away_color,
        "prediction": prediction,
        "live_state": _live_state(match, dt.datetime.utcnow()),
    }


_HORIZON_OPEN_DAYS = 13  # days ahead that render expanded by default
_ALWAYS_OPEN_GROUPS = 3  # first N matchdays always expanded, even mid-winter-break


def group_by_day(cards: list[dict], *, reverse: bool = False) -> list[dict]:
    """Group match-card dicts (anything with a `["match"].utc_kickoff`) into
    per-day sections, richly enough to drive both a simple flat list
    (index.html only ever reads `.label`/`.cards`, unchanged) and the
    matchday rail + collapsible day sections on competition.html.

    `reverse=True` orders day-groups newest-first — used for past-results
    history, where "most recent" should lead, while still sorting each day's
    own cards chronologically ascending.
    """
    today = dt.datetime.utcnow().date()
    buckets: dict[dt.date, list[dict]] = {}
    for card in cards:
        d = card["match"].utc_kickoff.date()
        buckets.setdefault(d, []).append(card)

    ordered_dates = sorted(buckets, reverse=reverse)
    out: list[dict] = []
    for i, d in enumerate(ordered_dates):
        day_cards = sorted(buckets[d], key=lambda c: c["match"].utc_kickoff)
        out.append(
            {
                "date": d,
                "iso": d.isoformat(),
                "anchor": f"d-{d.isoformat()}",
                "label": _day_label(d, today),
                "weekday": d.strftime("%a"),
                "day_num": d.day,
                "month_key": (d.year, d.month),
                "month_label": d.strftime("%B %Y"),
                "month_short": d.strftime("%b"),
                "is_today": d == today,
                "is_weekend": d.weekday() >= 5,
                "days_ahead": (d - today).days,
                "count": len(day_cards),
                "open": i < _ALWAYS_OPEN_GROUPS or 0 <= (d - today).days <= _HORIZON_OPEN_DAYS,
                "cards": day_cards,
            }
        )
    return out


def group_by_month(days: list[dict]) -> list[dict]:
    """Wraps group_by_day's output into month buckets for headers + the rail.
    Assumes `days` is in a single (either ascending or descending) date
    order, matching whatever order group_by_day produced it in."""
    out: list[dict] = []
    for day in days:
        if not out or out[-1]["key"] != day["month_key"]:
            out.append(
                {
                    "key": day["month_key"],
                    "label": day["month_label"],
                    "short": day["month_short"],
                    "days": [],
                    "count": 0,
                }
            )
        out[-1]["days"].append(day)
        out[-1]["count"] += day["count"]
    return out


def with_today_marker(days: list[dict], today: dt.date) -> list[dict]:
    """For the matchday rail specifically: a competition can easily have no
    fixture on today's actual date (e.g. Champions League between matchdays),
    which left the rail with no visible "you are here" reference at all.
    Inserts a zero-match marker day for `today` at its correct chronological
    slot if it's not already a real matchday, so the rail always has
    something to highlight. Does not affect the day-by-day fixture list
    (which should only ever show days with real matches) — callers pass the
    result to group_by_month for the rail only, keeping `days` itself as-is
    for everything else."""
    if any(d["date"] == today for d in days):
        return days
    marker = {
        "date": today,
        "iso": today.isoformat(),
        "anchor": None,
        "label": "Today",
        "weekday": today.strftime("%a"),
        "day_num": today.day,
        "month_key": (today.year, today.month),
        "month_label": today.strftime("%B %Y"),
        "month_short": today.strftime("%b"),
        "is_today": True,
        "is_weekend": today.weekday() >= 5,
        "days_ahead": 0,
        "count": 0,
        "open": False,
        "cards": [],
    }
    return sorted(days + [marker], key=lambda d: d["date"])


_RESULT_COLOR = {"W": "bg-green-600", "D": "bg-gray-400", "L": "bg-red-500"}


def _team_form(session: Session, team_id: int, *, before: dt.datetime, limit: int = 5) -> list[dict]:
    matches = (
        session.query(Match)
        .filter(
            or_(Match.home_team_id == team_id, Match.away_team_id == team_id),
            Match.status == "finished",
            Match.utc_kickoff < before,
        )
        .order_by(Match.utc_kickoff.desc())
        .limit(limit)
        .all()
    )
    rows = []
    for m in matches:
        is_home = m.home_team_id == team_id
        own_goals = m.home_goals if is_home else m.away_goals
        opp_goals = m.away_goals if is_home else m.home_goals
        opponent_id = m.away_team_id if is_home else m.home_team_id
        opponent = session.get(Team, opponent_id)
        letter = "W" if own_goals > opp_goals else ("D" if own_goals == opp_goals else "L")
        rows.append(
            {
                "match": m,
                "opponent": opponent.canonical_name,
                "venue": "home" if is_home else "away",
                "score": f"{m.home_goals}-{m.away_goals}",
                "result_letter": letter,
                "color_class": _RESULT_COLOR[letter],
                "date_display": m.utc_kickoff.strftime("%d %b"),
            }
        )
    return rows


def _head_to_head(session: Session, team_a: int, team_b: int, *, limit: int = 10) -> list[dict]:
    matches = (
        session.query(Match)
        .filter(
            or_(
                (Match.home_team_id == team_a) & (Match.away_team_id == team_b),
                (Match.home_team_id == team_b) & (Match.away_team_id == team_a),
            ),
            Match.status == "finished",
        )
        .order_by(Match.utc_kickoff.desc())
        .limit(limit)
        .all()
    )
    rows = []
    for m in matches:
        home = session.get(Team, m.home_team_id)
        away = session.get(Team, m.away_team_id)
        rows.append(
            {
                "match": m,
                "home_name": home.canonical_name,
                "away_name": away.canonical_name,
                "date_display": m.utc_kickoff.strftime("%d %b %Y"),
            }
        )
    return rows


def _head_to_head_tally(session: Session, team_a: int, team_b: int) -> dict:
    """All-time W/D/L from team_a's perspective, unlimited (unlike
    _head_to_head's display list, which caps at `limit` recent meetings)."""
    matches = (
        session.query(Match)
        .filter(
            or_(
                (Match.home_team_id == team_a) & (Match.away_team_id == team_b),
                (Match.home_team_id == team_b) & (Match.away_team_id == team_a),
            ),
            Match.status == "finished",
        )
        .all()
    )
    a_wins = b_wins = draws = 0
    for m in matches:
        if m.home_goals == m.away_goals:
            draws += 1
            continue
        winner_id = m.home_team_id if m.home_goals > m.away_goals else m.away_team_id
        if winner_id == team_a:
            a_wins += 1
        else:
            b_wins += 1
    return {"a_wins": a_wins, "b_wins": b_wins, "draws": draws, "total": len(matches)}


def _template_context(session: Session, request: Request, **extra) -> dict:
    uefa, leagues = _nav_competitions(session)
    freshness, warning = _data_freshness(session)
    return {
        "request": request,
        "nav_uefa": uefa,
        "nav_leagues": leagues,
        "current_slug": None,
        "data_freshness": freshness,
        "stale_warning": warning,
        "body_theme_class": "",
        **extra,
    }


# ---------- routes ----------


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


_SEARCH_MIN_CHARS = 2
_SEARCH_LIMIT = 8


def _like_escape(term: str) -> str:
    return term.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def search_teams_query(session: Session, term: str) -> list[Team]:
    """Case-insensitive match over Team.canonical_name and TeamAlias.alias,
    below _SEARCH_MIN_CHARS returns nothing. Dedupe is structural, not a
    DISTINCT: alias hits collapse to a set of team ids via `Team.id.in_(...)`,
    so a team matching through 2+ aliases (TeamAlias is only unique on
    (alias, source), not alias alone) still appears exactly once. Prefix
    matches sort first, then shorter names, then alphabetically."""
    term = term.strip()
    if len(term) < _SEARCH_MIN_CHARS:
        return []

    needle = _like_escape(term.lower())
    contains, prefix = f"%{needle}%", f"{needle}%"

    alias_hits = select(TeamAlias.team_id).where(func.lower(TeamAlias.alias).like(contains, escape="\\"))
    name_lc = func.lower(Team.canonical_name)
    return (
        session.query(Team)
        .filter(or_(name_lc.like(contains, escape="\\"), Team.id.in_(alias_hits)))
        .order_by(
            case((name_lc.like(prefix, escape="\\"), 0), else_=1),
            func.length(Team.canonical_name),
            Team.canonical_name,
        )
        .limit(_SEARCH_LIMIT)
        .all()
    )


@app.get("/search/teams", response_class=HTMLResponse)
def search_teams(request: Request, q: str = "", slot: str | None = None, other: int | None = None):
    term = q.strip()
    session = get_session()
    try:
        results = search_teams_query(session, term)

        rows = [
            {
                "id": t.id,
                "name": t.canonical_name,
                "country": t.country,
                "logo_url": t.logo_url,
                "color": t.primary_color or team_colors(t.canonical_name)[0],
            }
            for t in results
        ]
        # `slot`/`other` mean this search box is one side of the compare-teams
        # picker, not the header's "go straight to this team" search — same
        # backend query, a different results partial that links to
        # /compare?a=&b= (preserving whichever side is already picked)
        # instead of /team/{id}.
        if slot in ("a", "b"):
            return templates.TemplateResponse(
                request,
                "_compare_pick_results.html",
                {"request": request, "results": rows, "q": term, "slot": slot, "other": other},
            )
        return templates.TemplateResponse(
            request, "_search_results.html", {"request": request, "results": rows, "q": term}
        )
    finally:
        session.close()


# football-data.org's live-match window (~6h back covers a match that kicked
# off hours ago and is still somehow marked IN_PLAY by a source hiccup; ~15min
# forward is slack for ordinary kickoff-time drift between sources — this used
# to be a full hour to tolerate the football-data.co.uk BST bug, but that's
# fixed and backfilled now (see scripts/fix_footballdata_timezone.py), and a
# full hour of forward slack risked force-matching a match that genuinely
# hadn't kicked off yet to an unrelated live fixture for the same team pair).
_LIVE_SCORE_LOOKBACK = dt.timedelta(hours=6)
_LIVE_SCORE_LOOKAHEAD = dt.timedelta(minutes=15)


@app.get("/api/live-scores")
def live_scores():
    """Real score + clock for whichever of our tracked matches football-data.org
    currently has IN_PLAY/PAUSED. Returns two shapes of the same data:
    `matches` (keyed by our Match.id, for the frontend to drop straight into a
    card it already rendered) and `live` (a flat, self-contained list with
    team/competition names+crests, for the sitewide "Live now" strip, which
    has no already-rendered card to attach to on most pages). Best-effort: any
    resolution failure just drops that one match rather than erroring the
    whole response — see ingest/live_scores.py for why this never raises."""
    raw_matches = fetch_live_matches()
    if not raw_matches:
        return JSONResponse({"matches": {}, "live": []})

    session = get_session()
    try:
        teams_by_norm = {normalize(name): team_id for team_id, name in session.query(Team.id, Team.canonical_name)}
        now = dt.datetime.utcnow()

        matches: dict[str, dict] = {}
        live: list[dict] = []
        for row in raw_matches:
            if row["home_goals"] is None or row["away_goals"] is None:
                continue
            competition = session.query(Competition).filter_by(slug=row["competition_slug"]).one_or_none()
            if competition is None:
                continue
            home_id = teams_by_norm.get(normalize(FD_ORG_TO_CANONICAL.get(row["home_name"], row["home_name"])))
            away_id = teams_by_norm.get(normalize(FD_ORG_TO_CANONICAL.get(row["away_name"], row["away_name"])))
            if home_id is None or away_id is None:
                continue
            home, away = session.get(Team, home_id), session.get(Team, away_id)

            # .all(), not .one_or_none(): the same real fixture can exist as more
            # than one Match row when two sources disagree on the exact kickoff
            # time (confirmed happening, e.g. fixturedownload vs football_data.co.uk
            # times being UK-local but stored as UTC — off by exactly 1h during
            # BST, see ingest/footballdata_csv.py). Apply the live score to every
            # candidate rather than guessing which row is "the real one" — the
            # underlying duplicate-row gap is separate and not fixed by this route.
            candidates = (
                session.query(Match)
                .filter(
                    Match.competition_id == competition.id,
                    Match.home_team_id == home_id,
                    Match.away_team_id == away_id,
                    Match.status == "scheduled",
                    Match.utc_kickoff >= now - _LIVE_SCORE_LOOKBACK,
                    Match.utc_kickoff <= now + _LIVE_SCORE_LOOKAHEAD,
                )
                .all()
            )
            if not candidates:
                continue

            clock = "HT" if row["status"] == "PAUSED" else (f"{row['minute']}'" if row["minute"] is not None else None)
            for match in candidates:
                matches[str(match.id)] = {"home_goals": row["home_goals"], "away_goals": row["away_goals"], "clock": clock}

            home_primary = home.primary_color or team_colors(home.canonical_name)[0]
            away_primary = away.primary_color or team_colors(away.canonical_name)[0]
            home_color, away_color = resolve_match_colors(home_primary, away_primary, away.secondary_color)
            live.append(
                {
                    "id": candidates[0].id,
                    "competition_name": competition.name,
                    "competition_slug": competition.slug,
                    "home_name": home.canonical_name,
                    "away_name": away.canonical_name,
                    "home_abbrev": row["home_tla"] or home.canonical_name[:3].upper(),
                    "away_abbrev": row["away_tla"] or away.canonical_name[:3].upper(),
                    "home_logo": home.logo_url,
                    "away_logo": away.logo_url,
                    "home_color": home_color,
                    "away_color": away_color,
                    "home_goals": row["home_goals"],
                    "away_goals": row["away_goals"],
                    "clock": clock,
                }
            )

        return JSONResponse({"matches": matches, "live": live})
    finally:
        session.close()


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    session = get_session()
    try:
        model_run = _latest_model_run(session)
        now = dt.datetime.utcnow()
        matches = (
            session.query(Match)
            .filter(Match.status == "scheduled", Match.utc_kickoff >= now, Match.utc_kickoff <= now + dt.timedelta(days=7))
            .order_by(Match.utc_kickoff)
            .all()
        )
        cards = [build_match_card(session, m, model_run.id if model_run else None) for m in matches]
        days = group_by_day(cards)
        return templates.TemplateResponse(request, "index.html", _template_context(session, request, days=days))
    finally:
        session.close()


@app.get("/competition/{slug}", response_class=HTMLResponse)
def competition_page(request: Request, slug: str):
    session = get_session()
    try:
        competition = session.query(Competition).filter_by(slug=slug).one_or_none()
        if competition is None:
            raise HTTPException(status_code=404, detail="Competition not found")

        model_run = _latest_model_run(session)
        now = dt.datetime.utcnow()
        upcoming = (
            session.query(Match)
            .filter(Match.competition_id == competition.id, Match.status == "scheduled", Match.utc_kickoff >= now)
            .order_by(Match.utc_kickoff)
            .all()
        )
        cards = [build_match_card(session, m, model_run.id if model_run else None) for m in upcoming]
        days = group_by_day(cards)
        months = group_by_month(days)
        rail_months = group_by_month(with_today_marker(days, dt.datetime.utcnow().date()))

        no_upcoming_reason = None
        if not days:
            no_upcoming_reason = (
                "This competition's current-season fixture list isn't published by any free source yet. "
                "Historical results below still feed the model. Check back once the season's underway."
            )

        season_start_year = current_season_start_year()
        season_start = dt.date(season_start_year, 7, 1)
        season_start_dt = dt.datetime.combine(season_start, dt.time.min)
        current_season_label = f"{season_start_year}/{str(season_start_year + 1)[2:]}"

        # Full current-season history, day-grouped newest-first — replaces the
        # old flat 10-match "Recent Results" list (see plan item #14).
        recent = (
            session.query(Match)
            .filter(
                Match.competition_id == competition.id,
                Match.status == "finished",
                Match.utc_kickoff >= season_start_dt,
            )
            .order_by(Match.utc_kickoff.desc())
            .all()
        )
        result_cards = [
            {
                "match": m,
                "home_name": session.get(Team, m.home_team_id).canonical_name,
                "away_name": session.get(Team, m.away_team_id).canonical_name,
            }
            for m in recent
        ]
        results_days = group_by_day(result_cards, reverse=True)

        standings = None
        standings_empty_reason = None
        if competition.type == "league" or slug == "champions-league":
            standings = compute_standings(session, competition.id, season_start) or None
            if standings is None:
                standings_empty_reason = (
                    f"No matches have been played yet in the {current_season_label} season. "
                    "The table will appear once results start coming in."
                )
            else:
                # Tag each row with its relegation/play-off zone, or None. The
                # row count is passed so league_zone_for can refuse to band a
                # table whose size disagrees with the real league — colouring
                # the wrong club as relegated is worse than no colour at all.
                for row in standings:
                    row["zone"] = league_zone_for(slug, row["position"], len(standings))
        standings_has_zones = bool(standings) and any(r.get("zone") for r in standings)

        top_scorers = (
            session.query(PlayerStat)
            .filter_by(competition_id=competition.id, category="goals")
            .order_by(PlayerStat.rank)
            .all()
        )
        top_assists = (
            session.query(PlayerStat)
            .filter_by(competition_id=competition.id, category="assists")
            .order_by(PlayerStat.rank)
            .all()
        )
        player_stats_season = (top_scorers[0].season_label if top_scorers else None) or (
            top_assists[0].season_label if top_assists else None
        )

        stats_stale_reason = None
        player_stats_is_current = player_stats_season == current_season_label
        if competition.type == "uefa_cup":
            # A UEFA cup's entire participant pool changes every season
            # (different clubs qualify), so a prior-season top-scorers table
            # is actively misleading here, unlike a domestic league where
            # squads mostly persist year over year — see plan item #15.
            if player_stats_season and not player_stats_is_current:
                top_scorers, top_assists = [], []
                stats_stale_reason = (
                    f"Not yet available for the {current_season_label} season. The competition's "
                    "line-up changes every year, so last season's data isn't shown here."
                )

        ctx = _template_context(
            session,
            request,
            competition=competition,
            days=days,
            months=months,
            rail_months=rail_months,
            fixture_count=len(cards),
            no_upcoming_reason=no_upcoming_reason,
            results_days=results_days,
            standings=standings,
            standings_empty_reason=standings_empty_reason,
            standings_has_zones=standings_has_zones,
            top_scorers=top_scorers,
            top_assists=top_assists,
            player_stats_season=player_stats_season,
            player_stats_is_current=player_stats_is_current,
            stats_stale_reason=stats_stale_reason,
            current_season_label=current_season_label,
        )
        ctx["current_slug"] = slug
        ctx["body_theme_class"] = COMPETITION_THEME_CLASS.get(slug, "")
        return templates.TemplateResponse(request, "competition.html", ctx)
    finally:
        session.close()


@app.get("/match/{match_id}", response_class=HTMLResponse)
def match_page(request: Request, match_id: int):
    session = get_session()
    try:
        match = session.get(Match, match_id)
        if match is None:
            raise HTTPException(status_code=404, detail="Match not found")

        competition = session.get(Competition, match.competition_id)
        home = session.get(Team, match.home_team_id)
        away = session.get(Team, match.away_team_id)

        now = dt.datetime.utcnow()
        live_state = _live_state(match, now)

        prediction = None
        no_prediction_reason = None
        if match.status == "finished":
            # The latest model run's predictions never cover a match once
            # it's finished (generate_predictions only writes for currently-
            # scheduled matches) — the honest answer for "what did the site
            # predict before kickoff" is the most recent prediction actually
            # made before this match's own kickoff, same shape as
            # model/backtest.py::_latest_pre_kickoff_predictions but scoped
            # to one match.
            prediction = (
                session.query(Prediction)
                .filter(Prediction.match_id == match.id, Prediction.created_at < match.utc_kickoff)
                .order_by(Prediction.created_at.desc())
                .first()
            )
            if prediction is None:
                no_prediction_reason = "not_tracked"
        else:
            model_run = _latest_model_run(session)
            if model_run is not None:
                prediction = (
                    session.query(Prediction).filter_by(match_id=match.id, model_run_id=model_run.id).one_or_none()
                )
            if prediction is None:
                no_prediction_reason = "pending"

        home_primary = home.primary_color or team_colors(home.canonical_name)[0]
        away_primary = away.primary_color or team_colors(away.canonical_name)[0]
        home_color, away_color = resolve_match_colors(home_primary, away_primary, away.secondary_color)

        cutoff = match.utc_kickoff
        home_form = _team_form(session, home.id, before=cutoff)
        away_form = _team_form(session, away.id, before=cutoff)
        h2h = _head_to_head(session, home.id, away.id)

        market_probs = devigged_market_probs(match)
        market = None
        if market_probs is not None:
            market = {"home": market_probs[0] * 100, "draw": market_probs[1] * 100, "away": market_probs[2] * 100}

        ctx = _template_context(
            session,
            request,
            match=match,
            competition=competition,
            home=home,
            away=away,
            prediction=prediction,
            no_prediction_reason=no_prediction_reason,
            live_state=live_state,
            home_color=home_color,
            away_color=away_color,
            kickoff_display=_kickoff_display(match.utc_kickoff),
            home_form=home_form,
            away_form=away_form,
            head_to_head=h2h,
            market=market,
        )
        ctx["body_theme_class"] = COMPETITION_THEME_CLASS.get(competition.slug if competition else "", "")
        return templates.TemplateResponse(request, "match.html", ctx)
    finally:
        session.close()


@app.get("/team/{team_id}", response_class=HTMLResponse)
def team_page(request: Request, team_id: int):
    session = get_session()
    try:
        team = session.get(Team, team_id)
        if team is None:
            raise HTTPException(status_code=404, detail="Team not found")

        model_run = _latest_model_run(session)
        now = dt.datetime.utcnow()
        upcoming = (
            session.query(Match)
            .filter(
                or_(Match.home_team_id == team_id, Match.away_team_id == team_id),
                Match.status == "scheduled",
                Match.utc_kickoff >= now,
            )
            .order_by(Match.utc_kickoff)
            .limit(10)
            .all()
        )
        upcoming_cards = [build_match_card(session, m, model_run.id if model_run else None) for m in upcoming]
        recent_form = _team_form(session, team_id, before=now, limit=10)

        ratings = _cached_ratings(session)
        elo_rating = ratings.get(team_id, BASE_RATING)

        ctx = _template_context(
            session,
            request,
            team=team,
            upcoming_cards=upcoming_cards,
            recent_form=recent_form,
            elo_rating=elo_rating,
        )
        return templates.TemplateResponse(request, "team.html", ctx)
    finally:
        session.close()


@app.get("/compare", response_class=HTMLResponse)
def compare_page(request: Request, a: int | None = None, b: int | None = None):
    session = get_session()
    try:
        team_a = session.get(Team, a) if a else None
        team_b = session.get(Team, b) if b else None

        comparison = None
        upcoming_match = None
        color_a = color_b = None
        if team_a and team_b and team_a.id != team_b.id:
            now = dt.datetime.utcnow()
            ratings = _cached_ratings(session)
            primary_a = team_a.primary_color or team_colors(team_a.canonical_name)[0]
            primary_b = team_b.primary_color or team_colors(team_b.canonical_name)[0]
            color_a, color_b = resolve_match_colors(primary_a, primary_b, team_b.secondary_color)
            upcoming_match = (
                session.query(Match)
                .filter(
                    or_(
                        (Match.home_team_id == team_a.id) & (Match.away_team_id == team_b.id),
                        (Match.home_team_id == team_b.id) & (Match.away_team_id == team_a.id),
                    ),
                    Match.status == "scheduled",
                    Match.utc_kickoff >= now,
                )
                .order_by(Match.utc_kickoff)
                .first()
            )
            comparison = {
                "elo_a": ratings.get(team_a.id, BASE_RATING),
                "elo_b": ratings.get(team_b.id, BASE_RATING),
                "form_a": _team_form(session, team_a.id, before=now, limit=5),
                "form_b": _team_form(session, team_b.id, before=now, limit=5),
                "h2h": _head_to_head(session, team_a.id, team_b.id, limit=10),
                "h2h_tally": _head_to_head_tally(session, team_a.id, team_b.id),
                # Home/away here is arbitrary (team_a as "home") — this is a
                # neutral-ish estimate for comparison purposes, not a claim
                # about where either team would actually play.
                "prediction": _elo_bridge_estimate(session, team_a.id, team_b.id),
            }

        ctx = _template_context(
            session,
            request,
            team_a=team_a,
            team_b=team_b,
            color_a=color_a,
            color_b=color_b,
            comparison=comparison,
            upcoming_match=upcoming_match,
        )
        ctx["current_slug"] = "compare"
        return templates.TemplateResponse(request, "compare.html", ctx)
    finally:
        session.close()


@app.get("/accuracy", response_class=HTMLResponse)
def accuracy_page(request: Request):
    session = get_session()
    try:
        result_path = BASE_DIR / "data" / "backtest_results.json"
        result = json.loads(result_path.read_text(encoding="utf-8")) if result_path.exists() else None

        by_competition = []
        if result and not result.get("error"):
            by_competition = sorted(
                ({"slug": slug, **metrics} for slug, metrics in result["by_competition"].items()),
                key=lambda m: m["rps"],
            )

        live = backtest.live_tracking_summary(session)

        ctx = _template_context(session, request, result=result, by_competition=by_competition, live=live)
        ctx["current_slug"] = "accuracy"
        return templates.TemplateResponse(request, "accuracy.html", ctx)
    finally:
        session.close()


@app.get("/accuracy/tracked", response_class=HTMLResponse)
def tracked_predictions(request: Request, page: int = 1):
    session = get_session()
    try:
        tracked = backtest.tracked_predictions_page(session, page=page)
        ctx = _template_context(session, request, tracked=tracked)
        ctx["current_slug"] = "accuracy"
        return templates.TemplateResponse(request, "tracked_predictions.html", ctx)
    finally:
        session.close()

import datetime as dt
import json

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy import or_
from sqlalchemy.orm import Session

from app.colors import resolve_match_colors, team_colors
from app.config import BASE_DIR, settings
from app.db import SessionLocal, init_db
from app.models import Competition, EloRating, IngestLog, Match, ModelRun, Prediction, Team
from model.elo import BASE_RATING, compute_ratings

app = FastAPI(title="Football Match Prediction")
templates = Jinja2Templates(directory=str(BASE_DIR / "app" / "templates"))

static_dir = BASE_DIR / "app" / "static"
static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.on_event("startup")
def on_startup() -> None:
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
        warning = f"Data hasn't refreshed in {display} — predictions may be based on outdated form."
    return display, warning


def _kickoff_display(kickoff: dt.datetime) -> str:
    return kickoff.strftime("%a %d %b, %H:%M")


def _day_label(d: dt.date, today: dt.date) -> str:
    if d == today:
        return "Today"
    if d == today + dt.timedelta(days=1):
        return "Tomorrow"
    return d.strftime("%A %d %B")


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
        "competition_name": competition.name if competition else "",
        "round": match.round,
        "kickoff_display": _kickoff_display(match.utc_kickoff),
        "home_name": home.canonical_name,
        "away_name": away.canonical_name,
        "home_color": home_color,
        "away_color": away_color,
        "prediction": prediction,
    }


def group_by_day(cards: list[dict]) -> list[dict]:
    today = dt.datetime.utcnow().date()
    buckets: dict[dt.date, list[dict]] = {}
    for card in cards:
        d = card["match"].utc_kickoff.date()
        buckets.setdefault(d, []).append(card)
    return [
        {"label": _day_label(d, today), "cards": sorted(buckets[d], key=lambda c: c["match"].utc_kickoff)}
        for d in sorted(buckets)
    ]


_RESULT_COLOR = {"W": "text-green-600 dark:text-green-400", "D": "text-gray-500", "L": "text-red-600 dark:text-red-400"}


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
        **extra,
    }


# ---------- routes ----------


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


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
            .limit(40)
            .all()
        )
        cards = [build_match_card(session, m, model_run.id if model_run else None) for m in upcoming]
        days = group_by_day(cards)

        no_upcoming_reason = None
        if not days:
            no_upcoming_reason = (
                "This competition's current-season fixture list isn't published by any free source yet — "
                "historical results below still feed the model. Check back once the season's underway."
            )

        recent = (
            session.query(Match)
            .filter(Match.competition_id == competition.id, Match.status == "finished")
            .order_by(Match.utc_kickoff.desc())
            .limit(10)
            .all()
        )
        recent_results = [
            {
                "match": m,
                "home_name": session.get(Team, m.home_team_id).canonical_name,
                "away_name": session.get(Team, m.away_team_id).canonical_name,
            }
            for m in recent
        ]

        ctx = _template_context(
            session,
            request,
            competition=competition,
            days=days,
            no_upcoming_reason=no_upcoming_reason,
            recent_results=recent_results,
        )
        ctx["current_slug"] = slug
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

        model_run = _latest_model_run(session)
        prediction = None
        if model_run is not None:
            prediction = (
                session.query(Prediction).filter_by(match_id=match.id, model_run_id=model_run.id).one_or_none()
            )

        home_primary = home.primary_color or team_colors(home.canonical_name)[0]
        away_primary = away.primary_color or team_colors(away.canonical_name)[0]
        home_color, away_color = resolve_match_colors(home_primary, away_primary, away.secondary_color)

        cutoff = match.utc_kickoff
        home_form = _team_form(session, home.id, before=cutoff)
        away_form = _team_form(session, away.id, before=cutoff)
        h2h = _head_to_head(session, home.id, away.id)

        ctx = _template_context(
            session,
            request,
            match=match,
            competition=competition,
            home=home,
            away=away,
            prediction=prediction,
            home_color=home_color,
            away_color=away_color,
            kickoff_display=_kickoff_display(match.utc_kickoff),
            home_form=home_form,
            away_form=away_form,
            head_to_head=h2h,
        )
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

        ratings = compute_ratings(session)
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


@app.get("/accuracy", response_class=HTMLResponse)
def accuracy_page(request: Request):
    session = get_session()
    try:
        result_path = BASE_DIR / "data" / "backtest_results.json"
        result = json.loads(result_path.read_text(encoding="utf-8")) if result_path.exists() else None
        ctx = _template_context(session, request, result=result)
        return templates.TemplateResponse(request, "accuracy.html", ctx)
    finally:
        session.close()

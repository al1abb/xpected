"""SQLAlchemy ORM schema.

Design notes that matter for correctness elsewhere in the app:

- `Match` is unique on (competition_id, utc_kickoff, home_team_id, away_team_id).
  The same fixture can arrive from football-data.co.uk AND api-football; ingest
  upserts against this key rather than inserting duplicates.
- `TeamAlias` is how "Man United" / "Manchester United" / "Qarabağ FK" all
  resolve to one `Team` row. `UnresolvedAlias` is where names that couldn't be
  matched land, for manual review, instead of being silently dropped or guessed.
- `IngestLog` doubles as the freshness heartbeat the UI reads to warn about
  stale data (see app/config.py: stale_after_hours).
"""

from __future__ import annotations

import datetime as dt

from sqlalchemy import (
    JSON,
    Boolean,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Competition(Base):
    __tablename__ = "competitions"

    id: Mapped[int] = mapped_column(primary_key=True)
    slug: Mapped[str] = mapped_column(String(64), unique=True)
    name: Mapped[str] = mapped_column(String(128))
    country: Mapped[str] = mapped_column(String(64))
    type: Mapped[str] = mapped_column(String(16))  # "league" | "uefa_cup"
    fd_code: Mapped[str | None] = mapped_column(String(8), nullable=True)
    af_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    af_name: Mapped[str | None] = mapped_column(String(128), nullable=True)
    tier: Mapped[int] = mapped_column(Integer, default=1)
    sort_order: Mapped[int] = mapped_column(Integer, default=100)
    neutral_venue: Mapped[bool] = mapped_column(Boolean, default=False)

    seasons: Mapped[list["Season"]] = relationship(back_populates="competition")
    matches: Mapped[list["Match"]] = relationship(back_populates="competition")


class Season(Base):
    __tablename__ = "seasons"
    __table_args__ = (UniqueConstraint("competition_id", "label", name="uq_season_competition_label"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    competition_id: Mapped[int] = mapped_column(ForeignKey("competitions.id"))
    label: Mapped[str] = mapped_column(String(16))  # e.g. "2025/26"
    fd_season_code: Mapped[str | None] = mapped_column(String(8), nullable=True)  # e.g. "2526"
    af_season: Mapped[int | None] = mapped_column(Integer, nullable=True)  # e.g. 2025
    start_date: Mapped[dt.date | None] = mapped_column(Date, nullable=True)
    end_date: Mapped[dt.date | None] = mapped_column(Date, nullable=True)
    is_current: Mapped[bool] = mapped_column(Boolean, default=False)

    competition: Mapped[Competition] = relationship(back_populates="seasons")


class Team(Base):
    __tablename__ = "teams"

    id: Mapped[int] = mapped_column(primary_key=True)
    canonical_name: Mapped[str] = mapped_column(String(128), unique=True)
    country: Mapped[str | None] = mapped_column(String(64), nullable=True)
    primary_color: Mapped[str | None] = mapped_column(String(7), nullable=True)  # "#RRGGBB"
    secondary_color: Mapped[str | None] = mapped_column(String(7), nullable=True)
    logo_url: Mapped[str | None] = mapped_column(String(256), nullable=True)
    af_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    clubelo_name: Mapped[str | None] = mapped_column(String(128), nullable=True)

    aliases: Mapped[list["TeamAlias"]] = relationship(back_populates="team")


class TeamAlias(Base):
    __tablename__ = "team_aliases"
    __table_args__ = (UniqueConstraint("alias", "source", name="uq_alias_source"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"))
    alias: Mapped[str] = mapped_column(String(128))
    source: Mapped[str] = mapped_column(String(32))  # 'football_data' | 'api_football' | 'clubelo' | 'manual'

    team: Mapped[Team] = relationship(back_populates="aliases")


class UnresolvedAlias(Base):
    """Names ingest could not confidently map to a Team. Reviewed manually, never guessed."""

    __tablename__ = "unresolved_aliases"

    id: Mapped[int] = mapped_column(primary_key=True)
    raw_name: Mapped[str] = mapped_column(String(128))
    source: Mapped[str] = mapped_column(String(32))
    context: Mapped[str | None] = mapped_column(Text, nullable=True)
    first_seen_at: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow)
    resolved: Mapped[bool] = mapped_column(Boolean, default=False)
    resolved_team_id: Mapped[int | None] = mapped_column(ForeignKey("teams.id"), nullable=True)


class Match(Base):
    __tablename__ = "matches"
    __table_args__ = (
        UniqueConstraint(
            "competition_id", "utc_kickoff", "home_team_id", "away_team_id", name="uq_match_natural_key"
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    competition_id: Mapped[int] = mapped_column(ForeignKey("competitions.id"))
    season_id: Mapped[int | None] = mapped_column(ForeignKey("seasons.id"), nullable=True)
    utc_kickoff: Mapped[dt.datetime] = mapped_column(DateTime)
    status: Mapped[str] = mapped_column(String(16), default="scheduled")  # scheduled|finished|postponed|cancelled
    round: Mapped[str | None] = mapped_column(String(64), nullable=True)
    neutral_venue: Mapped[bool] = mapped_column(Boolean, default=False)

    home_team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"))
    away_team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"))

    home_goals: Mapped[int | None] = mapped_column(Integer, nullable=True)
    away_goals: Mapped[int | None] = mapped_column(Integer, nullable=True)
    home_goals_ht: Mapped[int | None] = mapped_column(Integer, nullable=True)
    away_goals_ht: Mapped[int | None] = mapped_column(Integer, nullable=True)

    home_shots: Mapped[int | None] = mapped_column(Integer, nullable=True)
    away_shots: Mapped[int | None] = mapped_column(Integer, nullable=True)
    home_shots_on_target: Mapped[int | None] = mapped_column(Integer, nullable=True)
    away_shots_on_target: Mapped[int | None] = mapped_column(Integer, nullable=True)
    home_corners: Mapped[int | None] = mapped_column(Integer, nullable=True)
    away_corners: Mapped[int | None] = mapped_column(Integer, nullable=True)
    home_yellow: Mapped[int | None] = mapped_column(Integer, nullable=True)
    away_yellow: Mapped[int | None] = mapped_column(Integer, nullable=True)
    home_red: Mapped[int | None] = mapped_column(Integer, nullable=True)
    away_red: Mapped[int | None] = mapped_column(Integer, nullable=True)
    home_xg: Mapped[float | None] = mapped_column(Float, nullable=True)
    away_xg: Mapped[float | None] = mapped_column(Float, nullable=True)

    source: Mapped[str] = mapped_column(String(32))  # 'football_data' | 'api_football'
    af_fixture_id: Mapped[int | None] = mapped_column(Integer, unique=True, nullable=True)

    competition: Mapped[Competition] = relationship(back_populates="matches")
    home_team: Mapped[Team] = relationship(foreign_keys=[home_team_id])
    away_team: Mapped[Team] = relationship(foreign_keys=[away_team_id])
    odds: Mapped[list["OddsSnapshot"]] = relationship(back_populates="match")
    predictions: Mapped[list["Prediction"]] = relationship(back_populates="match")


class OddsSnapshot(Base):
    """Benchmark only — never read by the model, only by the backtest."""

    __tablename__ = "odds_snapshots"

    id: Mapped[int] = mapped_column(primary_key=True)
    match_id: Mapped[int] = mapped_column(ForeignKey("matches.id"))
    bookmaker: Mapped[str | None] = mapped_column(String(64), nullable=True)
    home_odds: Mapped[float | None] = mapped_column(Float, nullable=True)
    draw_odds: Mapped[float | None] = mapped_column(Float, nullable=True)
    away_odds: Mapped[float | None] = mapped_column(Float, nullable=True)
    captured_at: Mapped[dt.datetime | None] = mapped_column(DateTime, nullable=True)
    source: Mapped[str] = mapped_column(String(32))

    match: Mapped[Match] = relationship(back_populates="odds")


class EloRating(Base):
    __tablename__ = "elo_ratings"
    __table_args__ = (UniqueConstraint("team_id", "as_of_date", "source", name="uq_elo_team_date_source"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"))
    as_of_date: Mapped[dt.date] = mapped_column(Date)
    elo: Mapped[float] = mapped_column(Float)
    source: Mapped[str] = mapped_column(String(32))  # 'clubelo' | 'internal'


class ModelRun(Base):
    __tablename__ = "model_runs"

    id: Mapped[int] = mapped_column(primary_key=True)
    run_at: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow)
    params: Mapped[dict] = mapped_column(JSON)  # decay xi, rho, league-strength multipliers, etc.
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)

    predictions: Mapped[list["Prediction"]] = relationship(back_populates="model_run")


class Prediction(Base):
    __tablename__ = "predictions"
    __table_args__ = (UniqueConstraint("match_id", "model_run_id", name="uq_prediction_match_run"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    match_id: Mapped[int] = mapped_column(ForeignKey("matches.id"))
    model_run_id: Mapped[int] = mapped_column(ForeignKey("model_runs.id"))

    home_win_prob: Mapped[float] = mapped_column(Float)
    draw_prob: Mapped[float] = mapped_column(Float)
    away_win_prob: Mapped[float] = mapped_column(Float)
    home_xg_pred: Mapped[float | None] = mapped_column(Float, nullable=True)
    away_xg_pred: Mapped[float | None] = mapped_column(Float, nullable=True)
    over_2_5_prob: Mapped[float | None] = mapped_column(Float, nullable=True)
    btts_prob: Mapped[float | None] = mapped_column(Float, nullable=True)
    top_scorelines: Mapped[list | None] = mapped_column(JSON, nullable=True)  # [{"score": "1-0", "prob": 0.12}, ...]
    confidence: Mapped[str] = mapped_column(String(16), default="normal")  # 'normal' | 'low'
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow)

    match: Mapped[Match] = relationship(back_populates="predictions")
    model_run: Mapped[ModelRun] = relationship(back_populates="predictions")


class IngestLog(Base):
    """Also the freshness heartbeat: the UI shows time-since-latest-success per source."""

    __tablename__ = "ingest_log"

    id: Mapped[int] = mapped_column(primary_key=True)
    source: Mapped[str] = mapped_column(String(32))
    competition_id: Mapped[int | None] = mapped_column(ForeignKey("competitions.id"), nullable=True)
    started_at: Mapped[dt.datetime] = mapped_column(DateTime)
    finished_at: Mapped[dt.datetime | None] = mapped_column(DateTime, nullable=True)
    status: Mapped[str] = mapped_column(String(16))  # 'ok' | 'error' | 'partial'
    rows_ingested: Mapped[int | None] = mapped_column(Integer, nullable=True)
    message: Mapped[str | None] = mapped_column(Text, nullable=True)


class ApiBudget(Base):
    """Daily request ledger — ingest refuses to call API-Football once cap is hit."""

    __tablename__ = "api_budget"

    id: Mapped[int] = mapped_column(primary_key=True)
    date: Mapped[dt.date] = mapped_column(Date, unique=True)
    requests_used: Mapped[int] = mapped_column(Integer, default=0)
    cap: Mapped[int] = mapped_column(Integer)

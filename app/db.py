import shutil
from collections.abc import Generator
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.config import BASE_DIR, settings

# Vercel's filesystem is read-only except /tmp. Rather than special-case every
# write in the app (ingest's disk cache, the ClubElo alias-resolution writes
# compute_ratings triggers as a side effect, init_db's create_all, ...), copy
# the committed snapshot into /tmp once per cold start and treat it as an
# ordinary writable SQLite file from there on — every other module keeps
# behaving exactly like local dev. Writes made during a cold start's lifetime
# are just incidental cache/alias enrichments, not data that needs to persist;
# the real source of truth is the committed snapshot, refreshed locally and
# pushed via git.
if settings.vercel:
    _tmp_db_path = Path("/tmp/app.db")
    if not _tmp_db_path.exists():
        shutil.copy(BASE_DIR / "data" / "app.db", _tmp_db_path)
    database_url = f"sqlite:///{_tmp_db_path.as_posix()}"
else:
    database_url = settings.database_url

connect_args = {"check_same_thread": False} if database_url.startswith("sqlite") else {}
engine = create_engine(database_url, connect_args=connect_args)
SessionLocal = sessionmaker(bind=engine)


def get_session() -> Generator[Session, None, None]:
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


def init_db() -> None:
    from sqlalchemy import inspect, text

    from app import models  # noqa: F401  (registers tables on Base.metadata)
    from app.models import Base

    Base.metadata.create_all(bind=engine)

    # `create_all` only creates missing TABLES, never adds columns to one that
    # already exists — this project has no migration tool, so a column added
    # to an existing table needs its own one-line bootstrap here, guarded to
    # run at most once per column. Add a tuple below whenever a new column
    # lands on a pre-existing table (new tables need nothing — create_all
    # handles those).
    _ADDED_COLUMNS = [
        ("matches", "bbs_match_id", "VARCHAR(36)"),
        ("matches", "home_formation", "VARCHAR(16)"),
        ("matches", "away_formation", "VARCHAR(16)"),
        ("lineups", "order_index", "INTEGER"),
        ("player_match_stats", "headshot_url", "VARCHAR(256)"),
        ("matches", "highlightly_match_id", "INTEGER"),
    ]
    inspector = inspect(engine)
    existing_tables = set(inspector.get_table_names())
    for table, column, coltype in _ADDED_COLUMNS:
        if table not in existing_tables:
            continue
        existing_columns = {c["name"] for c in inspector.get_columns(table)}
        if column not in existing_columns:
            with engine.begin() as conn:
                conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {column} {coltype}"))

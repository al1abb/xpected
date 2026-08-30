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
    from app import models  # noqa: F401  (registers tables on Base.metadata)
    from app.models import Base

    Base.metadata.create_all(bind=engine)

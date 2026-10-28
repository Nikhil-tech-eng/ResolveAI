"""Database engine and session setup."""

from collections.abc import Iterator
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from db.models import Base

DATABASE_PATH = Path(__file__).resolve().parents[1] / "data" / "support.db"
DATABASE_URL = f"sqlite:///{DATABASE_PATH.as_posix()}"

engine = create_engine(
	DATABASE_URL,
	connect_args={"check_same_thread": False},
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def init_db() -> None:
	"""Create all database tables if they do not already exist."""
	DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
	Base.metadata.create_all(bind=engine)


def get_session() -> Iterator[Session]:
	"""Yield a database session and ensure it is closed afterward."""
	session = SessionLocal()
	try:
		yield session
	finally:
		session.close()
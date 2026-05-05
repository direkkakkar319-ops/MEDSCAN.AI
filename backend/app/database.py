"""
database.py — SQLAlchemy engine and session factory.

Two separate engines are created here:
  - engine / SessionLocal      : synchronous (psycopg2 driver)
  - async_engine / AsyncSessionLocal : async (asyncpg driver)

Why two engines?
  FastAPI route handlers are async functions but SQLAlchemy's sync session
  works perfectly fine inside them via Depends(get_db).  Celery workers,
  however, are plain synchronous processes with no asyncio event loop, so
  they MUST use SessionLocal (psycopg2).  Mixing asyncpg with Celery causes
  "Future attached to a different loop" errors.

  AsyncSessionLocal is kept for any future async SQLAlchemy usage or
  background tasks that run inside an async context.
"""

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from app.config import settings

# ── Synchronous engine (psycopg2) ─────────────────────────────────────────────
# Used by FastAPI routes (via get_db) and Celery workers (via SessionLocal()).
# psycopg2 is thread-safe and works in any execution context.
engine = create_engine(settings.DATABASE_URL)

# ── Async engine (asyncpg) ────────────────────────────────────────────────────
# Derives the asyncpg URL from the sync URL when ASYNC_DATABASE_URL is not
# explicitly set.  asyncpg requires the postgresql+asyncpg:// scheme.
if settings.ASYNC_DATABASE_URL:
    async_db_url = settings.ASYNC_DATABASE_URL
else:
    # Replace the scheme in-place: postgresql:// → postgresql+asyncpg://
    async_db_url = settings.DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://")

async_engine = create_async_engine(async_db_url, echo=True)  # echo=True logs SQL

# ── Session factories ──────────────────────────────────────────────────────────
# autocommit=False  — every write requires an explicit db.commit() call,
#                     preventing accidental partial writes.
# autoflush=False   — SQLAlchemy won't automatically send pending changes to
#                     the DB before each query, giving us full control.
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,        # uses the synchronous psycopg2 engine
)

# Async session factory — expire_on_commit=False keeps ORM objects usable
# after commit without triggering a lazy-load (which would need another query).
AsyncSessionLocal = sessionmaker(
    async_engine,
    class_=AsyncSession,
    expire_on_commit=False,
)

# ── Declarative base ──────────────────────────────────────────────────────────
# All ORM model classes (Report, Task, User, …) inherit from Base.
# Base.metadata holds the schema definition used by create_all() and Alembic.
Base = declarative_base()


def get_db():
    """
    FastAPI dependency that provides a synchronous database session.

    Usage in a route:
        @router.get("/example")
        def my_route(db: Session = Depends(get_db)):
            ...

    How it works:
      - Opens a new session from SessionLocal before each request.
      - Yields it to the route handler (the route runs while the generator
        is paused here).
      - Closes the session in the finally block once the response has been
        sent, releasing the connection back to the psycopg2 connection pool.

    Why yield instead of return?
      yield turns this into a generator-based context manager, which FastAPI
      recognises as a dependency with cleanup logic.  The finally block
      always runs, even if the route raises an exception, preventing
      connection leaks.
    """
    db = SessionLocal()
    try:
        yield db       # route handler runs here with access to db
    finally:
        db.close()     # always release the connection, even on error

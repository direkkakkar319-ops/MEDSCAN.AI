"""
main.py — FastAPI application factory.

This is the entry point for the entire backend API. It:
  1. Creates the FastAPI app instance.
  2. Attaches rate-limiting middleware (slowapi) so individual IPs can't
     spam auth or upload endpoints.
  3. Attaches CORS middleware so the React frontend (localhost in dev,
     medscan-ai-s7t1.onrender.com in prod) can reach the API from a
     different origin.
  4. Runs Alembic database migrations automatically on startup so the
     production Postgres schema is always up to date without a manual
     migration step.
  5. Mounts all the API routers under their respective URL prefixes.
"""

import logging
from alembic.config import Config
from alembic import command
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from app.api import upload, status, reports, compare
from app.config import settings, BASE_DIR
from app.database import engine
from app import models
from app.auth.router import router as auth_router

logger = logging.getLogger(__name__)

# ── Rate limiter ──────────────────────────────────────────────────────────────
# slowapi reads the client's real IP (get_remote_address) and tracks request
# counts per IP per time window.  Limits are declared per-route with the
# @limiter.limit("N/period") decorator.
limiter = Limiter(key_func=get_remote_address)

# ── FastAPI app instance ──────────────────────────────────────────────────────
# PROJECT_NAME comes from app/config.py (reads MEDSCAN_PROJECT_NAME env var).
# The title shows up in the auto-generated /docs Swagger UI.
app = FastAPI(title=settings.PROJECT_NAME)

# Attach the limiter to app.state so slowapi can find it when a route fires.
app.state.limiter = limiter

# Register the default 429 handler so rate-limited requests get a proper JSON
# error body instead of a generic 500.
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ── CORS middleware ───────────────────────────────────────────────────────────
# The browser enforces the Same-Origin Policy, blocking JavaScript from calling
# APIs on a different domain.  CORS middleware responds to the browser's
# preflight OPTIONS request with the correct Allow-Origin header so the call
# is permitted.
#
# allow_origin_regex covers:
#   - http://localhost:<any port>    (Vite dev server, usually 5173)
#   - http://127.0.0.1:<any port>   (alternative localhost form)
#   - https://medscan-ai-s7t1.onrender.com  (production frontend)
#
# allow_credentials=True is required because the frontend sends the Bearer
# token in the Authorization header (not a cookie, but FastAPI still needs
# this flag for headers to be forwarded).
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$|^https://medscan-ai-s7t1\.onrender\.com$",
    allow_credentials=True,
    allow_methods=["*"],   # GET, POST, PUT, DELETE, OPTIONS, …
    allow_headers=["*"],   # Authorization, Content-Type, …
)


@app.on_event("startup")
def on_startup():
    """
    Run once when the Uvicorn server starts, before any HTTP request is handled.

    Steps:
      1. Try to run Alembic migrations to 'head' (the latest schema version).
         This ensures the Neon Postgres tables match the current codebase on
         every deploy without a manual migration command.
      2. If Alembic fails (e.g. alembic.ini not found in the container path),
         fall back to SQLAlchemy's create_all() which creates any missing
         tables without running versioned migrations.

    Why auto-migrate instead of a manual step?
      Render's free tier restarts containers on every deploy.  Auto-migrating
      on startup means zero-downtime schema updates: the new container applies
      migrations before serving traffic.
    """
    try:
        # Point Alembic at the alembic.ini file and the versions/ directory
        # relative to the project root (BASE_DIR).
        alembic_cfg = Config(str(BASE_DIR / "alembic.ini"))
        alembic_cfg.set_main_option("script_location", str(BASE_DIR / "alembic"))
        command.upgrade(alembic_cfg, "head")   # apply all unapplied migrations
        logger.info("Alembic migrations applied successfully")
    except Exception as e:
        # Fallback: create tables from the ORM metadata.
        # This does NOT track migration versions, so schema diffs may be missed,
        # but it keeps the service running if alembic.ini is misconfigured.
        logger.warning(f"Alembic migration failed ({e}), falling back to create_all")
        models.Base.metadata.create_all(bind=engine)


# ── Router registration ───────────────────────────────────────────────────────
# Each router is a mini FastAPI sub-application that handles one group of
# endpoints.  The prefix is prepended to every route inside the router.
#
#   upload.router  →  /api/upload
#   status.router  →  /api/status/{report_id}
#   reports.router →  /api/reports, /api/reports/{id}, /api/reports/{id}/download
#   compare.router →  /api/compare, /api/compare/{id}
#   auth_router    →  /auth/register, /auth/login, /auth/refresh
app.include_router(upload.router,  prefix="/api", tags=["upload"])
app.include_router(status.router,  prefix="/api", tags=["status"])
app.include_router(reports.router, prefix="/api", tags=["reports"])
app.include_router(compare.router, prefix="/api", tags=["compare"])
app.include_router(auth_router)   # already has prefix="/auth" inside the router


@app.get("/")
async def root():
    """
    Health-check route for the root URL.

    Render's health-check pings / to decide if the service is up.
    Returning a simple JSON object confirms the API process is alive
    and FastAPI is responding to HTTP requests.
    """
    return {"message": "Welcome to Disease Risk Predictor API"}

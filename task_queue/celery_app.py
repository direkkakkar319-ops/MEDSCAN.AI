"""
task_queue/celery_app.py — Celery worker configuration and lifecycle hooks.

This module creates the single shared `celery_app` object that every other
part of the project imports.  It configures:
  - Broker  : Redis (where Celery SENDS tasks to queue them)
  - Backend : Redis (where Celery STORES task results after completion)
  - Task serialization, timeouts, concurrency, SSL for Upstash rediss:// URLs

The Celery app is created here and NOT in tasks.py to avoid circular imports:
  tasks.py imports celery_app from here
  celery_app here lists tasks.py in `include`
If they were in the same file there would be no clean import order.

Worker lifecycle hooks:
  @worker_process_init  — fires inside EACH worker subprocess at startup
  @worker_ready         — fires once when the whole worker is connected to Redis
"""

from celery import Celery
from celery.signals import worker_ready, worker_process_init
import os
import logging

logger = logging.getLogger(__name__)

# ── Broker and backend URLs ───────────────────────────────────────────────────
# CELERY_BROKER_URL  — where Celery sends tasks (Redis acts as the message queue)
# CELERY_RESULT_BACKEND — where completed task results are stored so the API can
#                         retrieve them later with AsyncResult(task_id).get()
# Both default to the local Redis instance used in development.
# In production (Upstash) these are injected by docker-compose from .env.
CELERY_BROKER_URL     = os.getenv("CELERY_BROKER_URL",     "redis://localhost:6379/0")
CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/0")

# ── Celery application object ─────────────────────────────────────────────────
# "MEDSCAN AI" is just a display name shown in Flower and logs — it has no
# functional effect.
# `include` tells Celery where to find @celery_app.task decorated functions so
# they are registered before any worker tries to execute them.
celery_app = Celery(
    "MEDSCAN AI",
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND,
    include=["task_queue.tasks"],
)

# ── SSL detection for Upstash (rediss://) ─────────────────────────────────────
# Upstash free-tier Redis uses TLS — its URL starts with "rediss://" (double s).
# When TLS is detected we pass ssl_cert_reqs="none" to skip certificate
# verification.  This is acceptable for Redis traffic where we trust the
# Upstash endpoint; it avoids having to bundle CA certificates in the container.
_use_ssl = CELERY_BROKER_URL.startswith("rediss://")
_ssl_opts = {"ssl_cert_reqs": "none"} if _use_ssl else {}

# ── Worker configuration ──────────────────────────────────────────────────────
celery_app.conf.update(
    # Serialise all task arguments and results as JSON strings.
    # JSON is human-readable, language-agnostic, and safe (no pickle RCE risk).
    task_serializer    = "json",
    accept_content     = ["json"],
    result_serializer  = "json",

    # UTC timestamps in all task metadata so logs are consistent across zones.
    timezone           = "UTC",
    enable_utc         = True,

    # Hard timeout per task: 5 minutes.
    # PaddleOCR on a cold CPU is the bottleneck (~60-120 s on a 1-vCPU container).
    # 300 s gives headroom for slow images without blocking the queue indefinitely.
    task_time_limit    = 300,

    # Single worker process: PaddleOCR loads ~400 MB of model weights into RAM.
    # Running two concurrent processes would double RAM usage and crash on free-tier.
    # When throughput must scale, horizontal scaling (more workers) is preferred
    # over vertical (more concurrency per worker) for memory-heavy ML workloads.
    worker_concurrency = 1,

    # Task results expire from Redis after 1 hour.
    # The API reads results synchronously (status polling), so 1 h is more than enough.
    # After expiry the result is gone but the task row in the `tasks` DB table persists.
    result_expires     = 3600,

    # If Redis is unreachable at startup, keep retrying instead of crashing.
    # This prevents Docker Compose race conditions where Redis starts a few seconds
    # after the worker container.
    broker_connection_retry_on_startup = True,

    # TLS settings: passed only when the broker URL uses rediss://.
    broker_use_ssl        = _ssl_opts if _use_ssl else None,
    redis_backend_use_ssl = _ssl_opts if _use_ssl else None,

    # Route all tasks to the "default" queue — no priority lanes needed yet.
    # If a high-priority queue is added later, add its name here and update
    # the task decorator with queue="priority".
    task_default_queue = "default",
    task_routes        = {},
)


# ── Worker lifecycle hooks ────────────────────────────────────────────────────

@worker_process_init.connect
def _init_worker(**kwargs):
    """
    Fires inside EACH Celery worker subprocess immediately after it forks.

    Previously this pre-loaded the OCR engine to avoid cold-start delay on
    the first task.  Pre-loading is now skipped because:
      - get_ocr_runner() uses a module-level singleton — it loads on first use
        and is cached for all subsequent tasks in the same process.
      - Pre-loading in worker_process_init caused memory spikes on restart
        even when no tasks were queued.
    The log line confirms the worker subprocess is alive.
    """
    logger.info("Worker process initialising — OCR engine will load on first task")


@worker_ready.connect
def on_worker_ready(**kwargs):
    """
    Fires ONCE when the entire Celery worker is up and connected to Redis.

    This is the last step in the worker startup sequence:
      1. Process forks (worker_process_init fires inside each fork).
      2. Broker connection established.
      3. Task routes registered.
      4. This signal fires — the worker is now accepting tasks from the queue.

    The log line is the confirmation message that shows in docker-compose logs
    to tell you the pipeline is ready to process uploads.
    """
    logger.info("Celery worker is ready and listening for tasks.")


# ── Dev entrypoint ────────────────────────────────────────────────────────────
# Run `python -m task_queue.celery_app` in development to start the worker
# without Docker.  In production, docker-compose runs:
#   celery -A task_queue.celery_app worker --loglevel=info
if __name__ == "__main__":
    celery_app.start()

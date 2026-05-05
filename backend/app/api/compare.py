"""
api/compare.py — Report comparison API.

Three routes manage the full comparison lifecycle:

  POST /api/compare          — validate two reports and kick off comparison
  GET  /api/compare          — list the user's past comparisons
  GET  /api/compare/{id}     — poll status / fetch full comparison result

How comparison works end-to-end:
  1. User selects two completed same-type reports in the frontend.
  2. POST /api/compare creates a ReportComparison row (status="pending")
     and enqueues the compare_reports Celery task.
  3. Frontend polls GET /api/compare/{id} every 2 seconds.
  4. Worker fetches both reports' extracted_metrics and prediction_risks,
     runs ReportComparator, and writes the result back to the DB.
  5. Polling detects status="completed" and renders the comparison UI.

Validation enforced by this API:
  - Both reports must exist and belong to the requesting user.
  - Both reports must be the same report_type (can't compare blood vs lipid).
  - Both reports must have status="completed" (OCR and ML must be done).
"""

import uuid
import logging
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import Report, ReportComparison
from app.auth.dependencies import get_current_active_user
from app.auth.models import User
from task_queue.celery_app import celery_app

router = APIRouter()
logger = logging.getLogger(__name__)


class CompareRequest(BaseModel):
    """Request body for POST /api/compare."""
    report1_id: int   # ID of the first (newer) report
    report2_id: int   # ID of the second (baseline) report


@router.post("/compare")
async def trigger_comparison(
    body: CompareRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    Validate two reports, create a comparison row, and enqueue the Celery task.

    The comparison_id returned is a UUID string that the frontend uses to
    poll GET /api/compare/{comparison_id} until status="completed".

    Validation checks (all enforced before any DB write):
      1. Both report IDs exist in the database.
      2. Both reports belong to the requesting user (prevents cross-user access).
      3. Both reports have the same report_type (OCR patterns and feature
         vectors differ per type — comparing blood vs lipid has no meaning).
      4. Both reports have status="completed" (metrics and predictions must
         be available for the comparator to compute deltas).

    Error handling after validation:
      - If the DB INSERT for the comparison row fails: rollback and 500.
      - If the Celery task cannot be queued (Redis unreachable): mark the
        comparison row as "failed" immediately so the frontend stops polling.

    Response:
      { "comparison_id": "<uuid-string>" }

    Raises:
      404 — one or both report IDs not found or not owned by user.
      400 — reports are different types or not yet fully processed.
      500 — DB error creating the comparison row, or Celery queue failure.
    """
    # Load and validate both reports.  Ownership is checked by combining
    # the user_id filter with the id filter in a single query.
    r1 = db.query(Report).filter(
        Report.id == body.report1_id,
        Report.user_id == current_user.id,
    ).first()
    r2 = db.query(Report).filter(
        Report.id == body.report2_id,
        Report.user_id == current_user.id,
    ).first()

    if not r1 or not r2:
        raise HTTPException(status_code=404, detail="One or both reports not found")

    # Same-type check — the comparator aligns metrics by key name; different
    # report types have completely different feature sets.
    if r1.report_type != r2.report_type:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Reports must be the same type to compare "
                f"(got {r1.report_type} vs {r2.report_type})"
            ),
        )

    # Completion check — the worker needs extracted_metrics and prediction_risks
    # which are only written once the full pipeline finishes.
    if r1.status != "completed" or r2.status != "completed":
        raise HTTPException(
            status_code=400,
            detail="Both reports must be fully processed before comparing",
        )

    # Generate a UUID for this comparison.  Using str(uuid.uuid4()) gives a
    # standard 36-character hyphenated UUID e.g. "550e8400-e29b-41d4-a716-446655440000".
    comparison_id = str(uuid.uuid4())

    # Insert the comparison row BEFORE queuing the task so the worker can
    # find the row by ID when it calls _save_comparison().
    try:
        comp = ReportComparison(
            id=comparison_id,
            user_id=current_user.id,
            report1_id=body.report1_id,
            report2_id=body.report2_id,
            report_type=r1.report_type,
            status="pending",   # will be updated to "completed" or "failed" by worker
        )
        db.add(comp)
        db.commit()
    except Exception as exc:
        db.rollback()
        logger.error(f"[trigger_comparison] DB error creating comparison row: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Database error: {exc}")

    # Enqueue the comparison task.
    # send_task() is used instead of compare_reports.delay() because the
    # compare_reports task lives in task_queue.tasks and we don't import it
    # here to avoid circular dependencies (tasks.py imports app.models).
    try:
        celery_app.send_task(
            "task_queue.tasks.compare_reports",
            args=[comparison_id, body.report1_id, body.report2_id],
        )
    except Exception as exc:
        logger.error(f"[trigger_comparison] Failed to queue task: {exc}", exc_info=True)
        # Mark failed immediately so the frontend stops polling after this request.
        try:
            comp = db.query(ReportComparison).filter(
                ReportComparison.id == comparison_id
            ).first()
            if comp:
                comp.status = "failed"
                db.commit()
        except Exception:
            db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to queue comparison task: {exc}")

    return {"comparison_id": comparison_id}


@router.get("/compare")
async def list_comparisons(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    Return the current user's comparison history, newest first.

    Used by the Compare section's "Past Comparisons" panel so the user
    can reload a previous comparison without re-running the analysis.

    Response per item:
      {
        "comparison_id":  str,   UUID
        "report1_id":     int,
        "report2_id":     int,
        "report_type":    str,
        "status":         str,   pending / completed / failed
        "trend_analysis": str,   IMPROVING / WORSENING / STABLE — null if pending
        "created_at":     str,   ISO 8601
      }
    """
    comps = (
        db.query(ReportComparison)
        .filter(ReportComparison.user_id == current_user.id)
        .order_by(ReportComparison.created_at.desc())
        .all()
    )
    return [
        {
            "comparison_id":  c.id,
            "report1_id":     c.report1_id,
            "report2_id":     c.report2_id,
            "report_type":    c.report_type,
            "status":         c.status,
            "trend_analysis": c.trend_analysis,
            "created_at":     c.created_at.isoformat() if c.created_at else None,
        }
        for c in comps
    ]


@router.get("/compare/{comparison_id}")
async def get_comparison(
    comparison_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    Return the status and result of a specific comparison.

    The frontend polls this every 2 seconds after triggering a comparison.
    Once status="completed" the full comparison_data is included and the
    frontend stops polling and renders the comparison dashboard.

    Response when completed:
      {
        "comparison_id":       str,
        "status":              "completed",
        "report_type":         str,
        "report1_id":          int,
        "report2_id":          int,
        "trend_analysis":      str,   IMPROVING / WORSENING / STABLE
        "comparison_data": {
          "metrics":           list,  per-metric delta with improved flag
          "significant_changes": list, subset where |percent_change| > 5%
          "risk_comparison":   list,  per-disease risk delta
          "summary": {
            "overall_trend":   str,
            "improved_count":  int,
            "worsened_count":  int,
            "unchanged_count": int,
            "total_metrics":   int,
          }
        },
        "significant_changes": list,  mirror of comparison_data.significant_changes
        "created_at":          str,
      }

    Raises:
      404 — comparison not found or belongs to another user.
    """
    comp = db.query(ReportComparison).filter(
        ReportComparison.id == comparison_id,
        ReportComparison.user_id == current_user.id,
    ).first()

    if not comp:
        raise HTTPException(status_code=404, detail="Comparison not found")

    return {
        "comparison_id":       comp.id,
        "status":              comp.status,
        "report_type":         comp.report_type,
        "report1_id":          comp.report1_id,
        "report2_id":          comp.report2_id,
        "trend_analysis":      comp.trend_analysis,
        "comparison_data":     comp.comparison_data,         # null until completed
        "significant_changes": comp.significant_changes,     # null until completed
        "created_at":          comp.created_at.isoformat() if comp.created_at else None,
    }

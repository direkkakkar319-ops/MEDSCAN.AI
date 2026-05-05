"""
api/status.py — Report processing status endpoint.

GET /api/status/{report_id}

This is the endpoint the frontend polls every 3 seconds after uploading
a report.  It returns the current pipeline stage and — once processing is
complete — the full prediction result so the frontend can render the
analysis without a second request.

Status lifecycle (written by the Celery worker in tasks.py):
  "uploaded"      → file saved, task enqueued, worker not yet started
  "preprocessing" → worker started, file downloaded from S3 (if needed)
  "ocr_complete"  → PaddleOCR finished, metrics extracted, ML running
  "completed"     → prediction saved in tasks table
  "failed"        → any unrecoverable error during OCR or prediction

Security: report_id is validated against current_user.id.
  A user can never poll another user's report — they get a 404, not a 403,
  to avoid confirming that the report ID exists.
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import Report, Task
from app.auth.dependencies import get_current_active_user
from app.auth.models import User

router = APIRouter()


@router.get("/status/{report_id}")
async def get_report_status(
    report_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    Return the current processing status of an uploaded report.

    The frontend calls this every 3 seconds (up to 60 attempts = 5 min
    timeout) after upload.  On each poll it updates the progress indicator
    in the UI.

    How the result is found:
      The Celery worker writes prediction output into the tasks table with a
      task_id following the pattern "predict-{report_id}-{uuid8}".  We use a
      LIKE query to find it without needing to store the Celery task UUID in
      the reports table.

    Response fields:
      report_id        — echo of the requested ID
      status           — current pipeline stage (see lifecycle above)
      report_type      — blood / lipid / vitamin_d / hormone / kidney / liver
      filename         — original filename for display in the UI header
      ocr_confidence   — mean PaddleOCR confidence (0–1); null until ocr_complete
      extracted_metrics— {metric: {value, unit, source}} dict; only included
                         when status=="completed" to avoid sending partial data
      created_at       — ISO 8601 timestamp of the upload
      processed_at     — ISO 8601 timestamp of OCR completion (or failure)
      result           — full ML prediction dict (risks, risk_level, key_factors,
                         recommendations, shap_values, model_version, ocr_coverage,
                         raw_xgb_probas); null until status=="completed"

    Raises:
      401 UNAUTHORIZED — missing/expired token.
      404 NOT FOUND    — report_id doesn't exist OR belongs to a different user.
    """
    # Load the report and verify ownership in a single query.
    # Filtering by both id AND user_id means an attacker who guesses a
    # report_id of another user just gets a 404 — no information leak.
    report = db.query(Report).filter(
        Report.id == report_id,
        Report.user_id == current_user.id,
    ).first()

    if not report:
        raise HTTPException(status_code=404, detail="Report not found")

    # Find the prediction task row written by _save_prediction() in tasks.py.
    # The LIKE pattern "predict-42-%" matches any UUID suffix, so we don't
    # need to store the Celery task ID in the reports table.
    task = (
        db.query(Task)
        .filter(Task.task_id.like(f"predict-{report_id}-%"))
        .first()
    )

    return {
        "report_id":   report_id,
        "status":      report.status,
        "report_type": report.report_type,
        "filename":    report.filename,
        "ocr_confidence": report.ocr_confidence,

        # Only expose extracted_metrics once the pipeline is fully done.
        # During intermediate stages the dict may be partially populated or
        # None, which would confuse the frontend's transformResult() function.
        "extracted_metrics": report.extracted_metrics if report.status == "completed" else None,

        "created_at":  report.created_at.isoformat() if report.created_at else None,
        "processed_at": report.processed_at.isoformat() if report.processed_at else None,

        # task.result is None until the worker writes the prediction row.
        # The frontend checks result != null before rendering the dashboard.
        "result": task.result if task else None,
    }

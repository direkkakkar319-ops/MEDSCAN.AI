"""
api/reports.py — Report history and file download endpoints.

Three routes:
  GET /api/reports           — list all of the user's reports (history panel)
  GET /api/reports/{id}      — full detail for one report (for re-rendering)
  GET /api/reports/{id}/download — stream or redirect to the original file

These endpoints are used by the History section of the frontend and by the
Compare feature (which needs a list of completed reports to select from).

All queries are filtered by current_user.id so users can only ever see
their own data — there is no admin route here.
"""

import os
import mimetypes
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse, RedirectResponse
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import Report, Task
from app.auth.dependencies import get_current_active_user
from app.auth.models import User

router = APIRouter()


@router.get("/reports")
async def list_reports(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    Return a summary list of all reports for the current user, newest first.

    Used by:
      - History section: shows the report card grid with type, date, risk level.
      - Compare section: populates the two report-selector dropdowns.

    For each report we also join the corresponding tasks row (if it exists)
    to extract risk_level and the per-disease risks dict.  This avoids a
    second API call from the frontend just to get the risk summary.

    Response shape per item:
      {
        "id":             int,
        "filename":       str,
        "report_type":    str,   blood / lipid / …
        "status":         str,   uploaded / completed / failed / …
        "risk_level":     str,   low / moderate / high / critical — null if not done
        "risks":          dict,  {disease: float} — empty if not done
        "ocr_confidence": float, null if not done
        "created_at":     str,   ISO 8601
        "processed_at":   str,   ISO 8601 — null if not done
      }

    Performance note: N+1 query pattern (one tasks query per report row).
    Acceptable at current scale; could be replaced with a LEFT JOIN or
    subquery if the history list grows very large.
    """
    reports = (
        db.query(Report)
        .filter(Report.user_id == current_user.id)
        .order_by(Report.created_at.desc())   # newest first
        .all()
    )

    result = []
    for r in reports:
        # Find the matching task row using the same LIKE pattern as status.py.
        # task is None for reports that are still processing or failed before
        # the prediction step.
        task = (
            db.query(Task)
            .filter(Task.task_id.like(f"predict-{r.id}-%"))
            .first()
        )

        # Extract risk summary from the task result if available.
        risk_level = None
        risks = {}
        if task and task.result:
            risk_level = task.result.get("risk_level")
            risks      = task.result.get("risks", {})

        result.append({
            "id":             r.id,
            "filename":       r.filename,
            "report_type":    r.report_type,
            "status":         r.status,
            "risk_level":     risk_level,
            "risks":          risks,
            "ocr_confidence": r.ocr_confidence,
            "created_at":     r.created_at.isoformat() if r.created_at else None,
            "processed_at":   r.processed_at.isoformat() if r.processed_at else None,
        })

    return result


@router.get("/reports/{report_id}")
async def get_report(
    report_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    Return full detail for a single report including all OCR metrics and
    the complete ML prediction result.

    Used when the frontend needs to re-render a historical report without
    re-running the analysis.

    Response shape:
      {
        "id":                int,
        "filename":          str,
        "report_type":       str,
        "status":            str,
        "ocr_confidence":    float,
        "extracted_metrics": dict,  {metric: {value, unit, source}}
        "created_at":        str,
        "processed_at":      str,
        "prediction":        dict   full task.result — null if not completed
      }

    Raises:
      404 NOT FOUND — report doesn't exist or belongs to another user.
    """
    report = db.query(Report).filter(
        Report.id == report_id,
        Report.user_id == current_user.id,   # ownership check
    ).first()

    if not report:
        raise HTTPException(status_code=404, detail="Report not found")

    task = (
        db.query(Task)
        .filter(Task.task_id.like(f"predict-{report_id}-%"))
        .first()
    )

    return {
        "id":                report.id,
        "filename":          report.filename,
        "report_type":       report.report_type,
        "status":            report.status,
        "ocr_confidence":    report.ocr_confidence,
        "extracted_metrics": report.extracted_metrics,
        "created_at":        report.created_at.isoformat() if report.created_at else None,
        "processed_at":      report.processed_at.isoformat() if report.processed_at else None,
        "prediction":        task.result if task else None,
    }


@router.get("/reports/{report_id}/download")
async def download_report(
    report_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    Download the original uploaded file for a report.

    Two modes depending on USE_S3:

    S3 mode (production):
      Generates a presigned S3 URL valid for 1 hour and returns a 302
      redirect to it.  The browser follows the redirect and downloads
      directly from S3/Supabase.  This keeps large file transfers off
      the FastAPI process, saving bandwidth and memory.
      ResponseContentDisposition forces the filename in the browser's
      Save dialog.

    Local mode (dev):
      Returns the file directly via FastAPI's FileResponse, which streams
      it from /app/data/raw_uploads/.  content_type is guessed from the
      extension if not stored (mimetypes.guess_type).

    Note: In S3 mode the original file is DELETED from S3 after the
    Celery worker completes (see tasks.py: _get_s3().delete_object).
    The download endpoint will 500 in that case.  This is a known
    limitation — if original file access is needed after processing,
    remove the S3 cleanup step in tasks.py.

    Raises:
      404 NOT FOUND — report doesn't exist, belongs to another user,
                      or the local file was deleted.
      500           — S3 presign error.
    """
    report = db.query(Report).filter(
        Report.id == report_id,
        Report.user_id == current_user.id,
    ).first()

    if not report:
        raise HTTPException(status_code=404, detail="Report not found")

    file_path = report.file_path
    _USE_S3 = os.getenv("USE_S3", "false").lower() == "true"

    if _USE_S3:
        import boto3
        from botocore.config import Config
        s3_client = boto3.client(
            "s3",
            endpoint_url=os.getenv("S3_ENDPOINT_URL"),
            aws_access_key_id=os.getenv("S3_ACCESS_KEY"),
            aws_secret_access_key=os.getenv("S3_SECRET_KEY"),
            config=Config(signature_version="s3v4"),
            region_name=os.getenv("S3_REGION", "us-east-1"),
        )
        try:
            # generate_presigned_url creates a time-limited URL that includes
            # an HMAC signature — no auth header is needed by the browser.
            url = s3_client.generate_presigned_url(
                "get_object",
                Params={
                    "Bucket": os.getenv("S3_BUCKET_NAME"),
                    "Key": file_path,   # the S3 key stored in reports.file_path
                    # Forces "Save As" dialog in the browser with the original filename.
                    "ResponseContentDisposition": f'attachment; filename="{report.filename}"',
                },
                ExpiresIn=3600,   # URL valid for 1 hour
            )
            return RedirectResponse(url=url)   # 302 → browser downloads from S3
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"S3 download error: {str(e)}")
    else:
        # Local dev: check the file still exists before streaming it.
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found on server")

        return FileResponse(
            path=file_path,
            filename=report.filename,
            # Use stored content_type, fall back to mimetypes guess, then binary.
            media_type=(
                report.content_type
                or mimetypes.guess_type(file_path)[0]
                or "application/octet-stream"
            ),
        )

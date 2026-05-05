"""
api/upload.py — Report file upload endpoint.

Handles POST /api/upload which is the first step of the entire pipeline.

What this file does:
  1. Validates the report_type (must be one of 6 known types).
  2. Writes the uploaded file to storage — Supabase S3 in production,
     or a local /app/data/raw_uploads directory in local Docker dev.
  3. Creates a Report row in the database with status="uploaded".
  4. Enqueues a Celery background task (process_medical_report) that will
     pick up the file, run OCR, run ML predictions, and update the row.
  5. Returns a list of per-file results so the frontend knows which files
     succeeded and which failed.

Authentication: Bearer JWT required — each file is tied to current_user.id
                so users can only see their own reports.
"""

import os
import shutil
from pathlib import Path
from typing import List
from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException
from sqlalchemy.orm import Session
from app.database import get_db
from app.models import Report
from app.auth.dependencies import get_current_active_user
from app.auth.models import User
from task_queue.tasks import process_medical_report


router = APIRouter()

# The six report types the system understands.
# Anything else is rejected immediately at the API boundary so the ML
# models never receive a report they can't process.
VALID_REPORT_TYPES = {"blood", "lipid", "vitamin_d", "hormone", "kidney", "liver"}

# Local fallback directory for dev/Docker Compose.
# /app/data/raw_uploads is a Docker volume so files survive container restarts.
UPLOAD_DIR = Path("/app/data/raw_uploads")

# Read once at import time — avoids repeated os.getenv calls per request.
_USE_S3 = os.getenv("USE_S3", "false").lower() == "true"


def _get_s3_client():
    """
    Build and return a boto3 S3 client configured for Supabase Storage.

    Why Supabase S3?
      Supabase exposes an S3-compatible API endpoint, so standard boto3 works
      by pointing endpoint_url at the Supabase project URL.

    Why is this a function rather than a module-level object?
      boto3 is only imported when USE_S3=true, keeping memory usage low in
      local dev mode.  Instantiating it per-request is cheap because boto3
      re-uses an underlying urllib3 connection pool.

    Environment variables read:
      S3_ENDPOINT_URL  — e.g. https://<project>.supabase.co/storage/v1/s3
      S3_ACCESS_KEY    — Supabase S3 access key
      S3_SECRET_KEY    — Supabase S3 secret key
      S3_REGION        — defaults to "us-east-1" (S3 protocol requires a region
                         even when the service is not AWS)
    """
    import boto3
    from botocore.config import Config
    return boto3.client(
        "s3",
        endpoint_url=os.getenv("S3_ENDPOINT_URL"),
        aws_access_key_id=os.getenv("S3_ACCESS_KEY"),
        aws_secret_access_key=os.getenv("S3_SECRET_KEY"),
        config=Config(signature_version="s3v4"),  # Supabase requires SigV4
        region_name=os.getenv("S3_REGION", "us-east-1"),
    )


@router.post("/upload", response_model=List[dict])
async def upload_reports(
    files: List[UploadFile] = File(...),
    report_type: str = Form(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    Accept one or more medical report files and enqueue them for processing.

    Why accept multiple files at once?
      Future feature: batch upload of a full set of lab panels at once.
      Currently the frontend sends one file per request, but the endpoint
      is ready for multi-file without a code change.

    Request format (multipart/form-data):
      files[]     — one or more image/pdf files
      report_type — one of: blood, lipid, vitamin_d, hormone, kidney, liver

    Processing per file:
      1. Generate a unique filename: {user_id}_{8 random hex bytes}.{ext}
         The random bytes prevent filename collisions even if the same user
         uploads the same file twice.
      2. Write to S3 (prod) or local disk (dev).
      3. INSERT a Report row with status="uploaded".
      4. Call process_medical_report.delay() — this is asynchronous.
         The Celery broker (Redis) receives the job; the worker picks it up
         within milliseconds and the frontend polls /api/status/{id}.
      5. Append per-file success/failure dict to the response list.

    Per-file response:
      Success: { "id": int, "filename": str, "report_type": str, "status": "success" }
      Failure: { "filename": str, "status": "failed", "error": str }

    Raises:
      422 UNPROCESSABLE — invalid report_type.
      Individual file errors are caught and returned in the list rather than
      aborting the whole request, so a batch of 3 files can have 2 succeed
      and 1 fail gracefully.
    """
    # Validate report_type before touching any files.
    if report_type not in VALID_REPORT_TYPES:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Invalid report type '{report_type}'. "
                f"Must be one of: {', '.join(sorted(VALID_REPORT_TYPES))}"
            ),
        )

    # Ensure the local upload directory exists (no-op in S3 mode).
    if not _USE_S3:
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    uploaded_files_metadata = []

    for file in files:
        try:
            # Build a collision-resistant filename.
            # e.g. "7_a3f89bc12d4e56f0.pdf"
            file_extension = Path(file.filename).suffix
            unique_filename = f"{current_user.id}_{os.urandom(8).hex()}{file_extension}"

            if _USE_S3:
                # S3 key — the "path" inside the bucket.
                # "uploads/" prefix keeps files organised under one prefix.
                s3_key = f"uploads/{unique_filename}"
                _get_s3_client().upload_fileobj(
                    file.file,                     # file-like object from FastAPI
                    os.getenv("S3_BUCKET_NAME"),   # bucket name from env
                    s3_key,
                )
                stored_path = s3_key   # stored in DB as the S3 key, not a full URL
            else:
                # Local dev: write directly to the Docker volume.
                local_path = UPLOAD_DIR / unique_filename
                with local_path.open("wb") as buffer:
                    shutil.copyfileobj(file.file, buffer)
                stored_path = str(local_path)   # absolute path inside the container

            # Create the database record BEFORE enqueuing the task.
            # The worker needs the report_id, which is only available after
            # the INSERT + COMMIT.
            new_report = Report(
                filename=file.filename,           # original name for display
                content_type=file.content_type,
                file_path=stored_path,            # S3 key or /app/data/... path
                user_id=current_user.id,          # ownership — filters in all queries
                report_type=report_type,
                status="uploaded",                # first stage of the pipeline
            )
            db.add(new_report)
            db.commit()
            db.refresh(new_report)   # loads auto-generated id, created_at from DB

            # Enqueue the background task.
            # .delay() is Celery shorthand for .apply_async() with no extra options.
            # Redis receives the job immediately; the worker processes it asynchronously.
            process_medical_report.delay(
                new_report.id,   # worker needs this to update the correct DB row
                stored_path,     # where the file is (S3 key or local path)
                report_type,     # determines which OCR patterns + ML models to use
            )

            uploaded_files_metadata.append({
                "id":          new_report.id,
                "filename":    new_report.filename,
                "report_type": report_type,
                "status":      "success",
            })

        except Exception as e:
            # Per-file error — don't abort the whole batch.
            print(f"Error uploading {file.filename}: {e}")
            uploaded_files_metadata.append({
                "filename": file.filename,
                "status":   "failed",
                "error":    str(e),
            })

    return uploaded_files_metadata

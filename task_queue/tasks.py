"""
task_queue/tasks.py — The three Celery background tasks that power the pipeline.

Task execution order (triggered sequentially by the API after upload):
  1. process_medical_report  — reads the uploaded image with PaddleOCR,
                               calls the ML service, extracts lab values,
                               and saves OCR + prediction results to the DB.
  2. predict_disease_risk    — standalone prediction task (not currently
                               called by process_medical_report; reserved for
                               direct invocation when metrics are already known).
  3. compare_reports         — given two completed report IDs, runs ReportComparator
                               and saves the structured diff to the DB.

Why are tasks in their own file?
  The API (compare.py, upload.py) must enqueue tasks without importing the task
  functions directly — that would create circular imports because tasks.py imports
  app.models which imports app.database which is also imported by the API.
  Instead the API uses celery_app.send_task("task_queue.tasks.<name>", args=[...])
  which goes through Celery's string-based task registry.

Error strategy:
  Each task has max_retries set and calls self.retry(exc=exc) on failure.
  Celery uses exponential back-off by default.  After max_retries the exception
  propagates and the report/comparison is marked "failed" in the DB.
"""

import logging
import os
from datetime import datetime
from celery.exceptions import MaxRetriesExceededError
from task_queue.celery_app import celery_app

# Use the SYNCHRONOUS SessionLocal (psycopg2) in Celery workers.
# asyncpg connections are bound to an asyncio event loop; Celery workers run
# in plain synchronous threads with no event loop.  Using asyncpg here would
# raise "Future attached to a different loop" at runtime.
from app.database import SessionLocal

from app.models import Report as MedicalReport, Task as Prediction

# ReportComparison model may not exist in very early schema versions — guard
# the import so the older tasks still function even without the comparisons table.
try:
    from app.models import ReportComparison
except ImportError:
    ReportComparison = None

import requests
from services.ml_service import ReportComparator

# The ML service is a separate FastAPI process (ml_service container in Docker).
# It exposes POST /analyze which runs PaddleOCR + XGBoost and returns a combined
# result dict.  The URL is injected by docker-compose; defaults to empty string
# so local unit tests can detect "no ML service" without crashing on import.
ML_SERVICE_URL = os.getenv("ML_SERVICE_URL", "").rstrip("/")

logger = logging.getLogger(__name__)

# Parsed once at module load — avoids repeated os.getenv calls per task.
# True  → files are stored in Supabase S3; worker must download before reading.
# False → files are on the local Docker volume; worker reads them directly.
_USE_S3 = os.getenv("USE_S3", "false").lower() == "true"


# ── S3 client factory ─────────────────────────────────────────────────────────

def _get_s3():
    """
    Create and return a boto3 S3 client configured for Supabase Storage.

    A new client is created per call (not cached) because:
      - boto3 clients are not thread-safe when shared across tasks.
      - Tasks run in separate worker processes so module-level caching would
        not help across processes anyway.
      - Supabase S3 uses the same boto3 API as AWS — endpoint_url overrides the
        AWS endpoint so requests go to the Supabase bucket instead.

    Credentials come from environment variables injected by docker-compose:
      S3_ENDPOINT_URL — e.g. https://<project>.supabase.co/storage/v1/s3
      S3_ACCESS_KEY   — Supabase storage service key
      S3_SECRET_KEY   — Supabase storage secret
      S3_REGION       — defaults to "us-east-1" (Supabase ignores region but
                        boto3 requires a non-empty value)

    Returns a configured boto3 S3 client ready for download_file / delete_object.
    """
    import boto3
    from botocore.config import Config
    return boto3.client(
        "s3",
        endpoint_url=os.getenv("S3_ENDPOINT_URL"),
        aws_access_key_id=os.getenv("S3_ACCESS_KEY"),
        aws_secret_access_key=os.getenv("S3_SECRET_KEY"),
        config=Config(signature_version="s3v4"),
        region_name=os.getenv("S3_REGION", "us-east-1"),
    )


# ── DB session helper ─────────────────────────────────────────────────────────

def _get_db():
    """
    Yield a synchronous SQLAlchemy session and guarantee it is closed after use.

    This is a generator used as a context manager pattern, but the Celery
    helper functions below do NOT use it as `with _get_db() as db:` — they
    call SessionLocal() directly and close() in a finally block.
    This function exists for compatibility with code that may use it as a
    dependency injection pattern in non-async contexts.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ══════════════════════════════════════════════════════════════════════════════
#  Task 1 — OCR + Prediction (main pipeline task)
# ══════════════════════════════════════════════════════════════════════════════

@celery_app.task(
    bind=True,          # `self` gives access to retry(), update_state(), etc.
    max_retries=3,      # total 4 attempts (1 original + 3 retries) before giving up
    name="task_queue.tasks.process_medical_report"
)
def process_medical_report(self, report_id: int, file_path: str, report_type: str):
    """
    Main pipeline task: OCR → ML prediction → DB write.

    This is the task enqueued by POST /api/upload immediately after the file
    is saved.  It is the ONLY task that the upload API triggers; tasks 2 and 3
    are either called internally or triggered by the compare API.

    Full execution flow:
      1. Mark the report "preprocessing" so the frontend's status poller shows
         the correct stage.
      2. If USE_S3=true, download the file from Supabase S3 to /tmp/ on the
         worker container's local disk.  PaddleOCR can only read local paths.
      3. POST the local image to the ML service (a separate FastAPI container)
         at POST /analyze with the image bytes and report_type.
         The ML service runs PaddleOCR + XGBoost and returns a single JSON:
           { "raw_text": str, "structured_metrics": dict, "ocr_confidence": float,
             "prediction": { "risks": dict, "risk_level": str, ... } }
      4. Save the OCR results (raw_text, metrics, confidence) to the Report row.
      5. Save the ML prediction to a new Task row.
      6. If S3 mode: delete the uploaded file from the bucket (it has been
         processed; keeping it would cost storage).  Also remove the /tmp copy.

    Args:
        report_id   — ID of the Report row in the database.
        file_path   — S3 key (in S3 mode) or absolute local path (local mode).
        report_type — one of: blood, lipid, vitamin_d, hormone, kidney, liver.
                      Passed to the ML service to select the correct OCR parser
                      and XGBoost models.

    Retry behaviour:
        On any exception (network error, OCR crash, ML service timeout) the
        task retries up to 3 times.  Celery uses exponential back-off between
        retries.  After all retries are exhausted the report is marked "failed".

    State updates via self.update_state():
        These push intermediate status to Redis so Flower can show "PROGRESS"
        with the current step.  The FastAPI status endpoint does NOT read these —
        it reads the `status` column in the Report DB table instead.
    """
    logger.info(f"[process_medical_report] Starting for report {report_id}, file={file_path}")
    local_tmp = None
    try:
        # Mark the DB row so the frontend's polling sees "preprocessing".
        _update_report_status(report_id, "preprocessing")
        self.update_state(state="PROGRESS", meta={"step": "downloading_file"})

        if _USE_S3:
            # Download from Supabase S3 to a temp file on the worker's disk.
            # /tmp/ is writable in Docker containers and cleaned up in finally.
            local_tmp = f"/tmp/{os.path.basename(file_path)}"
            _get_s3().download_file(os.getenv("S3_BUCKET_NAME"), file_path, local_tmp)
            image_path = local_tmp
        else:
            # Local dev: the file was saved directly to the Docker volume
            # and the absolute path was stored in report.file_path.
            image_path = file_path

        self.update_state(state="PROGRESS", meta={"step": "calling_ml_service"})
        logger.info(f"[process_medical_report] Calling ML service at {ML_SERVICE_URL}")

        # Multipart POST: "file" key carries the image bytes; "report_type" is
        # a form field.  The ML service reads report_type to pick the parser
        # (e.g. _parse_blood_report) and the disease models (REPORT_MODEL_MAP).
        with open(image_path, "rb") as f:
            response = requests.post(
                f"{ML_SERVICE_URL}/analyze",
                files={"file": (os.path.basename(image_path), f)},
                data={"report_type": report_type},
                timeout=300,   # 5 min: must match task_time_limit in celery_app.py
            )
        response.raise_for_status()   # turns 4xx/5xx into an exception → triggers retry
        result = response.json()

        ocr_result = result
        prediction = result["prediction"]

        logger.info(f"[process_medical_report] ML service done: "
                    f"{len(ocr_result['structured_metrics'])} metrics, "
                    f"risk_level={prediction.get('risk_level')}")

        self.update_state(state="PROGRESS", meta={"step": "saving_results"})

        # Write the OCR layer first (marks report "ocr_complete").
        _save_ocr_results(
            report_id=report_id,
            raw_text=ocr_result["raw_text"],
            metrics=ocr_result["structured_metrics"],
            confidence=ocr_result.get("ocr_confidence", 0.0),
        )

        # Write the prediction layer (marks report "completed").
        _save_prediction(report_id, prediction)

        return {
            "status":           "completed",
            "report_id":        report_id,
            "metrics_extracted": len(ocr_result["structured_metrics"]),
            "risk_level":       prediction.get("risk_level"),
        }

    except Exception as exc:
        logger.error(f"[process_medical_report] failed for report {report_id}: {exc}", exc_info=True)
        _update_report_status(report_id, "failed")
        # self.retry() raises celery.exceptions.Retry — Celery catches it
        # and re-queues the task.  This is not a normal Python raise.
        raise self.retry(exc=exc)

    finally:
        # Always clean up /tmp/ regardless of success or failure.
        if local_tmp and os.path.exists(local_tmp):
            os.remove(local_tmp)

        # Delete the original S3 object after successful download + processing.
        # If download failed, the object stays in S3 so the retry can re-download it.
        # If delete fails (network blip), log a warning — don't retry just for cleanup.
        if _USE_S3:
            try:
                _get_s3().delete_object(Bucket=os.getenv("S3_BUCKET_NAME"), Key=file_path)
                logger.info(f"[process_medical_report] deleted S3 object {file_path}")
            except Exception as e:
                logger.warning(f"[process_medical_report] failed to delete S3 object {file_path}: {e}")


# ══════════════════════════════════════════════════════════════════════════════
#  Task 2 — Standalone Disease Risk Prediction
# ══════════════════════════════════════════════════════════════════════════════

@celery_app.task(
    bind=True,
    max_retries=2,
    default_retry_delay=30,   # wait 30 s before each retry
    name="task_queue.tasks.predict_disease_risk"
)
def predict_disease_risk(self, report_id: int, metrics: dict, report_type: str):
    """
    Run XGBoost disease-risk prediction on pre-extracted lab metrics.

    This task is NOT called by process_medical_report (which delegates to the
    ML service instead).  It exists as a standalone task for:
      - Direct invocation when OCR has already been done and only prediction
        needs to be re-run (e.g. after a model update).
      - Integration tests that bypass the OCR step.

    Flow:
      1. Instantiate RiskPredictor — loads XGBoost .pkl files from disk
         (cached after first load in _MODEL_CACHE).
      2. Call predictor.predict(metrics, report_type):
           a. Validates OCR coverage (logs which features were found/missing).
           b. Builds the feature vector (fills missing features with defaults).
           c. Runs all individual disease models → raw probabilities.
           d. Passes through the NeuralEnsemble meta-learner for calibration.
           e. Computes SHAP values from the highest-risk disease model.
           f. Returns risks, risk_level, key_factors, recommendations.
      3. Save the result to the Task table.

    Args:
        report_id   — links the prediction result back to the Report row.
        metrics     — structured_metrics dict: {key: {"value": float, "unit": str}}
        report_type — selects the right models and feature vector layout.

    Returns a summary dict:
        { "status": "completed", "report_id": int, "risk_level": str,
          "coverage_pct": float }
    """
    logger.info(f"[predict_disease_risk] Starting for report {report_id}")
    try:
        self.update_state(state="PROGRESS", meta={"step": "loading_models"})

        # Import here (not at module top) because RiskPredictor is in the ml_models
        # package which may not be on the path in every deployment configuration.
        from ml_models.predict import RiskPredictor
        predictor = RiskPredictor()

        self.update_state(state="PROGRESS", meta={"step": "predicting"})
        prediction_result = predictor.predict(
            metrics=metrics,
            report_type=report_type,
        )

        # Log the coverage summary for visibility into OCR quality.
        coverage = prediction_result.get("ocr_coverage", {})
        logger.info(
            f"[predict_disease_risk] report {report_id} — "
            f"risk_level={prediction_result.get('risk_level')}  "
            f"coverage={coverage.get('coverage_pct')}%  "
            f"risks={prediction_result.get('risks')}"
        )

        self.update_state(state="PROGRESS", meta={"step": "saving_results"})
        _save_prediction(report_id, prediction_result)

        return {
            "status":       "completed",
            "report_id":    report_id,
            "risk_level":   prediction_result.get("risk_level"),
            "coverage_pct": coverage.get("coverage_pct"),
        }

    except Exception as exc:
        logger.error(f"[predict_disease_risk] failed for report {report_id}: {exc}", exc_info=True)
        raise self.retry(exc=exc)


# ══════════════════════════════════════════════════════════════════════════════
#  Task 3 — Report Comparison
# ══════════════════════════════════════════════════════════════════════════════

@celery_app.task(
    bind=True,
    max_retries=2,
    default_retry_delay=30,
    name="task_queue.tasks.compare_reports"
)
def compare_reports(self, comparison_id: str, report_1_id: int, report_2_id: int):
    """
    Compare two completed reports and persist the structured diff.

    Triggered by POST /api/compare after the API creates the ReportComparison
    row with status="pending".  The frontend polls GET /api/compare/{id} until
    status="completed".

    Flow:
      1. Load both report rows from the DB via _get_report_data(), which also
         fetches the latest prediction_risks from the associated Task row.
      2. Pass both data dicts to ReportComparator.compare_medical_reports():
           - Aligns metrics by key name (intersection of both reports).
           - Computes diff, percent change, and "improved" flag per metric.
           - Determines overall trend: IMPROVING / WORSENING / STABLE.
           - Computes disease risk score deltas.
      3. Save the full comparison dict to the ReportComparison row and set
         status="completed".

    Error handling:
      - On retry: standard Celery retry with back-off.
      - After MaxRetriesExceeded: _mark_comparison_failed() sets status="failed"
        so the frontend stops polling and shows an error message.
        This is more explicit than letting the row hang in "pending" forever.

    Args:
        comparison_id — UUID string, primary key of the ReportComparison row.
        report_1_id   — ID of the newer (current) report.
        report_2_id   — ID of the older (baseline) report.
    """
    logger.info(f"[compare_reports] Starting for comparison {comparison_id}")
    try:
        self.update_state(state="PROGRESS", meta={"step": "fetching_reports"})

        # _get_report_data returns the structured_metrics and prediction_risks
        # needed by the comparator.  Returns {} if the report_id is not found.
        report_1 = _get_report_data(report_1_id)
        report_2 = _get_report_data(report_2_id)

        self.update_state(state="PROGRESS", meta={"step": "comparing"})
        comparator = ReportComparator()
        comparison = comparator.compare_medical_reports(report_1, report_2)

        self.update_state(state="PROGRESS", meta={"step": "saving_comparison"})
        _save_comparison(comparison_id, comparison)

        return {
            "status":        "completed",
            "comparison_id": comparison_id,
            "trend":         comparison["summary"]["overall_trend"],
        }

    except Exception as exc:
        logger.error(f"[compare_reports] failed for comparison {comparison_id}: {exc}", exc_info=True)
        try:
            raise self.retry(exc=exc)
        except MaxRetriesExceededError:
            # All retries exhausted — explicitly mark the row failed so the
            # frontend gets a definitive error state instead of polling forever.
            _mark_comparison_failed(comparison_id)
            raise


# ══════════════════════════════════════════════════════════════════════════════
#  Private DB helper functions
# ══════════════════════════════════════════════════════════════════════════════

def _update_report_status(report_id: int, status: str):
    """
    Write a new status value to the Report row identified by report_id.

    Also sets processed_at to utcnow() when status == "completed" so the
    frontend can display the completion timestamp.

    Opens and closes its own DB session (not reused across calls) so each
    helper is independently transactional — a failure in one helper does not
    roll back changes from a previous helper in the same task execution.

    Args:
        report_id — primary key of the Report row.
        status    — one of: "preprocessing", "ocr_complete", "completed", "failed".
    """
    db = SessionLocal()
    try:
        report = db.query(MedicalReport).filter(MedicalReport.id == report_id).first()
        if report:
            report.status = status
            if status == "completed":
                report.processed_at = datetime.utcnow()
            db.commit()
            logger.info(f"[_update_report_status] report {report_id} → {status}")
        else:
            logger.warning(f"[_update_report_status] report {report_id} not found")
    except Exception as e:
        db.rollback()
        logger.error(f"[_update_report_status] error: {e}", exc_info=True)
        raise
    finally:
        db.close()


def _save_ocr_results(report_id: int, raw_text: str, metrics: dict, confidence: float):
    """
    Persist OCR extraction results to the Report row.

    Called by process_medical_report after the ML service returns the OCR output.
    Sets status to "ocr_complete" (not "completed") — the report is only "completed"
    once _save_prediction() also runs.

    Stores:
      raw_text          — full concatenated text extracted from the image.  Useful
                          for debugging and for full-text search in the future.
      extracted_metrics — the structured {key: {value, unit, source}} dict used
                          by the prediction pipeline and the frontend biomarker cards.
      ocr_confidence    — mean confidence across all recognised text fragments (0–1).
                          Exposed in the UI as an image quality indicator.
      processed_at      — timestamp of OCR completion (not prediction completion).

    Args:
        report_id  — ID of the Report row to update.
        raw_text   — full OCR text string.
        metrics    — structured_metrics dict.
        confidence — mean PaddleOCR confidence score.
    """
    db = SessionLocal()
    try:
        report = db.query(MedicalReport).filter(MedicalReport.id == report_id).first()
        if report:
            report.raw_text         = raw_text
            report.extracted_metrics = metrics
            report.ocr_confidence   = confidence
            report.status           = "ocr_complete"
            report.processed_at     = datetime.utcnow()
            db.commit()
            logger.info(f"[_save_ocr_results] report {report_id} saved, "
                        f"{len(metrics)} metrics, confidence={confidence:.3f}")
        else:
            logger.warning(f"[_save_ocr_results] report {report_id} not found")
    except Exception as e:
        db.rollback()
        logger.error(f"[_save_ocr_results] error: {e}", exc_info=True)
        raise
    finally:
        db.close()


def _save_prediction(report_id: int, prediction_result: dict):
    """
    Create a Task row containing the ML prediction output and mark the report done.

    The Task row is queried later by the status API using:
        db.query(Task).filter(Task.task_id.like(f"predict-{report_id}-%")).first()

    The task_id format "predict-{report_id}-{uuid8}" lets the status endpoint
    find the result without storing the Celery task UUID in the Report table.
    The uuid8 suffix prevents collisions if a report is somehow re-processed.

    The Task.result JSON blob stores:
      risks          — {disease: float} — the calibrated disease probabilities.
      risk_level     — "low" / "moderate" / "high" / "critical".
      key_factors    — top 5 SHAP features driving the highest-risk disease.
      recommendations— actionable advice based on risk_level and specific diseases.
      model_version  — "neural-ensemble-{report_type}-v1" for model provenance.
      raw_xgb_probas — pre-ensemble XGBoost scores (useful for debugging ensemble).
      ocr_coverage   — {found, missing, coverage_pct} from validate_ocr_metrics.

    Also updates Report.status = "completed" and Report.processed_at in the
    same transaction so the two writes are atomic.

    Args:
        report_id         — ID of the parent Report row.
        prediction_result — dict returned by RiskPredictor.predict().
    """
    db = SessionLocal()
    try:
        import uuid
        task_row = Prediction(
            # Unique task_id following the "predict-{id}-{hex8}" pattern.
            task_id=f"predict-{report_id}-{uuid.uuid4().hex[:8]}",
            status="completed",
            result={
                "report_id":       report_id,
                "risks":           prediction_result.get("risks", {}),
                "risk_level":      prediction_result.get("risk_level"),
                "key_factors":     prediction_result.get("key_factors", []),
                "recommendations": prediction_result.get("recommendations", []),
                "model_version":   prediction_result.get("model_version"),
                "raw_xgb_probas":  prediction_result.get("raw_xgb_probas", {}),
                "ocr_coverage":    prediction_result.get("ocr_coverage", {}),
            },
        )
        db.add(task_row)

        # Update the parent report in the same transaction — both succeed or
        # both roll back together.
        report = db.query(MedicalReport).filter(MedicalReport.id == report_id).first()
        if report:
            report.status       = "completed"
            report.processed_at = datetime.utcnow()

        db.commit()
        logger.info(f"[_save_prediction] saved prediction for report {report_id}, "
                    f"risk_level={prediction_result.get('risk_level')}")
    except Exception as e:
        db.rollback()
        logger.error(f"[_save_prediction] error: {e}", exc_info=True)
        raise
    finally:
        db.close()


def _get_report_data(report_id: int) -> dict:
    """
    Load a Report row and its associated prediction results as a plain dict.

    Used by the compare_reports task to fetch both reports before running
    ReportComparator.  Returns a dict (not an ORM object) so the comparator
    receives JSON-safe data that does not depend on an active DB session.

    The prediction_risks sub-dict is pulled from the most recent Task row
    matching "predict-{report_id}-%" — the same LIKE pattern used by the
    status API.  The .desc() ordering ensures the latest prediction is used
    if the report was somehow re-processed.

    Returns {} (empty dict) if the report_id does not exist, which causes
    compare_medical_reports to find no shared metric keys and produce an
    empty comparison result rather than crashing.

    Args:
        report_id — ID of the Report row to fetch.

    Returns:
        {
          "structured_metrics": dict,   # {key: {value, unit, source}}
          "raw_text":           str,    # full OCR text
          "report_type":        str,    # blood / lipid / ...
          "created_at":         str,    # ISO 8601
          "prediction_risks":   dict,   # {disease: float} from Task.result
        }
    """
    db = SessionLocal()
    try:
        report = db.query(MedicalReport).filter(MedicalReport.id == report_id).first()
        if report:
            # Find the latest prediction task for this report.
            prediction = (
                db.query(Prediction)
                .filter(Prediction.task_id.like(f"predict-{report_id}-%"))
                .order_by(Prediction.created_at.desc())
                .first()
            )
            # Extract only the risks dict from the full prediction result.
            prediction_risks = (
                prediction.result.get("risks", {})
                if prediction and prediction.result else {}
            )
            return {
                "structured_metrics": report.extracted_metrics or {},
                "raw_text":           report.raw_text or "",
                "report_type":        report.report_type,
                "created_at":         report.created_at.isoformat(),
                "prediction_risks":   prediction_risks,
            }
        return {}
    finally:
        db.close()


def _mark_comparison_failed(comparison_id: str):
    """
    Set a ReportComparison row's status to "failed".

    Called by compare_reports after MaxRetriesExceededError so the frontend's
    polling loop receives a definitive terminal state.  Without this the row
    would stay "pending" forever and the UI would keep polling.

    The function is a no-op if ReportComparison was not importable (schema
    migration not yet applied) — guarded by the try/except at the top of the file.

    Args:
        comparison_id — UUID string primary key of the ReportComparison row.
    """
    if ReportComparison is None:
        return
    db = SessionLocal()
    try:
        comp = db.query(ReportComparison).filter(ReportComparison.id == comparison_id).first()
        if comp:
            comp.status = "failed"
            db.commit()
            logger.info(f"[_mark_comparison_failed] comparison {comparison_id} → failed")
    except Exception as e:
        db.rollback()
        logger.error(f"[_mark_comparison_failed] error: {e}", exc_info=True)
    finally:
        db.close()


def _save_comparison(comparison_id: str, comparison_data: dict):
    """
    Persist a completed comparison result to the ReportComparison row.

    Sets four fields on the row:
      status             — "completed" (stops frontend polling)
      comparison_data    — the full nested dict from ReportComparator:
                           { metrics, significant_changes, risk_comparison, summary }
      significant_changes — extracted subset so the API can return it without
                            deserialising the full comparison_data blob.
      trend_analysis     — "IMPROVING" / "WORSENING" / "STABLE" — stored at
                           the top level so GET /api/compare can include it in the
                           list response without loading comparison_data.

    The function is a no-op if ReportComparison was not importable.

    Args:
        comparison_id   — UUID string primary key.
        comparison_data — full dict returned by compare_medical_reports().
    """
    if ReportComparison is None:
        logger.warning("[_save_comparison] ReportComparison model not available, skipping save")
        return

    db = SessionLocal()
    try:
        comp = db.query(ReportComparison).filter(ReportComparison.id == comparison_id).first()
        if comp:
            comp.status             = "completed"
            comp.comparison_data    = comparison_data
            comp.significant_changes = comparison_data.get("significant_changes", [])
            comp.trend_analysis     = comparison_data["summary"]["overall_trend"]
            db.commit()
            logger.info(f"[_save_comparison] saved comparison {comparison_id}")
    except Exception as e:
        db.rollback()
        logger.error(f"[_save_comparison] error: {e}", exc_info=True)
        raise
    finally:
        db.close()

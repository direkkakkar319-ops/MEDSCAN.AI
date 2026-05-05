# MEDSCAN.AI — End-to-End Pipeline Workflow

> **Disclaimer:** This document is for educational and engineering reference only.  
> MEDSCAN.AI is not a licensed medical device and must not be used for clinical diagnosis.

---

## Table of Contents

1. [System Architecture Overview](#1-system-architecture-overview)
2. [Frontend Upload Flow](#2-frontend-upload-flow)
3. [Backend Storage & Queue Workflow](#3-backend-storage--queue-workflow)
4. [Frontend Real-Time Status & Results Display](#4-frontend-real-time-status--results-display)
5. [Machine Learning Pipeline](#5-machine-learning-pipeline)
6. [API Contracts & Database Transactions](#6-api-contracts--database-transactions)
7. [Testing Strategy](#7-testing-strategy)
8. [Deployment & Operations](#8-deployment--operations)

---

## 1. System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        BROWSER (React 19 + Vite)                    │
│  AuthModal → UploadReportModal → Results → History → Compare        │
└────────────────────────┬────────────────────────────────────────────┘
                         │  HTTPS  (Bearer JWT)
                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    FastAPI  (Render — medscan-api)                  │
│  /auth/*   /api/upload   /api/status/:id   /api/reports   /api/compare│
└─────┬──────────────────┬────────────────────┬────────────────────────┘
      │  SQLAlchemy       │  boto3 (S3)        │  Celery .delay()
      ▼                   ▼                    ▼
┌───────────┐   ┌─────────────────┐   ┌──────────────────────────────┐
│ Neon      │   │ Supabase S3     │   │  Redis (Upstash)             │
│ PostgreSQL│   │ medscan-uploads │   │  Celery Broker + Result       │
└───────────┘   └─────────────────┘   └──────────┬───────────────────┘
                                                  │
                                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│              Celery Worker  (same Render service)                   │
│   process_medical_report → predict_disease_risk → compare_reports  │
└────────────────────────┬────────────────────────────────────────────┘
                         │  HTTP POST /analyze
                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│          ML Microservice  (Hugging Face Space — port 7860)          │
│   PaddleOCR v5 (PP-OCRv5) + XGBoost + Neural Ensemble              │
└─────────────────────────────────────────────────────────────────────┘
```

### Services at a Glance

| Service | Technology | Host | URL |
|---|---|---|---|
| Frontend | React 19 + Vite + Tailwind | Render | https://medscan-ai-s7t1.onrender.com |
| Backend API | FastAPI + Uvicorn | Render | https://medscan-api-x9ne.onrender.com |
| Task Queue | Celery 5 + Redis | Co-located with API | — |
| ML/OCR Service | PaddleOCR 3 + XGBoost 2 | Hugging Face Space | https://direkkakkar-medscan-ai-ml-models.hf.space |
| Database | PostgreSQL 15 | Neon (eu-west-2) | — |
| Cache / Broker | Redis 7 | Upstash (ap-south-1) | — |
| Temprory File Storage | S3-compatible | Supabase | Bucket: `medscan-uploads` |

---
-
## 2. Frontend Upload Flow

### 2.1 Authentication Gate

Every request to the backend API requires a valid JWT **access token** stored in `localStorage`.

```
User opens AuthModal
        │
        ├─ SIGN UP path ──────────────────────────────────────────────┐
        │   POST /auth/register                                        │
        │   { email, username: email, password }                      │
        │   → 200 UserRead | 409 Conflict                              │
        │   ↓ (auto-login after register)                              │
        └─ LOG IN path ───────────────────────────────────────────────┤
            POST /auth/login  (application/x-www-form-urlencoded)     │
            username=<email>&password=<pass>                          │
            → 200 { access_token, refresh_token, token_type }         │
            ↓                                                          │
            localStorage.access_token  = <JWT, 30 min TTL>           │
            localStorage.refresh_token = <JWT, 7 day TTL>            │
            sessionStorage.loginSuccess = "true"                      │
            window.location.reload()  →  App.useEffect fires toast   │
```

**Rate limits** (enforced by slowapi per IP):

| Endpoint | Limit |
|---|---|
| POST /auth/register | 3 / minute |
| POST /auth/login | 5 / minute |
| POST /auth/refresh | 10 / minute |

**Token refresh** is handled silently inside `apiFetch()` (frontend/src/lib/api.js):
```
Request → 401 → tryRefresh() → POST /auth/refresh { refresh_token }
       → 200 → store new access_token → retry original request
       → 401 → clear both tokens → user must log in again
```

---

### 2.2 File Selection & Validation (UploadReportModal)

Before the file ever leaves the browser:

| Rule | Value |
|---|---|
| Accepted MIME types | image/jpeg, image/png, application/pdf |
| Max file size | 50 MB (client-side guard) |
| Report type | Required selection: blood / lipid / vitamin_d / hormone / kidney / liver |
| Auth required | Yes — modal blocks if no `access_token` in localStorage |

---

### 2.3 HTTP Upload Request

```
POST  https://medscan-api-x9ne.onrender.com/api/upload
Authorization: Bearer <access_token>
Content-Type: multipart/form-data

Body (FormData):
  files[]     = <File>            (one or more)
  report_type = "blood"           (one of the six valid types)
```

**Successful response** `200 OK`:
```json
[
  {
    "id": 42,
    "filename": "blood_panel_jan.pdf",
    "report_type": "blood",
    "status": "success"
  }
]
```

**Failure response** (per file):
```json
[
  {
    "filename": "bad_file.docx",
    "status": "failed",
    "error": "Invalid content type"
  }
]
```

**Error responses**:

| Code | Scenario |
|---|---|
| 401 | Missing / expired access_token |
| 422 | Invalid report_type or missing files field |
| 500 | S3 upload failure / DB error |

---

### 2.4 Post-Upload Frontend Actions

```javascript
// UploadReportModal.jsx — after successful upload response
const reportId = response[0].id;

// 1. Persist so polling survives a page refresh
localStorage.setItem('healthinsight_pending_report_id', reportId);

// 2. Notify Results section via custom event
window.dispatchEvent(new CustomEvent('reportUploaded', {
    detail: { reportIds: [reportId] }
}));

// 3. Smooth-scroll to results
document.getElementById('results').scrollIntoView({ behavior: 'smooth' });
```

---

## 3. Backend Storage & Queue Workflow

### 3.1 File Storage

When the API receives the upload it immediately writes the file to storage and creates a DB row:

```
POST /api/upload  received
        │
        ├─ USE_S3=true  ──────────────────────────────────────────────┐
        │   s3.upload_fileobj(file, bucket, "uploads/{uid}_{hex}.ext")│
        │   stored_path = "uploads/{uid}_{hex}.ext"                   │
        └─ USE_S3=false ─────────────────────────────────────────────►│
            shutil.copyfileobj(file, /app/data/raw_uploads/{name})   │
            stored_path = "/app/data/raw_uploads/{name}"             │
                                                                      │
        ◄─────────────────────────────────────────────────────────────┘
        │
        ▼
  INSERT INTO reports (filename, content_type, file_path,
                       user_id, report_type, status="uploaded")
        │
        ▼
  process_medical_report.delay(report_id, stored_path, report_type)
        │
        ▼
  Return JSON response to frontend
```

**Unique filename formula**: `{user_id}_{os.urandom(8).hex()}.{ext}`  
Example: `7_a3f89bc12d4e56f0.pdf`

---

### 3.2 Database Schema

#### `reports` table

| Column | Type | Description |
|---|---|---|
| id | INTEGER PK | Auto-increment primary key |
| filename | VARCHAR | Original filename as uploaded by user |
| content_type | VARCHAR | MIME type (image/jpeg, application/pdf…) |
| file_path | VARCHAR | S3 key or local absolute path |
| user_id | INTEGER | FK to auth_users.id |
| report_type | VARCHAR | blood / lipid / vitamin_d / hormone / kidney / liver |
| status | VARCHAR | uploaded → preprocessing → ocr_complete → completed \| failed |
| raw_text | TEXT | Full OCR-extracted text (set by worker) |
| extracted_metrics | JSONB | `{metric: {value, unit, source}}` (set by worker) |
| ocr_confidence | FLOAT | Mean PaddleOCR confidence score 0–1 |
| created_at | TIMESTAMPTZ | Auto-set on INSERT |
| processed_at | TIMESTAMP | Set when OCR completes or fails |

#### `tasks` table

| Column | Type | Description |
|---|---|---|
| id | INTEGER PK | Auto-increment |
| task_id | VARCHAR UNIQUE | Format: `predict-{report_id}-{uuid8}` |
| status | VARCHAR | always "completed" when written |
| result | JSONB | Full ML prediction output (risks, risk_level, key_factors…) |
| created_at | TIMESTAMPTZ | Auto-set on INSERT |

#### `report_comparisons` table

| Column | Type | Description |
|---|---|---|
| id | VARCHAR PK | UUID string |
| user_id | INTEGER | Owner |
| report1_id | INTEGER | FK to reports.id |
| report2_id | INTEGER | FK to reports.id |
| report_type | VARCHAR | Must match for both reports |
| status | VARCHAR | pending → completed \| failed |
| comparison_data | JSONB | metrics[], significant_changes[], risk_comparison[], summary{} |
| significant_changes | JSONB | Subset of metrics where \|pct\| > 5% |
| trend_analysis | VARCHAR | IMPROVING \| WORSENING \| STABLE |
| created_at | TIMESTAMPTZ | Auto-set |

#### `auth_users` table (auth system)

| Column | Type | Description |
|---|---|---|
| id | INTEGER PK | Auto-increment |
| username | VARCHAR UNIQUE | Used as JWT `sub` claim |
| email | VARCHAR UNIQUE | Registration email |
| hashed_password | VARCHAR | bcrypt (12 rounds) |
| is_active | BOOLEAN | True by default |
| created_at / updated_at | TIMESTAMPTZ | Audit timestamps |

---

### 3.3 Celery Task Queue

**Broker & Backend**: Redis (Upstash `rediss://` with SSL for production)  
**Queues**: `default`, `ocr`, `prediction`, `comparison`  
**Worker concurrency**: 1 process (CPU-bound OCR)  
**Task timeout**: 300 seconds  
**Result expiry**: 3600 seconds (tasks table is the durable store)

#### Task 1 — `process_medical_report`

```
process_medical_report(report_id, file_path, report_type)
    │
    ├─ DB: UPDATE reports SET status="preprocessing"
    │
    ├─ [S3 only] download file to /tmp/{basename}
    │
    ├─ HTTP POST {ML_SERVICE_URL}/analyze
    │     files={"file": <binary>}
    │     data={"report_type": "blood"}
    │     timeout=300s
    │
    ├─ Parse response:
    │     ocr_result["raw_text"]
    │     ocr_result["structured_metrics"]
    │     ocr_result["ocr_confidence"]
    │     ocr_result["prediction"]     ← full ML output from HF Space
    │
    ├─ _save_ocr_results()  → status="ocr_complete"
    │
    ├─ _save_prediction()   → INSERT tasks row, status="completed"
    │
    ├─ [S3 only] DELETE s3://{bucket}/{file_path}  (cleanup)
    │
    └─ Return {"status":"completed","report_id":42,"risk_level":"low"}

Retry policy: max_retries=3, exponential backoff (Celery default)
On final failure: DB status → "failed"
```

#### Task 2 — `predict_disease_risk`

```
predict_disease_risk(report_id, metrics, report_type)
    │
    ├─ Load RiskPredictor() (models cached in _MODEL_CACHE)
    ├─ predictor.predict(metrics, report_type)
    ├─ _save_prediction(report_id, result)
    └─ Return {"status":"completed","risk_level":"moderate","coverage_pct":87.5}

Retry policy: max_retries=2, default_retry_delay=30s
```

#### Task 3 — `compare_reports`

```
compare_reports(comparison_id, report1_id, report2_id)
    │
    ├─ _get_report_data(report1_id)  → metrics + prediction_risks
    ├─ _get_report_data(report2_id)
    ├─ ReportComparator().compare_medical_reports(r1, r2)
    ├─ _save_comparison(comparison_id, result)
    └─ Return {"status":"completed","trend":"IMPROVING"}

On MaxRetriesExceededError: _mark_comparison_failed(comparison_id)
```

---

### 3.4 Retry Logic

```
Task fails with exception
        │
        ├─ attempts < max_retries  → self.retry(exc=exc)
        │     Celery re-queues with exponential backoff
        │     Default: 2^retry_number seconds
        │
        └─ attempts >= max_retries → MaxRetriesExceededError raised
              process_medical_report: _update_report_status("failed")
              compare_reports:        _mark_comparison_failed(id)
              Frontend: stops polling, shows error banner
```

---

## 4. Frontend Real-Time Status & Results Display

### 4.1 Polling Architecture

```
Results.jsx mounts
        │
        ├─ Check localStorage.healthinsight_pending_report_id
        │   → exists: startPolling(id)
        │
        ├─ Check localStorage.healthinsight_latest_report_id
        │   → exists: GET /api/status/{id}, render if completed
        │
        └─ Listen for window 'reportUploaded' event
            → startPolling(e.detail.reportIds[0])

startPolling(reportId):
  setInterval(3000ms, max 60 attempts = 5 minute timeout)
  │
  └─ GET /api/status/{reportId}
        │
        ├─ status = "uploaded"       → show "Report received"
        ├─ status = "preprocessing"  → show "Preprocessing image…"
        ├─ status = "ocr_complete"   → show "OCR complete — running models…"
        ├─ status = "completed"
        │     stopPolling()
        │     transformResult(data) → setReportData()
        │     localStorage.latest = reportId
        │     localStorage.remove(pending)
        │
        ├─ status = "failed"         → stopPolling(), show error banner
        │
        └─ HTTP error / network      → keep polling (silent)
```

---

### 4.2 Data Transformation (`reportUtils.js → transformResult`)

```
API Response (GET /api/status/{id})
        │
        ▼
{
  result: {
    risks: { diabetes: 0.12, anemia: 0.05, infection: 0.03 },
    risk_level: "low",
    key_factors: [{ feature, impact, direction }],
    recommendations: ["Maintain current lifestyle..."]
  },
  report_type: "blood",
  filename: "bloodwork.pdf",
  extracted_metrics: {
    wbc: { value: 7.5, unit: "10³/µL", source: "text" }
  }
}
        │
        ▼ transformResult()
        │
        ├─ riskMetrics: risks entries → { name, value (×100), status, trend }
        │     riskToStatus(0.12) → "low"     (< 0.40)
        │     riskToStatus(0.45) → "moderate" (0.40–0.64)
        │     riskToStatus(0.67) → "high"    (0.65–0.84)
        │     riskToStatus(0.90) → "critical" (≥ 0.85)
        │
        ├─ biomarkers: top 4 from extracted_metrics → { name, value, unit, range, status }
        │     REFERENCE_RANGES["wbc"] → "4.5-11 ×10³/µL"
        │
        └─ { riskMetrics, biomarkers, riskLevel, recommendations, keyFactors,
             reportType, filename }
```

---

### 4.3 UI Components for Results

```
Results Section (Results.jsx)
├── Section Header (section number + live filename badge)
│
├── Processing Banner  [Loader2 spinner + step label]
│   uploaded | preprocessing | ocr_complete | predicting | completed
│
├── Error Banner       [AlertTriangle + error message]
│
├── Empty State        [Upload icon + "No analysis yet"]
│
├── Overall Risk Badge [colored left-border panel + glow dot]
│   LOW / MODERATE / HIGH / CRITICAL
│
└── Dashboard Grid (3 columns)
    │
    ├── RISK OVERVIEW
    │   └── Disease cards (clickable)
    │       ├── Disease name (e.g. "Vitamin D Deficiency")
    │       ├── StatusBadge: dot + outlined colored pill
    │       └── Progress bar + percentage (colored by risk level)
    │
    ├── RISK RADAR
    │   └── HTML Canvas radar chart
    │       ├── Grid circles (5 rings)
    │       ├── Axes (one per disease)
    │       ├── Data polygon (orange fill)
    │       └── Data points (orange dots)
    │
    └── KEY BIOMARKERS
        ├── Biomarker cards
        │   ├── Left border colored by status
        │   ├── Status badge (dot + outlined)
        │   ├── Value (colored by status) + unit
        │   └── Reference range
        ├── Key Factors (SHAP explanations)
        │   └── Feature + direction icon + "increases/decreases risk"
        └── Recommendation box (orange border)
```

---

### 4.4 Error Handling

| Scenario | Frontend Behaviour |
|---|---|
| Upload fails (network) | Per-file error in upload modal |
| Invalid report type | 422 caught, shown as modal error |
| 401 on upload | Silent refresh attempted; if fails → modal re-appears |
| Poll timeout (60 attempts) | Error banner: "Analysis timed out. Please try again." |
| Status = "failed" | Error banner: "Report analysis failed. Check image quality." |
| Empty extracted metrics | Results render with defaults; coverage warning surfaced |

---

## 5. Machine Learning Pipeline

### 5.1 ML Microservice Entry Point (`/analyze`)

The Celery worker sends a single HTTP request to the HF Space:

```
POST {ML_SERVICE_URL}/analyze
Content-Type: multipart/form-data

files: { "file": <binary image/pdf> }
data:  { "report_type": "blood" }
```

Response (200 OK):
```json
{
  "raw_text": "WBC 7.5 K/uL  RBC 4.5 M/uL ...",
  "structured_metrics": {
    "wbc":        { "value": 7.5,  "unit": "10³/µL", "source": "text" },
    "rbc":        { "value": 4.5,  "unit": "10⁶/µL", "source": "text" },
    "hemoglobin": { "value": 14.2, "unit": "g/dL",   "source": "text" }
  },
  "ocr_confidence": 0.923,
  "prediction": {
    "risks":        { "diabetes": 0.12, "anemia": 0.05, "infection": 0.03 },
    "risk_level":   "low",
    "key_factors":  [{ "feature": "hemoglobin", "impact": -0.18, "direction": "decreases" }],
    "recommendations": ["Maintain current lifestyle. Annual check-up recommended."],
    "model_version": "neural-ensemble-blood-v1",
    "ocr_coverage": { "found": ["wbc","rbc","hemoglobin","glucose"], "missing": ["hematocrit","platelets","creatinine","bun"], "coverage_pct": 50.0 },
    "raw_xgb_probas": { "diabetes": 0.1200, "anemia": 0.0500, "infection": 0.0300 }
  }
}
```

---

### 5.2 OCR Stage (PaddleOCR v5)

```
Image file received by ML service
        │
        ▼
OCRRunner.process_report(image_path, report_type)
        │
        ├─ Step 1: preprocess_image()
        │     OpenCV: BGR → Grayscale → fastNlMeansDenoising → CLAHE
        │     Write to {image_path}_temp.jpg, delete after OCR
        │
        ├─ Step 2: extract_text()
        │     PaddleOCR.predict(temp_path)
        │     Returns: [{ text, confidence, bbox }]
        │     Sorted top-to-bottom, left-to-right by bbox center
        │
        ├─ Step 3: extract_tables()  [PPStructureV3 — disabled if lang arg error]
        │     Falls back gracefully to []
        │
        ├─ Step 4: parse_<report_type>_report(text_data, tables)
        │     Joins all text items with spaces → full_text string
        │     Applies regex patterns to extract lab values
        │
        └─ Returns: { raw_text, structured_metrics, tables, average_confidence, text_items }
```

**Regex Pattern Design** (all parsers):

Old pattern (broke for many lab formats): `(?:WBC)[\s:)]+([0-9.]+)`  
New pattern (robust): `(?:WBC|White\s+Blood\s+Cell)[^0-9\n]{0,50}([0-9]+\.?[0-9]*)(?!\s*-)`

- `[^0-9\n]{0,50}` — matches any non-digit characters (units, brackets, pipes) up to 50 chars
- `(?!\s*-)` — negative lookahead prevents capturing the start of a reference range like `4.0-10.5`

---

### 5.3 Feature Engineering

```
structured_metrics (from OCR)
        │
        ▼
validate_ocr_metrics(metrics, report_type)
        │
        ├─ Checks each expected feature against OCR output
        ├─ found: ["wbc","rbc","hemoglobin"]   (value present)
        ├─ missing: ["hematocrit","platelets"]  (will use defaults)
        ├─ coverage_pct = len(found)/len(expected) × 100
        └─ Logs WARNING if coverage_pct < 50%

        ▼
normalize_units(metrics)
        │
        ├─ glucose mmol/L → mg/dL (×18.0182)    if value < 20 → infer mmol/L
        ├─ vitamin_d nmol/L → ng/mL (÷2.496)    if value > 200 → infer nmol/L
        ├─ cholesterol mmol/L → mg/dL (×38.67)
        └─ Unknown units: pass through unchanged

        ▼
build_feature_vector(metrics, report_type)
        │
        ├─ Blood (8 features):  wbc=7.0, rbc=4.9, hemoglobin=14.0, hematocrit=43.0,
        │                        platelets=250.0, glucose=90.0, creatinine=1.0, bun=15.0
        ├─ Lipid (5 features):  total_cholesterol, hdl, ldl, triglycerides, vldl
        ├─ Vitamin D (1 feat):  vitamin_d
        ├─ Hormone (10 feat):   tsh, t3, t4, testosterone, estradiol, progesterone,
        │                        prolactin, lh, fsh, cortisol
        ├─ Kidney (5 features): creatinine, bun, urea, uric_acid, egfr
        └─ Liver (7 features):  alt, ast, alp, bilirubin_total, bilirubin_direct,
                                 albumin, total_protein

        → Returns: ([float, ...], ["wbc","rbc",...])
        → Missing features use their default (population median) value
        → validate_value() range-checks glucose, hemoglobin, creatinine
```

---

### 5.4 XGBoost Individual Models

```
X = np.array(feature_vector, dtype=float32).reshape(1, -1)

For each (disease_label, pkl_name, model) in REPORT_MODEL_MAP[report_type]:
        │
        ▼
  model.predict_proba(X)  → [P(class0), P(class1), P(class2), ...]

  Severity encoding:
    Class 0 = healthy / no disease
    Class 1 = mild
    Class 2 = moderate
    Class 3 = severe  (some models only)

  disease_risk  = 1 - P(class 0)    ← probability of having ANY level of disease
  severity_class = argmax(all_probas) ← for SHAP explanation selection
```

**Model Registry** (`REPORT_MODEL_MAP`):

| Report Type | Disease Labels | pkl files |
|---|---|---|
| blood | diabetes, anemia, infection | diabetes.pkl, anemia.pkl, infection.pkl |
| lipid | heart_disease, stroke | heart.pkl, stroke.pkl |
| vitamin_d | vitamin_d_deficiency | vitamin_d.pkl |
| hormone | testosterone_imbalance, thyroid_disorder, hormonal_imbalance | testosterone.pkl, thyroid.pkl, hormone.pkl |
| liver | liver_disease, fatty_liver, hepatitis | liver.pkl, fatty_liver.pkl, hepatitis.pkl |
| kidney | kidney_disease, renal_failure | kidney.pkl, renal.pkl |

**Model cache**: `_MODEL_CACHE` dict at module level — models are loaded once per worker process via `joblib.load()`.

---

### 5.5 Neural Ensemble Meta-Learner

```
raw_probas = [0.12, 0.05, 0.03]   ← one per disease model (blood example)
        │
        ▼
load_ensemble(report_type, ensemble_path, n_diseases)
  ├─ Check _ENSEMBLE_CACHE[report_type] → return if cached
  ├─ Check ensemble_blood.pkl exists on disk
  │     → NeuralEnsemble.load(path)    (trained weights)
  └─     → NeuralEnsemble(n_diseases)  (identity passthrough if no weights)

ensemble.predict(raw_probas)
        │
  If trained:
        ├─ x = raw_probas.reshape(1, -1)
        ├─ h1 = ReLU(x  @ W1 + b1)   [shape: (1, 32)]
        ├─ h2 = ReLU(h1 @ W2 + b2)   [shape: (1, 16)]
        └─ out = Sigmoid(h2 @ W3 + b3) [shape: (1, n_diseases)]

  If not trained (passthrough):
        └─ return raw_probas unchanged

final_probas = [0.12, 0.05, 0.03]   ← calibrated scores
```

**Architecture**:
```
Input(n)  →  Dense(32, ReLU)  →  Dense(16, ReLU)  →  Dense(n, Sigmoid)
```
Pure NumPy implementation — no PyTorch/TensorFlow dependency.

---

### 5.6 Risk Level & SHAP Explanations

```
max_risk = max(final_probas)   → e.g. 0.12

_score_to_level(max_risk):
  ≥ 0.85 → "critical"
  ≥ 0.65 → "high"
  ≥ 0.40 → "moderate"
  else   → "low"

SHAP (top disease model only):
  top_disease = disease with highest final_proba
  shap.TreeExplainer(top_model).shap_values(X)
  → list[array] per class → pick explain_class = severity_class (min 1)
  → zip(feature_names, values) → dict

_top_factors(shap_values, feature_names, n=5):
  Sort by abs(shap_value) descending → top 5
  → [{ feature, impact, direction: "increases"|"decreases" }]

_recommendations(risk_level, risks):
  Base list by risk_level (low/moderate/high/critical)
  + Conditional additions:
    heart_disease > 0.6  → "lipid management advised"
    diabetes > 0.6       → "fasting glucose test recommended"
    renal_failure > 0.6  → "nephrology consultation advised"
    hepatitis > 0.6      → "further liver panel recommended"
    thyroid_disorder > 0.6 → "endocrinology referral advised"
```

---

### 5.7 Complete ML Output Schema

```json
{
  "risks": {
    "diabetes":  0.1200,
    "anemia":    0.0500,
    "infection": 0.0300
  },
  "risk_level": "low",
  "key_factors": [
    { "feature": "hemoglobin", "impact": -0.18, "direction": "decreases" },
    { "feature": "wbc",        "impact":  0.09, "direction": "increases" }
  ],
  "recommendations": [
    "Maintain current lifestyle. Annual check-up recommended."
  ],
  "shap_values": {
    "wbc": 0.09, "rbc": -0.02, "hemoglobin": -0.18,
    "hematocrit": 0.01, "platelets": 0.00,
    "glucose": 0.04, "creatinine": -0.01, "bun": 0.00
  },
  "model_version": "neural-ensemble-blood-v1",
  "severity": {
    "diabetes": 0, "anemia": 0, "infection": 0
  },
  "ocr_coverage": {
    "found": ["wbc", "rbc", "hemoglobin", "glucose"],
    "missing": ["hematocrit", "platelets", "creatinine", "bun"],
    "coverage_pct": 50.0
  },
  "raw_xgb_probas": {
    "diabetes": 0.1200, "anemia": 0.0500, "infection": 0.0300
  }
}
```

---

## 6. API Contracts & Database Transactions

### 6.1 Full API Contract Table

| Method | Path | Auth | Rate Limit | Success | Errors |
|---|---|---|---|---|---|
| POST | /auth/register | No | 3/min | 200 UserRead | 409 Conflict, 422 Validation |
| POST | /auth/login | No | 5/min | 200 TokenResponse | 401 Wrong credentials |
| POST | /auth/refresh | No | 10/min | 200 TokenResponse | 401 Invalid refresh token |
| POST | /api/upload | Yes | — | 200 List[FileResult] | 401, 422 bad type, 500 S3 |
| GET | /api/status/{id} | Yes | — | 200 StatusResponse | 401, 404 |
| GET | /api/reports | Yes | — | 200 List[ReportSummary] | 401 |
| GET | /api/reports/{id} | Yes | — | 200 ReportDetail | 401, 404 |
| GET | /api/reports/{id}/download | Yes | — | 200 File / 302 S3 redirect | 401, 404, 500 |
| POST | /api/compare | Yes | — | 200 {comparison_id} | 400 wrong type, 400 not completed, 404 |
| GET | /api/compare | Yes | — | 200 List[ComparisonSummary] | 401 |
| GET | /api/compare/{id} | Yes | — | 200 ComparisonDetail | 401, 404 |
| GET | / | No | — | 200 {message} | — |

---

### 6.2 Database Transactions per Operation

**Upload flow** (FastAPI process):
```sql
-- 1. Insert report row
INSERT INTO reports (filename, content_type, file_path, user_id, report_type, status)
VALUES ('file.pdf', 'application/pdf', 's3key', 7, 'blood', 'uploaded');

-- Report ID returned to frontend immediately
-- Celery task queued asynchronously (no DB write yet)
```

**OCR complete** (Celery worker — _save_ocr_results):
```sql
UPDATE reports
SET raw_text = '...', extracted_metrics = '{...}',
    ocr_confidence = 0.923, status = 'ocr_complete', processed_at = NOW()
WHERE id = 42;
```

**Prediction saved** (Celery worker — _save_prediction):
```sql
-- Atomic transaction: both writes must succeed together
INSERT INTO tasks (task_id, status, result)
VALUES ('predict-42-a1b2c3d4', 'completed', '{risks:{...}, risk_level:"low", ...}');

UPDATE reports SET status = 'completed', processed_at = NOW() WHERE id = 42;
COMMIT;
```

**Comparison triggered** (FastAPI process):
```sql
INSERT INTO report_comparisons (id, user_id, report1_id, report2_id, report_type, status)
VALUES ('uuid-string', 7, 42, 41, 'blood', 'pending');
-- Celery task queued
```

**Comparison complete** (Celery worker — _save_comparison):
```sql
UPDATE report_comparisons
SET status = 'completed',
    comparison_data = '{metrics:[...], ...}',
    significant_changes = '[...]',
    trend_analysis = 'IMPROVING'
WHERE id = 'uuid-string';
```

---

### 6.3 Error Scenarios

| Scenario | Detection | Response | Recovery |
|---|---|---|---|
| S3 upload fails | boto3 exception in upload.py | 500 to frontend, file not in DB | User retries upload |
| OCR returns empty text | `len(text_data) == 0` | coverage_pct = 0, defaults used | Warn in logs, predictions use defaults |
| ML service unreachable | `requests.post` timeout/error | Task retried up to 3× | After 3 fails → status="failed" |
| Model file missing | FileNotFoundError | None placeholder, proba=0.0 | Worker continues with 0 risk for that disease |
| DB write fails (task) | SQLAlchemy exception | db.rollback(), re-raised | Task retried by Celery |
| Refresh token expired | 401 on /auth/refresh | tokens cleared | User must log in again |
| Comparison type mismatch | Pre-check in compare.py | 400 Bad Request | Frontend validates before sending |
| Poll timeout (frontend) | 60 attempts × 3s = 180s | Error banner shown | User may retry upload |

---

### 6.4 Performance Characteristics

| Operation | Expected Duration | Notes |
|---|---|---|
| File upload to S3 | < 2s | 50 MB max |
| DB INSERT (report row) | < 50ms | Neon PostgreSQL |
| Celery task queued | < 100ms | Redis publish |
| HF Space cold start | 30–120s | Hugging Face free tier wakes up |
| PaddleOCR on CPU | 5–30s | Depends on image size/quality |
| XGBoost inference | < 500ms | All models in memory |
| Neural ensemble | < 10ms | Pure NumPy matrix ops |
| SHAP computation | 1–5s | TreeExplainer on XGBoost |
| Total pipeline (warm) | ~15–45s | OCR dominates |
| Total pipeline (cold start) | ~90–180s | HF Space wake-up included |

---

## 7. Testing Strategy

### 7.1 Unit Tests

#### Auth Security (`test_security.py`)
```python
def test_password_hash_and_verify():
    hashed = get_password_hash("MySecret123")
    assert verify_password("MySecret123", hashed) is True
    assert verify_password("WrongPass",   hashed) is False

def test_access_token_contains_correct_claims():
    token = create_access_token({"sub": "user@example.com"})
    payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
    assert payload["sub"] == "user@example.com"
    assert payload["type"] == "access"

def test_refresh_token_has_longer_expiry():
    access  = create_access_token({"sub": "user"})
    refresh = create_refresh_token({"sub": "user"})
    a_exp = jwt.decode(access,  SECRET_KEY, algorithms=["HS256"])["exp"]
    r_exp = jwt.decode(refresh, SECRET_KEY, algorithms=["HS256"])["exp"]
    assert r_exp > a_exp
```

#### Feature Engineering (`test_feature_engineering.py`)
```python
def test_blood_coverage_full():
    metrics = {
        "wbc":        {"value": 7.5,  "unit": "10³/µL"},
        "rbc":        {"value": 4.5,  "unit": "10⁶/µL"},
        "hemoglobin": {"value": 14.0, "unit": "g/dL"},
        "hematocrit": {"value": 43.0, "unit": "%"},
        "platelets":  {"value": 250,  "unit": "10³/µL"},
        "glucose":    {"value": 90.0, "unit": "mg/dL"},
        "creatinine": {"value": 1.0,  "unit": "mg/dL"},
        "bun":        {"value": 15.0, "unit": "mg/dL"},
    }
    _, coverage = validate_ocr_metrics(metrics, "blood")
    assert coverage["coverage_pct"] == 100.0
    assert coverage["missing"] == []

def test_missing_metric_uses_default():
    metrics = {"wbc": {"value": 7.5, "unit": "10³/µL"}}
    vector, names = build_feature_vector(metrics, "blood")
    # rbc defaults to 4.9, hemoglobin to 14.0, etc.
    assert vector[names.index("rbc")] == 4.9

def test_glucose_unit_normalization():
    metrics = {"glucose": {"value": 5.0, "unit": "mmol/L"}}
    normalized = normalize_units(metrics)
    assert abs(normalized["glucose"]["value"] - 90.09) < 0.1  # 5.0 × 18.0182
    assert normalized["glucose"]["unit"] == "mg/dL"

def test_validate_value_rejects_out_of_range_glucose():
    result = validate_value("glucose", 5.0)   # 5.0 < 20 → treated as mmol/L input
    assert result == 90.0                      # falls back to default
```

#### OCR Regex Patterns (`test_ocr_patterns.py`)
```python
import re

def test_wbc_plain_format():
    text = "WBC 7.5 K/uL"
    pattern = r'(?:WBC|White\s+Blood\s+Cell)[^0-9\n]{0,50}([0-9]+\.?[0-9]*)(?!\s*-)'
    match = re.findall(pattern, text, re.IGNORECASE)
    assert match == ["7.5"]

def test_wbc_with_unit_between_label_and_value():
    text = "WBC (K/µL) 7.5"
    pattern = r'(?:WBC|White\s+Blood\s+Cell)[^0-9\n]{0,50}([0-9]+\.?[0-9]*)(?!\s*-)'
    match = re.findall(pattern, text, re.IGNORECASE)
    assert match == ["7.5"]

def test_does_not_capture_reference_range_start():
    text = "WBC 7.5 4.0-11.0 K/uL"
    pattern = r'(?:WBC)[^0-9\n]{0,50}([0-9]+\.?[0-9]*)(?!\s*-)'
    match = re.findall(pattern, text, re.IGNORECASE)
    assert match[0] == "7.5"   # not "4.0"

def test_vitamin_d_various_formats():
    texts = [
        "Vitamin D 45.2 ng/mL",
        "25-OH Vitamin D: 45.2",
        "Vit D  45.2",
    ]
    pattern = r'(?:25[\s\-]?(?:OH|Hydroxy)?\s*Vitamin\s*D|Vitamin\s*D|Vit\s*D)[^0-9\n]{0,50}([0-9]+\.?[0-9]*)(?!\s*-)'
    for text in texts:
        match = re.findall(pattern, text, re.IGNORECASE)
        assert match == ["45.2"], f"Failed for: {text}"
```

#### Neural Ensemble (`test_neural_ensemble.py`)
```python
def test_identity_passthrough_when_untrained():
    ensemble = NeuralEnsemble(n_diseases=3)
    input_probs = np.array([0.2, 0.5, 0.8], dtype=np.float32)
    output = ensemble.predict(input_probs)
    np.testing.assert_array_equal(output, input_probs)

def test_trained_ensemble_outputs_valid_probabilities():
    ensemble = NeuralEnsemble(n_diseases=2)
    X = np.random.rand(100, 2).astype(np.float32)
    y = (X > 0.5).astype(np.float32)
    ensemble.fit(X, y, epochs=50)
    out = ensemble.predict(np.array([0.3, 0.7]))
    assert all(0.0 <= p <= 1.0 for p in out)

def test_save_and_load_roundtrip(tmp_path):
    path = str(tmp_path / "ensemble_test.pkl")
    e1 = NeuralEnsemble(n_diseases=2)
    e1.fit(np.random.rand(50, 2), np.random.rand(50, 2), epochs=10)
    e1.save(path)
    e2 = NeuralEnsemble.load(path)
    out1 = e1.predict(np.array([0.3, 0.7]))
    out2 = e2.predict(np.array([0.3, 0.7]))
    np.testing.assert_array_almost_equal(out1, out2)
```

---

### 7.2 Integration Tests

#### Upload → Poll → Result (`test_upload_flow.py`)
```python
@pytest.fixture
def auth_headers(client, test_user):
    resp = client.post("/auth/login",
        data={"username": test_user.email, "password": "TestPass123"})
    token = resp.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}

def test_upload_queues_task_and_report_row_created(client, auth_headers, mock_celery):
    with open("tests/fixtures/sample_blood.jpg", "rb") as f:
        resp = client.post("/api/upload",
            files={"files": ("blood.jpg", f, "image/jpeg")},
            data={"report_type": "blood"},
            headers=auth_headers)
    assert resp.status_code == 200
    data = resp.json()
    assert data[0]["status"] == "success"
    assert "id" in data[0]
    # Celery task should have been queued
    mock_celery.assert_called_once()

def test_status_returns_uploaded_before_processing(client, auth_headers, db, report_fixture):
    resp = client.get(f"/api/status/{report_fixture.id}", headers=auth_headers)
    assert resp.json()["status"] == "uploaded"

def test_status_returns_completed_with_result(client, auth_headers, db, completed_report):
    resp = client.get(f"/api/status/{completed_report.id}", headers=auth_headers)
    body = resp.json()
    assert body["status"] == "completed"
    assert body["result"]["risk_level"] in ("low","moderate","high","critical")

def test_user_cannot_access_another_users_report(client, auth_headers_user2, report_user1):
    resp = client.get(f"/api/status/{report_user1.id}", headers=auth_headers_user2)
    assert resp.status_code == 404
```

#### Compare Flow (`test_compare_flow.py`)
```python
def test_compare_requires_same_report_type(client, auth_headers, blood_report, lipid_report):
    resp = client.post("/api/compare",
        json={"report1_id": blood_report.id, "report2_id": lipid_report.id},
        headers=auth_headers)
    assert resp.status_code == 400
    assert "same type" in resp.json()["detail"]

def test_compare_requires_both_completed(client, auth_headers, blood_r1, blood_r2_pending):
    resp = client.post("/api/compare",
        json={"report1_id": blood_r1.id, "report2_id": blood_r2_pending.id},
        headers=auth_headers)
    assert resp.status_code == 400

def test_compare_returns_trend(client, auth_headers, two_completed_blood_reports, mock_celery):
    r1_id, r2_id = two_completed_blood_reports
    resp = client.post("/api/compare",
        json={"report1_id": r1_id, "report2_id": r2_id},
        headers=auth_headers)
    assert resp.status_code == 200
    comparison_id = resp.json()["comparison_id"]
    assert len(comparison_id) == 36  # UUID format
```

#### Task Worker (`test_tasks.py`)
```python
@patch("task_queue.tasks.requests.post")
def test_process_medical_report_saves_ocr_and_prediction(mock_post, db, report):
    mock_post.return_value.json.return_value = {
        "raw_text": "WBC 7.5",
        "structured_metrics": {"wbc": {"value": 7.5, "unit": "10³/µL", "source": "text"}},
        "ocr_confidence": 0.95,
        "prediction": {
            "risks": {"diabetes": 0.05},
            "risk_level": "low",
            "key_factors": [],
            "recommendations": ["Annual check-up."],
            "model_version": "neural-ensemble-blood-v1",
            "raw_xgb_probas": {"diabetes": 0.05},
            "ocr_coverage": {"found":["wbc"], "missing":[], "coverage_pct":12.5},
        }
    }
    mock_post.return_value.raise_for_status = lambda: None

    result = process_medical_report(report.id, "/tmp/test.jpg", "blood")

    assert result["status"] == "completed"
    db.refresh(report)
    assert report.status == "completed"
    assert report.extracted_metrics == {"wbc": {"value": 7.5, "unit": "10³/µL", "source": "text"}}

def test_process_medical_report_retries_on_network_error(mock_post, report):
    mock_post.side_effect = ConnectionError("Network down")
    with pytest.raises(MaxRetriesExceededError):
        process_medical_report.apply(args=[report.id, "/tmp/test.jpg", "blood"],
                                     retries=3)
    db.refresh(report)
    assert report.status == "failed"
```

---

### 7.3 Edge Case Validation

| Edge Case | Test Approach |
|---|---|
| 0% OCR coverage | Pass empty metrics dict, assert all features use defaults |
| All features at defaults | Assert predictions complete without error |
| Model pkl missing | Mock FileNotFoundError, assert proba=0.0 for that disease |
| Glucose in mmol/L | Assert unit inference and conversion applied |
| vitamin_d 0.0% coverage | Assert warning logged and fallback default 30.0 used |
| Concurrent uploads | Two parallel requests with same user — assert separate rows |
| PDF upload | Mock OCR to handle PDF → image conversion |
| Expired access token | Assert 401, then assert refresh attempted, then retry |
| Report not owned by user | Assert 404 returned, not 403 |
| Compare same report to itself | Should succeed technically (0% change all metrics) |

---

## 8. Deployment & Operations

### 8.1 Docker Compose (Local Development)

```yaml
services:

  db:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: medscan
      POSTGRES_USER: medscan
      POSTGRES_PASSWORD: medscan
    ports: ["5432:5432"]
    volumes: [postgres_data:/var/lib/postgresql/data]
    healthcheck:
      test: ["CMD", "pg_isready", "-U", "medscan"]
      interval: 5s

  redis:
    image: redis:7-alpine
    ports: ["6379:6379"]
    volumes: [redis_data:/data]
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]

  api:
    build: ./backend
    command: uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
    ports: ["8000:8000"]
    depends_on: [db, redis]
    env_file: .env
    volumes:
      - ./backend:/app
      - upload_data:/app/data/raw_uploads

  worker:
    build: ./backend
    command: >
      celery -A task_queue.celery_app worker
        --loglevel=info --concurrency=2
        -Q default,ocr,prediction,comparison
    depends_on: [db, redis]
    env_file: .env
    environment:
      PADDLE_USE_GPU: "false"

  flower:
    image: mher/flower
    command: celery flower --broker=${REDIS_URL} --port=5555
    ports: ["5555:5555"]
    depends_on: [redis]
```

**Start local stack**:
```bash
docker compose up --build
# API available at http://localhost:8000
# Flower dashboard at http://localhost:5555
```

---

### 8.2 Environment Variables Reference

```bash
# ── Auth ──────────────────────────────────────
SECRET_KEY=<256-bit hex>              # JWT signing key — NEVER commit to git
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30
REFRESH_TOKEN_EXPIRE_DAYS=7

# ── Database ──────────────────────────────────
NEON_DB_URL=postgresql://user:pass@host/db?sslmode=require
NEON_ASYNC_DB_URL=postgresql+asyncpg://user:pass@host/db

# ── Redis / Celery ────────────────────────────
CELERY_BROKER_URL=rediss://:<password>@host:port/0   # rediss:// = SSL
CELERY_RESULT_BACKEND=rediss://:<password>@host:port/0

# ── S3 Storage ────────────────────────────────
USE_S3=true
S3_ENDPOINT_URL=https://<project>.supabase.co/storage/v1/s3
S3_ACCESS_KEY=<key>
S3_SECRET_KEY=<secret>
S3_BUCKET_NAME=medscan-uploads
S3_REGION=us-east-1

# ── ML Service ────────────────────────────────
ML_SERVICE_URL=https://direkkakkar-medscan-ai-ml-models.hf.space

# ── OCR tuning ────────────────────────────────
PADDLE_USE_GPU=false
PADDLE_OCR_LANG=en
```

---

### 8.3 Production Deployment (Render)

**API service** (`render.yaml` excerpt):
```yaml
services:
  - type: web
    name: medscan-api
    runtime: python
    buildCommand: pip install -r requirements.txt
    startCommand: >
      uvicorn app.main:app
        --host 0.0.0.0
        --port $PORT
        --workers 2
        --proxy-headers
        --forwarded-allow-ips='*'
    envVars:
      - key: SECRET_KEY
        sync: false
      - key: NEON_DB_URL
        sync: false

  - type: worker
    name: medscan-worker
    runtime: python
    buildCommand: pip install -r requirements.txt
    startCommand: >
      celery -A task_queue.celery_app worker
        --loglevel=info --concurrency=1
        -Q default,ocr,prediction,comparison
```

**ML service** (Hugging Face Space):
- Runtime: Docker (Python 3.10)
- Port: 7860
- `app.py` starts FastAPI on 7860
- Models downloaded from PaddleX on first startup (cached in `/root/.paddlex/`)
- XGBoost `.pkl` files bundled in the Space repository

---

### 8.4 Logging

**Backend (Python)**:
```python
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
```

Key log messages to monitor:

| Logger | Message | Significance |
|---|---|---|
| `task_queue.tasks` | `[process_medical_report] Starting for report {id}` | Task picked up |
| `task_queue.tasks` | `ML service done: N metrics, risk_level=X` | OCR + ML complete |
| `task_queue.tasks` | `[process_medical_report] failed for report {id}` | Task error — check exc |
| `ml_models.xgboost.feature_engineering` | `Low OCR coverage (X%)` | Image quality issue |
| `ml_models.predict` | `Individual XGBoost risks` | Per-model scores |
| `ml_models.paddle_ocr.ocr_runner` | `PPStructureV3 failed to initialise` | Table extraction off |

---

### 8.5 Monitoring Alerts (Recommended)

| Alert | Condition | Action |
|---|---|---|
| OCR coverage < 30% | coverage_pct < 30 in worker log | Check image format / OCR version |
| Task failure rate > 5% | Failed tasks in Flower dashboard | Check ML service health |
| HF Space cold starts | Response time > 120s | Upgrade to paid tier or pre-warm |
| DB connection errors | SQLAlchemy errors in API logs | Check Neon connection pool |
| Redis connection loss | Celery can't reach broker | Check Upstash status |
| Token refresh failures > 10/min | 401s on /auth/refresh | Check SECRET_KEY consistency |

**Flower dashboard** (`http://localhost:5555` locally) provides:
- Active / reserved / failed task counts
- Task execution times
- Worker online status
- Per-queue depth

---

## Appendix — Complete Request/Response Lifecycle (One Upload)

```
00:00  User selects blood_panel.pdf, chooses "blood", clicks Upload
00:00  POST /api/upload  multipart/form-data  →  FastAPI
00:00    Validate report_type="blood"  ✓
00:00    Upload file → Supabase S3: uploads/7_a3f89bc1.pdf
00:00    INSERT reports (status="uploaded", id=42)
00:00    process_medical_report.delay(42, "uploads/7_a3f89bc1.pdf", "blood")
00:00  ← 200 [{ id:42, filename:"blood_panel.pdf", status:"success" }]

00:00  Frontend: localStorage.pending=42
00:00  Frontend: starts polling GET /api/status/42 every 3s

00:01  Celery worker picks up task
00:01    UPDATE reports SET status="preprocessing" WHERE id=42
00:01    Download file from S3 → /tmp/7_a3f89bc1.pdf
00:01    POST {ML_SERVICE_URL}/analyze  (timeout 300s)

00:03  Frontend poll #1  →  { status: "preprocessing" }
00:06  Frontend poll #2  →  { status: "preprocessing" }

~00:20 ML service (HF Space) finishes OCR + prediction
00:20    Worker receives { raw_text, structured_metrics, ocr_confidence, prediction }
00:20    UPDATE reports SET status="ocr_complete", extracted_metrics={...}
00:20    INSERT tasks (task_id="predict-42-a1b2", result={risks, risk_level, ...})
00:20    UPDATE reports SET status="completed"
00:20    DELETE s3://medscan-uploads/uploads/7_a3f89bc1.pdf

00:21  Frontend poll #7  →  { status: "completed", result: { risks:{...}, risk_level:"low" } }
00:21    stopPolling()
00:21    transformResult() → riskMetrics, biomarkers, riskLevel
00:21    setReportData(transformed)
00:21    localStorage.latest=42
00:21    Render: Overall Risk badge, Radar chart, Biomarker cards, Recommendation
```

Total wall-clock time (warm HF Space): **~20 seconds**  
Total wall-clock time (cold HF Space start): **~90–150 seconds**

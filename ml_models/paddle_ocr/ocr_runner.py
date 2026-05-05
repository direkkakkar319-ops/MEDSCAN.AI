"""
ml_models/paddle_ocr/ocr_runner.py — PaddleOCR wrapper and lab-value parsers.

This module is the entry point for all OCR work in the ML service.
get_ocr_runner() returns the process-wide OCRRunner singleton; callers call
ocr_runner.process_report(image_path, report_type) and get back a dict with:
  raw_text            — full concatenated OCR text
  structured_metrics  — {key: {"value": float, "unit": str, "source": str}}
  tables              — list of table dicts from PPStructureV3 (may be empty)
  average_confidence  — mean OCR confidence across all text fragments
  text_items          — number of text fragments detected

Architecture decisions:
  Singleton pattern  — PaddleOCR loads ~400 MB of ONNX model weights.  Creating
                       a new instance per request would exceed RAM on free-tier
                       containers and add 5–30 s cold-start per task.
  Two engines        — ocr_engine (PP-OCRv5) handles text detection & recognition.
                       structure_engine (PPStructureV3) handles table layout.
                       Structure is tried after text regex fails (fallback path).
  Per-type parsers   — each report type (blood, lipid, etc.) has its own regex
                       pattern dict tuned to that lab format.  A single universal
                       parser would require far more complex patterns and produce
                       more false positives.
"""

import cv2
import numpy as np
from typing import Dict, List, Any, Optional
import logging
import re
import os

# Guard the PaddleOCR import so that files that import OCRRunner for testing
# don't crash on machines where PaddleOCR is not installed.
try:
    from paddleocr import PaddleOCR, PPStructureV3
    _PADDLE_AVAILABLE = True
except ImportError:
    _PADDLE_AVAILABLE = False

logger = logging.getLogger(__name__)


class OCRRunner:
    """
    Singleton wrapper around PP-OCRv5 and PP-StructureV3.

    Only ONE instance of this class ever exists per worker process.
    The singleton is enforced at two levels:
      1. __new__   — returns the cached instance instead of allocating a new one.
      2. __init__  — guards against re-initialising the already-loaded engines.

    Public API:
      process_report(image_path, report_type) → full result dict
      extract_text(image_path)                → list of {text, confidence, bbox}
      extract_tables(image_path)              → list of table dicts
    """

    # Class-level singleton storage — shared across ALL instances.
    _instance = None

    def __new__(cls):
        """
        Python's object allocator — runs BEFORE __init__ on every OCRRunner() call.

        If the singleton has not been created yet, allocates a new object and
        marks it as uninitialised.  If it already exists, returns the cached
        object.  The _initialized flag on the instance prevents __init__ from
        re-running the expensive PaddleOCR setup on the second and subsequent
        OCRRunner() calls.
        """
        if cls._instance is None:
            cls._instance = super(OCRRunner, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """
        Initialise the PaddleOCR text engine and PPStructureV3 table engine.

        Guard 1 (_initialized check) — skips re-init if the engines are already
        loaded.  This is what makes the singleton "lazy": the first OCRRunner()
        call does all the work; every subsequent call returns immediately.

        Guard 2 (_PADDLE_AVAILABLE check) — raises RuntimeError with a helpful
        install command if paddleocr is not installed (prevents a cryptic
        AttributeError later).

        Engine configuration:
          use_doc_orientation_classify=False — skips whole-page rotation detection.
            Most lab reports are portrait-upright; the classifier adds ~200 ms
            per image for no benefit.
          use_doc_unwarping=False — skips perspective dewarping for book scans.
            Lab reports are flat scans, not photos of curved pages.
          use_textline_orientation=True — keeps per-line angle correction on.
            Some printers produce slightly rotated text lines; this corrects them.

        Fallback for PaddleOCR init failure:
          If the full parameter set fails (e.g. a future API change), a minimal
          PaddleOCR(lang=...) call is attempted.  This keeps the service alive
          at the cost of slightly lower OCR accuracy.

        PPStructureV3 failure is non-fatal:
          If the table engine fails to load, structure_engine is set to None and
          table extraction is silently disabled.  The text-based regex parsers
          still run, so partial extraction continues.
        """
        if self._initialized:
            return

        if not _PADDLE_AVAILABLE:
            raise RuntimeError(
                "PaddleOCR is not installed.\n"
                "Run: pip install paddlepaddle paddleocr"
            )

        # Read configuration from environment so docker-compose can override
        # without changing code (PADDLE_USE_GPU=true for GPU containers).
        self.use_gpu = os.getenv('PADDLE_USE_GPU', 'false').lower() == 'true'
        self.lang    = os.getenv('PADDLE_OCR_LANG', 'en')

        logger.info(f"Initializing PaddleOCR (GPU: {self.use_gpu}, Lang: {self.lang})")

        # PaddleOCR 3.4.0 changed its constructor API — the old parameters
        # (use_gpu, use_angle_cls, show_log, enable_mkldnn, ocr_version,
        # limit_side_len, det_db_thresh, etc.) no longer exist.  The new API
        # only accepts model-selection and pipeline-toggle parameters.
        try:
            self.ocr_engine = PaddleOCR(
                lang=self.lang,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=True,
            )
            logger.info("PaddleOCR (text engine) initialised successfully")
        except Exception as e:
            logger.warning(f"PaddleOCR init failed ({e}), retrying with minimal params")
            self.ocr_engine = PaddleOCR(lang=self.lang)
            logger.info("PaddleOCR (text engine) initialised successfully (minimal params)")

        try:
            self.structure_engine = PPStructureV3(
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                lang=self.lang,
            )
            logger.info("PPStructureV3 (table engine) initialised successfully")
        except Exception as e:
            logger.warning(f"PPStructureV3 failed to initialise ({e}). Table extraction disabled.")
            self.structure_engine = None

        self._initialized = True
        logger.info("OCRRunner fully initialised")

    # ── Image preprocessing ───────────────────────────────────────────────────

    def preprocess_image(self, image_path: str) -> np.ndarray:
        """
        Apply a 3-step image enhancement pipeline to improve OCR accuracy.

        Step 1 — Grayscale conversion:
          PaddleOCR works on greyscale internally.  Converting before passing
          to the engine removes colour noise (different coloured text vs background)
          and reduces data size by 3×.

        Step 2 — Denoising (fastNlMeansDenoising):
          Scanned lab reports often have scanner noise (speckle, compression
          artefacts from JPEG).  h=10 is a mild strength; h=7 pixel neighbourhood
          size; h=21 search window size.  These values were tuned to remove noise
          without blurring thin font strokes.

        Step 3 — CLAHE (Contrast Limited Adaptive Histogram Equalisation):
          Improves local contrast — particularly useful for faded prints or reports
          with uneven lighting across the page.  clipLimit=2.0 prevents over-
          amplification of noise in very dark regions.  tileGridSize=(8,8) divides
          the image into 64 local regions for adaptive equalisation.

        The processed image is written to a temp file (not returned from this
        method) by the caller (extract_text) because PaddleOCR accepts a file
        path, not a numpy array.

        Args:
            image_path — absolute path to the original uploaded image.

        Returns:
            Preprocessed greyscale numpy array (not yet saved to disk).

        Raises:
            ValueError — if cv2.imread returns None (file doesn't exist or
                         format is unreadable by OpenCV).
        """
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Cannot load image: {image_path}")

        gray     = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        clahe    = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)

        return enhanced

    # ── Text extraction ───────────────────────────────────────────────────────

    def extract_text(self, image_path: str, preprocess: bool = True) -> List[Dict]:
        """
        Run PP-OCRv5 on an image and return detected text in reading order.

        Flow:
          1. (If preprocess=True) Enhance the image and write to a temp file.
             PaddleOCR 3.4 requires a file path — it cannot accept a numpy array.
          2. Call ocr_engine.predict(input_path) — returns a list of result dicts,
             one per image page.
          3. Parse the PaddleOCR 3.4 result format:
               rec_texts  — list of recognised strings
               rec_scores — list of confidence scores (0–1)
               rec_polys  — list of bounding polygon arrays (4 × 2 or N × 2)
          4. Sort all text fragments top-to-bottom, left-to-right by bounding
             box centre coordinates.  This gives reading order regardless of how
             PaddleOCR internally ordered its detections.
          5. Delete the temp file in the finally block.

        Args:
            image_path — absolute path to the input image.
            preprocess — if True, applies preprocess_image() first (recommended
                         for real reports).  Set False in unit tests.

        Returns:
            List of dicts sorted in reading order, each:
              {"text": str, "confidence": float, "bbox": [[x,y], ...]}
        """
        temp_path = None
        try:
            if preprocess:
                temp_path = f"{image_path}_temp.jpg"
                processed = self.preprocess_image(image_path)
                cv2.imwrite(temp_path, processed)
                input_path = temp_path
            else:
                input_path = image_path

            # PaddleOCR 3.4 API: use predict(), not ocr() (deprecated).
            results = self.ocr_engine.predict(input_path)

        except Exception as e:
            logger.error(f"Text extraction failed: {e}", exc_info=True)
            raise

        finally:
            # Always remove the temp file even if predict() raises.
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)

        extracted: List[Dict] = []

        # PaddleOCR 3.4 returns: [{"rec_texts": [...], "rec_scores": [...],
        #                           "rec_polys": [array, ...], ...}, ...]
        # One dict per page — we only process single-page lab reports.
        for page_result in (results or []):
            if not page_result:
                continue
            texts  = page_result.get('rec_texts', [])
            scores = page_result.get('rec_scores', [])
            polys  = page_result.get('rec_polys', [])

            for text, score, poly in zip(texts, scores, polys):
                if not text.strip():
                    continue
                # Convert numpy polygon to plain list for JSON serialisability.
                bbox = poly.tolist() if hasattr(poly, 'tolist') else list(poly)
                extracted.append({
                    'text':       text,
                    'confidence': float(score),
                    'bbox':       bbox,
                })

        # Sort by (y_centre, x_centre) — top-to-bottom primary, left-to-right secondary.
        extracted.sort(key=lambda x: (
            self._get_center(x['bbox'])[1],
            self._get_center(x['bbox'])[0],
        ))

        return extracted

    # ── Table extraction ──────────────────────────────────────────────────────

    def extract_tables(self, image_path: str) -> List[Dict]:
        """
        Extract structured table data using PP-StructureV3.

        PP-StructureV3 understands page layout — it identifies table regions,
        separates them from prose text, and returns cell data.  This is used
        as a fallback when the regex text parsers fail to find any metrics
        (e.g. when a lab report uses a pure-table format with no keyword labels).

        Each returned table dict:
          { "html": str,    — HTML representation of the table
            "data": list,   — list of rows, each row is a list of cell strings
            "bbox": list }  — bounding box of the table on the page

        Returns [] if:
          - structure_engine is None (failed to initialise at startup).
          - predict() raises an exception (caught and logged).

        Args:
            image_path — absolute path to the input image.

        Returns:
            List of table dicts, empty list on any failure.
        """
        if self.structure_engine is None:
            return []

        try:
            result = self.structure_engine.predict(input=image_path)
            tables = []

            for res in result:
                # PPStructureV3 result items may be dict-like objects or
                # custom result types — access both ways defensively.
                item_type = (
                    res.get('type', '') if isinstance(res, dict)
                    else getattr(res, 'type', '')
                )
                if item_type == 'table':
                    item_res = (
                        res.get('res', {}) if isinstance(res, dict)
                        else getattr(res, 'res', {})
                    )
                    tables.append({
                        'html': item_res.get('html', '') if isinstance(item_res, dict) else '',
                        'data': item_res.get('data', []) if isinstance(item_res, dict) else [],
                        'bbox': res.get('bbox', []) if isinstance(res, dict) else [],
                    })

            return tables

        except Exception as e:
            logger.error(f"Table extraction failed: {e}", exc_info=True)
            return []

    # ── Main report pipeline ──────────────────────────────────────────────────

    def process_report(self, image_path: str, report_type: str) -> Dict[str, Any]:
        """
        Run the full OCR pipeline for a medical report and return structured data.

        This is the ONLY public method called by the ML service's /analyze endpoint.
        All other public methods (extract_text, extract_tables) are exposed for
        testing but are not called directly in production.

        Steps:
          1. Extract text — runs PP-OCRv5 with preprocessing on the image.
          2. Extract tables — runs PP-StructureV3 for table-format reports.
          3. Parse metrics — calls the correct _parse_* method based on report_type.
          4. Compute confidence — mean of all per-fragment OCR confidence scores.

        Parser selection:
          The parsers dict maps report_type strings to bound methods.
          parse_fn = parsers.get(report_type) — returns None for unknown types.
          The fallback lambda calls _parse_general() which just returns raw lines.

        Args:
            image_path  — absolute path to the uploaded image file.
            report_type — one of: blood, lipid, vitamin_d, hormone, kidney, liver.

        Returns dict:
          raw_text           — full joined OCR text (useful for debugging).
          structured_metrics — {key: {"value": float, "unit": str, "source": str}}.
          tables             — list of table dicts from PPStructureV3.
          average_confidence — float 0–1, mean recognition confidence.
          text_items         — int, count of text fragments detected.
        """
        logger.info(f"Processing {report_type} report: {image_path}")

        text_data = self.extract_text(image_path)
        full_text = '\n'.join([item['text'] for item in text_data])

        tables = self.extract_tables(image_path)

        parsers = {
            'blood':     self._parse_blood_report,
            'lipid':     self._parse_lipid_profile,
            'vitamin_d': self._parse_vitamin_d,
            'hormone':   self._parse_hormone_report,
            'kidney':    self._parse_kidney_function_report,
            'liver':     self._parse_liver_function_report,
        }

        parse_fn = parsers.get(report_type, lambda t, _: self._parse_general(t))
        metrics  = parse_fn(text_data, tables)

        avg_confidence = (
            sum(item['confidence'] for item in text_data) / len(text_data)
            if text_data else 0
        )

        return {
            'raw_text':           full_text,
            'structured_metrics': metrics,
            'tables':             tables,
            'average_confidence': round(avg_confidence, 3),
            'text_items':         len(text_data),
        }

    # ── Report-type parsers ───────────────────────────────────────────────────

    def _parse_blood_report(self, text_data: List[Dict], tables: List[Dict]) -> Dict[str, Any]:
        """
        Extract CBC and metabolic panel metrics from blood report text.

        Strategy:
          1. Join all OCR text fragments into one flat string (space-separated).
             This handles cases where a metric name and its value appear as
             separate text fragments side by side on the page.
          2. Apply regex patterns for each expected metric.  Each pattern:
               - Matches the metric label (with common abbreviations and spacing).
               - Captures the numeric value ([0-9.]+) that follows.
          3. If any pattern matches, take the first match (matches[0]) and
             store {value, unit, source}.
          4. If NO metrics were found from text AND tables exist, fall back to
             the table parser which scans cell data for numeric values.

        Regex pattern design:
          [\s:)]+ between label and value allows for common lab formats:
            "WBC 7.5"      → one space separator
            "WBC: 7.5"     → colon separator
            "WBC(K/µL)7.5" → closing paren separator

        Args:
            text_data — list of {text, confidence, bbox} dicts from extract_text().
            tables    — list of table dicts from extract_tables() (fallback).

        Returns:
            Dict of {metric_key: {"value": float, "unit": str, "source": str}}.
        """
        text = ' '.join([item['text'] for item in text_data])

        patterns = {
            'wbc':        r'(?:WBC|White\s+Blood\s+Cell)[\s:)]+([0-9.]+)',
            'rbc':        r'(?:RBC|Red\s+Blood\s+Cell)[\s:)]+([0-9.]+)',
            'hemoglobin': r'(?:Hemoglobin|HGB|Hb)[\s:)]+([0-9.]+)',
            'hematocrit': r'(?:Hematocrit|HCT)[\s:)]+([0-9.]+)',
            'platelets':  r'(?:Platelets|PLT)[\s:)]+([0-9.]+)',
            'glucose':    r'(?:Glucose|Fasting\s+Glucose)[\s:)]+([0-9.]+)',
            'creatinine': r'(?:Creatinine)[\s:)]+([0-9.]+)',
            'bun':        r'(?:BUN)[\s:)]+([0-9.]+)',
        }

        metrics: Dict[str, Any] = {}
        for key, pattern in patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                try:
                    metrics[key] = {
                        'value':  float(matches[0]),
                        'unit':   self._infer_unit(key),
                        'source': 'text',
                    }
                except ValueError:
                    continue

        if tables and not metrics:
            metrics = self._extract_from_tables(tables)

        return metrics

    def _parse_lipid_profile(self, text_data: List[Dict], tables: List[Dict]) -> Dict[str, Any]:
        """
        Extract lipid panel metrics: total cholesterol, HDL, LDL, TG, VLDL.

        All lipid values are stored with unit 'mg/dL' — the universal standard
        for lipid panels.  Labs that report in mmol/L are handled by the
        normalize_units() function in feature_engineering.py.

        Args:
            text_data — OCR text fragments from extract_text().
            tables    — table fallback from extract_tables().

        Returns:
            Dict of {metric_key: {"value": float, "unit": "mg/dL", "source": "text"}}.
        """
        text = ' '.join([item['text'] for item in text_data])

        patterns = {
            'total_cholesterol': r'(?:Total\s+Cholesterol|T\.?\s*Chol)[\s:)]+([0-9.]+)',
            'hdl':               r'(?:HDL)[\s:)]+([0-9.]+)',
            'ldl':               r'(?:LDL)[\s:)]+([0-9.]+)',
            'triglycerides':     r'(?:Triglycerides|TG)[\s:)]+([0-9.]+)',
            'vldl':              r'(?:VLDL)[\s:)]+([0-9.]+)',
        }

        metrics: Dict[str, Any] = {}
        for key, pattern in patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                try:
                    metrics[key] = {
                        'value':  float(matches[0]),
                        'unit':   'mg/dL',
                        'source': 'text',
                    }
                except ValueError:
                    continue

        if tables and not metrics:
            metrics = self._extract_from_tables(tables)

        return metrics

    def _parse_vitamin_d(self, text_data: List[Dict], tables: List[Dict]) -> Dict[str, Any]:
        """
        Extract the single Vitamin D metric from a Vitamin D report.

        The pattern covers multiple common label formats:
          "25-OH Vitamin D"   — clinical chemistry label
          "25 Hydroxy Vitamin D" — spelled-out version
          "Vitamin D"         — shortened form
          "Vit D"             — common abbreviation

        The [\s-]? between "25" and "OH" handles both "25-OH" and "25 OH".

        Args:
            text_data — OCR text fragments.
            tables    — fallback table data.

        Returns:
            {"vitamin_d": {"value": float, "unit": "ng/mL", "source": "text"}}
            or {} if not found.
        """
        text = ' '.join(item["text"] for item in text_data)

        patterns = {
            'vitamin_d': r'(?:25[\s\-]?(?:OH|Hydroxy)?\s*Vitamin\s*D|Vitamin\s*D|Vit\s*D)[\s:)]+([0-9.]+)',
        }

        metrics: Dict[str, Any] = {}
        for key, pattern in patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                try:
                    metrics[key] = {
                        'value':  float(matches[0]),
                        'unit':   'ng/mL',
                        'source': 'text',
                    }
                except ValueError:
                    continue

        if tables and not metrics:
            metrics = self._extract_from_tables(tables)

        return metrics

    def _parse_hormone_report(self, text_data: List[Dict], tables: List[Dict]) -> Dict[str, Any]:
        """
        Extract thyroid and sex hormone metrics from a hormone panel report.

        Covers 10 hormone markers across thyroid, sex hormones, and adrenal axes:
          Thyroid: TSH, T3, T4
          Sex hormones: testosterone, estradiol, progesterone
          Pituitary: prolactin, LH, FSH
          Adrenal: cortisol

        T3/T4 patterns match both "T3" alone and "Free T3" / "Triiodothyronine"
        to handle different lab naming conventions.

        Args:
            text_data — OCR text fragments.
            tables    — fallback table data.

        Returns:
            Dict of extracted hormone metrics with appropriate units.
        """
        text = ' '.join(item["text"] for item in text_data)

        patterns = {
            'tsh':          r'(?:TSH|T\.?\s*S\.?\s*H)[\s:)]+([0-9.]+)',
            't3':           r'(?:T3|Free\s*T3|Triiodothyronine)[\s:)]+([0-9.]+)',
            't4':           r'(?:T4|Free\s*T4|Thyroxine)[\s:)]+([0-9.]+)',
            'testosterone': r'(?:Testosterone|Total\s*Testosterone)[\s:)]+([0-9.]+)',
            'estradiol':    r'(?:Estradiol|Estrogen|E2)[\s:)]+([0-9.]+)',
            'progesterone': r'(?:Progesterone)[\s:)]+([0-9.]+)',
            'prolactin':    r'(?:Prolactin)[\s:)]+([0-9.]+)',
            'lh':           r'(?:LH|Luteinizing\s*Hormone)[\s:)]+([0-9.]+)',
            'fsh':          r'(?:FSH|Follicle\s*Stimulating\s*Hormone)[\s:)]+([0-9.]+)',
            'cortisol':     r'(?:Cortisol)[\s:)]+([0-9.]+)',
        }

        metrics: Dict[str, Any] = {}
        for key, pattern in patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                try:
                    metrics[key] = {
                        'value':  float(matches[0]),
                        'unit':   self._infer_unit(key),
                        'source': 'text',
                    }
                except ValueError:
                    continue

        if tables and not metrics:
            metrics = self._extract_from_tables(tables)

        return metrics

    def _parse_kidney_function_report(self, text_data: List[Dict], tables: List[Dict]) -> Dict[str, Any]:
        """
        Extract kidney function metrics: creatinine, BUN, urea, uric acid, eGFR.

        BUN pattern accepts both "BUN" (US standard) and "Blood Urea Nitrogen"
        (written out on some international lab reports).

        eGFR matches both "eGFR" and "GFR" — some reports omit the "estimated" prefix.

        Args:
            text_data — OCR text fragments.
            tables    — fallback table data.

        Returns:
            Dict of kidney function metrics with clinical units.
        """
        text = ' '.join(item["text"] for item in text_data)

        patterns = {
            'creatinine': r'(?:Creatinine)[\s:)]+([0-9.]+)',
            'bun':        r'(?:BUN|Blood\s+Urea\s+Nitrogen)[\s:)]+([0-9.]+)',
            'urea':       r'(?:Urea)[\s:)]+([0-9.]+)',
            'uric_acid':  r'(?:Uric\s+Acid)[\s:)]+([0-9.]+)',
            'egfr':       r'(?:eGFR|GFR)[\s:)]+([0-9.]+)',
        }

        metrics: Dict[str, Any] = {}
        for key, pattern in patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                try:
                    metrics[key] = {
                        'value':  float(matches[0]),
                        'unit':   self._infer_unit(key),
                        'source': 'text',
                    }
                except ValueError:
                    continue

        if tables and not metrics:
            metrics = self._extract_from_tables(tables)

        return metrics

    def _parse_liver_function_report(self, text_data: List[Dict], tables: List[Dict]) -> Dict[str, Any]:
        """
        Extract liver function test (LFT) metrics.

        Covers the standard liver panel:
          ALT/SGPT, AST/SGOT  — aminotransferases (hepatocyte damage markers)
          ALP / Alkaline Phosphatase — biliary and bone enzyme
          Total bilirubin, Direct bilirubin, Indirect bilirubin — bile pigments
          Albumin  — protein synthesis capacity (low = chronic liver disease)
          Total protein — albumin + globulins combined

        SGPT and SGOT are alternate names used in South Asian lab reports.

        Args:
            text_data — OCR text fragments.
            tables    — fallback table data.

        Returns:
            Dict of LFT metrics with clinical units.
        """
        text = ' '.join(item["text"] for item in text_data)

        patterns = {
            'bilirubin_total':    r'(?:Total\s+Bilirubin|Bilirubin\s+Total)[\s:)]+([0-9.]+)',
            'bilirubin_direct':   r'(?:Direct\s+Bilirubin)[\s:)]+([0-9.]+)',
            'bilirubin_indirect': r'(?:Indirect\s+Bilirubin)[\s:)]+([0-9.]+)',
            'alt':                r'(?:ALT|SGPT)[\s:)]+([0-9.]+)',
            'ast':                r'(?:AST|SGOT)[\s:)]+([0-9.]+)',
            'alp':                r'(?:ALP|Alkaline\s+Phosphatase)[\s:)]+([0-9.]+)',
            'albumin':            r'(?:Albumin)[\s:)]+([0-9.]+)',
            'total_protein':      r'(?:Total\s+Protein)[\s:)]+([0-9.]+)',
        }

        metrics: Dict[str, Any] = {}
        for key, pattern in patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                try:
                    metrics[key] = {
                        'value':  float(matches[0]),
                        'unit':   self._infer_unit(key),
                        'source': 'text',
                    }
                except ValueError:
                    continue

        if tables and not metrics:
            metrics = self._extract_from_tables(tables)

        return metrics

    def _parse_general(self, text_data: List[Dict]) -> Dict[str, Any]:
        """
        Fallback parser for unknown report types.

        Returns the raw OCR text lines as a list rather than a structured dict.
        This lets the API return something useful even for unrecognised report
        formats, while the frontend can display the raw text as a last resort.

        Args:
            text_data — OCR text fragments from extract_text().

        Returns:
            {"extracted_lines": [str], "line_count": int}
        """
        return {
            'extracted_lines': [item['text'] for item in text_data],
            'line_count':      len(text_data),
        }

    # ── Table fallback parser ─────────────────────────────────────────────────

    def _extract_from_tables(self, tables: List[Dict]) -> Dict[str, Any]:
        """
        Extract metric key-value pairs from PPStructureV3 table data.

        Called when regex text parsing returns no metrics.  Iterates over every
        row in every detected table and tries to interpret it as:
          row[0] — metric name (normalised to snake_case)
          row[1] — numeric value (float-cast)
          row[2] — unit (optional, falls back to "unknown")

        Rows where row[1] can't be cast to float are silently skipped.

        This is a best-effort parser — table cell content varies significantly
        across lab formats.  It may extract metadata rows (e.g. patient name)
        that look like metric rows; validate_value() in feature_engineering.py
        filters physiologically impossible values downstream.

        Args:
            tables — list of table dicts from extract_tables().

        Returns:
            Dict of {normalised_key: {"value": float, "unit": str, "source": "table"}}.
        """
        metrics = {}
        for table in tables:
            for row in table.get('data', []):
                if len(row) >= 2:
                    # Normalise the label: lowercase, spaces → underscores, drop dots.
                    name = str(row[0]).lower().replace(' ', '_').replace('.', '')
                    try:
                        value = float(row[1])
                        metrics[name] = {
                            'value':  value,
                            'unit':   row[2] if len(row) > 2 else 'unknown',
                            'source': 'table',
                        }
                    except (ValueError, IndexError):
                        continue
        return metrics

    # ── Utility helpers ───────────────────────────────────────────────────────

    def _infer_unit(self, metric_name: str) -> str:
        """
        Return the standard measurement unit for a known metric name.

        Used by the parsers to attach a unit string to each extracted value.
        The unit is looked up from a static dict rather than being parsed from
        the report text — OCR unit strings are unreliable (e.g. "K/uL" vs
        "×10³/µL" vs "thou/µL").

        Returns "unknown" for any metric not in the lookup dict.
        Downstream code (normalize_units in feature_engineering.py) handles
        the "unknown" unit by preserving the value as-is.

        Args:
            metric_name — one of the parser keys (e.g. "wbc", "tsh", "alt").

        Returns:
            Unit string (e.g. "g/dL", "mg/dL", "U/L") or "unknown".
        """
        units = {
            'wbc':                '10^3/uL',
            'rbc':                '10^6/uL',
            'hemoglobin':         'g/dL',
            'hematocrit':         '%',
            'platelets':          '10^3/uL',
            'glucose':            'mg/dL',
            'creatinine':         'mg/dL',
            'bun':                'mg/dL',
            'urea':               'mg/dL',
            'uric_acid':          'mg/dL',
            'egfr':               'mL/min/1.73m²',
            'tsh':                'mIU/L',
            't3':                 'pg/mL',
            't4':                 'ng/dL',
            'testosterone':       'ng/dL',
            'estradiol':          'pg/mL',
            'progesterone':       'ng/mL',
            'prolactin':          'ng/mL',
            'lh':                 'mIU/mL',
            'fsh':                'mIU/mL',
            'cortisol':           'µg/dL',
            'bilirubin_total':    'mg/dL',
            'bilirubin_direct':   'mg/dL',
            'bilirubin_indirect': 'mg/dL',
            'alt':                'U/L',
            'ast':                'U/L',
            'alp':                'U/L',
            'albumin':            'g/dL',
            'total_protein':      'g/dL',
        }
        return units.get(metric_name, 'unknown')

    def _get_center(self, bbox: List[List[float]]) -> tuple:
        """
        Compute the geometric centre of a bounding polygon.

        PaddleOCR 3.4 returns bounding polygons as a list of N [x, y] coordinate
        pairs (typically 4 corners for a rectangle, but may be more for skewed text).
        The centre is the mean of all x-coordinates and the mean of all y-coordinates.

        Used by extract_text() to sort detected text fragments into reading order
        (top-to-bottom, left-to-right) after OCR returns them in arbitrary order.

        Args:
            bbox — list of [x, y] coordinate pairs, e.g. [[10,5],[80,5],[80,20],[10,20]].

        Returns:
            (x_centre, y_centre) tuple of floats.
        """
        x = sum(p[0] for p in bbox) / len(bbox)
        y = sum(p[1] for p in bbox) / len(bbox)
        return (x, y)


# ── Module-level singleton accessor ──────────────────────────────────────────

# The module-level variable holds the singleton after first creation.
# Checking it here (before calling OCRRunner()) adds a second protection layer
# on top of OCRRunner.__new__ in case the class-level _instance is somehow reset.
_ocr_runner: Optional[OCRRunner] = None


def get_ocr_runner() -> OCRRunner:
    """
    Return the process-wide OCRRunner singleton, creating it on first call.

    Two protection layers prevent accidental double-initialisation:
      1. This function checks _ocr_runner before calling OCRRunner().
      2. OCRRunner.__new__ + _initialized guard inside __init__ prevent a second
         full init even if OCRRunner() is called directly.

    Called by the ML service's /analyze endpoint once per task.  On the first
    call the full PaddleOCR engine loads (~5–15 s on CPU).  All subsequent
    calls return the already-loaded instance immediately.

    Returns:
        The singleton OCRRunner instance, fully initialised and ready.
    """
    global _ocr_runner
    if _ocr_runner is None:
        _ocr_runner = OCRRunner()
    return _ocr_runner

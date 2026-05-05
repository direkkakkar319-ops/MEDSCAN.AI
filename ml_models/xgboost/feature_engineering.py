"""
ml_models/xgboost/feature_engineering.py — Feature vector construction for XGBoost.

This module bridges the raw OCR output (unstructured metric dicts with mixed units)
and the fixed-length numpy arrays that XGBoost expects.

Key responsibilities:
  1. Unit normalisation  — convert any metric to the standard unit XGBoost was
                           trained on (e.g. glucose mmol/L → mg/dL × 18.0182).
  2. Value validation    — reject physiologically impossible values (out of
                           plausible biological range) and replace with defaults.
  3. Coverage reporting  — log which expected features were actually extracted by
                           OCR and which will use default values.
  4. Feature vector      — produce a deterministic-length list of floats in the
                           exact order XGBoost was trained on.

Column order matters: XGBoost is not order-agnostic.  If a new feature is added,
the model must be retrained; inserting features mid-list silently corrupts existing
models.  Always append new features at the END of each FEATURE list.
"""

import logging
from typing import Any, Dict, List, Tuple, TypedDict

logger = logging.getLogger(__name__)


class MetricsCoverage(TypedDict):
    """
    Returned by validate_ocr_metrics — describes how much of the expected
    feature set was successfully extracted from the report image.

    Fields:
      found        — list of feature keys that OCR found with a valid value.
      missing      — list of feature keys that will use their default value.
      coverage_pct — percentage of expected features that were found (0–100).
                     Used to log a warning when coverage < 50%.
    """
    found: List[str]
    missing: List[str]
    coverage_pct: float


# ── Unit normalisation table ──────────────────────────────────────────────────
# Maps metric key → (standard_unit, {alternative_unit: multiplier_to_standard}).
#
# The XGBoost models were trained on data in the "standard" unit column.
# When OCR extracts a value in an alternative unit, multiplying by the
# corresponding factor converts it to the training-time scale.
#
# Example: glucose 5.5 mmol/L × 18.0182 = 99.1 mg/dL
#
# Metrics not listed here are passed through unchanged because they are either
# always reported in a single unit (e.g. WBC in ×10³/µL) or the training data
# was unit-agnostic.
UNIT_NORMALIZATION = {
    "glucose":      ("mg/dL",  {"mmol/L": 18.0182}),
    "bun":          ("mg/dL",  {"mmol/L": 2.8}),
    "creatinine":   ("mg/dL",  {"µmol/L": 0.0113}),

    "vitamin_d":    ("ng/mL",  {"nmol/L": 2.496}),
    "testosterone": ("ng/dL",  {"nmol/L": 28.8}),

    "total_cholesterol": ("mg/dL", {"mmol/L": 38.67}),
    "ldl":               ("mg/dL", {"mmol/L": 38.67}),
    "hdl":               ("mg/dL", {"mmol/L": 38.67}),
    "triglycerides":     ("mg/dL", {"mmol/L": 88.57}),
}


# ── Feature definitions ───────────────────────────────────────────────────────
# Each list defines the features for one report type.  Tuple format:
#   (feature_name, metrics_key, default_value)
#
#   feature_name  — name used in the output feature vector and SHAP labels.
#   metrics_key   — key in the structured_metrics dict from OCR.
#   default_value — clinically normal reference value used when OCR missed
#                   the metric.  Using a normal-range centre as the default
#                   is better than 0.0 because 0 WBC / 0 hemoglobin would
#                   push models toward extreme disease predictions.
#
# COLUMN ORDER IS FIXED — do not reorder without retraining the models.

BLOOD_FEATURES: List[Tuple[str, str, float]] = [
    ("wbc",        "wbc",        7.0),    # White Blood Cell count (×10³/µL)
    ("rbc",        "rbc",        4.9),    # Red Blood Cell count (×10⁶/µL)
    ("hemoglobin", "hemoglobin", 14.0),   # g/dL
    ("hematocrit", "hematocrit", 43.0),   # %
    ("platelets",  "platelets",  250.0),  # ×10³/µL
    ("glucose",    "glucose",    90.0),   # mg/dL (fasting normal ~90)
    ("creatinine", "creatinine", 1.0),    # mg/dL
    ("bun",        "bun",        15.0),   # mg/dL
]

LIPID_FEATURES: List[Tuple[str, str, float]] = [
    ("total_cholesterol", "total_cholesterol", 180.0),  # mg/dL
    ("hdl",               "hdl",               55.0),   # mg/dL (higher is better)
    ("ldl",               "ldl",               100.0),  # mg/dL (lower is better)
    ("triglycerides",     "triglycerides",     130.0),  # mg/dL
    ("vldl",              "vldl",               26.0),   # mg/dL (VLDL = TG / 5)
]

VITAMIND_FEATURES: List[Tuple[str, str, float]] = [
    ("vitamin_d", "vitamin_d", 30.0),    # ng/mL — low end of normal (20–50 range)
]

HORMONE_FEATURES: List[Tuple[str, str, float]] = [
    ("tsh",          "tsh",          2.5),    # mIU/L
    ("t3",           "t3",           1.2),    # pg/mL (Free T3)
    ("t4",           "t4",           8.0),    # ng/dL (Free T4)
    ("testosterone", "testosterone", 500.0),  # ng/dL (mid-range adult male)
    ("estradiol",    "estradiol",    25.0),   # pg/mL
    ("progesterone", "progesterone", 1.0),    # ng/mL
    ("prolactin",    "prolactin",    10.0),   # ng/mL
    ("lh",           "lh",           5.0),    # mIU/mL
    ("fsh",          "fsh",          5.0),    # mIU/mL
    ("cortisol",     "cortisol",     10.0),   # µg/dL (morning reference)
]

KIDNEY_FEATURES: List[Tuple[str, str, float]] = [
    ("creatinine", "creatinine", 1.0),    # mg/dL
    ("bun",        "bun",        15.0),   # mg/dL
    ("urea",       "urea",       25.0),   # mg/dL (urea ≈ BUN × 2.14)
    ("uric_acid",  "uric_acid",  5.0),    # mg/dL
    ("egfr",       "egfr",       90.0),   # mL/min/1.73m² (normal ≥ 60)
]

LIVER_FEATURES: List[Tuple[str, str, float]] = [
    ("alt",              "alt",              25.0),   # U/L (SGPT)
    ("ast",              "ast",              25.0),   # U/L (SGOT)
    ("alp",              "alp",              100.0),  # U/L
    ("bilirubin_total",  "bilirubin_total",  1.0),    # mg/dL
    ("bilirubin_direct", "bilirubin_direct", 0.3),    # mg/dL
    ("albumin",          "albumin",          4.5),    # g/dL
    ("total_protein",    "total_protein",    7.0),    # g/dL
]

GENERAL_FEATURES: List[Tuple[str, str, float]] = []   # no-op for unknown types

# FEATURE_MAP maps report_type → the correct feature list.
FEATURE_MAP: Dict[str, List[Tuple[str, str, float]]] = {
    "blood":     BLOOD_FEATURES,
    "lipid":     LIPID_FEATURES,
    "vitamin_d": VITAMIND_FEATURES,
    "hormone":   HORMONE_FEATURES,
    "kidney":    KIDNEY_FEATURES,
    "liver":     LIVER_FEATURES,
    "general":   GENERAL_FEATURES,
}


# ── Helper functions ──────────────────────────────────────────────────────────

def infer_unit(key: str, value: float) -> str:
    """
    Guess the measurement unit from the numeric value when the unit is absent.

    OCR sometimes strips unit labels, leaving bare numbers.  For a small set
    of metrics where the two common units have non-overlapping value ranges,
    we can reliably infer which unit was used:

      glucose:
        mmol/L range: 1–20 (typical 3–10)
        mg/dL  range: 18–360 (typical 54–180)
        Threshold: < 20 → mmol/L, else mg/dL

      vitamin_d:
        ng/mL  range: 5–150
        nmol/L range: 12–375 (ng/mL × 2.496)
        Threshold: > 200 → nmol/L (extremely unlikely in ng/mL)

    For all other metrics "unknown" is returned and the value is used as-is.
    This is safe because normalize_units() keeps the value if the unit matches
    the standard, and falls back to the raw value when the unit is unknown.

    Args:
        key   — metric name (e.g. "glucose", "vitamin_d").
        value — numeric value extracted by OCR (unit uncertain).

    Returns:
        Inferred unit string, or "unknown" if inference is not possible.
    """
    if key == "glucose":
        return "mmol/L" if value < 20 else "mg/dL"

    if key == "vitamin_d":
        return "nmol/L" if value > 200 else "ng/mL"

    return "unknown"


def validate_value(key: str, value: float) -> float:
    """
    Reject physiologically impossible values and replace with the metric's default.

    OCR occasionally misreads a digit, producing values like glucose=9000 or
    hemoglobin=0.5.  These are outside any plausible human range and would push
    XGBoost toward extreme probability outputs.  Replacing them with the default
    (a clinically normal reference value) is safer than propagating the error.

    Current validation rules:
      glucose    — valid range 20–600 mg/dL.  Below 20 is below the lowest
                   recorded hypoglycaemia; above 600 is diabetic ketoacidosis range.
      hemoglobin — valid range 3–25 g/dL.  Below 3 indicates severe anaemia;
                   above 25 is physiologically impossible.
      creatinine — valid range 0.1–15 mg/dL.  Above 15 indicates end-stage renal
                   failure even in dialysis patients.

    Args:
        key   — metric name used to select the validation rule.
        value — the numeric value (already unit-converted) to validate.

    Returns:
        The original value if it passes validation, or the hardcoded default
        if it is out of range.
    """
    if key == "glucose" and not (20 <= value <= 600):
        return 90.0

    if key == "hemoglobin" and not (3 <= value <= 25):
        return 14.0

    if key == "creatinine" and not (0.1 <= value <= 15):
        return 1.0

    return value


# ── Public functions ──────────────────────────────────────────────────────────

def normalize_units(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert all metrics in the dict to their standard training-time units.

    Iterates over every metric in the dict and, for metrics listed in
    UNIT_NORMALIZATION, applies the conversion factor if the unit differs from
    the standard.  Metrics not in UNIT_NORMALIZATION are passed through unchanged.

    Input format — each value is either:
      dict:  {"value": float, "unit": str, "source": str}  (from OCR parser)
      float: raw numeric value (from older pipeline versions)

    Output format — same structure but with value converted and unit set to
    the standard unit string.

    Special cases:
      - value is None: skip the metric entirely (don't include it in output).
      - unit is None: call infer_unit() to guess from the value magnitude.
      - unit is unknown / not in conversions: keep value as-is with standard
        unit label (better than crashing on an unexpected unit string).
      - raw float input: wrap in {"value": float, "unit": "unknown"} for uniformity.

    Args:
        metrics — structured_metrics dict from OCR, possibly with mixed units.

    Returns:
        New dict with the same keys but values guaranteed to be in the training
        units.  The original dict is NOT mutated.
    """
    normalized = {}

    for key, entry in metrics.items():

        if isinstance(entry, dict):
            value = entry.get("value")
            unit  = entry.get("unit")

            if value is None:
                continue   # metric was in the dict but had no numeric value

            if key in UNIT_NORMALIZATION:
                std_unit, conversions = UNIT_NORMALIZATION[key]

                # Try to guess unit from value magnitude when the label is absent.
                if unit is None:
                    unit = infer_unit(key, float(value))

                if unit == std_unit:
                    # Already in the correct unit — just cast to float.
                    normalized[key] = {"value": float(value), "unit": std_unit}

                elif unit in conversions:
                    # Convert: multiply by the factor (e.g. mmol/L × 18.0182).
                    normalized[key] = {
                        "value": float(value) * conversions[unit],
                        "unit":  std_unit,
                    }

                else:
                    # Unknown unit (e.g. an unusual abbreviation the lab uses).
                    # Keep the raw value rather than silently discarding it.
                    normalized[key] = {"value": float(value), "unit": std_unit}

            else:
                # Metric is not in UNIT_NORMALIZATION — no conversion needed.
                normalized[key] = {"value": float(value), "unit": unit}

        else:
            # Raw float (no unit metadata) — wrap in dict for uniform downstream
            # handling by build_feature_vector.
            normalized[key] = {"value": float(entry), "unit": "unknown"}

    return normalized


def validate_ocr_metrics(
    metrics: Dict[str, Any],
    report_type: str,
) -> Tuple[Dict[str, Any], MetricsCoverage]:
    """
    Check which expected features were actually extracted and log coverage quality.

    After PaddleOCR runs, not every expected lab metric may be present —
    low image quality, unusual formatting, or a truncated report can all cause
    missed extractions.  This function:
      - Identifies which of the expected features for the report_type were found.
      - Identifies which are missing and will fall back to their default values.
      - Logs a summary at INFO level for every prediction (visibility into quality).
      - Logs an additional WARNING when coverage < 50% (prediction reliability reduced).

    The function does NOT mutate the metrics dict and does NOT fill in defaults —
    that is done by build_feature_vector().  Here we only measure and report.

    Args:
        metrics     — structured_metrics dict from OCR:
                      {key: {"value": float, "unit": str, "source": str}}
        report_type — one of blood / lipid / vitamin_d / hormone / kidney / liver.
                      Used to look up the expected feature list in FEATURE_MAP.

    Returns:
        (metrics, coverage) where:
          metrics  — the original dict, unchanged.
          coverage — MetricsCoverage TypedDict:
                       found        → list of keys with valid numeric values
                       missing      → list of keys that will use defaults
                       coverage_pct → float, percentage of features found
    """
    feature_defs  = FEATURE_MAP.get(report_type, GENERAL_FEATURES)
    expected_keys = [feat_name for feat_name, _, _ in feature_defs]

    found:   List[str] = []
    missing: List[str] = []

    for key in expected_keys:
        entry = metrics.get(key)
        if entry is not None:
            if isinstance(entry, dict):
                # Dict format: check that "value" is a non-None numeric.
                val = entry.get("value")
                if val is not None:
                    found.append(key)
                    continue
            else:
                # Raw float format: try to cast.
                try:
                    float(entry)
                    found.append(key)
                    continue
                except (TypeError, ValueError):
                    pass
        missing.append(key)

    total        = len(expected_keys)
    coverage_pct = round((len(found) / total * 100) if total > 0 else 0.0, 1)

    coverage: MetricsCoverage = {
        "found":        found,
        "missing":      missing,
        "coverage_pct": coverage_pct,
    }

    logger.info(
        f"[{report_type}] OCR metric coverage: {len(found)}/{total} "
        f"({coverage_pct}%)  |  "
        f"found={found}  |  "
        f"missing (will use defaults)={missing}"
    )

    if coverage_pct < 50.0:
        logger.warning(
            f"[{report_type}] Low OCR coverage ({coverage_pct}%). "
            "Prediction reliability may be reduced. "
            "Check image quality or report format."
        )

    return metrics, coverage


def build_feature_vector(
    metrics: Dict[str, Any],
    report_type: str,
) -> Tuple[List[float], List[str]]:
    """
    Build the fixed-length numeric feature vector for XGBoost inference.

    This is the final step before calling model.predict_proba().  It:
      1. Normalises units in the metrics dict (calls normalize_units()).
      2. For each expected feature (in training order):
           - Reads the normalised value if present.
           - Falls back to the feature's hardcoded default if absent.
           - Runs validate_value() to replace physiologically impossible numbers.
      3. Returns (feature_vector, feature_names) where both lists are in the same
         order as FEATURE_MAP[report_type].

    Why defaults instead of NaN / imputation?
      XGBoost can handle missing values internally, but only if trained with
      missing values.  Our models were trained with filled defaults, so using the
      same defaults at inference time is consistent with training.

    Args:
        metrics     — structured_metrics dict (may have mixed units, handled inside).
        report_type — selects the correct FEATURE_MAP entry.

    Returns:
        feature_vector — list of floats, length = len(FEATURE_MAP[report_type]).
        feature_names  — list of feature name strings in the same order.

    Example for report_type="blood":
        feature_vector = [7.2, 4.8, 13.5, 41.0, 245.0, 92.0, 0.9, 14.0]
        feature_names  = ["wbc","rbc","hemoglobin","hematocrit","platelets",
                          "glucose","creatinine","bun"]
    """
    feature_defs       = FEATURE_MAP.get(report_type, GENERAL_FEATURES)
    normalized_metrics = normalize_units(metrics)

    feature_vector: List[float] = []
    feature_names:  List[str]   = []

    for feat_name, metrics_key, default in feature_defs:

        entry = normalized_metrics.get(metrics_key)

        if isinstance(entry, dict):
            # Dict format from normalize_units() — read .value, fall back to default
            # if .value is somehow None (e.g. the key existed but had null value).
            value = float(entry.get("value", default))
        elif entry is not None:
            # Raw float that survived normalize_units() as-is.
            value = float(entry)
        else:
            # Feature completely absent from OCR output — use the normal-range default.
            value = default

        # Final sanity check: reject physiologically impossible values.
        value = validate_value(metrics_key, value)

        feature_vector.append(value)
        feature_names.append(feat_name)

    return feature_vector, feature_names

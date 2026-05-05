"""
ml_models/model_utils.py — Model registry and file path helpers.

Central place that maps report types → disease models and resolves the
filesystem paths where .pkl files live.

Why a separate registry file?
  predict.py, the Celery tasks, and any future training scripts all need
  to know which models exist for each report type.  Keeping the mapping
  here avoids duplication and makes adding a new report type a one-file
  change (add an entry to REPORT_MODEL_MAP, drop the new .pkl in xgboost/).
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple

# ── Risk level thresholds ────────────────────────────────────────────────────
# These float values are the boundaries between risk categories.
# Applied to max(final_probas) — the highest disease probability across all
# diseases for the given report type.
#
# Calibration note: these are conservative thresholds.  A value of 0.40 for
# "moderate" means a 40% model-estimated probability of having the disease.
# They were chosen to match clinical screening guidelines where a low FPR
# matters more than a low FNR.
RISK_THRESHOLDS = {
    "critical": 0.85,   # ≥ 85% probability → seek immediate medical attention
    "high":     0.65,   # ≥ 65% probability → schedule doctor visit soon
    "moderate": 0.40    # ≥ 40% probability → follow up within 3 months
    # < 0.40           → "low" — maintain current lifestyle
}

# ── File system layout ────────────────────────────────────────────────────────
# _BASE_DIR points to the ml_models/ directory (where this file lives).
# _XGBOOST_DIR points to ml_models/xgboost/ where all .pkl files are stored.
_BASE_DIR    = Path(__file__).parent
_XGBOOST_DIR = _BASE_DIR / "xgboost"

# Valid report types — used for input validation in model_utils functions.
ALLOWED_REPORT_TYPE = {"blood", "lipid", "vitamin_d", "hormone", "kidney", "liver"}

# ── Model registry ────────────────────────────────────────────────────────────
# Maps each report type to its list of individual disease models.
#
# Each entry is a tuple: (disease_label, pkl_filename_without_extension)
#   disease_label   — the key used in the output "risks" dict
#                     e.g. {"diabetes": 0.12, "anemia": 0.05, ...}
#   pkl_filename    — the .pkl file name (without extension) in _XGBOOST_DIR
#                     e.g. "diabetes" → ml_models/xgboost/diabetes.pkl
#
# ORDER MATTERS:  The position of each entry determines its index in the
# neural ensemble input vector.  Do not reorder without retraining the ensemble.
REPORT_MODEL_MAP: Dict[str, List[Tuple[str, str]]] = {
    "blood": [
        ("diabetes",   "diabetes"),    # XGBoost model trained on CBC + metabolic panel
        ("anemia",     "anemia"),      # trained on Hgb, Hct, RBC, MCV
        ("infection",  "infection"),   # trained on WBC, neutrophils, CRP proxy
    ],
    "lipid": [
        ("heart_disease", "heart"),    # trained on total cholesterol, LDL, HDL, TG
        ("stroke",        "stroke"),   # trained on lipid panel + BP proxies
    ],
    "vitamin_d": [
        ("vitamin_d_deficiency", "vitamin_d"),   # single-feature model
    ],
    "hormone": [
        ("testosterone_imbalance", "testosterone"),  # trained on sex hormones
        ("thyroid_disorder",       "thyroid"),        # trained on TSH, T3, T4
        ("hormonal_imbalance",     "hormone"),        # broad panel model
    ],
    "liver": [
        ("liver_disease", "liver"),       # trained on ALT, AST, ALP, bilirubin, albumin
        ("fatty_liver",   "fatty_liver"), # trained on liver enzymes + metabolic markers
        ("hepatitis",     "hepatitis"),   # trained on bilirubin, albumin, AST/ALT ratio
    ],
    "kidney": [
        ("kidney_disease", "kidney"),    # trained on creatinine, BUN, eGFR, urea
        ("renal_failure",  "renal"),     # trained on creatinine, eGFR, uric acid
    ],
}


def get_disease_model_path(pkl_name: str) -> str:
    """
    Resolve the absolute filesystem path to an individual disease model .pkl.

    Args:
        pkl_name : The short pkl name from REPORT_MODEL_MAP, e.g. "diabetes".
                   The function appends ".pkl" and resolves the full path.

    Returns:
        Absolute path string, e.g. "/app/ml_models/xgboost/diabetes.pkl"

    Raises:
        FileNotFoundError — if the .pkl file doesn't exist at the expected path.
                            This happens when a model hasn't been trained yet or
                            the pkl_name is misspelled.  The caller (predict.py)
                            catches this and inserts a None placeholder so the
                            pipeline continues with 0.0 probability for that disease.
    """
    path = _XGBOOST_DIR / f"{pkl_name}.pkl"
    if not path.exists():
        raise FileNotFoundError(
            f"Disease model not found: {path}\n"
            "Train the model or verify the filename."
        )
    return str(path)


def get_ensemble_path(report_type: str) -> str:
    """
    Resolve the path where the neural ensemble weights for a report type
    would be stored.

    The file may not exist yet — the ensemble is optional.  If it doesn't
    exist, NeuralEnsemble falls back to an identity passthrough (raw XGBoost
    probabilities are returned unchanged).

    File naming convention:
        ml_models/xgboost/ensemble_{report_type}.pkl
        e.g.  ensemble_blood.pkl, ensemble_lipid.pkl, …

    The caller (load_ensemble in neural_ensemble.py) checks os.path.exists()
    before loading.

    Args:
        report_type : e.g. "blood", "lipid", "vitamin_d"

    Returns:
        Absolute path string — even if the file doesn't exist yet.
    """
    return str(_XGBOOST_DIR / f"ensemble_{report_type}.pkl")


def get_model_path(report_type: str) -> str:
    """
    Legacy helper kept for backwards compatibility with older code.

    Before the multi-model architecture, a single model_{report_type}.pkl
    file was used per report type.  This function resolves that old path.

    Resolution order:
      1. XGBOOST_MODEL_PATH env var (override for testing/deployment)
      2. ml_models/xgboost/model_{report_type}.pkl
      3. ml_models/xgboost/model.pkl (generic fallback)

    Args:
        report_type : Must be in ALLOWED_REPORT_TYPE.

    Returns:
        Absolute path string to the legacy combined model file.

    Raises:
        ValueError       — report_type not in ALLOWED_REPORT_TYPE.
        FileNotFoundError — none of the fallback paths exist.
    """
    if report_type not in ALLOWED_REPORT_TYPE:
        raise ValueError(f"Invalid report type: {report_type}")

    # Allow the deployment to override the model path via env var.
    env_path = os.getenv("XGBOOST_MODEL_PATH")
    if env_path and os.path.exists(env_path):
        return env_path

    # Try the type-specific legacy model file.
    typed = _XGBOOST_DIR / f"model_{report_type}.pkl"
    if typed.exists():
        return str(typed)

    # Final fallback — a generic model.pkl that covers all types.
    fallback = _XGBOOST_DIR / "model.pkl"
    if not fallback.exists():
        raise FileNotFoundError("No model file found for the given report-type")
    return str(fallback)

"""
ml_models/predict.py — Main prediction orchestrator.

This module is the single entry point for running disease risk inference.
RiskPredictor.predict() ties together every part of the ML pipeline:

  OCR metrics (dict)
      ↓  validate_ocr_metrics()   — checks which features were found vs. missing
      ↓  build_feature_vector()   — builds a numpy array, filling gaps with defaults
      ↓  _load_all_models()       — loads all XGBoost .pkl files for the report type
      ↓  _run_individual_models() — runs each XGBoost model → raw disease probabilities
      ↓  _apply_ensemble()        — passes through NeuralEnsemble meta-learner
      ↓  _compute_shap()          — explains the highest-risk disease with SHAP
      ↓  _score_to_level()        — converts max probability to risk_level string
      ↓  _recommendations()       — returns actionable advice based on risk_level
      ↓
  Full prediction dict

Module-level caching:
  _MODEL_CACHE stores loaded XGBoost objects by pkl_name.  Within a single Celery
  worker process this means a model is read from disk only once — all subsequent
  tasks reuse the in-memory object.  The cache is NOT shared across worker processes
  (each process has its own memory space).
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np

from ml_models.model_utils import (
    REPORT_MODEL_MAP,
    RISK_THRESHOLDS,
    get_disease_model_path,
    get_ensemble_path,
)
from ml_models.neural_ensemble import load_ensemble
from ml_models.xgboost.feature_engineering import build_feature_vector, validate_ocr_metrics

logger = logging.getLogger(__name__)

# Process-level model cache: {pkl_name: loaded_model_object}
# Example: {"diabetes": <XGBClassifier>, "anemia": <XGBClassifier>, ...}
# Populated lazily on first predict() call; reused for all subsequent calls.
_MODEL_CACHE: Dict[str, Any] = {}


# ── Model loading ─────────────────────────────────────────────────────────────

def _load_disease_model(pkl_name: str) -> Any:
    """
    Load and cache a single XGBoost disease model from its .pkl file.

    On first call the model is read from disk with joblib.load() and stored in
    _MODEL_CACHE.  On subsequent calls the cached object is returned immediately
    without touching the filesystem.  This is important because PaddleOCR +
    XGBoost together use significant RAM; avoiding repeated disk reads keeps
    task throughput consistent.

    Args:
        pkl_name — short name used as the cache key and for path resolution,
                   e.g. "diabetes" resolves to ml_models/xgboost/diabetes.pkl.

    Returns:
        The loaded model object (XGBClassifier or any sklearn-compatible model).

    Raises:
        FileNotFoundError — if get_disease_model_path() cannot find the .pkl.
                            The caller (_load_all_models) catches this and inserts
                            a None placeholder so prediction continues with 0.0.
    """
    if pkl_name in _MODEL_CACHE:
        return _MODEL_CACHE[pkl_name]

    path = get_disease_model_path(pkl_name)
    model = joblib.load(path)
    _MODEL_CACHE[pkl_name] = model
    logger.info(f"Loaded disease model '{pkl_name}' from {path}")
    return model


def _load_all_models(report_type: str) -> List[Tuple[str, str, Any]]:
    """
    Load every disease model registered for the given report type.

    Iterates over REPORT_MODEL_MAP[report_type] and calls _load_disease_model()
    for each entry.  If a model file is missing (FileNotFoundError), a None
    placeholder is inserted instead of crashing — the downstream code checks for
    None and uses a 0.0 probability for that disease.

    Why allow missing models?
      During development not all 14 models may be trained yet.  The system should
      still return predictions for the models that DO exist rather than failing
      entirely.  Missing models log an error for visibility.

    Args:
        report_type — must be a key in REPORT_MODEL_MAP (blood, lipid, etc.).

    Returns:
        List of (disease_label, pkl_name, model_or_None) tuples in the order
        defined by REPORT_MODEL_MAP.  ORDER MATTERS — it determines the index
        positions passed to the NeuralEnsemble.

    Raises:
        ValueError — if report_type is not in REPORT_MODEL_MAP.
    """
    entries = REPORT_MODEL_MAP.get(report_type, [])
    if not entries:
        raise ValueError(f"Unknown report type: '{report_type}'")

    loaded = []
    for disease_label, pkl_name in entries:
        try:
            model = _load_disease_model(pkl_name)
            loaded.append((disease_label, pkl_name, model))
        except FileNotFoundError as exc:
            logger.error(exc)
            # None placeholder keeps indices aligned for the ensemble input vector.
            loaded.append((disease_label, pkl_name, None))

    return loaded


# ── Inference helpers ─────────────────────────────────────────────────────────

def _disease_probability(model: Any, X: np.ndarray) -> Tuple[float, int]:
    """
    Extract a single disease risk probability from a multi-class XGBoost model.

    All 14 disease models use severity-level class encoding:
        Class 0 — healthy / no disease
        Class 1 — mild disease
        Class 2 — moderate disease
        Class 3 — severe disease  (only in 4-class models)

    "Disease risk" is defined as:
        risk = 1 - P(class 0) = P(any disease)

    This formulation correctly handles both 3-class and 4-class models because
    it only looks at P(healthy) and inverts it.  It gives the probability that
    the patient has the disease at ANY severity level.

    The "severity_class" is the argmax of the full probability vector — the
    most likely single class.  It is logged for debugging and passed to SHAP
    so explanations are computed for the predicted class, not always class 1.

    Args:
        model — loaded XGBClassifier with predict_proba() support.
        X     — numpy array of shape (1, n_features) — single patient's features.

    Returns:
        (disease_risk, severity_class)
          disease_risk   — float in [0, 1]
          severity_class — int: 0=none, 1=mild, 2=moderate, 3=severe
    """
    all_probas = model.predict_proba(X)[0]       # shape: (n_classes,)
    p_healthy  = float(all_probas[0])            # P(class 0 = no disease)
    disease_risk = round(1.0 - p_healthy, 6)

    # argmax gives the most probable single class — used for SHAP explanation.
    severity_class = int(np.argmax(all_probas))

    return disease_risk, severity_class


def _run_individual_models(
    loaded_models: List[Tuple[str, str, Any]],
    X: np.ndarray,
) -> Tuple[List[str], np.ndarray, List[int]]:
    """
    Run every individual disease model on the shared feature vector X.

    Iterates over all (disease_label, pkl_name, model) tuples for the report
    type.  For each model:
      - If model is None (missing .pkl): appends 0.0 risk and severity 0.
      - If model has predict_proba(): uses _disease_probability() to get 1-P(class0).
      - If model only has predict() (unusual): treats the output as a binary score.
      - On inference exception: logs a warning and uses 0.0 so the pipeline
        continues for the remaining models.

    Args:
        loaded_models — list from _load_all_models(), tuples of (label, name, model).
        X             — numpy array shape (1, n_features).

    Returns:
        disease_labels   — list of str, same order as loaded_models.
        raw_probas       — 1-D float32 array, one probability per disease.
        severity_classes — list of int, predicted severity class per disease.
    """
    disease_labels:   List[str]   = []
    raw_probas:       List[float] = []
    severity_classes: List[int]   = []

    for disease_label, pkl_name, model in loaded_models:
        disease_labels.append(disease_label)

        if model is None:
            # Missing model — use zero so this disease doesn't influence risk_level.
            raw_probas.append(0.0)
            severity_classes.append(0)
            continue

        try:
            if hasattr(model, "predict_proba"):
                proba, sev = _disease_probability(model, X)
            else:
                # Legacy binary regressor path (shouldn't occur with current models).
                proba = float(model.predict(X)[0])
                sev   = 1 if proba >= 0.5 else 0
        except Exception as exc:
            logger.warning(f"Inference failed for '{pkl_name}': {exc}")
            proba, sev = 0.0, 0

        raw_probas.append(proba)
        severity_classes.append(sev)
        logger.debug(f"  {disease_label}: risk={proba:.4f}  severity_class={sev}")

    return disease_labels, np.array(raw_probas, dtype=np.float32), severity_classes


def _apply_ensemble(
    report_type: str,
    raw_probas: np.ndarray,
) -> np.ndarray:
    """
    Pass raw XGBoost probabilities through the NeuralEnsemble meta-learner.

    The ensemble is a small 3-layer MLP trained to calibrate the raw XGBoost
    outputs — it learns systematic biases across diseases for the same report
    type (e.g. if the blood models tend to over-predict diabetes, the ensemble
    corrects for that).

    If no trained ensemble weights exist (ensemble_{report_type}.pkl not found),
    load_ensemble() returns an untrained NeuralEnsemble whose predict() method
    is an identity pass-through — the raw probabilities are returned unchanged.
    This means the system works correctly even before the ensemble is trained.

    Args:
        report_type — used to find the correct ensemble .pkl file.
        raw_probas  — 1-D array of shape (n_diseases,) from _run_individual_models.

    Returns:
        np.ndarray of shape (n_diseases,) — calibrated probabilities.
    """
    ensemble_path = get_ensemble_path(report_type)
    n_diseases    = len(raw_probas)
    ensemble      = load_ensemble(report_type, ensemble_path, n_diseases)
    return ensemble.predict(raw_probas)


# ── SHAP & recommendations ────────────────────────────────────────────────────

def _compute_shap(
    model: Any,
    X: np.ndarray,
    feature_names: List[str],
    severity_class: int = 1,
) -> Optional[Dict]:
    """
    Compute SHAP feature importance for a single XGBoost model prediction.

    SHAP (SHapley Additive exPlanations) assigns each feature a "contribution
    score" for the predicted output.  A positive SHAP value means the feature
    pushed the model toward a higher disease risk; negative means it reduced risk.

    For multi-class XGBoost, shap.TreeExplainer returns one SHAP array per class:
      sv = [array_class0, array_class1, array_class2, ...]
    We explain the severity_class that the model predicted.  If the model predicted
    class 0 (healthy), we fall back to class 1 (mild) so the explanation always
    shows disease-relevant drivers rather than "why is the patient healthy."

    This function is called ONLY on the highest-risk disease model so the SHAP
    values shown to the user are relevant to their biggest concern.

    Args:
        model          — trained XGBClassifier.
        X              — numpy array shape (1, n_features).
        feature_names  — list of feature name strings matching X's columns.
        severity_class — predicted class from _disease_probability(), used to
                         select which class to explain.

    Returns:
        Dict mapping feature name → SHAP value (float), or None if SHAP fails.
        None is handled gracefully by _top_factors() which returns [].
    """
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X)   # list[(1, n_feat)] or (1, n_feat) for binary

        if isinstance(sv, list):
            # Multi-class: pick the class to explain.
            explain_class = severity_class if severity_class > 0 else 1
            # Guard: clamp to valid class index in case model has fewer classes.
            explain_class = min(explain_class, len(sv) - 1)
            values = sv[explain_class][0]
        else:
            # Binary model (unusual but handled).
            values = sv[0]

        return {name: round(float(v), 6) for name, v in zip(feature_names, values)}
    except Exception as exc:
        logger.debug(f"SHAP skipped: {exc}")
        return None


def _top_factors(
    shap_values: Optional[Dict],
    feature_names: List[str],
    n: int = 5,
) -> List[Dict]:
    """
    Return the top N features by absolute SHAP magnitude.

    These become the "key_factors" displayed in the Results section of the UI
    under "What's driving your risk?"

    Factors are sorted by |SHAP value| descending so the most influential feature
    appears first regardless of direction (a large negative SHAP that strongly
    reduces risk is equally notable as a large positive one).

    Args:
        shap_values   — dict from _compute_shap(), or None.
        feature_names — ordered list of feature names (not used here since
                        shap_values keys already contain names; kept for signature
                        consistency with earlier versions).
        n             — number of top factors to return (default 5).

    Returns:
        List of dicts: [{"feature": str, "impact": float, "direction": str}, ...]
        Returns [] if shap_values is None.
    """
    if not shap_values:
        return []
    top = sorted(shap_values.items(), key=lambda kv: abs(kv[1]), reverse=True)[:n]
    return [
        {
            "feature":   k,
            "impact":    v,
            "direction": "increases" if v > 0 else "decreases",
        }
        for k, v in top
    ]


def _score_to_level(score: float) -> str:
    """
    Convert a max disease risk probability (0–1) to a risk level string.

    Thresholds are defined in RISK_THRESHOLDS (model_utils.py) and are applied
    to the MAXIMUM risk across all diseases for the report type.  Using the max
    ensures the overall risk level reflects the single most concerning disease.

    Threshold logic (waterfall — first match wins):
      score ≥ 0.85 → "critical"   seek immediate medical attention
      score ≥ 0.65 → "high"       schedule a doctor visit soon
      score ≥ 0.40 → "moderate"   follow up within 3 months
      otherwise    → "low"        maintain current lifestyle

    Args:
        score — float in [0, 1], the highest disease probability after ensemble.

    Returns:
        One of: "critical", "high", "moderate", "low".
    """
    if score >= RISK_THRESHOLDS["critical"]:  return "critical"
    if score >= RISK_THRESHOLDS["high"]:      return "high"
    if score >= RISK_THRESHOLDS["moderate"]:  return "moderate"
    return "low"


def _recommendations(risk_level: str, risks: Dict[str, float]) -> List[str]:
    """
    Build a list of actionable health recommendations.

    Combines two sources:
      1. Base recommendations keyed by risk_level — general advice for everyone
         at that level (e.g. "Consult your doctor within 3 months" for moderate).
      2. Disease-specific recommendations — appended when a specific disease
         probability exceeds 0.60, giving targeted advice for the patient's
         particular pattern (e.g. cardiovascular diet if heart_disease > 0.6).

    The 0.60 threshold for disease-specific messages is deliberately lower than
    the "high" risk threshold (0.65) so the message appears while there is still
    time to act rather than only after the risk is already rated "high".

    Args:
        risk_level — string from _score_to_level().
        risks      — {disease: float} dict of calibrated probabilities.

    Returns:
        List of recommendation strings, base first then disease-specific.
    """
    base = {
        "low":      ["Maintain current lifestyle. Annual check-up recommended."],
        "moderate": [
            "Consult your doctor within 3 months.",
            "Consider dietary adjustments.",
            "Increase physical activity.",
        ],
        "high": [
            "Schedule a doctor appointment soon.",
            "Review medication and diet with your physician.",
            "Monitor key metrics more frequently.",
        ],
        "critical": [
            "Seek immediate medical attention.",
            "Do not delay consulting a healthcare professional.",
        ],
        "unknown": ["Please consult a healthcare professional for interpretation."],
    }

    recs = list(base.get(risk_level, base["unknown"]))

    # Append disease-specific advice when the individual disease risk is notable.
    if risks.get("heart_disease", 0) > 0.6:
        recs.append("High cardiovascular risk — lipid management advised.")
    if risks.get("diabetes", 0) > 0.6:
        recs.append("Elevated diabetes risk — fasting glucose test recommended.")
    if risks.get("renal_failure", 0) > 0.6:
        recs.append("Elevated renal risk — nephrology consultation advised.")
    if risks.get("hepatitis", 0) > 0.6:
        recs.append("Hepatitis marker elevated — further liver panel recommended.")
    if risks.get("thyroid_disorder", 0) > 0.6:
        recs.append("Thyroid irregularity detected — endocrinology referral advised.")

    return recs


# ── Main prediction class ─────────────────────────────────────────────────────

class RiskPredictor:
    """
    High-level orchestrator for the full disease risk prediction pipeline.

    Instantiated once per Celery task call (but models are cached globally in
    _MODEL_CACHE so the cost is just the object allocation).

    For each predict() call:
      1. Load all XGBoost disease models for the report type.
      2. Validate OCR metric coverage and build the feature vector.
      3. Run each model independently → raw probabilities (1 - P(class 0)).
      4. Pass raw probabilities through the NeuralEnsemble meta-learner.
      5. Compute SHAP values on the highest-risk disease model.
      6. Return the full prediction dict.
    """

    def __init__(self):
        # Models are loaded lazily per-call and cached in _MODEL_CACHE at module level.
        pass

    def predict(
        self,
        metrics: Dict[str, Any],
        report_type: str = "blood",
    ) -> Dict[str, Any]:
        """
        Run the full disease risk prediction pipeline on extracted lab metrics.

        Args:
            metrics     — structured_metrics dict from OCR:
                          {key: {"value": float, "unit": str, "source": str}}
            report_type — selects which models and feature layout to use.
                          Must be in REPORT_MODEL_MAP.

        Returns a dict with:
          risks           — {disease: float}  calibrated probability per disease.
          risk_level      — "low" / "moderate" / "high" / "critical".
          key_factors     — top 5 SHAP drivers for the highest-risk disease.
          recommendations — actionable advice list.
          shap_values     — full SHAP dict for the highest-risk disease, or None.
          model_version   — "neural-ensemble-{report_type}-v1".
          severity        — {disease: int}  predicted severity class per disease.
          ocr_coverage    — {found, missing, coverage_pct} from validation.
          raw_xgb_probas  — pre-ensemble XGBoost scores (for debugging).

        Returns _fallback_response() if the report_type is unknown.
        """

        # ── 1. Load all disease models for this report type ───────────────
        try:
            loaded_models = _load_all_models(report_type)
        except ValueError as exc:
            logger.error(exc)
            return self._fallback_response()

        # ── 2. Validate OCR coverage, then build the feature vector ───────
        # validate_ocr_metrics logs which features were found/missing and
        # returns an unchanged metrics dict + a coverage report.
        # build_feature_vector normalises units and fills missing features
        # with clinically sensible default values.
        metrics, coverage = validate_ocr_metrics(metrics, report_type)
        feature_vector, feature_names = build_feature_vector(metrics, report_type)
        X = np.array(feature_vector, dtype=np.float32).reshape(1, -1)

        # ── 3. Run each individual disease model ──────────────────────────
        disease_labels, raw_probas, severity_classes = _run_individual_models(loaded_models, X)

        logger.info(
            f"[{report_type}] Individual XGBoost risks (1-P(class0)): "
            + ", ".join(
                f"{lbl}={p:.4f}(sev={s})"
                for lbl, p, s in zip(disease_labels, raw_probas, severity_classes)
            )
        )

        # ── 4. Neural ensemble: calibrate combined probabilities ──────────
        final_probas = _apply_ensemble(report_type, raw_probas)

        logger.info(
            f"[{report_type}] Ensemble calibrated risks: "
            + ", ".join(
                f"{lbl}={p:.4f}" for lbl, p in zip(disease_labels, final_probas)
            )
        )

        # ── 5. Build the risks output dict ────────────────────────────────
        risks: Dict[str, float] = {
            label: round(float(p), 4)
            for label, p in zip(disease_labels, final_probas)
        }

        # Overall risk level is determined by the single highest disease probability.
        max_risk   = max(risks.values()) if risks else 0.0
        risk_level = _score_to_level(max_risk)

        # ── 6. SHAP on the disease with the highest calibrated risk ───────
        # Computing SHAP for every model would be expensive.  Only the disease
        # the patient should be most concerned about needs an explanation.
        shap_values: Optional[Dict] = None
        if risks:
            top_disease = max(risks, key=risks.get)
            top_idx     = disease_labels.index(top_disease)
            _, _, top_model = loaded_models[top_idx]
            top_severity    = severity_classes[top_idx]
            if top_model is not None:
                shap_values = _compute_shap(
                    top_model, X, feature_names, severity_class=top_severity
                )

        key_factors = _top_factors(shap_values, feature_names)

        return {
            "risks":           risks,
            "risk_level":      risk_level,
            "key_factors":     key_factors,
            "recommendations": _recommendations(risk_level, risks),
            "shap_values":     shap_values,
            "model_version":   f"neural-ensemble-{report_type}-v1",
            # Per-disease severity class: 0=none, 1=mild, 2=moderate, 3=severe.
            "severity": {
                label: sev
                for label, sev in zip(disease_labels, severity_classes)
            },
            "ocr_coverage":   coverage,
            # Pre-ensemble scores preserved for debugging and ensemble retraining.
            "raw_xgb_probas": {
                label: round(float(p), 4)
                for label, p in zip(disease_labels, raw_probas)
            },
        }

    @staticmethod
    def _fallback_response() -> Dict[str, Any]:
        """
        Return a safe empty-result dict when prediction cannot run.

        Triggered when the report_type is not in REPORT_MODEL_MAP (e.g. a future
        report type that hasn't been registered yet).  Returns risk_level="unknown"
        so the frontend shows a neutral state rather than a misleading low/high risk.
        """
        return {
            "risks":           {},
            "risk_level":      "unknown",
            "key_factors":     [],
            "recommendations": [
                "Model not available. Please consult a healthcare professional."
            ],
            "shap_values":     None,
            "model_version":   "none",
            "raw_xgb_probas":  {},
        }

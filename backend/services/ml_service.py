"""
services/ml_service.py — Report comparison logic.

This module contains ReportComparator, which computes a structured diff
between two reports of the same type.  It is called by the
compare_reports Celery task (task_queue/tasks.py).

Key concepts:
  - "lower is better" metrics (e.g. LDL, glucose): a decrease is an improvement.
  - "higher is better" metrics (e.g. HDL, eGFR): an increase is an improvement.
  - "significant change": absolute percent change > 5% (arbitrary clinical threshold).
  - "trend": overall IMPROVING / WORSENING / STABLE based on improved vs worsened count.

Note on report ordering:
  report_1 is treated as the CURRENT (newer) report.
  report_2 is treated as the BASELINE (older) report.
  diff = r1_val - r2_val  (positive diff means the value went up from baseline to current)
  improved = moved in the clinically beneficial direction.
"""

# Metrics where a LOWER value indicates better health.
# e.g. high LDL increases cardiovascular risk, so a drop is "improved".
_LOWER_IS_BETTER = {
    "glucose", "hba1c", "total_cholesterol", "ldl", "triglycerides",
    "alt", "ast", "bilirubin", "creatinine", "bun", "uric_acid",
    "tsh", "cortisol", "insulin", "ferritin",
}

# Metrics where a HIGHER value indicates better health.
# e.g. low HDL is a cardiovascular risk factor, so an increase is "improved".
_HIGHER_IS_BETTER = {
    "hdl", "egfr", "vitamin_d", "testosterone", "hemoglobin",
    "albumin", "t3", "t4",
}


def _lower_is_better(metric_name: str) -> bool:
    """
    Determine the clinical direction for a metric name.

    The metric name is normalised (lowercased, spaces/hyphens → underscores)
    before lookup so "Total Cholesterol", "total-cholesterol", and
    "total_cholesterol" all map to the same entry.

    Returns:
      True  — lower is better (a decrease from r2 to r1 is an improvement).
      False — higher is better, or neutral / unknown direction.
              Returning False for neutral is a conservative default: the UI
              shows such metrics as neither improved nor worsened.
    """
    name = metric_name.lower().replace(" ", "_").replace("-", "_")
    if name in _LOWER_IS_BETTER:
        return True
    if name in _HIGHER_IS_BETTER:
        return False
    return False   # default: treat as neutral / higher-not-worse


class ReportComparator:
    """
    Compares the extracted lab metrics and disease risk scores from two
    same-type reports and produces a structured diff for the frontend.

    Instantiated fresh by each compare_reports task call — no shared state.
    """

    def compare_medical_reports(self, report_1: dict, report_2: dict) -> dict:
        """
        Compute a full structured comparison between two reports.

        Args:
            report_1: dict returned by _get_report_data() for the newer report.
                      Keys: structured_metrics, report_type, created_at, prediction_risks
            report_2: same structure for the baseline (older) report.

        Algorithm:
          1. Find the intersection of metric keys present in both reports.
             Metrics only in one report are skipped (can't compute a diff).
          2. For each shared metric compute:
               diff    = r1_val - r2_val
               percent = diff / r2_val × 100  (change from baseline)
               improved = did the value move in the clinically beneficial direction?
               is_significant = |percent| > 5%
          3. Count improved / worsened metrics to determine overall trend.
          4. Compute risk score deltas from the ML prediction output.
          5. Return the full comparison dict.

        Returns a dict with keys:
          "metrics"              — list of per-metric comparison dicts
          "significant_changes"  — subset of metrics where |percent| > 5%
          "risk_comparison"      — list of per-disease risk score deltas
          "summary"              — { overall_trend, improved_count, worsened_count,
                                     unchanged_count, total_metrics }
        """
        # Extract the metric dicts.  Use empty dict if the key is missing or None
        # so the intersection below produces an empty set rather than crashing.
        metrics_1 = report_1.get("structured_metrics") or {}
        metrics_2 = report_2.get("structured_metrics") or {}

        # Only compare metrics that appear in BOTH reports.
        # A metric only in report_1 (e.g. a newly ordered test) has no baseline,
        # so we can't compute a meaningful percentage change.
        shared_keys = set(metrics_1.keys()) & set(metrics_2.keys())

        metric_results = []
        for key in sorted(shared_keys):   # sorted for deterministic output order
            try:
                # Metrics can be stored as {"value": float, ...} dicts OR as raw
                # floats (from older pipeline versions).  float() handles both.
                v1 = float(metrics_1[key])
                v2 = float(metrics_2[key])
            except (TypeError, ValueError):
                # Skip if either value is not numeric (e.g. None or a string).
                continue

            diff = round(v1 - v2, 4)

            # Percent change from the baseline (r2).
            # Guard against division by zero when the baseline value is 0.
            pct = round((diff / v2) * 100, 2) if v2 != 0 else 0.0

            lib = _lower_is_better(key)

            # "improved" means the change moved in the beneficial direction:
            #   lower-is-better metrics: improved if v1 < v2 (diff < 0)
            #   higher-is-better metrics: improved if v1 > v2 (diff > 0)
            improved = (diff < 0) if lib else (diff > 0)

            metric_results.append({
                "name":            key,
                "r1_val":          v1,
                "r2_val":          v2,
                "diff":            diff,
                "percent":         pct,
                "lower_is_better": lib,
                "improved":        improved,
                "is_significant":  abs(pct) > 5,   # > 5% change is clinically notable
            })

        # Tally improved vs worsened to determine the overall trend.
        improved_count  = sum(1 for m in metric_results if m["improved"])
        # "worsened" only counts if the change was also significant (> 5%).
        # Small reversals in neutral directions shouldn't count as worsening.
        worsened_count  = sum(
            1 for m in metric_results if not m["improved"] and m["is_significant"]
        )
        unchanged_count = len(metric_results) - improved_count - worsened_count

        # Overall trend: whichever side has more metrics wins; tie → STABLE.
        if improved_count > worsened_count:
            trend = "IMPROVING"
        elif worsened_count > improved_count:
            trend = "WORSENING"
        else:
            trend = "STABLE"

        # Significant changes: metrics worth highlighting in the UI summary.
        significant = [m for m in metric_results if m["is_significant"]]

        # ── Risk score comparison ─────────────────────────────────────────────
        # Compare the ML-predicted disease risk probabilities (0–1) between
        # the two reports.  For disease risk, lower is ALWAYS better.
        risks_1 = report_1.get("prediction_risks", {})
        risks_2 = report_2.get("prediction_risks", {})
        shared_diseases = set(risks_1.keys()) & set(risks_2.keys())

        risk_comparison = []
        for disease in sorted(shared_diseases):
            r1 = round(float(risks_1[disease]), 4)
            r2 = round(float(risks_2[disease]), 4)
            diff = round(r1 - r2, 4)
            improved = diff < 0   # lower disease risk is always better
            risk_comparison.append({
                "disease":  disease,
                "r1_risk":  r1,
                "r2_risk":  r2,
                "diff":     diff,
                "improved": improved,
            })

        return {
            "metrics":             metric_results,
            "significant_changes": significant,
            "risk_comparison":     risk_comparison,
            "summary": {
                "overall_trend":   trend,
                "improved_count":  improved_count,
                "worsened_count":  worsened_count,
                "unchanged_count": unchanged_count,
                "total_metrics":   len(metric_results),
            },
        }

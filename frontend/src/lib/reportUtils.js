/**
 * frontend/src/lib/reportUtils.js — Shared formatting and data-transform utilities.
 *
 * Three exported utilities used throughout the Results and History sections:
 *
 *   REFERENCE_RANGES — display-ready normal range strings per metric key.
 *   riskToStatus()   — converts a 0–1 probability to a status category string.
 *   formatLabel()    — converts snake_case metric keys to Title Case display names.
 *   transformResult()— converts the raw API status response into the UI data shape.
 *
 * Why centralise these here?
 *   Both the Results section (live analysis) and the History section (re-rendering
 *   a past report) need to display risk metrics and biomarkers in the same format.
 *   Keeping the transformation logic here avoids duplication and ensures that
 *   a change to how risk levels map to colours (for example) is made in one place.
 */

/**
 * Normal reference ranges for common lab metrics, keyed by structured_metrics key.
 *
 * These are displayed in the Biomarker card's "Range" field in the UI.
 * Values follow standard adult clinical reference ranges; they are for display
 * only and are NOT used in any risk calculation.
 *
 * Keys match the metric keys produced by the OCR parsers in ocr_runner.py so
 * that REFERENCE_RANGES[key] can be looked up directly from the structured_metrics
 * dict without any translation step.
 */
export const REFERENCE_RANGES = {
    glucose:           '70-100 mg/dL',
    hemoglobin:        '12-17 g/dL',
    hematocrit:        '36-52 %',
    wbc:               '4.5-11 ×10³/µL',
    rbc:               '4.2-5.9 ×10⁶/µL',
    platelets:         '150-400 ×10³/µL',
    creatinine:        '0.6-1.2 mg/dL',
    bun:               '7-20 mg/dL',
    total_cholesterol: '<200 mg/dL',
    hdl:               '>40 mg/dL',
    ldl:               '<100 mg/dL',
    triglycerides:     '<150 mg/dL',
    vldl:              '<30 mg/dL',
    alt:               '7-56 U/L',
    ast:               '10-40 U/L',
    alp:               '44-147 U/L',
    bilirubin_total:   '0.2-1.2 mg/dL',
    albumin:           '3.5-5.0 g/dL',
    vitamin_d:         '20-50 ng/mL',
    tsh:               '0.4-4.0 mIU/L',
    testosterone:      '300-1000 ng/dL',
    egfr:              '>60 mL/min/1.73m²',
};

/**
 * Convert a raw disease probability (0–1) to a status category string.
 *
 * Thresholds mirror RISK_THRESHOLDS in ml_models/model_utils.py — they must
 * stay in sync so the frontend status badge matches the backend risk_level label.
 *
 *   ≥ 0.85 → 'critical'  (seek immediate medical attention)
 *   ≥ 0.65 → 'high'      (schedule a doctor visit soon)
 *   ≥ 0.40 → 'moderate'  (follow up within 3 months)
 *   < 0.40 → 'low'       (maintain current lifestyle)
 *
 * The returned string is used as:
 *   - A CSS class key in Results.jsx (each status maps to a colour palette).
 *   - The label text in the StatusBadge component.
 *   - The `status` field on each riskMetric object returned by transformResult().
 *
 * @param {number} value — Disease probability in [0, 1].
 * @returns {'critical'|'high'|'moderate'|'low'} Status category string.
 */
export function riskToStatus(value) {
    if (value >= 0.85) return 'critical';
    if (value >= 0.65) return 'high';
    if (value >= 0.40) return 'moderate';
    return 'low';
}

/**
 * Format a snake_case metric or disease key as a human-readable Title Case label.
 *
 * Examples:
 *   "heart_disease"          → "Heart Disease"
 *   "vitamin_d_deficiency"   → "Vitamin D Deficiency"
 *   "total_cholesterol"      → "Total Cholesterol"
 *   "egfr"                   → "Egfr"  (single-word keys are just capitalised)
 *
 * This is applied to:
 *   - Disease names in the Risk Cards (from result.risks keys).
 *   - Biomarker names in the Biomarker Cards (from extracted_metrics keys).
 *   - SHAP feature names in the Key Factors section.
 *
 * The replace chain:
 *   1. /_/g → spaces        ("heart_disease" → "heart disease")
 *   2. \b\w → uppercase     first letter of each word → "Heart Disease"
 *
 * @param {string} key — snake_case metric key from the API response.
 * @returns {string} Title Case display string.
 */
export function formatLabel(key) {
    return key.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
}

/**
 * Transform the raw GET /api/status/{report_id} response into the shape
 * expected by the Results section components.
 *
 * The API returns a flat response with `result` (ML prediction), `report_type`,
 * `filename`, and `extracted_metrics`.  The Results section needs two arrays:
 *
 *   riskMetrics — one item per disease, with name, percentage, and status.
 *   biomarkers  — one item per extracted lab metric (up to 4), with value,
 *                 unit, reference range, and status.
 *
 * Why cap biomarkers at 4?
 *   The biomarker card grid shows 4 cards in a 2×2 layout.  The OCR may
 *   extract 5–10 metrics but the UI only has space for 4.  slice(0, 4) takes
 *   the first 4 in insertion order (which is roughly the order the OCR parser
 *   defined them — the most important metrics for that report type).
 *
 * Disease probability conversion:
 *   The backend returns raw probabilities (e.g. 0.12 for 12% diabetes risk).
 *   The UI displays percentages, so Math.round(value * 100) converts 0.12 → 12.
 *
 * Biomarker status:
 *   Currently hardcoded to 'normal' — a future enhancement would compare the
 *   extracted value against REFERENCE_RANGES to determine actual status.
 *
 * @param {Object} statusData — Raw response from GET /api/status/{report_id}.
 *   @param {Object|null} statusData.result          — ML prediction (null until completed).
 *   @param {string}      statusData.report_type     — blood / lipid / ...
 *   @param {string}      statusData.filename        — original uploaded filename.
 *   @param {Object|null} statusData.extracted_metrics — {key: {value, unit, source}}.
 *
 * @returns {Object|null} Transformed data ready for the Results components, or
 *   null if result is not yet available (report still processing).
 *
 *   Return shape:
 *   {
 *     riskMetrics:     Array<{name, value, status, trend}>,
 *     biomarkers:      Array<{name, value, unit, range, status}>,
 *     riskLevel:       string,
 *     recommendations: string[],
 *     keyFactors:      Array<{feature, impact, direction}>,
 *     reportType:      string,
 *     filename:        string,
 *   }
 */
export function transformResult(statusData) {
    const { result, report_type, filename, extracted_metrics } = statusData;

    // Return null if the ML result is not yet available (still processing).
    // The polling loop in Results.jsx checks for null before rendering the dashboard.
    if (!result) return null;

    const { risks = {}, risk_level, key_factors = [], recommendations = [] } = result;

    // Build one riskMetric object per disease in the risks dict.
    // value is converted from probability (0–1) to percentage integer (0–100).
    // trend is hardcoded to 'stable' — the comparison feature sets this properly
    // when comparing two reports, but single-report analysis has no trend.
    const riskMetrics = Object.entries(risks).map(([name, value]) => ({
        name:   formatLabel(name),
        value:  Math.round(value * 100),
        status: riskToStatus(value),
        trend:  'stable',
    }));

    // Build biomarker cards from the extracted lab metrics.
    // Use optional chaining on entry because extracted_metrics may be null
    // during intermediate pipeline stages (status endpoint returns null until
    // the pipeline completes — but transformResult is only called on completion).
    const biomarkers = Object.entries(extracted_metrics || {})
        .slice(0, 4)   // show only the first 4 metrics in the 2×2 card grid
        .map(([key, entry]) => {
            // extracted_metrics values can be dict-style {value, unit, source}
            // or raw floats from older pipeline versions.
            const value = typeof entry === 'object' ? entry?.value : entry;
            const unit  = typeof entry === 'object' ? (entry?.unit ?? '') : '';
            return {
                name:   formatLabel(key),
                value:  value ?? '—',     // '—' em-dash shown when value is null
                unit,
                range:  REFERENCE_RANGES[key] ?? 'N/A',
                status: 'normal',         // TODO: derive from value vs. REFERENCE_RANGES
            };
        });

    return {
        riskMetrics,
        biomarkers,
        riskLevel:       risk_level,
        recommendations,
        keyFactors:      key_factors,
        reportType:      report_type,
        filename,
    };
}

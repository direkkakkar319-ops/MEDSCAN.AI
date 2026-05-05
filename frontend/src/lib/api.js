/**
 * frontend/src/lib/api.js — Authenticated HTTP client for the MEDSCAN API.
 *
 * All API calls in the frontend go through apiFetch() or apiGet() rather than
 * calling fetch() directly.  This gives one central place to:
 *   1. Attach the JWT access token to every request.
 *   2. Handle 401 Unauthorized by silently refreshing the token and retrying.
 *   3. Clear both tokens and force re-login when the refresh token is also expired.
 *
 * Token storage:
 *   access_token  — localStorage, expires in 30 minutes (short-lived).
 *   refresh_token — localStorage, expires in 7 days (long-lived).
 *   Both tokens are written by the login/register flow in AuthModal.jsx.
 *
 * Silent refresh flow (happens transparently to the user):
 *   1. apiFetch() attaches the access_token and makes the request.
 *   2. If the server returns 401 (access_token expired), tryRefresh() runs.
 *   3. tryRefresh() POSTs the refresh_token to /auth/refresh.
 *   4. If successful: stores the new access_token, retries the original request.
 *   5. If the refresh also fails: clears both tokens → user sees the login screen.
 *
 * Why only ONE retry after refresh?
 *   A retry loop would mask real 401 errors (e.g. a route that legitimately
 *   requires a permission the user doesn't have).  One retry is enough to cover
 *   the token expiry case without infinite looping.
 */

/** Base URL for all API calls.
 *  In development: Vite proxies /api → localhost:8000, so API_BASE can be empty.
 *  In production: VITE_API_BASE is set to the Render backend URL in .env.production.
 */
const API_BASE = import.meta.env.VITE_API_BASE || 'https://medscan-api-x9ne.onrender.com';

/**
 * Attempt to refresh the access token using the stored refresh token.
 *
 * Called by apiFetch() when it receives a 401 response.  The function is
 * intentionally not exported — it is an internal implementation detail of the
 * retry flow and should never be called directly by components.
 *
 * Flow:
 *   1. Read refresh_token from localStorage.  If missing, return false immediately
 *      (the user was never logged in or already logged out).
 *   2. POST to /auth/refresh with the refresh_token in the JSON body.
 *   3. On success: store the new access_token.  Also update refresh_token if the
 *      server issued a new one (the backend currently returns the same refresh
 *      token, but this handles a future rotation policy without code changes).
 *   4. On failure (network error OR non-2xx from /auth/refresh): clear both tokens
 *      so the user is treated as logged out.  Return false.
 *
 * @returns {Promise<boolean>} true if a new access_token was obtained, false otherwise.
 */
async function tryRefresh() {
    const refreshToken = localStorage.getItem('refresh_token');
    if (!refreshToken) return false;

    try {
        const res = await fetch(`${API_BASE}/auth/refresh`, {
            method:  'POST',
            headers: { 'Content-Type': 'application/json' },
            body:    JSON.stringify({ refresh_token: refreshToken }),
        });
        if (!res.ok) throw new Error('Refresh failed');

        const data = await res.json();
        localStorage.setItem('access_token', data.access_token);

        // Backend returns the same refresh_token; store it anyway in case a
        // future server update starts issuing rotating refresh tokens.
        if (data.refresh_token) localStorage.setItem('refresh_token', data.refresh_token);

        return true;
    } catch {
        // Any failure (network error, expired refresh token, server error) means
        // the user's session is unrecoverable — clear storage so they see the login.
        localStorage.removeItem('access_token');
        localStorage.removeItem('refresh_token');
        return false;
    }
}

/**
 * Authenticated fetch wrapper — the primary API call helper.
 *
 * Automatically attaches the Bearer token from localStorage to every request.
 * On a 401 response, silently refreshes the token and retries the request once.
 * Returns the raw Response object so callers can inspect status, headers, etc.
 *
 * Usage:
 *   const res = await apiFetch('/api/reports', { method: 'GET' });
 *   const data = await res.json();
 *
 *   // Or with a body:
 *   const res = await apiFetch('/api/compare', {
 *     method: 'POST',
 *     headers: { 'Content-Type': 'application/json' },
 *     body: JSON.stringify({ report1_id: 1, report2_id: 2 }),
 *   });
 *
 * Note on FormData uploads:
 *   Do NOT set 'Content-Type' manually for multipart/form-data — the browser
 *   sets it automatically with the correct boundary string.  Passing it manually
 *   breaks the upload.
 *
 * @param {string}      path — API path, e.g. '/api/reports'.
 * @param {RequestInit} opts — Standard fetch options (method, headers, body, etc.).
 * @returns {Promise<Response>} The raw fetch Response (may be error status).
 */
export async function apiFetch(path, opts = {}) {
    const token = localStorage.getItem('access_token');

    // Spread the caller's headers first, then overlay the Authorization header.
    // If the user is not logged in (no token), the Authorization header is omitted
    // entirely — some public endpoints (e.g. /auth/register) don't require it.
    const headers = {
        ...(opts.headers ?? {}),
        ...(token ? { Authorization: `Bearer ${token}` } : {}),
    };

    let res = await fetch(`${API_BASE}${path}`, { ...opts, headers });

    // Silent token refresh on 401.
    if (res.status === 401) {
        const refreshed = await tryRefresh();
        if (refreshed) {
            // Pick up the new token that tryRefresh() stored in localStorage.
            const newToken = localStorage.getItem('access_token');
            headers.Authorization = `Bearer ${newToken}`;
            // Retry the original request with the fresh token.
            res = await fetch(`${API_BASE}${path}`, { ...opts, headers });
        }
        // If refresh failed, res is still the 401 — the caller receives it and
        // typically shows an "unauthorised" error or redirects to login.
    }

    return res;
}

/**
 * Convenience GET wrapper — makes a GET request and returns parsed JSON.
 *
 * Covers the most common frontend pattern: fetch data and use it directly.
 * For non-GET requests, for responses that need header inspection, or for
 * file uploads, use apiFetch() directly instead.
 *
 * Throws an Error if the response status is not 2xx, with the status code
 * and path in the message to make debugging easier.
 *
 * @param {string} path — API path, e.g. '/api/reports'.
 * @returns {Promise<any>} Parsed JSON response body.
 * @throws {Error} On any non-2xx response status.
 */
export async function apiGet(path) {
    const res = await apiFetch(path);
    if (!res.ok) throw new Error(`API ${res.status}: ${path}`);
    return res.json();
}
